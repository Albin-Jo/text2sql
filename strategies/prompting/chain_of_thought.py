# text2sql_mvp/strategies/prompting/chain_of_thought.py
"""
Chain-of-thought Text-to-SQL generation strategy.
Guides the LLM through step-by-step reasoning before generating SQL.
"""

import time
from typing import Any, Optional

from core.gemini_client import GeminiClient
from core.sql_parser import SQLParser
from core.prompt_builder import SchemaFormat
from schema.formatter import SchemaFormatter
from strategies.base import (
    BaseStrategy,
    GenerationResult,
    GenerationContext,
    StrategyType,
)
from strategies.registry import register_strategy


# Chain-of-thought reasoning templates
COT_TEMPLATES = {
    "standard": """Let me think through this step by step:

1. **Understanding the Question**: What information is being requested?
2. **Identifying Tables**: Which tables contain the needed data?
3. **Selecting Columns**: What columns should appear in the output?
4. **Join Conditions**: How are the tables related?
5. **Filtering**: What WHERE conditions apply?
6. **Aggregation**: Is GROUP BY needed? What aggregate functions?
7. **Ordering**: Should results be sorted?
8. **Limiting**: Is there a limit on results?

Now, based on this analysis:""",

    "execution_plan": """I'll approach this like a database would execute the query:

1. **FROM/JOIN Analysis**: Identify source tables and their relationships
2. **WHERE Filtering**: Determine which rows to include
3. **GROUP BY Aggregation**: If grouping is needed, identify the groups
4. **HAVING Filtering**: Filter after aggregation if needed
5. **SELECT Projection**: Choose output columns
6. **ORDER BY Sorting**: Determine sort order
7. **LIMIT/OFFSET**: Apply result limiting

Based on this execution plan:""",

    "minimal": """Let me analyze:
- Tables needed: 
- Join conditions:
- Filters:
- Output columns:

SQL:""",

    "detailed": """## Step-by-Step Analysis

### Step 1: Parse the Question
What is being asked? What are the key entities and conditions?

### Step 2: Identify Required Tables
Which tables contain the data we need? Check the schema for relevant tables.

### Step 3: Determine Relationships
How are these tables connected? What foreign keys exist?

### Step 4: Plan the SELECT Clause
What columns need to be in the output? Are any calculations needed?

### Step 5: Plan JOIN Conditions
What JOIN type is appropriate? What are the ON conditions?

### Step 6: Plan WHERE Filters
What conditions limit the data? Are there date ranges, status filters, etc.?

### Step 7: Check for Aggregation
Does the question ask for totals, averages, counts? If so, what GROUP BY?

### Step 8: Plan Ordering
Should results be sorted? By what column(s)?

### Step 9: Consider Edge Cases
- NULL values handling
- Division by zero
- Empty results

### Final SQL Query:""",
}


@register_strategy
class ChainOfThoughtStrategy(BaseStrategy):
    """
    Chain-of-thought SQL generation strategy.
    
    Prompts the LLM to reason through the problem step-by-step
    before generating SQL. This improves accuracy on complex queries
    by breaking down the problem.
    
    Expected accuracy: 65-75% on typical benchmarks, with stronger
    performance on complex multi-step queries.
    
    Example:
        ```python
        strategy = ChainOfThoughtStrategy(
            gemini_client,
            cot_style="execution_plan"
        )
        result = await strategy.generate(question, context)
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
        cot_style: str = "standard",  # "standard", "execution_plan", "minimal", "detailed"
        extract_reasoning: bool = True,
    ):
        """
        Initialize chain-of-thought strategy.
        
        Args:
            llm_client: Gemini client instance
            schema_format: Format for schema representation
            temperature: Generation temperature
            cot_style: Style of chain-of-thought prompting
            extract_reasoning: Whether to extract reasoning from response
        """
        self._llm = llm_client
        self._schema_format = schema_format
        self._temperature = temperature
        self._cot_style = cot_style
        self._extract_reasoning = extract_reasoning
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "chain_of_thought"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return f"Chain-of-thought reasoning ({self._cot_style} style)"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _get_cot_template(self) -> str:
        """Get the CoT template for the selected style."""
        return COT_TEMPLATES.get(self._cot_style, COT_TEMPLATES["standard"])
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for CoT."""
        return """You are an expert SQL developer who thinks through problems methodically.

When given a natural language question, you:
1. First analyze the question and schema carefully
2. Reason through the solution step by step
3. Then generate the correct SQL query

Your reasoning helps ensure accuracy, especially for complex queries involving:
- Multiple table joins
- Subqueries
- Aggregations with conditions
- Window functions

Always show your reasoning, then provide the final SQL query."""
    
    def _build_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> tuple[str, str]:
        """Build the chain-of-thought prompt."""
        # Format schema
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        # Get CoT template
        cot_template = self._get_cot_template()
        
        # Build user prompt
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        # Add relationships summary
        relationships = self._formatter.get_relationships_summary(context.schema)
        if relationships and "No foreign key" not in relationships:
            user_parts.extend([
                "",
                "## Table Relationships",
                relationships,
            ])
        
        # Add column descriptions if available
        if context.column_descriptions:
            user_parts.extend([
                "",
                "## Column Descriptions",
                context.column_descriptions,
            ])
        
        # Add hints if provided
        if context.hints:
            user_parts.extend([
                "",
                "## Hints",
                *[f"- {hint}" for hint in context.hints],
            ])
        
        # Add question and CoT template
        user_parts.extend([
            "",
            "## Question",
            question,
            "",
            "## Your Analysis",
            cot_template,
        ])
        
        system_prompt = self._build_system_prompt()
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    def _extract_sql_and_reasoning(
        self,
        response_text: str
    ) -> tuple[str, Optional[str]]:
        """
        Extract SQL and reasoning from response.
        
        Returns:
            Tuple of (sql, reasoning)
        """
        # First try to extract SQL using parser
        parse_result = self._parser.parse(response_text)
        sql = parse_result.sql if parse_result.is_valid else ""
        
        # Extract reasoning (everything before SQL)
        reasoning = None
        if self._extract_reasoning and sql:
            # Find where SQL starts in the response
            sql_lower = sql.lower().strip()
            response_lower = response_text.lower()
            
            # Look for common SQL start patterns
            for pattern in ["select ", "with "]:
                idx = response_lower.find(pattern)
                if idx > 0:
                    reasoning = response_text[:idx].strip()
                    # Clean up reasoning
                    if reasoning.endswith(":"):
                        reasoning = reasoning[:-1].strip()
                    break
        
        return sql, reasoning
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using chain-of-thought reasoning.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        
        # Build prompt
        system_prompt, user_prompt = self._build_prompt(question, context)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate with slightly higher max tokens to allow for reasoning
        client = self._ensure_client()
        response = await client.generate(
            prompt=full_prompt,
            temperature=self._temperature,
            max_tokens=4096,  # Allow space for reasoning
        )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Handle failure
        if not response.success:
            return GenerationResult(
                sql=None,
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                error=response.error,
                error_type=response.error_type.value if response.error_type else None,
                prompt_used=full_prompt,
                model=response.metrics.model,
            )
        
        # Extract SQL and reasoning
        sql, reasoning = self._extract_sql_and_reasoning(response.text)
        
        # Validate SQL
        if sql:
            parse_result = self._parser.parse(sql, extract_first=False)
        else:
            parse_result = self._parser.parse(response.text)
            sql = parse_result.sql
        
        if not parse_result.is_valid:
            return GenerationResult(
                sql=response.text.strip(),
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                prompt_tokens=response.metrics.prompt_tokens,
                completion_tokens=response.metrics.completion_tokens,
                total_tokens=response.metrics.total_tokens,
                temperature=self._temperature,
                model=response.metrics.model,
                error=f"SQL parse error: {parse_result.error}",
                error_type="parse_error",
                raw_response=response.text,
                prompt_used=full_prompt,
            )
        
        return GenerationResult(
            sql=parse_result.sql,
            success=True,
            strategy_name=self.name,
            strategy_type=self.strategy_type,
            latency_ms=latency_ms,
            prompt_tokens=response.metrics.prompt_tokens,
            completion_tokens=response.metrics.completion_tokens,
            total_tokens=response.metrics.total_tokens,
            temperature=self._temperature,
            model=response.metrics.model,
            raw_response=response.text,
            prompt_used=full_prompt,
            metadata={
                "schema_format": self._schema_format.value,
                "cot_style": self._cot_style,
                "reasoning": reasoning,
                "reasoning_length": len(reasoning) if reasoning else 0,
                "complexity_score": parse_result.complexity_score,
                "tables_used": parse_result.tables,
            },
        )
    
    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "version": self.version,
            "schema_format": self._schema_format.value,
            "temperature": self._temperature,
            "cot_style": self._cot_style,
            "extract_reasoning": self._extract_reasoning,
        }
