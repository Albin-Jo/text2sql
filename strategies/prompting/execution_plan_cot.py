# text2sql_mvp/strategies/prompting/execution_plan_cot.py
"""
Execution Plan Chain-of-Thought strategy for SQL generation.

Uses database execution plan reasoning to guide SQL generation.
The model thinks through how a query would be executed step-by-step,
which helps with complex JOINs, aggregations, and filtering logic.
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


@register_strategy
class ExecutionPlanCoTStrategy(BaseStrategy):
    """
    SQL generation using execution plan reasoning.
    
    This strategy prompts the model to think through query execution:
    1. Identify source tables
    2. Plan table joins and order
    3. Determine filter conditions
    4. Plan aggregations and groupings
    5. Generate the final SQL
    
    This approach is particularly effective for:
    - Multi-table queries requiring correct JOIN order
    - Complex aggregations with filtering
    - Queries with subqueries or CTEs
    
    Example:
        ```python
        strategy = ExecutionPlanCoTStrategy(llm_client=client)
        result = await strategy.generate(
            "Find customers who spent more than the average customer",
            context
        )
        # result.metadata["execution_plan"] contains the reasoning
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
        include_plan_in_output: bool = True,
        two_pass: bool = False,  # Generate plan first, then SQL
    ):
        """
        Initialize execution plan CoT strategy.
        
        Args:
            llm_client: Gemini client instance
            schema_format: Schema format to use
            temperature: Generation temperature
            include_plan_in_output: Include reasoning in metadata
            two_pass: Generate plan and SQL in separate calls
        """
        self._llm = llm_client
        self._schema_format = schema_format
        self._temperature = temperature
        self._include_plan_in_output = include_plan_in_output
        self._two_pass = two_pass
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "execution_plan_cot"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return "Chain-of-thought using database execution plan reasoning"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _build_single_pass_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> tuple[str, str]:
        """Build prompt for single-pass generation with reasoning."""
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        # Get foreign key relationships for reference
        relationships = context.schema.get_relationships()
        fk_text = ""
        if relationships:
            fk_lines = [
                f"- {r[0]}.{r[1]} -> {r[2]}.{r[3]}"
                for r in relationships
            ]
            fk_text = "\n\nForeign Key Relationships:\n" + "\n".join(fk_lines)
        
        system_prompt = """You are an expert SQL developer who thinks through queries systematically.

For each question, you will:
1. Analyze which tables are needed
2. Plan the execution path (JOINs, filters, aggregations)
3. Generate the optimal SQL query

Think like a query optimizer - consider the most efficient way to retrieve the data."""
        
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        if fk_text:
            user_parts.append(fk_text)
        
        if context.hints:
            user_parts.extend([
                "",
                "## Hints",
                *[f"- {hint}" for hint in context.hints],
            ])
        
        user_parts.extend([
            "",
            "## Question",
            question,
            "",
            """## Query Execution Plan

Think through the query step by step:

### Step 1: Data Sources
What tables contain the data we need? List them and their relevant columns.

### Step 2: Join Strategy
How should tables be connected? In what order? What are the join conditions?

### Step 3: Filtering
What WHERE conditions are needed? Any HAVING clauses for post-aggregation filtering?

### Step 4: Aggregation & Grouping
Do we need GROUP BY? What aggregations (COUNT, SUM, AVG, etc.)?

### Step 5: Final Output
What columns should be in SELECT? Any ordering or limits?

### Step 6: SQL Query
Based on the above plan, write the final SQL query.

Now analyze and generate:""",
        ])
        
        return system_prompt, "\n".join(user_parts)
    
    async def _generate_plan(
        self,
        question: str,
        context: GenerationContext,
        client: GeminiClient,
    ) -> str:
        """Generate execution plan only (first pass)."""
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
        )
        
        prompt = f"""You are a database query planner. Create an execution plan for the following query.

Database Schema:
{schema_text}

Question: {question}

Create a detailed execution plan:

1. **Data Sources**: Which tables are needed and why?
2. **Join Strategy**: How to connect tables (include join conditions)?
3. **Filtering**: What conditions filter the data?
4. **Aggregation**: Any grouping or aggregation needed?
5. **Output**: What columns in the final result?

Execution Plan:"""
        
        response = await client.generate(prompt, temperature=0.0)
        return response.text if response.success else ""
    
    async def _generate_sql_from_plan(
        self,
        question: str,
        context: GenerationContext,
        execution_plan: str,
        client: GeminiClient,
    ) -> str:
        """Generate SQL from execution plan (second pass)."""
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
        )
        
        prompt = f"""You are an expert SQL developer. Convert this execution plan into a SQL query.

Database Schema:
{schema_text}

Original Question: {question}

Execution Plan:
{execution_plan}

Based on this plan, generate the SQL query. Output ONLY the SQL, nothing else.

SQL:"""
        
        response = await client.generate(prompt, temperature=self._temperature)
        return response.text if response.success else ""
    
    def _extract_sql_and_plan(self, response_text: str) -> tuple[str, str]:
        """Extract SQL and execution plan from response."""
        text = response_text.strip()
        
        # Find SQL in response
        # Look for code blocks first
        import re
        
        # Try to find SQL code block
        sql_block_match = re.search(r'```sql\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if sql_block_match:
            sql = sql_block_match.group(1).strip()
            # Everything before the SQL block is the plan
            plan = text[:sql_block_match.start()].strip()
            return sql, plan
        
        # Try generic code block
        code_block_match = re.search(r'```\s*(SELECT.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if code_block_match:
            sql = code_block_match.group(1).strip()
            plan = text[:code_block_match.start()].strip()
            return sql, plan
        
        # Look for "SQL:" or "SQL Query:" marker
        sql_marker_match = re.search(
            r'(?:SQL Query|SQL|Final SQL|Query)[:\s]*\n*(SELECT\s+.*)',
            text,
            re.DOTALL | re.IGNORECASE
        )
        if sql_marker_match:
            sql = sql_marker_match.group(1).strip()
            plan = text[:sql_marker_match.start()].strip()
            return sql, plan
        
        # Last resort: find SELECT statement
        select_match = re.search(r'(SELECT\s+.*)', text, re.DOTALL | re.IGNORECASE)
        if select_match:
            sql = select_match.group(1).strip()
            plan = text[:select_match.start()].strip()
            return sql, plan
        
        # If no SQL found, return empty
        return "", text
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using execution plan reasoning.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        client = self._ensure_client()
        
        execution_plan = ""
        
        if self._two_pass:
            # Two-pass: generate plan first, then SQL
            execution_plan = await self._generate_plan(question, context, client)
            
            if execution_plan:
                sql_text = await self._generate_sql_from_plan(
                    question, context, execution_plan, client
                )
            else:
                sql_text = ""
        else:
            # Single pass: generate plan and SQL together
            system_prompt, user_prompt = self._build_single_pass_prompt(question, context)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            response = await client.generate(
                prompt=full_prompt,
                temperature=self._temperature,
            )
            
            if not response.success:
                latency_ms = (time.perf_counter() - start_time) * 1000
                return GenerationResult(
                    sql=None,
                    success=False,
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    latency_ms=latency_ms,
                    error=response.error,
                    error_type=response.error_type.value if response.error_type else None,
                    model=response.metrics.model,
                )
            
            sql_text, execution_plan = self._extract_sql_and_plan(response.text)
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Parse SQL
        if sql_text:
            parse_result = self._parser.parse(sql_text)
        else:
            parse_result = None
        
        if parse_result and parse_result.is_valid:
            return GenerationResult(
                sql=parse_result.sql,
                success=True,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                temperature=self._temperature,
                model=client.model,
                metadata={
                    "execution_plan": execution_plan if self._include_plan_in_output else None,
                    "two_pass": self._two_pass,
                    "complexity_score": parse_result.complexity_score,
                    "tables_used": parse_result.tables,
                },
            )
        else:
            return GenerationResult(
                sql=sql_text if sql_text else None,
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                temperature=self._temperature,
                model=client.model,
                error=f"Failed to extract valid SQL from response",
                metadata={
                    "execution_plan": execution_plan if self._include_plan_in_output else None,
                    "two_pass": self._two_pass,
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
            "include_plan_in_output": self._include_plan_in_output,
            "two_pass": self._two_pass,
        }
