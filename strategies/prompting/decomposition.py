# text2sql_mvp/strategies/prompting/decomposition.py
"""
Decomposition Text-to-SQL generation strategy.
Breaks complex questions into sub-questions before generating SQL.
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
class DecompositionStrategy(BaseStrategy):
    """
    Decomposition-based SQL generation strategy.
    
    Breaks complex questions into simpler sub-questions, then
    combines the parts into a complete SQL query. Particularly
    effective for multi-step questions and complex aggregations.
    
    This strategy uses a two-phase approach:
    1. Decompose the question into logical sub-parts
    2. Generate SQL that addresses all sub-parts
    
    Expected accuracy: 65-75% on typical benchmarks, with stronger
    performance on complex multi-part questions.
    
    Example:
        ```python
        strategy = DecompositionStrategy(gemini_client)
        result = await strategy.generate(
            "Find customers who spent more than the average and list their recent transactions",
            context
        )
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
        two_phase: bool = True,  # If True, decompose first, then generate
        max_subquestions: int = 5,
    ):
        """
        Initialize decomposition strategy.
        
        Args:
            llm_client: Gemini client instance
            schema_format: Format for schema representation
            temperature: Generation temperature
            two_phase: Whether to use two separate LLM calls
            max_subquestions: Maximum number of sub-questions
        """
        self._llm = llm_client
        self._schema_format = schema_format
        self._temperature = temperature
        self._two_phase = two_phase
        self._max_subquestions = max_subquestions
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "decomposition"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return "Question decomposition for complex queries"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _build_decomposition_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> str:
        """Build prompt for decomposing the question."""
        schema_text = self._formatter.format(
            context.schema,
            format_type=SchemaFormat.COMPACT,  # Use compact for decomposition
            include_descriptions=False,
        )
        
        return f"""You are a SQL query planning expert. Your task is to decompose a complex natural language question into simpler sub-questions that can be combined into a single SQL query.

## Database Tables
{schema_text}

## Question
{question}

## Task
Break down this question into {self._max_subquestions} or fewer logical sub-questions. Each sub-question should:
1. Be answerable with a simple SQL operation
2. Build toward the final answer
3. Reference specific tables and columns from the schema

Format your response as:
Sub-question 1: [What data do we need first?]
Sub-question 2: [What conditions or filters apply?]
Sub-question 3: [What aggregations or calculations are needed?]
...

Then explain how these combine into the final query structure.

## Decomposition:"""
    
    def _build_generation_prompt(
        self,
        question: str,
        context: GenerationContext,
        decomposition: Optional[str] = None,
    ) -> tuple[str, str]:
        """Build prompt for SQL generation after decomposition."""
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        system_prompt = """You are an expert SQL developer. Generate a SQL query that answers all parts of the decomposed question.

Guidelines:
- Address each sub-question in your SQL
- Use subqueries, CTEs, or JOINs as needed to combine parts
- Ensure the final query is syntactically correct
- Handle NULL values appropriately
- Output only the SQL query"""
        
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        # Add decomposition if available
        if decomposition:
            user_parts.extend([
                "",
                "## Question Breakdown",
                decomposition,
            ])
        
        # Add original question
        user_parts.extend([
            "",
            "## Original Question",
            question,
        ])
        
        # Add hints
        if context.hints:
            user_parts.extend([
                "",
                "## Hints",
                *[f"- {hint}" for hint in context.hints],
            ])
        
        user_parts.extend([
            "",
            "## SQL Query (addressing all sub-questions):",
        ])
        
        return system_prompt, "\n".join(user_parts)
    
    def _build_single_phase_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> tuple[str, str]:
        """Build prompt for single-phase decomposition and generation."""
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        system_prompt = """You are an expert SQL developer who breaks down complex questions methodically.

Your approach:
1. First, decompose the question into logical sub-parts
2. Plan how each part maps to SQL operations
3. Generate a complete SQL query that addresses all parts

This systematic approach helps with:
- Multi-condition queries
- Queries requiring subqueries or CTEs
- Complex aggregations
- Multi-table joins with conditions"""
        
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        # Add relationships
        relationships = self._formatter.get_relationships_summary(context.schema)
        if relationships and "No foreign key" not in relationships:
            user_parts.extend([
                "",
                "## Relationships",
                relationships,
            ])
        
        # Add hints
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
            "## Decomposition",
            "Break this question into sub-parts:",
            "",
            f"1. [First sub-question about data needed]",
            f"2. [Second sub-question about conditions/filters]",
            f"3. [Third sub-question about output/aggregation]",
            "(Add more if needed, max {self._max_subquestions})",
            "",
            "## Query Plan",
            "How the sub-parts combine (JOINs, subqueries, CTEs, etc.):",
            "",
            "## Final SQL Query:",
        ])
        
        return system_prompt, "\n".join(user_parts)
    
    async def _decompose_question(
        self,
        question: str,
        context: GenerationContext,
    ) -> Optional[str]:
        """
        Decompose question into sub-questions.
        
        Returns:
            Decomposition text or None on failure
        """
        prompt = self._build_decomposition_prompt(question, context)
        
        client = self._ensure_client()
        response = await client.generate(
            prompt=prompt,
            temperature=0.0,  # Deterministic for decomposition
            max_tokens=1024,
        )
        
        if response.success:
            return response.text.strip()
        return None
    
    def _parse_decomposition(self, text: str) -> list[str]:
        """Parse sub-questions from decomposition text."""
        subquestions = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered sub-questions
            if line and (
                line.startswith("Sub-question") or
                line.startswith("1.") or line.startswith("2.") or
                line.startswith("3.") or line.startswith("4.") or
                line.startswith("5.")
            ):
                # Extract the question part
                if ':' in line:
                    subquestions.append(line.split(':', 1)[1].strip())
                elif '.' in line[:3]:
                    subquestions.append(line.split('.', 1)[1].strip())
        
        return subquestions[:self._max_subquestions]
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using decomposition strategy.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        
        client = self._ensure_client()
        decomposition = None
        subquestions = []
        
        if self._two_phase:
            # Phase 1: Decompose
            decomposition = await self._decompose_question(question, context)
            if decomposition:
                subquestions = self._parse_decomposition(decomposition)
            
            # Phase 2: Generate with decomposition
            system_prompt, user_prompt = self._build_generation_prompt(
                question, context, decomposition
            )
        else:
            # Single phase: decompose and generate together
            system_prompt, user_prompt = self._build_single_phase_prompt(
                question, context
            )
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Generate SQL
        response = await client.generate(
            prompt=full_prompt,
            temperature=self._temperature,
            max_tokens=4096,
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
        
        # Parse SQL from response
        parse_result = self._parser.parse(response.text)
        
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
                "two_phase": self._two_phase,
                "decomposition": decomposition,
                "subquestions": subquestions,
                "num_subquestions": len(subquestions),
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
            "two_phase": self._two_phase,
            "max_subquestions": self._max_subquestions,
        }
