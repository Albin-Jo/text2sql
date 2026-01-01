# text2sql_mvp/strategies/prompting/zero_shot.py
"""
Zero-shot Text-to-SQL generation strategy.
Generates SQL using only the schema and question, without examples.
"""

import time
from typing import Any, Optional

from core.gemini_client import GeminiClient
from core.sql_parser import SQLParser
from core.prompt_builder import PromptBuilder, SchemaFormat
from schema.formatter import SchemaFormatter
from strategies.base import (
    BaseStrategy,
    GenerationResult,
    GenerationContext,
    StrategyType,
)
from strategies.registry import register_strategy


@register_strategy
class ZeroShotStrategy(BaseStrategy):
    """
    Zero-shot SQL generation strategy.
    
    Uses only the database schema and natural language question
    to generate SQL without any examples. This is the simplest
    baseline strategy.
    
    Expected accuracy: 50-60% on typical benchmarks.
    
    Example:
        ```python
        strategy = ZeroShotStrategy(gemini_client)
        result = await strategy.generate(
            question="How many customers are there?",
            context=GenerationContext(schema=schema, question=question)
        )
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
    ):
        """
        Initialize zero-shot strategy.
        
        Args:
            llm_client: Gemini client instance (created if not provided)
            schema_format: Format for schema representation
            temperature: Generation temperature (0.0 = deterministic)
        """
        self._llm = llm_client
        self._schema_format = schema_format
        self._temperature = temperature
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "zero_shot"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return "Basic zero-shot SQL generation using schema and question only"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _build_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> tuple[str, str]:
        """
        Build the zero-shot prompt.
        
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Format schema
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        # Build prompt
        builder = PromptBuilder()
        builder.use_system_prompt("default")
        builder.set_schema(schema_text, format=self._schema_format)
        builder.set_question(question)
        builder.set_output_format(
            "Generate only the SQL query. Do not include explanations or markdown formatting."
        )
        
        # Add hints if provided
        if context.hints:
            builder.add_hints(context.hints)
        
        return builder.build()
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL from question using zero-shot prompting.
        
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
        
        # Generate
        client = self._ensure_client()
        response = await client.generate(
            prompt=full_prompt,
            temperature=self._temperature,
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
        }
