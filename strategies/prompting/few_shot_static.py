# text2sql_mvp/strategies/prompting/few_shot_static.py
"""
Few-shot static Text-to-SQL generation strategy.
Uses a fixed set of examples selected at initialization time.
"""

import time
from pathlib import Path
from typing import Any, Optional

from core.gemini_client import GeminiClient
from core.sql_parser import SQLParser
from core.prompt_builder import PromptBuilder, SchemaFormat
from data.few_shot_pool import FewShotPool, FewShotExample
from schema.formatter import SchemaFormatter
from strategies.base import (
    BaseStrategy,
    GenerationResult,
    GenerationContext,
    StrategyType,
)
from strategies.registry import register_strategy


@register_strategy
class FewShotStaticStrategy(BaseStrategy):
    """
    Few-shot SQL generation with static (fixed) examples.
    
    Uses a predetermined set of examples that don't change based on
    the input question. Good for consistent behavior and when you have
    carefully curated examples.
    
    Expected accuracy: 60-70% on typical benchmarks.
    
    Example:
        ```python
        strategy = FewShotStaticStrategy(
            gemini_client,
            examples_path="examples.json",
            num_examples=3
        )
        result = await strategy.generate(question, context)
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        examples: Optional[list[FewShotExample]] = None,
        examples_path: Optional[str | Path] = None,
        num_examples: int = 3,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
        include_explanations: bool = False,
        example_selection: str = "diverse",  # "diverse", "random", "first"
    ):
        """
        Initialize few-shot static strategy.
        
        Args:
            llm_client: Gemini client instance
            examples: Pre-loaded examples (takes priority)
            examples_path: Path to JSON file with examples
            num_examples: Number of examples to include
            schema_format: Format for schema representation
            temperature: Generation temperature
            include_explanations: Include example explanations
            example_selection: How to select examples ("diverse", "random", "first")
        """
        self._llm = llm_client
        self._num_examples = num_examples
        self._schema_format = schema_format
        self._temperature = temperature
        self._include_explanations = include_explanations
        self._example_selection = example_selection
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
        
        # Load examples
        self._pool = FewShotPool()
        if examples:
            self._pool.add_examples(examples)
        elif examples_path:
            self._pool.load_from_json(Path(examples_path))
        
        # Pre-select static examples
        self._static_examples: list[FewShotExample] = []
        self._select_static_examples()
    
    @property
    def name(self) -> str:
        return "few_shot_static"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return f"Few-shot with {self._num_examples} static examples"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _select_static_examples(self) -> None:
        """Select static examples based on selection strategy."""
        if self._pool.size == 0:
            self._static_examples = []
            return
        
        if self._example_selection == "diverse":
            # Select diverse examples by difficulty
            self._static_examples = self._pool.get_diverse_examples(
                n=self._num_examples,
                by="difficulty"
            )
        elif self._example_selection == "random":
            self._static_examples = self._pool.get_random_examples(
                n=self._num_examples
            )
        else:  # "first"
            self._static_examples = self._pool.get_examples(
                n=self._num_examples,
                random_order=False
            )
    
    def set_examples(self, examples: list[FewShotExample]) -> None:
        """
        Set new static examples.
        
        Args:
            examples: New examples to use
        """
        self._pool = FewShotPool()
        self._pool.add_examples(examples)
        self._select_static_examples()
    
    def load_examples(self, path: str | Path) -> None:
        """
        Load examples from file.
        
        Args:
            path: Path to examples JSON file
        """
        self._pool.load_from_json(Path(path))
        self._select_static_examples()
    
    def _format_examples(self) -> str:
        """Format examples for prompt."""
        if not self._static_examples:
            return ""
        
        formatted = []
        for i, example in enumerate(self._static_examples, 1):
            example_text = f"Example {i}:\n"
            example_text += f"Question: {example.question}\n"
            if self._include_explanations and example.explanation:
                example_text += f"Reasoning: {example.explanation}\n"
            example_text += f"SQL: {example.sql}"
            formatted.append(example_text)
        
        return "\n\n".join(formatted)
    
    def _build_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> tuple[str, str]:
        """Build the few-shot prompt."""
        # Format schema
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        # System prompt
        system_prompt = """You are an expert SQL developer. Your task is to convert natural language questions into accurate SQL queries.

Study the examples provided carefully, then generate a SQL query for the given question.

Guidelines:
- Use the exact table and column names from the schema
- Follow the patterns shown in the examples
- Generate only the SQL query without explanations
- Handle NULL values appropriately
- Use proper JOIN conditions"""
        
        # Build user prompt
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        # Add examples
        examples_text = self._format_examples()
        if examples_text:
            user_parts.extend([
                "",
                "## Examples",
                examples_text,
            ])
        
        # Add hints if provided
        if context.hints:
            user_parts.extend([
                "",
                "## Hints",
                *[f"- {hint}" for hint in context.hints],
            ])
        
        # Add question
        user_parts.extend([
            "",
            "## Your Task",
            f"Question: {question}",
            "",
            "SQL:",
        ])
        
        user_prompt = "\n".join(user_parts)
        
        return system_prompt, user_prompt
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using few-shot static prompting.
        
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
                "num_examples": len(self._static_examples),
                "example_selection": self._example_selection,
                "include_explanations": self._include_explanations,
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
            "num_examples": self._num_examples,
            "actual_examples": len(self._static_examples),
            "example_selection": self._example_selection,
            "include_explanations": self._include_explanations,
            "pool_size": self._pool.size,
        }
