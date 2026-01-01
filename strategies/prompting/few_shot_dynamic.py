# text2sql_mvp/strategies/prompting/few_shot_dynamic.py
"""
Few-shot dynamic Text-to-SQL generation strategy.
Selects examples based on semantic similarity to the input question.
"""

import time
from pathlib import Path
from typing import Any, Optional

from core.gemini_client import GeminiClient
from core.sql_parser import SQLParser
from core.prompt_builder import SchemaFormat
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
class FewShotDynamicStrategy(BaseStrategy):
    """
    Few-shot SQL generation with dynamic example selection.
    
    Selects the most relevant examples for each question using
    semantic similarity. Supports multiple selection strategies:
    - similarity: Select most similar examples
    - mmr: Maximum Marginal Relevance for diversity
    - table_match: Select examples using similar tables
    
    Expected accuracy: 65-75% on typical benchmarks.
    
    Example:
        ```python
        strategy = FewShotDynamicStrategy(
            gemini_client,
            examples_path="examples.json",
            selection_method="mmr"
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
        selection_method: str = "similarity",  # "similarity", "mmr", "table_match"
        mmr_lambda: float = 0.5,
        min_similarity: float = 0.0,
        include_explanations: bool = False,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize few-shot dynamic strategy.
        
        Args:
            llm_client: Gemini client instance
            examples: Pre-loaded examples
            examples_path: Path to JSON file with examples
            num_examples: Number of examples to include
            schema_format: Format for schema representation
            temperature: Generation temperature
            selection_method: How to select examples
            mmr_lambda: Lambda for MMR (balance relevance vs diversity)
            min_similarity: Minimum similarity threshold
            include_explanations: Include example explanations
            embedding_model: Sentence transformer model name
        """
        self._llm = llm_client
        self._num_examples = num_examples
        self._schema_format = schema_format
        self._temperature = temperature
        self._selection_method = selection_method
        self._mmr_lambda = mmr_lambda
        self._min_similarity = min_similarity
        self._include_explanations = include_explanations
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
        
        # Load examples
        self._pool = FewShotPool(embedding_model=embedding_model)
        if examples:
            self._pool.add_examples(examples)
        elif examples_path:
            self._pool.load_from_json(Path(examples_path))
    
    @property
    def name(self) -> str:
        return "few_shot_dynamic"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return f"Few-shot with {self._selection_method}-based example selection"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def set_examples(self, examples: list[FewShotExample]) -> None:
        """Set new examples for the pool."""
        self._pool = FewShotPool()
        self._pool.add_examples(examples)
    
    def load_examples(self, path: str | Path) -> None:
        """Load examples from file."""
        self._pool.load_from_json(Path(path))
    
    def _select_examples(
        self,
        question: str,
        context: GenerationContext,
    ) -> list[FewShotExample]:
        """
        Select relevant examples for the question.
        
        Args:
            question: The input question
            context: Generation context (for table info)
            
        Returns:
            Selected examples
        """
        if self._pool.size == 0:
            return []
        
        if self._selection_method == "similarity":
            try:
                return self._pool.get_similar_examples(
                    question=question,
                    n=self._num_examples,
                    min_similarity=self._min_similarity,
                )
            except ImportError:
                # Fall back to random if embeddings unavailable
                return self._pool.get_random_examples(n=self._num_examples)
        
        elif self._selection_method == "mmr":
            try:
                return self._pool.get_mmr_examples(
                    question=question,
                    n=self._num_examples,
                    lambda_param=self._mmr_lambda,
                )
            except ImportError:
                return self._pool.get_random_examples(n=self._num_examples)
        
        elif self._selection_method == "table_match":
            # Select based on matching tables
            tables = context.schema.table_names()
            examples = self._pool.get_by_tables(tables, n=self._num_examples)
            
            # Fill remaining with random if needed
            if len(examples) < self._num_examples:
                remaining = self._pool.get_random_examples(
                    n=self._num_examples - len(examples)
                )
                # Avoid duplicates
                example_set = {ex.question for ex in examples}
                for ex in remaining:
                    if ex.question not in example_set:
                        examples.append(ex)
                        if len(examples) >= self._num_examples:
                            break
            
            return examples[:self._num_examples]
        
        else:
            # Default to random
            return self._pool.get_random_examples(n=self._num_examples)
    
    def _format_examples(self, examples: list[FewShotExample]) -> str:
        """Format examples for prompt."""
        if not examples:
            return ""
        
        formatted = []
        for i, example in enumerate(examples, 1):
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
        examples: list[FewShotExample],
    ) -> tuple[str, str]:
        """Build the few-shot prompt with selected examples."""
        # Format schema
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        # System prompt
        system_prompt = """You are an expert SQL developer. Your task is to convert natural language questions into accurate SQL queries.

Study the examples provided carefully - they are selected to be similar to your task. Generate a SQL query following their patterns.

Guidelines:
- Use the exact table and column names from the schema
- Follow the patterns shown in the examples
- Generate only the SQL query without explanations
- Handle NULL values appropriately
- Use proper JOIN conditions based on foreign key relationships"""
        
        # Build user prompt
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        # Add examples
        examples_text = self._format_examples(examples)
        if examples_text:
            user_parts.extend([
                "",
                "## Similar Examples",
                examples_text,
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
        Generate SQL using few-shot dynamic prompting.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        
        # Select relevant examples
        selected_examples = self._select_examples(question, context)
        
        # Build prompt
        system_prompt, user_prompt = self._build_prompt(
            question, context, selected_examples
        )
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
                "num_examples_selected": len(selected_examples),
                "selection_method": self._selection_method,
                "example_questions": [ex.question for ex in selected_examples],
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
            "selection_method": self._selection_method,
            "mmr_lambda": self._mmr_lambda,
            "min_similarity": self._min_similarity,
            "include_explanations": self._include_explanations,
            "pool_size": self._pool.size,
        }
