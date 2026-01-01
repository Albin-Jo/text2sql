# text2sql_mvp/strategies/prompting/few_shot_mmr.py
"""
Few-shot SQL generation with Maximum Marginal Relevance (MMR) example selection.

MMR balances relevance to the query with diversity among selected examples,
preventing redundant examples and improving coverage of SQL patterns.
"""

import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

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
class FewShotMMRStrategy(BaseStrategy):
    """
    Few-shot SQL generation with MMR-based example selection.
    
    Maximum Marginal Relevance (MMR) selects examples that are:
    1. Relevant to the input question
    2. Diverse from each other (not redundant)
    
    The MMR score for a candidate example is:
    MMR = λ * sim(candidate, query) - (1-λ) * max(sim(candidate, selected))
    
    Where λ controls the relevance-diversity tradeoff:
    - λ = 1.0: Pure relevance (like standard similarity search)
    - λ = 0.0: Pure diversity
    - λ = 0.5: Balanced (recommended)
    
    Example:
        ```python
        strategy = FewShotMMRStrategy(
            llm_client=client,
            examples_path="examples.json",
            lambda_param=0.6,  # Slightly favor relevance
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
        lambda_param: float = 0.5,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        temperature: float = 0.0,
        embedding_model: Optional[str] = None,
        include_explanations: bool = True,
        fallback_to_random: bool = True,
    ):
        """
        Initialize MMR few-shot strategy.
        
        Args:
            llm_client: Gemini client instance
            examples: Pre-loaded examples
            examples_path: Path to JSON file with examples
            num_examples: Number of examples to select
            lambda_param: MMR lambda (0-1, higher = more relevance focus)
            schema_format: Format for schema representation
            temperature: Generation temperature
            embedding_model: Sentence transformer model name
            include_explanations: Include example explanations
            fallback_to_random: Fall back to random if embedding fails
        """
        self._llm = llm_client
        self._num_examples = num_examples
        self._lambda_param = lambda_param
        self._schema_format = schema_format
        self._temperature = temperature
        self._include_explanations = include_explanations
        self._fallback_to_random = fallback_to_random
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
        
        # Initialize embedding model
        self._embedding_model_name = embedding_model or "all-MiniLM-L6-v2"
        self._embedder = None
        self._embeddings_cache: dict[str, np.ndarray] = {}
        
        # Load examples
        self._pool = FewShotPool(embedding_model=embedding_model)
        if examples:
            self._pool.add_examples(examples)
        elif examples_path:
            self._pool.load_from_json(Path(examples_path))
    
    @property
    def name(self) -> str:
        return "few_shot_mmr"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.PROMPTING
    
    @property
    def description(self) -> str:
        return f"Few-shot with MMR selection (λ={self._lambda_param})"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _ensure_embedder(self):
        """Ensure embedding model is loaded."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self._embedding_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for MMR selection. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, with caching."""
        if text not in self._embeddings_cache:
            embedder = self._ensure_embedder()
            self._embeddings_cache[text] = embedder.encode(text, convert_to_numpy=True)
        return self._embeddings_cache[text]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def _select_mmr_examples(
        self,
        question: str,
        all_examples: list[FewShotExample],
        n: int,
    ) -> list[FewShotExample]:
        """
        Select examples using Maximum Marginal Relevance.
        
        Args:
            question: Input question
            all_examples: Pool of all available examples
            n: Number of examples to select
            
        Returns:
            Selected examples
        """
        if not all_examples:
            return []
        
        if len(all_examples) <= n:
            return all_examples
        
        # Get query embedding
        query_embedding = self._get_embedding(question)
        
        # Get embeddings for all examples
        example_embeddings = [
            self._get_embedding(ex.question) for ex in all_examples
        ]
        
        # Compute relevance scores (similarity to query)
        relevance_scores = [
            self._cosine_similarity(query_embedding, emb)
            for emb in example_embeddings
        ]
        
        # Initialize
        selected_indices: list[int] = []
        remaining_indices = list(range(len(all_examples)))
        
        # Greedy selection
        for _ in range(min(n, len(all_examples))):
            best_idx = -1
            best_mmr_score = float('-inf')
            
            for idx in remaining_indices:
                # Relevance component
                relevance = relevance_scores[idx]
                
                # Diversity component (max similarity to already selected)
                if selected_indices:
                    max_sim_to_selected = max(
                        self._cosine_similarity(
                            example_embeddings[idx],
                            example_embeddings[sel_idx]
                        )
                        for sel_idx in selected_indices
                    )
                else:
                    max_sim_to_selected = 0.0
                
                # MMR score
                mmr_score = (
                    self._lambda_param * relevance -
                    (1 - self._lambda_param) * max_sim_to_selected
                )
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        return [all_examples[i] for i in selected_indices]
    
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
        """Build the generation prompt."""
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        system_prompt = """You are an expert SQL developer. Convert natural language questions into accurate SQL queries.

Study the examples provided - they are selected to demonstrate diverse but relevant SQL patterns.

Guidelines:
- Use exact table and column names from the schema
- Follow patterns from the examples when applicable
- Generate only the SQL query without explanations
- Handle NULL values appropriately
- Use proper JOIN conditions"""
        
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        examples_text = self._format_examples(examples)
        if examples_text:
            user_parts.extend([
                "",
                "## Diverse Examples",
                examples_text,
            ])
        
        if context.hints:
            user_parts.extend([
                "",
                "## Hints",
                *[f"- {hint}" for hint in context.hints],
            ])
        
        user_parts.extend([
            "",
            "## Your Task",
            f"Question: {question}",
            "",
            "SQL:",
        ])
        
        return system_prompt, "\n".join(user_parts)
    
    def set_examples(self, examples: list[FewShotExample]) -> None:
        """Set new examples for the pool."""
        self._pool = FewShotPool(embedding_model=self._embedding_model_name)
        self._pool.add_examples(examples)
        self._embeddings_cache.clear()
    
    def load_examples(self, path: str | Path) -> None:
        """Load examples from file."""
        self._pool.load_from_json(Path(path))
        self._embeddings_cache.clear()
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using MMR-selected few-shot examples.
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with generated SQL
        """
        start_time = time.perf_counter()
        
        # Select examples using MMR
        all_examples = self._pool.get_all()
        selected_examples: list[FewShotExample] = []
        selection_method = "mmr"
        
        try:
            selected_examples = self._select_mmr_examples(
                question, all_examples, self._num_examples
            )
        except ImportError as e:
            if self._fallback_to_random:
                selected_examples = self._pool.get_random_examples(self._num_examples)
                selection_method = "random_fallback"
            else:
                raise e
        except Exception as e:
            if self._fallback_to_random:
                selected_examples = self._pool.get_random_examples(self._num_examples)
                selection_method = "random_fallback"
            else:
                raise e
        
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
        
        if not response.success:
            return GenerationResult(
                sql=None,
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                error=response.error,
                error_type=response.error_type.value if response.error_type else None,
                model=response.metrics.model,
                metadata={
                    "selection_method": selection_method,
                    "num_examples": len(selected_examples),
                },
            )
        
        # Parse SQL
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
                metadata={
                    "selection_method": selection_method,
                    "num_examples": len(selected_examples),
                    "lambda_param": self._lambda_param,
                },
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
                "selection_method": selection_method,
                "num_examples_selected": len(selected_examples),
                "lambda_param": self._lambda_param,
                "example_questions": [ex.question for ex in selected_examples],
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
            "lambda_param": self._lambda_param,
            "embedding_model": self._embedding_model_name,
            "include_explanations": self._include_explanations,
            "fallback_to_random": self._fallback_to_random,
            "pool_size": self._pool.size,
        }
