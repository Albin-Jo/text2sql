# text2sql_mvp/strategies/generation/temperature_diversity.py
"""
Temperature diversity strategy for multi-path SQL generation.

Generates multiple SQL candidates at different temperatures to explore
the solution space. Lower temperatures produce more deterministic outputs,
while higher temperatures encourage creativity and alternative formulations.
"""

import asyncio
import time
from dataclasses import dataclass, field
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
from strategies.generation.candidate_pool import (
    CandidatePool,
    CandidateSource,
    EnrichedCandidate,
    PoolConfig,
)


@dataclass
class TemperatureConfig:
    """Configuration for temperature diversity sampling."""
    temperatures: list[float] = field(
        default_factory=lambda: [0.0, 0.3, 0.7]
    )
    samples_per_temperature: int = 1
    max_total_candidates: int = 10
    concurrent_generation: bool = True
    max_concurrent: int = 5
    include_zero_shot: bool = True
    include_few_shot: bool = True
    few_shot_examples: int = 3


@register_strategy
class TemperatureDiversityStrategy(BaseStrategy):
    """
    Multi-temperature SQL generation strategy.
    
    Generates SQL candidates at multiple temperature settings to create
    a diverse pool of solutions. This is one of the simplest and most
    effective ways to improve accuracy through candidate diversity.
    
    Based on research showing test-time compute scaling via sampling
    can significantly improve accuracy when combined with good selection.
    
    Example:
        ```python
        strategy = TemperatureDiversityStrategy(
            llm_client=client,
            temperatures=[0.0, 0.3, 0.5, 0.7],
            samples_per_temperature=2,
        )
        result = await strategy.generate(question, context)
        # result.metadata["candidate_pool"] contains all candidates
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        config: Optional[TemperatureConfig] = None,
        temperatures: Optional[list[float]] = None,
        samples_per_temperature: int = 1,
        schema_format: SchemaFormat = SchemaFormat.DDL,
        few_shot_examples: Optional[list] = None,
    ):
        """
        Initialize temperature diversity strategy.
        
        Args:
            llm_client: Gemini client instance
            config: Full temperature configuration
            temperatures: List of temperatures to sample (shortcut)
            samples_per_temperature: Samples at each temperature
            schema_format: Schema format to use
            few_shot_examples: Optional few-shot examples
        """
        self._llm = llm_client
        self._config = config or TemperatureConfig()
        
        # Override config with direct parameters
        if temperatures is not None:
            self._config.temperatures = temperatures
        if samples_per_temperature > 1:
            self._config.samples_per_temperature = samples_per_temperature
        
        self._schema_format = schema_format
        self._few_shot_examples = few_shot_examples or []
        self._parser = SQLParser()
        self._formatter = SchemaFormatter()
    
    @property
    def name(self) -> str:
        return "temperature_diversity"
    
    @property
    def strategy_type(self) -> StrategyType:
        return StrategyType.GENERATION
    
    @property
    def description(self) -> str:
        temps = ", ".join(str(t) for t in self._config.temperatures)
        return f"Multi-temperature sampling at [{temps}]"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _build_base_prompt(
        self,
        question: str,
        context: GenerationContext,
    ) -> tuple[str, str]:
        """Build the base prompt for generation."""
        # Format schema
        schema_text = self._formatter.format(
            context.schema,
            format_type=self._schema_format,
            include_descriptions=True,
            include_foreign_keys=True,
        )
        
        # System prompt
        system_prompt = """You are an expert SQL developer. Convert natural language questions into accurate SQL queries.

Guidelines:
- Use exact table and column names from the schema
- Generate only the SQL query without explanations
- Ensure proper JOIN conditions using foreign key relationships
- Handle NULL values appropriately
- Use appropriate aggregations and groupings"""
        
        # Build user prompt
        user_parts = [
            "## Database Schema",
            schema_text,
        ]
        
        # Add few-shot examples if available
        if self._few_shot_examples and self._config.include_few_shot:
            examples_text = self._format_examples(
                self._few_shot_examples[:self._config.few_shot_examples]
            )
            if examples_text:
                user_parts.extend([
                    "",
                    "## Examples",
                    examples_text,
                ])
        
        # Add context examples if provided
        if context.examples:
            examples_text = self._format_context_examples(context.examples)
            if examples_text:
                user_parts.extend([
                    "",
                    "## Similar Examples",
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
            "## Question",
            question,
            "",
            "SQL:",
        ])
        
        return system_prompt, "\n".join(user_parts)
    
    def _format_examples(self, examples: list) -> str:
        """Format few-shot examples."""
        if not examples:
            return ""
        
        formatted = []
        for i, ex in enumerate(examples, 1):
            if hasattr(ex, 'question') and hasattr(ex, 'sql'):
                formatted.append(f"Q{i}: {ex.question}\nSQL: {ex.sql}")
            elif isinstance(ex, dict):
                formatted.append(f"Q{i}: {ex.get('question', '')}\nSQL: {ex.get('sql', '')}")
        
        return "\n\n".join(formatted)
    
    def _format_context_examples(self, examples: list) -> str:
        """Format examples from context."""
        return self._format_examples(examples)
    
    async def _generate_at_temperature(
        self,
        prompt: str,
        temperature: float,
        client: GeminiClient,
    ) -> list[GenerationResult]:
        """Generate SQL at a specific temperature."""
        results = []
        
        for sample_idx in range(self._config.samples_per_temperature):
            start_time = time.perf_counter()
            
            response = await client.generate(
                prompt=prompt,
                temperature=temperature,
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            if not response.success:
                results.append(GenerationResult(
                    sql=None,
                    success=False,
                    strategy_name=self.name,
                    strategy_type=self.strategy_type,
                    latency_ms=latency_ms,
                    temperature=temperature,
                    error=response.error,
                    model=response.metrics.model,
                    metadata={"sample_index": sample_idx},
                ))
                continue
            
            # Parse SQL
            parse_result = self._parser.parse(response.text)
            
            results.append(GenerationResult(
                sql=parse_result.sql if parse_result.is_valid else response.text.strip(),
                success=parse_result.is_valid,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=latency_ms,
                prompt_tokens=response.metrics.prompt_tokens,
                completion_tokens=response.metrics.completion_tokens,
                total_tokens=response.metrics.total_tokens,
                temperature=temperature,
                model=response.metrics.model,
                raw_response=response.text,
                metadata={
                    "sample_index": sample_idx,
                    "complexity_score": parse_result.complexity_score if parse_result.is_valid else 0,
                    "tables_used": parse_result.tables if parse_result.is_valid else [],
                },
            ))
        
        return results
    
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL using temperature diversity.
        
        Generates multiple candidates at different temperatures and
        returns the best one (lowest temperature successful candidate).
        
        The full candidate pool is available in result.metadata["candidate_pool"].
        
        Args:
            question: Natural language question
            context: Generation context with schema
            
        Returns:
            GenerationResult with best SQL and candidate pool
        """
        start_time = time.perf_counter()
        client = self._ensure_client()
        
        # Build prompt
        system_prompt, user_prompt = self._build_base_prompt(question, context)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Create candidate pool
        pool = CandidatePool(PoolConfig(
            max_candidates=self._config.max_total_candidates,
            deduplicate=True,
        ))
        
        # Generate at all temperatures
        all_results: list[GenerationResult] = []
        
        if self._config.concurrent_generation:
            # Concurrent generation
            tasks = [
                self._generate_at_temperature(full_prompt, temp, client)
                for temp in self._config.temperatures
            ]
            
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self._config.max_concurrent)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            temperature_results = await asyncio.gather(
                *[limited_task(t) for t in tasks]
            )
            
            for results in temperature_results:
                all_results.extend(results)
        else:
            # Sequential generation
            for temp in self._config.temperatures:
                results = await self._generate_at_temperature(
                    full_prompt, temp, client
                )
                all_results.extend(results)
        
        # Add all results to pool
        for result in all_results:
            pool.add(result, CandidateSource.TEMPERATURE_DIVERSITY)
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Select best candidate (prefer lower temperature, then success)
        best_result = self._select_best(all_results)
        
        # Build final result
        if best_result and best_result.success:
            return GenerationResult(
                sql=best_result.sql,
                success=True,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=total_latency_ms,
                prompt_tokens=sum(r.prompt_tokens for r in all_results),
                completion_tokens=sum(r.completion_tokens for r in all_results),
                total_tokens=sum(r.total_tokens for r in all_results),
                attempts=len(all_results),
                temperature=best_result.temperature,
                model=best_result.model,
                prompt_used=full_prompt,
                metadata={
                    "candidate_pool": pool.to_dict(),
                    "total_candidates": pool.size,
                    "unique_candidates": pool.unique_count,
                    "diversity_score": pool.get_diversity_score(),
                    "temperatures_used": self._config.temperatures,
                    "best_temperature": best_result.temperature,
                    "complexity_score": best_result.metadata.get("complexity_score", 0),
                    "tables_used": best_result.metadata.get("tables_used", []),
                },
            )
        else:
            # No successful generation
            error_msgs = [r.error for r in all_results if r.error]
            return GenerationResult(
                sql=best_result.sql if best_result else None,
                success=False,
                strategy_name=self.name,
                strategy_type=self.strategy_type,
                latency_ms=total_latency_ms,
                attempts=len(all_results),
                error="; ".join(error_msgs[:3]) if error_msgs else "All candidates failed",
                prompt_used=full_prompt,
                metadata={
                    "candidate_pool": pool.to_dict(),
                    "total_candidates": pool.size,
                    "temperatures_used": self._config.temperatures,
                },
            )
    
    def _select_best(
        self,
        results: list[GenerationResult],
    ) -> Optional[GenerationResult]:
        """
        Select the best result from candidates.
        
        Priority:
        1. Successful results
        2. Lower temperature (more deterministic)
        3. Earlier in sequence
        """
        if not results:
            return None
        
        # Sort by (success DESC, temperature ASC, index ASC)
        sorted_results = sorted(
            enumerate(results),
            key=lambda x: (
                -int(x[1].success),  # Success first
                x[1].temperature,     # Lower temp preferred
                x[0],                 # Earlier preferred
            ),
        )
        
        return sorted_results[0][1] if sorted_results else None
    
    async def generate_pool(
        self,
        question: str,
        context: GenerationContext,
    ) -> CandidatePool:
        """
        Generate candidates and return the pool directly.
        
        Use this when you want full access to all candidates
        for custom selection logic.
        
        Args:
            question: Natural language question
            context: Generation context
            
        Returns:
            CandidatePool with all generated candidates
        """
        result = await self.generate(question, context)
        
        # Reconstruct pool from metadata
        pool_data = result.metadata.get("candidate_pool", {})
        pool = CandidatePool(PoolConfig(
            max_candidates=self._config.max_total_candidates,
            deduplicate=True,
        ))
        
        # Re-add candidates from result
        for candidate_dict in pool_data.get("candidates", []):
            pool.add_sql(
                sql=candidate_dict["sql"],
                source=CandidateSource(candidate_dict.get("source", "temperature_diversity")),
                strategy_name=candidate_dict.get("strategy_name", self.name),
                temperature=candidate_dict.get("temperature", 0.0),
            )
        
        return pool
    
    def get_config(self) -> dict[str, Any]:
        """Get strategy configuration."""
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "version": self.version,
            "temperatures": self._config.temperatures,
            "samples_per_temperature": self._config.samples_per_temperature,
            "max_total_candidates": self._config.max_total_candidates,
            "concurrent_generation": self._config.concurrent_generation,
            "schema_format": self._schema_format.value,
        }
