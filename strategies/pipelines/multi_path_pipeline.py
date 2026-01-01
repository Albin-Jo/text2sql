# text2sql_mvp/strategies/pipelines/multi_path_pipeline.py
"""
Multi-path SQL generation pipeline.

Combines multiple generation strategies (CHASE-SQL inspired) to produce
diverse SQL candidates, then selects the best one. This is the recommended
approach for maximum accuracy.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.gemini_client import GeminiClient
from core.sql_parser import SQLParser
from schema.manager import SchemaInfo
from strategies.base import (
    BaseStrategy,
    BasePipeline,
    GenerationResult,
    GenerationContext,
    PipelineResult,
    StrategyType,
    SQLCandidate,
)
from strategies.registry import register_strategy
from strategies.generation.candidate_pool import (
    CandidatePool,
    CandidateSource,
    EnrichedCandidate,
    PoolConfig,
)


@dataclass
class MultiPathConfig:
    """Configuration for multi-path pipeline."""
    # Strategy selection
    use_temperature_diversity: bool = True
    use_divide_conquer: bool = True
    use_synthetic_examples: bool = False  # More expensive, optional
    use_few_shot_dynamic: bool = True
    use_execution_plan_cot: bool = False  # Optional for complex queries
    
    # Temperature diversity settings
    temperatures: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.7])
    samples_per_temperature: int = 1
    
    # Candidate pool settings
    max_candidates: int = 10
    deduplicate: bool = True
    
    # Selection settings
    selection_method: str = "first_valid"  # "first_valid", "majority", "best_complexity"
    
    # Execution
    concurrent_generation: bool = True
    max_concurrent: int = 5
    
    # Complexity threshold for divide-conquer
    dc_complexity_threshold: int = 2


@register_strategy
class MultiPathPipeline(BasePipeline):
    """
    Multi-path SQL generation pipeline (CHASE-SQL inspired).
    
    This pipeline:
    1. Generates SQL using multiple strategies in parallel
    2. Collects all candidates in a pool
    3. Deduplicates and validates candidates
    4. Selects the best candidate
    
    Strategies used:
    - Temperature diversity sampling
    - Divide and conquer (for complex queries)
    - Few-shot dynamic example selection
    - Optional: Synthetic examples, Execution plan CoT
    
    Expected improvement: +6-12% over single-strategy approaches
    
    Example:
        ```python
        pipeline = MultiPathPipeline(llm_client=client)
        result = await pipeline.run(
            "Find customers with above-average spending",
            schema=schema
        )
        print(result.final_sql)
        ```
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        config: Optional[MultiPathConfig] = None,
        few_shot_examples: Optional[list] = None,
        few_shot_path: Optional[str] = None,
    ):
        """
        Initialize multi-path pipeline.
        
        Args:
            llm_client: Gemini client instance
            config: Pipeline configuration
            few_shot_examples: Few-shot example list
            few_shot_path: Path to few-shot examples JSON
        """
        self._llm = llm_client
        self._config = config or MultiPathConfig()
        self._few_shot_examples = few_shot_examples
        self._few_shot_path = few_shot_path
        self._parser = SQLParser()
        
        # Lazy-loaded strategies
        self._temp_diversity_strategy = None
        self._divide_conquer_strategy = None
        self._synthetic_strategy = None
        self._few_shot_strategy = None
        self._exec_plan_strategy = None
    
    @property
    def name(self) -> str:
        return "multi_path_pipeline"
    
    @property
    def description(self) -> str:
        strategies = []
        if self._config.use_temperature_diversity:
            strategies.append("temp_diversity")
        if self._config.use_divide_conquer:
            strategies.append("divide_conquer")
        if self._config.use_few_shot_dynamic:
            strategies.append("few_shot")
        if self._config.use_synthetic_examples:
            strategies.append("synthetic")
        return f"Multi-path generation using: {', '.join(strategies)}"
    
    def _ensure_client(self) -> GeminiClient:
        """Ensure LLM client is initialized."""
        if self._llm is None:
            self._llm = GeminiClient()
        return self._llm
    
    def _get_temp_diversity_strategy(self):
        """Get or create temperature diversity strategy."""
        if self._temp_diversity_strategy is None:
            from strategies.generation.temperature_diversity import (
                TemperatureDiversityStrategy,
                TemperatureConfig,
            )
            self._temp_diversity_strategy = TemperatureDiversityStrategy(
                llm_client=self._ensure_client(),
                config=TemperatureConfig(
                    temperatures=self._config.temperatures,
                    samples_per_temperature=self._config.samples_per_temperature,
                    max_total_candidates=self._config.max_candidates,
                    concurrent_generation=self._config.concurrent_generation,
                ),
            )
        return self._temp_diversity_strategy
    
    def _get_divide_conquer_strategy(self):
        """Get or create divide and conquer strategy."""
        if self._divide_conquer_strategy is None:
            from strategies.generation.divide_and_conquer import (
                DivideAndConquerStrategy,
            )
            self._divide_conquer_strategy = DivideAndConquerStrategy(
                llm_client=self._ensure_client(),
                complexity_threshold=self._config.dc_complexity_threshold,
            )
        return self._divide_conquer_strategy
    
    def _get_synthetic_strategy(self):
        """Get or create synthetic examples strategy."""
        if self._synthetic_strategy is None:
            from strategies.generation.synthetic_examples import (
                SyntheticExamplesStrategy,
            )
            self._synthetic_strategy = SyntheticExamplesStrategy(
                llm_client=self._ensure_client(),
            )
        return self._synthetic_strategy
    
    def _get_few_shot_strategy(self):
        """Get or create few-shot dynamic strategy."""
        if self._few_shot_strategy is None:
            from strategies.prompting.few_shot_dynamic import (
                FewShotDynamicStrategy,
            )
            self._few_shot_strategy = FewShotDynamicStrategy(
                llm_client=self._ensure_client(),
                examples=self._few_shot_examples,
                examples_path=self._few_shot_path,
            )
        return self._few_shot_strategy
    
    def _get_exec_plan_strategy(self):
        """Get or create execution plan CoT strategy."""
        if self._exec_plan_strategy is None:
            from strategies.prompting.execution_plan_cot import (
                ExecutionPlanCoTStrategy,
            )
            self._exec_plan_strategy = ExecutionPlanCoTStrategy(
                llm_client=self._ensure_client(),
            )
        return self._exec_plan_strategy
    
    async def _generate_with_strategy(
        self,
        strategy: BaseStrategy,
        question: str,
        context: GenerationContext,
        source: CandidateSource,
    ) -> tuple[list[GenerationResult], CandidateSource]:
        """Generate SQL with a single strategy."""
        try:
            result = await strategy.generate(question, context)
            return [result], source
        except Exception as e:
            # Return failed result
            return [GenerationResult(
                sql=None,
                success=False,
                strategy_name=strategy.name,
                strategy_type=strategy.strategy_type,
                error=str(e),
            )], source
    
    def _select_best_candidate(
        self,
        pool: CandidatePool,
    ) -> Optional[EnrichedCandidate]:
        """
        Select the best candidate from the pool.
        
        Selection strategies:
        - first_valid: First successful candidate (fastest)
        - majority: Most common SQL pattern
        - best_complexity: Highest complexity score (often more complete)
        """
        candidates = pool.get_candidates()
        if not candidates:
            return None
        
        # Get valid candidates
        valid_candidates = [c for c in candidates if c.is_valid_syntax]
        if not valid_candidates:
            # Return first candidate even if invalid
            return candidates[0]
        
        if self._config.selection_method == "first_valid":
            return valid_candidates[0]
        
        elif self._config.selection_method == "best_complexity":
            # Prefer more complex queries (likely more complete)
            return max(valid_candidates, key=lambda c: c.complexity)
        
        elif self._config.selection_method == "majority":
            # Group by SQL hash and pick most common
            hash_groups: dict[str, list[EnrichedCandidate]] = {}
            for c in valid_candidates:
                h = c.sql_hash
                if h not in hash_groups:
                    hash_groups[h] = []
                hash_groups[h].append(c)
            
            # Find largest group
            largest_group = max(hash_groups.values(), key=len)
            return largest_group[0]
        
        else:
            return valid_candidates[0]
    
    async def run(
        self,
        question: str,
        schema: SchemaInfo,
        execute: bool = False,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Run the multi-path generation pipeline.
        
        Args:
            question: Natural language question
            schema: Database schema
            execute: Whether to execute (not implemented in this phase)
            **kwargs: Additional arguments
            
        Returns:
            PipelineResult with final SQL and all candidates
        """
        start_time = time.perf_counter()
        
        # Create generation context
        context = GenerationContext(
            schema=schema,
            question=question,
            examples=kwargs.get("examples", []),
            hints=kwargs.get("hints", []),
        )
        
        # Create candidate pool
        pool = CandidatePool(PoolConfig(
            max_candidates=self._config.max_candidates,
            deduplicate=self._config.deduplicate,
        ))
        
        # Collect generation tasks
        tasks = []
        
        # Temperature diversity
        if self._config.use_temperature_diversity:
            strategy = self._get_temp_diversity_strategy()
            tasks.append(
                self._generate_with_strategy(
                    strategy, question, context,
                    CandidateSource.TEMPERATURE_DIVERSITY
                )
            )
        
        # Divide and conquer (for complex queries)
        if self._config.use_divide_conquer:
            strategy = self._get_divide_conquer_strategy()
            tasks.append(
                self._generate_with_strategy(
                    strategy, question, context,
                    CandidateSource.DIVIDE_CONQUER
                )
            )
        
        # Few-shot dynamic
        if self._config.use_few_shot_dynamic:
            strategy = self._get_few_shot_strategy()
            tasks.append(
                self._generate_with_strategy(
                    strategy, question, context,
                    CandidateSource.FEW_SHOT
                )
            )
        
        # Synthetic examples (optional)
        if self._config.use_synthetic_examples:
            strategy = self._get_synthetic_strategy()
            tasks.append(
                self._generate_with_strategy(
                    strategy, question, context,
                    CandidateSource.SYNTHETIC_EXAMPLES
                )
            )
        
        # Execution plan CoT (optional)
        if self._config.use_execution_plan_cot:
            strategy = self._get_exec_plan_strategy()
            tasks.append(
                self._generate_with_strategy(
                    strategy, question, context,
                    CandidateSource.EXECUTION_PLAN_COT
                )
            )
        
        # Run all strategies
        all_results: list[GenerationResult] = []
        stage_latencies: dict[str, float] = {}
        
        if self._config.concurrent_generation:
            # Run concurrently with semaphore
            semaphore = asyncio.Semaphore(self._config.max_concurrent)
            
            async def limited_task(task):
                async with semaphore:
                    return await task
            
            task_results = await asyncio.gather(
                *[limited_task(t) for t in tasks],
                return_exceptions=True
            )
            
            for task_result in task_results:
                if isinstance(task_result, Exception):
                    continue
                results, source = task_result
                for result in results:
                    all_results.append(result)
                    pool.add(result, source)
                    stage_latencies[source.value] = result.latency_ms
        else:
            # Run sequentially
            for task in tasks:
                results, source = await task
                for result in results:
                    all_results.append(result)
                    pool.add(result, source)
                    stage_latencies[source.value] = result.latency_ms
        
        # Select best candidate
        best_candidate = self._select_best_candidate(pool)
        
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Build pipeline result
        pipeline_result = PipelineResult(
            final_sql=best_candidate.sql if best_candidate else None,
            success=best_candidate is not None and best_candidate.is_valid_syntax,
            pipeline_name=self.name,
            stages_completed=list(stage_latencies.keys()),
            generation_results=all_results,
            total_latency_ms=total_latency_ms,
            stage_latencies=stage_latencies,
            total_tokens=sum(r.total_tokens for r in all_results),
            metadata={
                "candidate_pool": pool.get_statistics(),
                "selection_method": self._config.selection_method,
                "strategies_used": [r.strategy_name for r in all_results if r.sql],
                "unique_candidates": pool.unique_count,
                "diversity_score": pool.get_diversity_score(),
            },
        )
        
        if not pipeline_result.success:
            error_msgs = [r.error for r in all_results if r.error]
            pipeline_result.error = "; ".join(error_msgs[:3]) if error_msgs else "No valid SQL generated"
        
        return pipeline_result
    
    def get_config(self) -> dict[str, Any]:
        """Get pipeline configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "use_temperature_diversity": self._config.use_temperature_diversity,
            "use_divide_conquer": self._config.use_divide_conquer,
            "use_few_shot_dynamic": self._config.use_few_shot_dynamic,
            "use_synthetic_examples": self._config.use_synthetic_examples,
            "use_execution_plan_cot": self._config.use_execution_plan_cot,
            "temperatures": self._config.temperatures,
            "max_candidates": self._config.max_candidates,
            "selection_method": self._config.selection_method,
            "concurrent_generation": self._config.concurrent_generation,
        }
