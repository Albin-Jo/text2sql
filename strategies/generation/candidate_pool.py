# text2sql_mvp/strategies/generation/candidate_pool.py
"""
Candidate pool management for multi-path SQL generation.

Collects, deduplicates, and manages SQL candidates from multiple
generation strategies for subsequent selection.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from strategies.base import SQLCandidate, GenerationResult


class CandidateSource(str, Enum):
    """Source of a SQL candidate."""
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    DIVIDE_CONQUER = "divide_conquer"
    EXECUTION_PLAN_COT = "execution_plan_cot"
    SYNTHETIC_EXAMPLES = "synthetic_examples"
    TEMPERATURE_DIVERSITY = "temperature_diversity"
    DECOMPOSITION = "decomposition"
    CORRECTION = "correction"
    UNKNOWN = "unknown"


@dataclass
class PoolConfig:
    """Configuration for candidate pool."""
    max_candidates: int = 20
    deduplicate: bool = True
    normalize_for_dedup: bool = True  # Normalize SQL before comparing
    track_sources: bool = True
    min_candidates_required: int = 1


@dataclass
class EnrichedCandidate:
    """
    SQL candidate with additional metadata for selection.
    
    Extends SQLCandidate with generation context and scoring info.
    """
    # Core candidate info
    sql: str
    source: CandidateSource
    strategy_name: str
    
    # Generation context
    temperature: float = 0.0
    attempt_number: int = 1
    generation_latency_ms: float = 0.0
    
    # Validation status
    is_valid_syntax: bool = False
    is_executable: bool = False
    execution_result: Optional[Any] = None
    execution_error: Optional[str] = None
    
    # Scoring (filled by selection phase)
    syntax_score: float = 0.0
    semantic_score: float = 0.0
    confidence_score: float = 0.0
    final_score: float = 0.0
    
    # SQL analysis
    complexity: int = 0
    tables_used: list[str] = field(default_factory=list)
    has_aggregation: bool = False
    has_subquery: bool = False
    has_joins: bool = False
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    # Hash for deduplication
    _sql_hash: Optional[str] = field(default=None, repr=False)
    
    @property
    def sql_hash(self) -> str:
        """Get hash of normalized SQL for deduplication."""
        if self._sql_hash is None:
            normalized = self._normalize_sql(self.sql)
            self._sql_hash = hashlib.md5(normalized.encode()).hexdigest()
        return self._sql_hash
    
    @staticmethod
    def _normalize_sql(sql: str) -> str:
        """Normalize SQL for comparison."""
        import re
        # Convert to lowercase
        normalized = sql.lower()
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        # Remove trailing semicolon
        normalized = normalized.strip().rstrip(';')
        return normalized
    
    def to_sql_candidate(self) -> SQLCandidate:
        """Convert to base SQLCandidate."""
        return SQLCandidate(
            sql=self.sql,
            strategy=self.strategy_name,
            score=self.final_score,
            is_executable=self.is_executable,
            execution_result=self.execution_result,
            execution_error=self.execution_error,
            complexity=self.complexity,
            tables_used=self.tables_used,
            metadata={
                "source": self.source.value,
                "temperature": self.temperature,
                **self.metadata,
            },
        )
    
    @classmethod
    def from_generation_result(
        cls,
        result: GenerationResult,
        source: CandidateSource,
    ) -> Optional["EnrichedCandidate"]:
        """Create EnrichedCandidate from GenerationResult."""
        if not result.sql:
            return None
        
        return cls(
            sql=result.sql,
            source=source,
            strategy_name=result.strategy_name,
            temperature=result.temperature,
            generation_latency_ms=result.latency_ms,
            is_valid_syntax=result.success,
            complexity=result.metadata.get("complexity_score", 0),
            tables_used=result.metadata.get("tables_used", []),
            metadata={
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "model": result.model,
            },
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sql": self.sql,
            "source": self.source.value,
            "strategy_name": self.strategy_name,
            "temperature": self.temperature,
            "is_valid_syntax": self.is_valid_syntax,
            "is_executable": self.is_executable,
            "execution_error": self.execution_error,
            "complexity": self.complexity,
            "tables_used": self.tables_used,
            "final_score": self.final_score,
            "sql_hash": self.sql_hash,
        }


class CandidatePool:
    """
    Pool for collecting and managing SQL candidates.
    
    Features:
    - Collect candidates from multiple sources
    - Automatic deduplication (optional)
    - Track generation metadata
    - Prepare candidates for selection
    
    Example:
        ```python
        pool = CandidatePool()
        
        # Add candidates from different strategies
        pool.add(result1, CandidateSource.ZERO_SHOT)
        pool.add(result2, CandidateSource.FEW_SHOT)
        pool.add_batch(results, CandidateSource.TEMPERATURE_DIVERSITY)
        
        # Get unique candidates for selection
        candidates = pool.get_candidates()
        ```
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        """
        Initialize candidate pool.
        
        Args:
            config: Pool configuration
        """
        self._config = config or PoolConfig()
        self._candidates: list[EnrichedCandidate] = []
        self._seen_hashes: set[str] = set()
        self._source_counts: dict[CandidateSource, int] = {}
    
    @property
    def size(self) -> int:
        """Number of candidates in pool."""
        return len(self._candidates)
    
    @property
    def unique_count(self) -> int:
        """Number of unique SQL queries."""
        return len(self._seen_hashes)
    
    @property
    def is_full(self) -> bool:
        """Check if pool is at max capacity."""
        return self.size >= self._config.max_candidates
    
    @property
    def source_distribution(self) -> dict[str, int]:
        """Get count of candidates by source."""
        return {k.value: v for k, v in self._source_counts.items()}
    
    def add(
        self,
        result: GenerationResult,
        source: CandidateSource,
    ) -> bool:
        """
        Add a generation result to the pool.
        
        Args:
            result: Generation result with SQL
            source: Source strategy type
            
        Returns:
            True if candidate was added (not duplicate/full)
        """
        if not result.sql or not result.sql.strip():
            return False
        
        if self.is_full:
            return False
        
        candidate = EnrichedCandidate.from_generation_result(result, source)
        if candidate is None:
            return False
        
        return self._add_candidate(candidate)
    
    def add_sql(
        self,
        sql: str,
        source: CandidateSource,
        strategy_name: str = "",
        temperature: float = 0.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Add raw SQL to the pool.
        
        Args:
            sql: SQL query string
            source: Source strategy type
            strategy_name: Name of strategy that generated it
            temperature: Temperature used for generation
            metadata: Additional metadata
            
        Returns:
            True if candidate was added
        """
        if not sql or not sql.strip():
            return False
        
        if self.is_full:
            return False
        
        candidate = EnrichedCandidate(
            sql=sql.strip(),
            source=source,
            strategy_name=strategy_name or source.value,
            temperature=temperature,
            metadata=metadata or {},
        )
        
        return self._add_candidate(candidate)
    
    def add_batch(
        self,
        results: list[GenerationResult],
        source: CandidateSource,
    ) -> int:
        """
        Add multiple generation results.
        
        Args:
            results: List of generation results
            source: Source strategy type
            
        Returns:
            Number of candidates added
        """
        added = 0
        for result in results:
            if self.is_full:
                break
            if self.add(result, source):
                added += 1
        return added
    
    def _add_candidate(self, candidate: EnrichedCandidate) -> bool:
        """Internal method to add a candidate with dedup check."""
        # Check for duplicate if deduplication enabled
        if self._config.deduplicate:
            sql_hash = candidate.sql_hash
            if sql_hash in self._seen_hashes:
                return False
            self._seen_hashes.add(sql_hash)
        
        self._candidates.append(candidate)
        
        # Track source distribution
        self._source_counts[candidate.source] = (
            self._source_counts.get(candidate.source, 0) + 1
        )
        
        return True
    
    def get_candidates(self) -> list[EnrichedCandidate]:
        """Get all candidates in the pool."""
        return list(self._candidates)
    
    def get_sql_candidates(self) -> list[SQLCandidate]:
        """Get candidates as SQLCandidate objects."""
        return [c.to_sql_candidate() for c in self._candidates]
    
    def get_by_source(self, source: CandidateSource) -> list[EnrichedCandidate]:
        """Get candidates from a specific source."""
        return [c for c in self._candidates if c.source == source]
    
    def get_valid_syntax(self) -> list[EnrichedCandidate]:
        """Get candidates with valid syntax."""
        return [c for c in self._candidates if c.is_valid_syntax]
    
    def get_executable(self) -> list[EnrichedCandidate]:
        """Get candidates that executed successfully."""
        return [c for c in self._candidates if c.is_executable]
    
    def get_top_k(self, k: int, by: str = "final_score") -> list[EnrichedCandidate]:
        """
        Get top K candidates by score.
        
        Args:
            k: Number of candidates to return
            by: Score field to sort by
            
        Returns:
            Top K candidates sorted by score (descending)
        """
        sorted_candidates = sorted(
            self._candidates,
            key=lambda c: getattr(c, by, 0),
            reverse=True,
        )
        return sorted_candidates[:k]
    
    def update_candidate(
        self,
        index: int,
        **updates: Any,
    ) -> bool:
        """
        Update a candidate's fields.
        
        Args:
            index: Candidate index
            **updates: Fields to update
            
        Returns:
            True if updated successfully
        """
        if index < 0 or index >= len(self._candidates):
            return False
        
        candidate = self._candidates[index]
        for field, value in updates.items():
            if hasattr(candidate, field):
                setattr(candidate, field, value)
        
        return True
    
    def mark_executable(
        self,
        index: int,
        is_executable: bool,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Mark a candidate's execution status."""
        if 0 <= index < len(self._candidates):
            self._candidates[index].is_executable = is_executable
            self._candidates[index].execution_result = result
            self._candidates[index].execution_error = error
    
    def clear(self) -> None:
        """Clear all candidates from the pool."""
        self._candidates.clear()
        self._seen_hashes.clear()
        self._source_counts.clear()
    
    def get_diversity_score(self) -> float:
        """
        Calculate diversity score of the pool.
        
        Returns:
            Score from 0-1 indicating query diversity
        """
        if self.size < 2:
            return 0.0
        
        # Count unique structural patterns
        patterns = set()
        for candidate in self._candidates:
            pattern = self._get_sql_pattern(candidate.sql)
            patterns.add(pattern)
        
        # Diversity = unique patterns / total candidates
        return len(patterns) / self.size
    
    @staticmethod
    def _get_sql_pattern(sql: str) -> str:
        """Extract structural pattern from SQL."""
        import re
        
        # Normalize
        sql_upper = sql.upper()
        
        # Extract key structural elements
        pattern_parts = []
        
        # Check for main clauses
        if "SELECT" in sql_upper:
            if "DISTINCT" in sql_upper:
                pattern_parts.append("SELECT_DISTINCT")
            else:
                pattern_parts.append("SELECT")
        
        if "JOIN" in sql_upper:
            join_count = sql_upper.count("JOIN")
            pattern_parts.append(f"JOIN_{join_count}")
        
        if "WHERE" in sql_upper:
            pattern_parts.append("WHERE")
        
        if "GROUP BY" in sql_upper:
            pattern_parts.append("GROUP_BY")
        
        if "HAVING" in sql_upper:
            pattern_parts.append("HAVING")
        
        if "ORDER BY" in sql_upper:
            pattern_parts.append("ORDER_BY")
        
        if "LIMIT" in sql_upper:
            pattern_parts.append("LIMIT")
        
        # Check for subqueries
        subquery_count = len(re.findall(r'\(\s*SELECT', sql_upper))
        if subquery_count > 0:
            pattern_parts.append(f"SUBQUERY_{subquery_count}")
        
        # Check for window functions
        if "OVER" in sql_upper:
            pattern_parts.append("WINDOW")
        
        # Check for aggregations
        agg_funcs = ["COUNT", "SUM", "AVG", "MIN", "MAX"]
        has_agg = any(f"{f}(" in sql_upper for f in agg_funcs)
        if has_agg:
            pattern_parts.append("AGGREGATE")
        
        return "_".join(pattern_parts) or "SIMPLE"
    
    def get_statistics(self) -> dict[str, Any]:
        """Get pool statistics."""
        valid_count = len(self.get_valid_syntax())
        executable_count = len(self.get_executable())
        
        return {
            "total_candidates": self.size,
            "unique_candidates": self.unique_count,
            "valid_syntax_count": valid_count,
            "executable_count": executable_count,
            "valid_syntax_rate": valid_count / self.size if self.size > 0 else 0,
            "executable_rate": executable_count / self.size if self.size > 0 else 0,
            "source_distribution": self.source_distribution,
            "diversity_score": self.get_diversity_score(),
            "is_full": self.is_full,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert pool to dictionary."""
        return {
            "config": {
                "max_candidates": self._config.max_candidates,
                "deduplicate": self._config.deduplicate,
            },
            "statistics": self.get_statistics(),
            "candidates": [c.to_dict() for c in self._candidates],
        }
