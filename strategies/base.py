# text2sql_mvp/strategies/base.py
"""
Base classes and interfaces for Text-to-SQL generation strategies.
All strategies must inherit from BaseStrategy and implement the generate method.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from schema.manager import SchemaInfo


class StrategyType(str, Enum):
    """Classification of strategy types."""
    PROMPTING = "prompting"
    GENERATION = "generation"
    SCHEMA_LINKING = "schema_linking"
    CORRECTION = "correction"
    SELECTION = "selection"
    PIPELINE = "pipeline"


class DifficultyLevel(str, Enum):
    """Query difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class GenerationContext:
    """
    Context for SQL generation.
    
    Contains all information needed to generate a SQL query.
    """
    schema: SchemaInfo
    question: str
    dialect: str = "bigquery"
    
    # Optional context
    hints: list[str] = field(default_factory=list)
    sample_data: Optional[str] = None
    column_descriptions: Optional[str] = None
    
    # Few-shot examples (populated by strategy if needed)
    examples: list[Any] = field(default_factory=list)
    
    # Previous attempts (for correction strategies)
    previous_sql: Optional[str] = None
    previous_error: Optional[str] = None
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_examples(self) -> bool:
        """Check if few-shot examples are available."""
        return len(self.examples) > 0
    
    @property
    def needs_correction(self) -> bool:
        """Check if this is a correction context."""
        return self.previous_sql is not None and self.previous_error is not None


@dataclass
class GenerationResult:
    """
    Result of a SQL generation attempt.
    
    Contains the generated SQL along with metadata about the generation process.
    """
    # Core results
    sql: Optional[str] = None
    success: bool = False
    
    # Generation metadata
    strategy_name: str = ""
    strategy_type: StrategyType = StrategyType.PROMPTING
    
    # Performance metrics
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Generation details
    attempts: int = 1
    temperature: float = 0.0
    model: str = ""
    
    # Correction tracking
    corrections_applied: list[str] = field(default_factory=list)
    
    # Confidence and scoring
    confidence_score: float = 0.0
    
    # Error information
    error: Optional[str] = None
    error_type: Optional[str] = None
    
    # Raw outputs for debugging
    raw_response: Optional[str] = None
    prompt_used: Optional[str] = None
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def is_valid_sql(self) -> bool:
        """Check if result contains valid SQL."""
        return self.sql is not None and len(self.sql.strip()) > 0
    
    def add_correction(self, correction_type: str) -> None:
        """Record that a correction was applied."""
        self.corrections_applied.append(correction_type)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sql": self.sql,
            "success": self.success,
            "strategy_name": self.strategy_name,
            "strategy_type": self.strategy_type.value,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "attempts": self.attempts,
            "temperature": self.temperature,
            "model": self.model,
            "corrections_applied": self.corrections_applied,
            "confidence_score": self.confidence_score,
            "error": self.error,
            "error_type": self.error_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseStrategy(ABC):
    """
    Abstract base class for all Text-to-SQL generation strategies.
    
    Subclasses must implement:
    - name: Property returning strategy identifier
    - strategy_type: Property returning StrategyType
    - generate: Async method for generating SQL
    
    Example:
        ```python
        class ZeroShotStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "zero_shot"
            
            @property
            def strategy_type(self) -> StrategyType:
                return StrategyType.PROMPTING
            
            async def generate(
                self,
                question: str,
                context: GenerationContext,
            ) -> GenerationResult:
                # Implementation here
                pass
        ```
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this strategy."""
        pass
    
    @property
    @abstractmethod
    def strategy_type(self) -> StrategyType:
        """Type classification for this strategy."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description of the strategy."""
        return f"{self.name} strategy"
    
    @property
    def version(self) -> str:
        """Strategy version for tracking."""
        return "1.0.0"
    
    @abstractmethod
    async def generate(
        self,
        question: str,
        context: GenerationContext,
    ) -> GenerationResult:
        """
        Generate SQL from natural language question.
        
        Args:
            question: Natural language question
            context: Generation context with schema and other info
            
        Returns:
            GenerationResult with generated SQL and metadata
        """
        pass
    
    async def validate(self, result: GenerationResult) -> bool:
        """
        Validate a generation result.
        
        Override in subclasses for strategy-specific validation.
        
        Args:
            result: Generation result to validate
            
        Returns:
            True if result is valid
        """
        return result.is_valid_sql
    
    def get_config(self) -> dict[str, Any]:
        """
        Get strategy configuration.
        
        Override to expose configurable parameters.
        """
        return {
            "name": self.name,
            "type": self.strategy_type.value,
            "version": self.version,
        }


@dataclass
class PipelineResult:
    """
    Result from a complete pipeline execution.
    
    Contains the final SQL plus all intermediate results.
    """
    # Final output
    final_sql: Optional[str] = None
    success: bool = False
    
    # Pipeline metadata
    pipeline_name: str = ""
    stages_completed: list[str] = field(default_factory=list)
    
    # All generation results from each stage
    generation_results: list[GenerationResult] = field(default_factory=list)
    
    # Execution results (if query was executed)
    executed: bool = False
    execution_success: bool = False
    execution_result: Optional[Any] = None
    execution_error: Optional[str] = None
    row_count: int = 0
    
    # Timing
    total_latency_ms: float = 0.0
    stage_latencies: dict[str, float] = field(default_factory=dict)
    
    # Token usage
    total_tokens: int = 0
    
    # Errors
    error: Optional[str] = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def best_result(self) -> Optional[GenerationResult]:
        """Get the best generation result (last successful one)."""
        for result in reversed(self.generation_results):
            if result.success:
                return result
        return self.generation_results[-1] if self.generation_results else None
    
    def add_stage_result(
        self,
        stage_name: str,
        result: GenerationResult,
        latency_ms: float
    ) -> None:
        """Record a stage result."""
        self.stages_completed.append(stage_name)
        self.generation_results.append(result)
        self.stage_latencies[stage_name] = latency_ms
        self.total_latency_ms += latency_ms
        self.total_tokens += result.total_tokens
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "final_sql": self.final_sql,
            "success": self.success,
            "pipeline_name": self.pipeline_name,
            "stages_completed": self.stages_completed,
            "generation_results": [r.to_dict() for r in self.generation_results],
            "executed": self.executed,
            "execution_success": self.execution_success,
            "execution_error": self.execution_error,
            "row_count": self.row_count,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies": self.stage_latencies,
            "total_tokens": self.total_tokens,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class BasePipeline(ABC):
    """
    Abstract base class for generation pipelines.
    
    A pipeline combines multiple strategies into a complete workflow.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline identifier."""
        pass
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return f"{self.name} pipeline"
    
    @abstractmethod
    async def run(
        self,
        question: str,
        schema: SchemaInfo,
        execute: bool = False,
        **kwargs: Any,
    ) -> PipelineResult:
        """
        Run the complete pipeline.
        
        Args:
            question: Natural language question
            schema: Database schema
            execute: Whether to execute the generated query
            **kwargs: Additional pipeline-specific arguments
            
        Returns:
            PipelineResult with final SQL and intermediate results
        """
        pass
    
    def get_config(self) -> dict[str, Any]:
        """Get pipeline configuration."""
        return {
            "name": self.name,
            "description": self.description,
        }


@dataclass
class SQLCandidate:
    """
    A candidate SQL query for selection.
    
    Used when generating multiple candidates for selection.
    """
    sql: str
    strategy: str
    score: float = 0.0
    
    # Execution info
    is_executable: bool = False
    execution_result: Optional[Any] = None
    execution_error: Optional[str] = None
    
    # Features
    complexity: int = 0
    tables_used: list[str] = field(default_factory=list)
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def syntax_score(self) -> float:
        """Get syntax validity score."""
        return 1.0 if self.is_executable else 0.5
