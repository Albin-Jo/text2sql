# text2sql_mvp/strategies/pipelines/__init__.py
"""
End-to-end SQL generation pipelines.

Pipelines combine multiple strategies into complete workflows:
- Simple pipeline: Basic generate-execute
- Validated pipeline: With syntax/execution validation
- Multi-path pipeline: CHASE-SQL style multi-candidate generation
- Production pipeline: Full production-ready with all features
"""

# Import pipelines (some may not exist yet)

try:
    from strategies.pipelines.simple_pipeline import SimplePipeline
except ImportError:
    SimplePipeline = None

try:
    from strategies.pipelines.validated_pipeline import ValidatedPipeline
except ImportError:
    ValidatedPipeline = None

# Phase 3 pipeline
from strategies.pipelines.multi_path_pipeline import MultiPathPipeline, MultiPathConfig

try:
    from strategies.pipelines.chase_sql_pipeline import ChASESQLPipeline
except ImportError:
    ChASESQLPipeline = None

try:
    from strategies.pipelines.multi_level_pipeline import MultiLevelPipeline
except ImportError:
    MultiLevelPipeline = None

try:
    from strategies.pipelines.production_pipeline import ProductionPipeline
except ImportError:
    ProductionPipeline = None


__all__ = [
    # Phase 2 pipelines
    "SimplePipeline",
    "ValidatedPipeline",
    # Phase 3 pipeline
    "MultiPathPipeline",
    "MultiPathConfig",
    # Future pipelines
    "ChASESQLPipeline",
    "MultiLevelPipeline",
    "ProductionPipeline",
]
