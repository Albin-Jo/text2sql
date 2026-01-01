# text2sql_mvp/strategies/prompting/__init__.py
"""
Prompting strategies for Text-to-SQL generation.

This module contains strategies based on different prompting techniques:
- Zero-shot (basic and enhanced)
- Few-shot (static, dynamic, and MMR-based)
- Chain-of-thought variations
- Question decomposition
"""

# Import all prompting strategies to trigger registration
# Note: Some may not exist yet (from earlier phases)

try:
    from strategies.prompting.zero_shot import ZeroShotStrategy
except ImportError:
    ZeroShotStrategy = None

try:
    from strategies.prompting.zero_shot_enhanced import ZeroShotEnhancedStrategy
except ImportError:
    ZeroShotEnhancedStrategy = None

try:
    from strategies.prompting.few_shot_static import FewShotStaticStrategy
except ImportError:
    FewShotStaticStrategy = None

try:
    from strategies.prompting.few_shot_dynamic import FewShotDynamicStrategy
except ImportError:
    FewShotDynamicStrategy = None

try:
    from strategies.prompting.chain_of_thought import ChainOfThoughtStrategy
except ImportError:
    ChainOfThoughtStrategy = None

try:
    from strategies.prompting.decomposition import DecompositionStrategy
except ImportError:
    DecompositionStrategy = None

# Phase 3 strategies
from strategies.prompting.execution_plan_cot import ExecutionPlanCoTStrategy
from strategies.prompting.few_shot_mmr import FewShotMMRStrategy


__all__ = [
    # Phase 2 strategies (may be None if not yet implemented)
    "ZeroShotStrategy",
    "ZeroShotEnhancedStrategy",
    "FewShotStaticStrategy",
    "FewShotDynamicStrategy",
    "ChainOfThoughtStrategy",
    "DecompositionStrategy",
    # Phase 3 strategies
    "ExecutionPlanCoTStrategy",
    "FewShotMMRStrategy",
]
