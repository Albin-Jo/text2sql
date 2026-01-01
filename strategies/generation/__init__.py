# text2sql_mvp/strategies/generation/__init__.py
"""
Multi-path SQL generation strategies.

This module implements CHASE-SQL inspired diverse candidate generation:
- Temperature diversity sampling
- Divide and conquer decomposition
- Synthetic example generation
- Candidate pool management

These strategies generate multiple SQL candidates to improve accuracy
through selection from a diverse pool.
"""

# Core components that don't require external API dependencies
from strategies.generation.candidate_pool import (
    CandidatePool,
    CandidateSource,
    PoolConfig,
    EnrichedCandidate,
)

# Lazy imports for strategies that require API clients
def get_temperature_diversity_strategy():
    """Lazy load TemperatureDiversityStrategy."""
    from strategies.generation.temperature_diversity import TemperatureDiversityStrategy
    return TemperatureDiversityStrategy

def get_divide_and_conquer_strategy():
    """Lazy load DivideAndConquerStrategy."""
    from strategies.generation.divide_and_conquer import DivideAndConquerStrategy
    return DivideAndConquerStrategy

def get_synthetic_examples_strategy():
    """Lazy load SyntheticExamplesStrategy."""
    from strategies.generation.synthetic_examples import SyntheticExamplesStrategy
    return SyntheticExamplesStrategy


__all__ = [
    # Candidate management
    "CandidatePool",
    "CandidateSource",
    "PoolConfig",
    "EnrichedCandidate",
    # Lazy loaders for strategies
    "get_temperature_diversity_strategy",
    "get_divide_and_conquer_strategy",
    "get_synthetic_examples_strategy",
]
