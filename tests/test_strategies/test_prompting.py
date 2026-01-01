# text2sql_mvp/tests/test_strategies/test_prompting.py
"""
Tests for prompting strategies.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.gemini_client import GeminiResponse, GenerationMetrics
from strategies.base import GenerationContext, StrategyType
from strategies.prompting.zero_shot import ZeroShotStrategy
from strategies.prompting.zero_shot_enhanced import ZeroShotEnhancedStrategy
from strategies.prompting.few_shot_static import FewShotStaticStrategy
from strategies.prompting.few_shot_dynamic import FewShotDynamicStrategy
from strategies.prompting.chain_of_thought import ChainOfThoughtStrategy
from strategies.prompting.decomposition import DecompositionStrategy


def create_mock_response(sql: str = "SELECT * FROM customers") -> GeminiResponse:
    """Create a mock Gemini response."""
    return GeminiResponse(
        text=sql,
        success=True,
        metrics=GenerationMetrics(
            latency_ms=100.0,
            prompt_tokens=50,
            completion_tokens=10,
            total_tokens=60,
            model="gemini-2.0-flash-exp",
            temperature=0.0,
        ),
    )


class TestZeroShotStrategy:
    """Tests for zero-shot strategy."""
    
    @pytest.mark.asyncio
    async def test_generate_success(self, sample_schema, mock_gemini_client):
        """Test successful generation."""
        strategy = ZeroShotStrategy(gemini_client=mock_gemini_client)
        
        context = GenerationContext(
            schema=sample_schema,
            question="How many customers are there?",
            dialect="bigquery",
        )
        
        result = await strategy.generate("How many customers are there?", context)
        
        assert result.success or result.sql is not None
        assert result.strategy_name == "zero_shot"
        assert result.strategy_type == StrategyType.PROMPTING
    
    def test_strategy_properties(self):
        """Test strategy properties."""
        strategy = ZeroShotStrategy()
        
        assert strategy.name == "zero_shot"
        assert strategy.strategy_type == StrategyType.PROMPTING
        assert "zero-shot" in strategy.description.lower()
    
    def test_config(self):
        """Test strategy configuration."""
        strategy = ZeroShotStrategy(temperature=0.5)
        config = strategy.get_config()
        
        assert config["name"] == "zero_shot"
        assert config["temperature"] == 0.5


class TestZeroShotEnhancedStrategy:
    """Tests for enhanced zero-shot strategy."""
    
    @pytest.mark.asyncio
    async def test_generate_with_rules(self, sample_schema, mock_gemini_client):
        """Test generation includes dialect rules."""
        strategy = ZeroShotEnhancedStrategy(
            llm_client=mock_gemini_client,
            dialect="bigquery"
        )
        
        context = GenerationContext(
            schema=sample_schema,
            question="What is the total order amount?",
            dialect="bigquery",
        )
        
        result = await strategy.generate("What is the total order amount?", context)
        
        # Should include BigQuery rules in prompt
        if result.prompt_used:
            assert "bigquery" in result.prompt_used.lower() or "BIGQUERY" in result.prompt_used
    
    def test_dialect_rules_loaded(self):
        """Test that dialect rules are loaded."""
        strategy = ZeroShotEnhancedStrategy(dialect="bigquery")
        rules = strategy._get_rules()
        
        assert len(rules) > 0
        # Should have both default and BigQuery rules
        assert any("JOIN" in rule.upper() for rule in rules)


class TestFewShotStaticStrategy:
    """Tests for static few-shot strategy."""
    
    @pytest.mark.asyncio
    async def test_generate_with_examples(
        self, sample_schema, mock_gemini_client, sample_few_shot_examples
    ):
        """Test generation includes examples in prompt."""
        strategy = FewShotStaticStrategy(
            llm_client=mock_gemini_client,
            examples=sample_few_shot_examples,
            num_examples=2,
        )
        
        context = GenerationContext(
            schema=sample_schema,
            question="List all orders",
            dialect="bigquery",
        )
        
        result = await strategy.generate("List all orders", context)
        
        # Prompt should include examples
        if result.prompt_used:
            assert "Example" in result.prompt_used or "example" in result.prompt_used
    
    def test_num_examples_config(self):
        """Test number of examples configuration."""
        strategy = FewShotStaticStrategy(num_examples=5)
        config = strategy.get_config()
        
        assert config["num_examples"] == 5


class TestFewShotDynamicStrategy:
    """Tests for dynamic few-shot strategy."""
    
    def test_selection_methods(self):
        """Test different selection methods are supported."""
        methods = ["similarity", "mmr", "table_match"]
        
        for method in methods:
            strategy = FewShotDynamicStrategy(selection_method=method)
            assert strategy._selection_method == method
    
    def test_config_includes_selection_method(self):
        """Test config includes selection method."""
        strategy = FewShotDynamicStrategy(selection_method="mmr", mmr_lambda=0.7)
        config = strategy.get_config()
        
        assert config["selection_method"] == "mmr"
        assert config["mmr_lambda"] == 0.7


class TestChainOfThoughtStrategy:
    """Tests for chain-of-thought strategy."""
    
    @pytest.mark.asyncio
    async def test_generate_with_reasoning(self, sample_schema, mock_gemini_client):
        """Test generation includes reasoning steps."""
        strategy = ChainOfThoughtStrategy(llm_client=mock_gemini_client)
        
        context = GenerationContext(
            schema=sample_schema,
            question="What is the average order value per customer?",
            dialect="bigquery",
        )
        
        result = await strategy.generate(
            "What is the average order value per customer?", context
        )
        
        # Prompt should include reasoning instructions
        if result.prompt_used:
            # Should have step-by-step reasoning
            has_reasoning = any(word in result.prompt_used.lower() for word in 
                              ["step", "think", "reason", "analyze", "first"])
            assert has_reasoning or result.success
    
    def test_strategy_properties(self):
        """Test strategy properties."""
        strategy = ChainOfThoughtStrategy()
        
        assert strategy.name == "chain_of_thought"
        assert strategy.strategy_type == StrategyType.PROMPTING


class TestDecompositionStrategy:
    """Tests for decomposition strategy."""
    
    @pytest.mark.asyncio
    async def test_complex_question_decomposition(self, sample_schema, mock_gemini_client):
        """Test decomposition of complex question."""
        strategy = DecompositionStrategy(llm_client=mock_gemini_client)
        
        context = GenerationContext(
            schema=sample_schema,
            question="What is the total revenue for customers who joined last year and placed more than 5 orders?",
            dialect="bigquery",
        )
        
        result = await strategy.generate(
            "What is the total revenue for customers who joined last year and placed more than 5 orders?",
            context
        )
        
        # Should attempt decomposition
        assert result.strategy_name == "decomposition"
    
    def test_strategy_properties(self):
        """Test strategy properties."""
        strategy = DecompositionStrategy()
        
        assert strategy.name == "decomposition"
        assert strategy.strategy_type == StrategyType.PROMPTING


class TestStrategyRegistry:
    """Tests for strategy registration."""
    
    def test_strategies_registered(self):
        """Test that all strategies are registered."""
        from strategies.registry import list_strategies, get_strategy
        
        # Import to ensure registration
        import strategies.prompting  # noqa
        
        strategies_list = list_strategies()
        
        expected = [
            "zero_shot",
            "zero_shot_enhanced",
            "few_shot_static",
            "few_shot_dynamic",
            "chain_of_thought",
            "decomposition",
        ]
        
        for name in expected:
            assert name in strategies_list, f"Strategy {name} not registered"
    
    def test_get_strategy_by_name(self):
        """Test getting strategy by name."""
        from strategies.registry import get_strategy
        import strategies.prompting  # noqa
        
        strategy = get_strategy("zero_shot")
        assert strategy is not None
        assert strategy.name == "zero_shot"
    
    def test_get_nonexistent_strategy(self):
        """Test getting non-existent strategy returns None."""
        from strategies.registry import get_strategy
        
        strategy = get_strategy("nonexistent_strategy")
        assert strategy is None
