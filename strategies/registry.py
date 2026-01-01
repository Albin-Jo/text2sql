# text2sql_mvp/strategies/registry.py
"""
Strategy registry for discovering and instantiating strategies.
Provides decorator-based registration and lookup functionality.
"""

from typing import Callable, Optional, Type

from strategies.base import BaseStrategy, StrategyType


class StrategyRegistry:
    """
    Singleton registry for Text-to-SQL strategies.
    
    Supports:
    - Decorator-based registration
    - Lookup by name or type
    - Lazy instantiation
    
    Example:
        ```python
        @register_strategy
        class MyStrategy(BaseStrategy):
            ...
        
        # Later
        strategy = get_strategy("my_strategy")
        ```
    """
    
    _instance: Optional["StrategyRegistry"] = None
    _strategies: dict[str, Type[BaseStrategy]]
    _instances: dict[str, BaseStrategy]
    
    def __new__(cls) -> "StrategyRegistry":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies = {}
            cls._instance._instances = {}
        return cls._instance
    
    def register(
        self,
        strategy_class: Type[BaseStrategy],
        name: Optional[str] = None
    ) -> Type[BaseStrategy]:
        """
        Register a strategy class.
        
        Args:
            strategy_class: Strategy class to register
            name: Optional override name (uses class.name property if not provided)
            
        Returns:
            The registered class (for decorator chaining)
        """
        # Create temporary instance to get name
        try:
            temp_instance = strategy_class.__new__(strategy_class)
            strategy_name = name or temp_instance.name
        except Exception:
            # Fallback to class name
            strategy_name = name or strategy_class.__name__.lower()
        
        self._strategies[strategy_name] = strategy_class
        return strategy_class
    
    def get(
        self,
        name: str,
        **kwargs
    ) -> Optional[BaseStrategy]:
        """
        Get a strategy instance by name.
        
        Args:
            name: Strategy name
            **kwargs: Arguments to pass to strategy constructor
            
        Returns:
            Strategy instance or None if not found
        """
        if name not in self._strategies:
            return None
        
        # Create new instance with kwargs if provided
        if kwargs:
            return self._strategies[name](**kwargs)
        
        # Return cached instance or create new one
        if name not in self._instances:
            self._instances[name] = self._strategies[name]()
        
        return self._instances[name]
    
    def get_class(self, name: str) -> Optional[Type[BaseStrategy]]:
        """Get strategy class by name without instantiating."""
        return self._strategies.get(name)
    
    def list_strategies(self) -> list[str]:
        """List all registered strategy names."""
        return list(self._strategies.keys())
    
    def list_by_type(self, strategy_type: StrategyType) -> list[str]:
        """List strategies of a specific type."""
        result = []
        for name, cls in self._strategies.items():
            try:
                instance = self.get(name)
                if instance and instance.strategy_type == strategy_type:
                    result.append(name)
            except Exception:
                continue
        return result
    
    def get_info(self, name: str) -> Optional[dict]:
        """Get strategy info without full instantiation."""
        if name not in self._strategies:
            return None
        
        try:
            instance = self.get(name)
            if instance:
                return {
                    "name": instance.name,
                    "type": instance.strategy_type.value,
                    "description": instance.description,
                    "version": instance.version,
                    "config": instance.get_config(),
                }
        except Exception:
            pass
        
        return {"name": name, "class": self._strategies[name].__name__}
    
    def list_all_info(self) -> list[dict]:
        """Get info for all registered strategies."""
        return [
            info for name in self.list_strategies()
            if (info := self.get_info(name)) is not None
        ]
    
    def clear_instances(self) -> None:
        """Clear cached instances (useful for testing)."""
        self._instances.clear()
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a strategy.
        
        Args:
            name: Strategy name to unregister
            
        Returns:
            True if strategy was unregistered
        """
        if name in self._strategies:
            del self._strategies[name]
            if name in self._instances:
                del self._instances[name]
            return True
        return False
    
    def is_registered(self, name: str) -> bool:
        """Check if a strategy is registered."""
        return name in self._strategies


# Global registry instance
_registry = StrategyRegistry()


def register_strategy(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
    """
    Decorator to register a strategy class.
    
    Example:
        ```python
        @register_strategy
        class ZeroShotStrategy(BaseStrategy):
            @property
            def name(self) -> str:
                return "zero_shot"
            ...
        ```
    """
    _registry.register(cls)
    return cls


def register_strategy_with_name(name: str) -> Callable[[Type[BaseStrategy]], Type[BaseStrategy]]:
    """
    Decorator factory to register with a specific name.
    
    Example:
        ```python
        @register_strategy_with_name("custom_name")
        class MyStrategy(BaseStrategy):
            ...
        ```
    """
    def decorator(cls: Type[BaseStrategy]) -> Type[BaseStrategy]:
        _registry.register(cls, name=name)
        return cls
    return decorator


def get_strategy(name: str, **kwargs) -> Optional[BaseStrategy]:
    """
    Get a strategy instance by name.
    
    Args:
        name: Strategy name
        **kwargs: Arguments to pass to constructor
        
    Returns:
        Strategy instance or None
    """
    return _registry.get(name, **kwargs)


def get_strategy_class(name: str) -> Optional[Type[BaseStrategy]]:
    """Get strategy class by name."""
    return _registry.get_class(name)


def list_strategies() -> list[str]:
    """List all registered strategy names."""
    return _registry.list_strategies()


def list_strategies_by_type(strategy_type: StrategyType) -> list[str]:
    """List strategies of a specific type."""
    return _registry.list_by_type(strategy_type)


def get_registry() -> StrategyRegistry:
    """Get the global registry instance."""
    return _registry
