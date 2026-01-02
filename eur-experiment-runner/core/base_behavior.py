"""Base class for experiment behaviors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from .context import Context


class BaseBehavior(ABC):
    """Abstract base class for experiment behaviors.

    Behaviors orchestrate experiment execution:
    1. Build INI config from roles
    2. Create AEPsych Strategy
    3. Run warmup (if specified)
    4. Execute sampling loop with Oracle
    5. Collect results
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return behavior name.

        Returns:
            Behavior identifier string
        """
        pass

    @abstractmethod
    def get_default_config(self) -> str:
        """Return default TOML configuration string.

        Returns:
            TOML config string with default parameters
        """
        pass

    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, error_messages)
        """
        pass

    @abstractmethod
    def run(self, roles: List['BaseRole'], config: Dict[str, Any],
            context: Context) -> Context:
        """Execute experiment.

        Args:
            roles: List of roles to use
            config: Configuration dictionary
            context: Shared context

        Returns:
            Updated context with results
        """
        pass
