"""Base class for acquisition method roles."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseRole(ABC):
    """Abstract base class for acquisition methods.

    Roles generate INI configuration sections for AEPsych acquisition methods.
    They do NOT implement acquisition logic - that's handled by AEPsych.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return role name (e.g., 'eur', 'sobol').

        Returns:
            Role identifier string
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
    def generate_ini_section(self, config: Dict[str, Any]) -> str:
        """Generate INI config section for this acquisition method.

        Args:
            config: Configuration dictionary from TOML

        Returns:
            INI string with [strategy_name] and [generator_name] sections
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for INI (e.g., 'eur_strat', 'sobol_strat').

        Returns:
            Strategy identifier for INI config
        """
        pass
