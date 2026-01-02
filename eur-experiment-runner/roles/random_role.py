"""Random role - Uniform random sampling."""

from typing import Dict, Any
from core.base_role import BaseRole


class RandomRole(BaseRole):
    """Random sampling role.

    Uses AEPsych's RandomGenerator for uniform random sampling.
    """

    def get_name(self) -> str:
        """Return role name."""
        return "random"

    def get_default_config(self) -> str:
        """Return default TOML configuration."""
        return """
[role.random]
# Random采样没有特殊参数 / Random sampling has no special parameters
# 采样数量由behavior的budget控制 / Number of samples controlled by behavior budget
"""

    def generate_ini_section(self, config: Dict[str, Any]) -> str:
        """Generate INI config section for Random sampling.

        Args:
            config: Configuration dictionary

        Returns:
            INI string with Random strategy section
        """
        budget = config.get('budget', 50)
        model = config.get('model', 'GPClassificationModel')

        ini = f"""
[random_strat]
min_asks = {budget}
generator = RandomGenerator
model = {model}
"""
        return ini

    def get_strategy_name(self) -> str:
        """Return strategy name for INI."""
        return "random_strat"
