"""Sobol role - Quasi-random Sobol sampling."""

from typing import Dict, Any
from core.base_role import BaseRole


class SobolRole(BaseRole):
    """Sobol quasi-random sampling role.

    Uses AEPsych's SobolGenerator for quasi-random sampling.
    """

    def get_name(self) -> str:
        """Return role name."""
        return "sobol"

    def get_default_config(self) -> str:
        """Return default TOML configuration."""
        return """
[role.sobol]
# Sobol采样没有特殊参数 / Sobol sampling has no special parameters
# 采样数量由behavior的budget控制 / Number of samples controlled by behavior budget
"""

    def generate_ini_section(self, config: Dict[str, Any]) -> str:
        """Generate INI config section for Sobol sampling.

        Args:
            config: Configuration dictionary

        Returns:
            INI string with Sobol strategy section
        """
        budget = config.get('budget', 50)
        model = config.get('model', 'GPClassificationModel')

        ini = f"""
[sobol_strat]
min_asks = {budget}
generator = SobolGenerator
model = {model}
"""
        return ini

    def get_strategy_name(self) -> str:
        """Return strategy name for INI."""
        return "sobol_strat"
