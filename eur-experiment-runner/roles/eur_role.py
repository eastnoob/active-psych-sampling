"""EUR Role - KEY_CESHI 1:1 Replication.

Exact replication of KEY_CESHI/20251212 EUR configuration:
- ManualGenerator for warmup
- CustomPoolBasedGenerator with EURAnovaMultiAcqf for EUR sampling
- GPRegressionModel with CustomBaseGPResidualFactory
- Mixed parameter types (ordinal + categorical)
- Continuous outcome
"""

from typing import Dict, Any
from pathlib import Path
from loguru import logger
from core.base_role import BaseRole


class EURRole(BaseRole):
    """EUR (Expected Uncertainty Reduction) acquisition role.

    1:1 replication of KEY_CESHI/20251212 configuration.
    """

    def get_name(self) -> str:
        """Return role name."""
        return "eur"

    def get_default_config(self) -> str:
        """Return default TOML configuration."""
        return """
[role.eur]
ini_config_path = ""  # 用户编辑的INI配置文件路径 / Path to user-edited INI config file (REQUIRED)
"""

    def get_strategy_name(self) -> str:
        """Return primary strategy name for single-strategy mode."""
        return "eur_strat"

    def get_strategy_names(self) -> list:
        """Return all strategy names for multi-strategy mode."""
        return ["init_strat", "eur_strat"]

    def generate_ini_section(self, config: Dict[str, Any]) -> str:
        """Load INI configuration from user-provided file.

        Args:
            config: Configuration dictionary with:
                - ini_config_path: Path to user-edited INI file (REQUIRED)

        Returns:
            INI configuration string loaded from file
        """
        ini_path = config.get('ini_config_path', '')

        if not ini_path:
            logger.error("ini_config_path not specified in [role.eur] configuration")
            raise ValueError(
                "ini_config_path is required in [role.eur] configuration. "
                "Please specify the path to your INI config file."
            )

        ini_file = Path(ini_path)
        if not ini_file.exists():
            logger.error(f"INI config file not found: {ini_path}")
            raise FileNotFoundError(f"INI config file not found: {ini_path}")

        logger.info(f"Loading INI config from: {ini_path}")
        with open(ini_file, 'r', encoding='utf-8') as f:
            ini_content = f.read()

        return ini_content
