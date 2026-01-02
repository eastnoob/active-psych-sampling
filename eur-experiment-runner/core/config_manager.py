"""Configuration manager for TOML configs."""

import toml
from pathlib import Path
from typing import Dict, Any
from loguru import logger


class ConfigManager:
    """Manages TOML configuration loading and validation."""

    def __init__(self, config_path: Path):
        """Initialize config manager.

        Args:
            config_path: Path to TOML config file
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """Load configuration from TOML file.

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            toml.TomlDecodeError: If config file is invalid
        """
        if not self.config_path.exists():
            logger.error(f"Config file not found: {self.config_path}")
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        logger.info(f"Loading config from: {self.config_path}")
        try:
            self.config = toml.load(self.config_path)
            logger.debug(f"Config loaded successfully: {len(self.config)} sections")
            return self.config
        except toml.TomlDecodeError as e:
            logger.error(f"Invalid TOML config: {e}")
            raise

    def get(self, section: str, key: str = None, default=None) -> Any:
        """Get configuration value.

        Args:
            section: Config section name
            key: Key within section (optional)
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section.

        Args:
            section: Section name

        Returns:
            Section dictionary or empty dict if not found
        """
        return self.config.get(section, {})
