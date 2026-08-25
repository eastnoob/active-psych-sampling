import toml
import os
import subprocess
import tempfile
from typing import Dict, Any, List, Tuple
from pathlib import Path
from loguru import logger

class ConfigManager:
    """Handles TOML configuration generation, editing, and validation."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self.template_dir = config_dir / "templates"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)

    def get_template(self, module_name: str) -> str:
        """Read a template from the templates directory."""
        template_path = self.template_dir / f"{module_name}.toml"
        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def list_configs(self) -> List[Path]:
        """List all TOML files in the config directory (excluding templates)."""
        return [f for f in self.config_dir.glob("*.toml") if f.name != "temp_config.toml"]

    def create_combined_config(self, modules_info: List[Tuple[str, str]], filename: str = "temp_config.toml") -> Path:
        """
        Combines default configs from multiple modules into one TOML file.
        modules_info: List of (module_name, default_config_str)
        """
        combined_content = "# EUR-Warmup Unified Configuration\n"
        combined_content += "# Please edit the parameters below. Comments are in Chinese.\n\n"
        
        for name, config_str in modules_info:
            # Try to get from template first, fallback to provided string
            template_content = self.get_template(name)
            content = template_content if template_content else config_str
            
            combined_content += f"[{name}]\n"
            combined_content += content + "\n\n"
            
        config_path = self.config_dir / filename
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(combined_content)
        return config_path

    def open_editor(self, file_path: Path):
        """Opens the config file in the system's default editor."""
        editor = os.environ.get('EDITOR', 'notepad' if os.name == 'nt' else 'nano')
        logger.info(f"Opening config in editor: {editor}")
        try:
            subprocess.run([editor, str(file_path)], check=True)
        except Exception as e:
            logger.error(f"Failed to open editor: {e}")
            print(f"\n[Error] Could not open editor automatically. Please manually edit: {file_path}")
            input("Press Enter after you have finished editing and saved the file...")

    def load_and_validate(self, file_path: Path) -> Dict[str, Any]:
        """Loads the TOML file and returns a dict."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return toml.load(f)
        except Exception as e:
            logger.error(f"Failed to parse TOML: {e}")
            raise ValueError(f"Invalid TOML format: {e}")
