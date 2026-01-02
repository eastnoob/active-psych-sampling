"""Context - Shared state container for experiment execution."""

from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd


class Context:
    """Shared state passed between components during experiment execution."""

    def __init__(self):
        self.design_space: Optional[pd.DataFrame] = None
        self.oracle: Optional[Any] = None  # Subject simulator
        self.strategy: Optional[Any] = None  # AEPsych Strategy
        self.results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.output_dir: Optional[Path] = None
        self.ini_config_path: Optional[Path] = None
        self.sampling_history: list = []

    def set(self, key: str, value: Any):
        """Store arbitrary metadata."""
        self.metadata[key] = value

    def get(self, key: str, default=None) -> Any:
        """Retrieve stored metadata."""
        return self.metadata.get(key, default)

    def add_result(self, role_name: str, data: Any):
        """Add result for a specific role."""
        self.results[role_name] = data

    def get_result(self, role_name: str) -> Optional[Any]:
        """Get result for a specific role."""
        return self.results.get(role_name)
