from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
from .context import Context

class BaseModule(ABC):
    """Abstract base class for all warmup steps."""
    
    @abstractmethod
    def get_default_config(self) -> str:
        """
        Return a default TOML configuration string with Chinese comments.
        Using a string allows us to preserve comments which dicts don't.
        """
        pass
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate the configuration for this module.
        Returns (is_valid, error_messages).
        """
        pass
    
    @abstractmethod
    def run(self, config: Dict[str, Any], context: Context) -> Context:
        """
        Execute the module logic.
        Returns the updated context.
        """
        pass
