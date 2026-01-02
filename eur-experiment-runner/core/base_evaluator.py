"""Base class for evaluators."""

from abc import ABC, abstractmethod
from typing import Dict, Any
from .context import Context


class BaseEvaluator(ABC):
    """Abstract base class for evaluators.

    Evaluators analyze experiment results and generate reports.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Return evaluator name.

        Returns:
            Evaluator identifier string
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
    def evaluate(self, config: Dict[str, Any], context: Context) -> Dict[str, Any]:
        """Perform evaluation.

        Args:
            config: Configuration dictionary
            context: Shared context with results

        Returns:
            Dictionary of evaluation results
        """
        pass

    @abstractmethod
    def generate_report(self, results: Dict[str, Any], context: Context):
        """Generate and save report.

        Args:
            results: Evaluation results from evaluate()
            context: Shared context
        """
        pass
