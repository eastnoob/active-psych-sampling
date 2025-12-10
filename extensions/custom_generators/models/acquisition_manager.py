"""
Acquisition Function Manager

Manages acquisition function creation, caching, and evaluation.
"""

from typing import Optional, Any, Dict
import torch
from botorch.acquisition import AcquisitionFunction
from aepsych.models.base import AEPsychModelMixin
from loguru import logger


class AcquisitionManager:
    """Manages acquisition function lifecycle and caching."""

    def __init__(
        self,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize AcquisitionManager.

        Args:
            acqf: Acquisition function class
            acqf_kwargs: Additional acquisition function arguments
        """
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs or {}
        self._acqf_instance: Optional[AcquisitionFunction] = None
        self._acqf_instance_model: Optional[AEPsychModelMixin] = None
        self._last_model_train_size: Optional[int] = None
        self._last_train_tensor_id: Optional[int] = None

    def get_instance(
        self, model: Optional[AEPsychModelMixin]
    ) -> Optional[AcquisitionFunction]:
        """
        Get or create acquisition function instance.

        Caches instance to ensure consistency between acquisitions
        in the same generation cycle.

        Args:
            model: AEPsych model instance

        Returns:
            Acquisition function instance or None
        """
        if model is None:
            logger.debug("[AcqfMgr] No model available for acquisition function")
            return None

        try:
            # Check if refit occurred by comparing model training data
            current_train_size = None
            current_tensor_id = None

            if hasattr(model, "train_inputs") and model.train_inputs:
                current_train_size = model.train_inputs[0].shape[0]
                current_tensor_id = id(model.train_inputs[0])

            # Invalidate cache if model was refitted
            if current_train_size != self._last_model_train_size:
                logger.debug(
                    f"[AcqfMgr] Model refit detected: "
                    f"{self._last_model_train_size} -> {current_train_size}"
                )
                self._acqf_instance = None
                self._last_model_train_size = current_train_size

            if current_tensor_id != self._last_train_tensor_id:
                self._last_train_tensor_id = current_tensor_id

            # Create new instance if needed
            if self._acqf_instance is None or self._acqf_instance_model != model:
                logger.debug(f"[AcqfMgr] Creating new acquisition function instance")
                self._acqf_instance = self.acqf(model=model, **self.acqf_kwargs)
                self._acqf_instance_model = model
                logger.debug(
                    f"[AcqfMgr] Instance: {type(self._acqf_instance).__name__} "
                    f"with model: {type(model).__name__}"
                )
            else:
                logger.debug("[AcqfMgr] Reusing cached acquisition function instance")

            return self._acqf_instance

        except Exception as e:
            logger.error(f"[AcqfMgr] Failed to create acquisition function: {e}")
            return None

    def evaluate(
        self,
        points: torch.Tensor,
        model: Optional[AEPsychModelMixin],
    ) -> Optional[torch.Tensor]:
        """
        Evaluate acquisition function on given points.

        Args:
            points: Points to evaluate [n_points, n_dim]
            model: AEPsych model instance

        Returns:
            Acquisition values or None
        """
        try:
            acqf_instance = self.get_instance(model)
            if acqf_instance is None:
                logger.debug("[AcqfMgr] Cannot evaluate: no acquisition function")
                return None

            # Evaluate acquisition function
            with torch.no_grad():
                acqf_values = acqf_instance(points)

            logger.debug(
                f"[AcqfMgr] Evaluated {points.shape[0]} points, "
                f"range: [{acqf_values.min():.4f}, {acqf_values.max():.4f}]"
            )

            return acqf_values

        except Exception as e:
            logger.error(f"[AcqfMgr] Failed to evaluate acquisition function: {e}")
            return None

    def invalidate_cache(self) -> None:
        """Invalidate cached acquisition function instance."""
        self._acqf_instance = None
        self._acqf_instance_model = None
        self._last_model_train_size = None
        self._last_train_tensor_id = None
        logger.debug("[AcqfMgr] Acquisition function cache invalidated")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information about acquisition function."""
        return {
            "acqf_type": type(self.acqf).__name__,
            "has_instance": self._acqf_instance is not None,
            "cached_model": (
                type(self._acqf_instance_model).__name__
                if self._acqf_instance_model
                else None
            ),
            "last_train_size": self._last_model_train_size,
        }
