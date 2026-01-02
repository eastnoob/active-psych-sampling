"""Oracle - Subject simulators for testing."""

import numpy as np
from typing import Dict, Any, List, Tuple
from loguru import logger


class LinearOracle:
    """Simple linear oracle for testing acquisition methods.

    Simulates subject responses using a linear model with optional interactions.
    """

    def __init__(self, seed: int = 42, noise_std: float = 0.5,
                 output_type: str = 'binary', likert_levels: int = 5,
                 likert_mode: str = 'tanh', likert_sensitivity: float = 2.0,
                 weights: List[float] = None, bias: float = 0.0,
                 interactions: Dict[Tuple[int, int], float] = None):
        """Initialize linear oracle.

        Args:
            seed: Random seed
            noise_std: Standard deviation of observation noise
            output_type: 'binary', 'likert', or 'continuous'
            likert_levels: Number of Likert levels (if output_type='likert')
            likert_mode: 'tanh' or 'sigmoid' for Likert mapping
            likert_sensitivity: Sensitivity parameter for Likert mapping
            weights: Linear weights for each parameter
            bias: Intercept term
            interactions: Dictionary of interaction terms {(i, j): weight}
        """
        np.random.seed(seed)
        self.seed = seed
        self.noise_std = noise_std
        self.output_type = output_type
        self.likert_levels = likert_levels
        self.likert_mode = likert_mode
        self.likert_sensitivity = likert_sensitivity

        self.weights = np.array(weights) if weights is not None else np.array([0.3, 0.2, -0.4, 0.1, 0.25, -0.15])
        self.bias = bias
        self.interactions = interactions or {}

        logger.info(f"LinearOracle initialized: seed={seed}, noise_std={noise_std}, "
                   f"output_type={output_type}, n_params={len(self.weights)}")

    def query(self, x: np.ndarray) -> Any:
        """Query oracle response for parameter vector.

        Args:
            x: Parameter vector (n_params,)

        Returns:
            Response value (type depends on output_type)
        """
        # Main effects
        linear = self.bias + np.dot(self.weights, x)

        # Interaction effects
        for (i, j), weight in self.interactions.items():
            linear += weight * x[i] * x[j]

        # Add noise
        raw = linear + np.random.randn() * self.noise_std

        # Convert to output type
        if self.output_type == 'binary':
            return 1 if raw > 0 else 0
        elif self.output_type == 'likert':
            return self._to_likert(raw)
        else:  # continuous
            return float(raw)

    def _to_likert(self, raw: float) -> int:
        """Convert continuous value to Likert scale.

        Args:
            raw: Continuous value

        Returns:
            Likert integer (1 to likert_levels)
        """
        L = self.likert_levels
        sens = self.likert_sensitivity

        if self.likert_mode == 'tanh':
            v = np.tanh(raw * sens)
            likert_float = v * (L - 1) / 2 + (L + 1) / 2
        else:  # sigmoid
            s = 1.0 / (1.0 + np.exp(-raw * sens))
            likert_float = s * (L - 1) + 1

        likert_int = int(np.round(likert_float))
        return int(np.clip(likert_int, 1, L))

    def get_model_spec(self) -> Dict[str, Any]:
        """Get oracle model specification.

        Returns:
            Dictionary with model parameters
        """
        return {
            'model_type': 'linear',
            'seed': self.seed,
            'bias': float(self.bias),
            'noise_std': self.noise_std,
            'weights': self.weights.tolist(),
            'interactions': {f"x{i}*x{j}": w for (i, j), w in self.interactions.items()},
            'output_type': self.output_type,
            'likert_levels': self.likert_levels if self.output_type == 'likert' else None,
            'likert_mode': self.likert_mode if self.output_type == 'likert' else None,
            'likert_sensitivity': self.likert_sensitivity if self.output_type == 'likert' else None,
        }
