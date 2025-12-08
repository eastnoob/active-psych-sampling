from pathlib import Path
import numpy as np
from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject


class SingleOutputLatentSubject(MixedEffectsLatentSubject):
    """
    Subclass that ensures the output is a single scalar dependent variable.
    This preserves all behavior (noise, factor loadings, item_noises) of
    MixedEffectsLatentSubject but sets num_observed_vars=1 by default.
    """

    def __init__(self, *args, **kwargs):
        # Force single output
        kwargs.setdefault("num_observed_vars", 1)
        super().__init__(*args, **kwargs)

    def __call__(self, X):
        ratings = super().__call__(X)
        # ratings might be [y]; return scalar y while preserving type
        if isinstance(ratings, list) and len(ratings) == 1:
            return float(ratings[0])
        # If not list or multiple outputs, try to coerce
        try:
            return float(ratings[0])
        except Exception:
            # As a fallback, convert the entire vector to a scalar by mean
            arr = np.array(ratings, dtype=float)
            return float(np.mean(arr))
