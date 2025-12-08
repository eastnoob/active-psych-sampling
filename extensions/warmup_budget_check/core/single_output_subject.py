from pathlib import Path
import numpy as np
import sys

# 尝试导入MixedEffectsLatentSubject
try:
    from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject
except ImportError:
    # 如果直接导入失败，尝试从models位置导入（推荐最新版本）
    here = Path(__file__).resolve()
    imported = False
    for p in [here] + list(here.parents):
        models_subject = p / "tools" / "archive" / "simulate_subject" / "models"
        if models_subject.is_dir():
            sys.path.insert(0, str(models_subject))
            try:
                from subject_models.MixedEffectsLatentSubject import MixedEffectsLatentSubject
                imported = True
                break
            except ImportError:
                pass

    # Fallback: 尝试从archive位置导入
    if not imported:
        for p in [here] + list(here.parents):
            archive_subject = p / "tools" / "archive" / "simulate_subject" / "archive"
            if archive_subject.is_dir():
                sys.path.insert(0, str(archive_subject))
                break
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
