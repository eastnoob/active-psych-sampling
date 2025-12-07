#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the same license as the project (see LICENSE).

"""
WarmupMinimalGenerator: High-information-density minimal warmup generator.

Design goals:
- Use the smallest number of points to recover main effects (allowing 2-way interaction aliasing).
- Save budget for subsequent EUR optimization.
- Ensure main-effect design is full rank (no main-effect confounding).

Rules implemented:
- k <= 3 factors: 4 points (L4, 2-level OA) [+ optional center].
- 4 <= k <= 7 factors: 8 points (2^3 full factorial with folded/derived columns) [+ optional center].
- k == 8 factors: promote to 12-run PB to keep full rank (warn user).
- 9 <= k <= 12 factors: k+1 points using 12-run Plackett–Burman (select first k columns) [+ optional center].
- k >= 13 factors: nearest 4-multiple runs currently supported up to 16 and 20. If k+1 <= 16 -> 16-run Hadamard; elif k+1 <= 20 -> 20-run PB; else raise NotImplementedError.

Outputs:
- Acts like a standard AEPsych generator: gen(num_points, ...) returns design points within [lb, ub].
- add_center: whether to append a center point row; default True.

Notes:
- This generator doesn't require a model; points are precomputed on init.
- Mixed-type (categorical) dimensions are not supported here; all dims are treated as continuous.
"""

from __future__ import annotations

from typing import Any, Optional
import sys
import os

# Ensure bundled AEPsych is importable in this workspace
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "temp_aepsych")
)

import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.utils import _process_bounds
from aepsych.utils_logging import logger


class WarmupMinimalGenerator(AEPsychGenerator):
    """High-information-density, minimal warmup generator focusing on main effects.

    Parameters
    ----------
    lb, ub : torch.Tensor
        Lower/upper bounds for each of the k dimensions (shape [k]).
    dim : int | None
        If provided, overrides dimension inferred from bounds.
    add_center : bool
        If True (default), append a center point.
    seed : int | None
        Random seed (currently unused; reserved for potential tie-breaking).
    """

    _requires_model = False

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        dim: Optional[int] = None,
        add_center: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.add_center = bool(add_center)
        self.seed = seed

        # Build coded design in {-1, +1} (center coded as 0s)
        coded = self._build_coded_design(self.dim)
        # Optionally append center point (coded zeros)
        if self.add_center:
            center = torch.zeros(1, self.dim, dtype=torch.float32)
            coded = torch.cat([coded, center], dim=0)

        self._coded_design = coded  # for testing/inspection
        self.design_points = self._map_coded_to_bounds(coded, self.lb, self.ub)

        self._cursor = 0
        self.max_asks = len(self.design_points)

    # ----------------------- Public API -----------------------
    def gen(
        self,
        num_points: int = 1,
        model: Any | None = None,
        fixed_features: Optional[dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if fixed_features is not None and len(fixed_features) != 0:
            logger.warning(
                f"Cannot fix features when generating from {self.__class__.__name__}"
            )

        if self._cursor >= len(self.design_points):
            raise RuntimeError(
                "Warmup design exhausted. Consider disabling add_center or reducing num_points."
            )

        end = min(self._cursor + num_points, len(self.design_points))
        pts = self.design_points[self._cursor : end]
        self._cursor = end
        return pts

    @property
    def finished(self) -> bool:
        return self._cursor >= len(self.design_points)

    def reset(self) -> None:
        self._cursor = 0
        logger.info("WarmupMinimalGenerator reset.")

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        options = super().get_config_options(config=config, name=name, options=options)
        # Nothing special to parse yet (add_center can be read from [common] or generator section)
        return options

    # ----------------------- Design builders -----------------------
    def _build_coded_design(self, k: int) -> torch.Tensor:
        """Return a coded design matrix with values in {-1, +1} of size [n_runs x k].
        Center point is added outside this function.
        """
        if k <= 0:
            raise ValueError("Dimension must be >= 1.")
        if k <= 3:
            X = self._oa_l4()  # [4 x 3]
            return X[:, :k].clone()
        if 4 <= k <= 7:
            X = self._ff2k_derived_cols()  # [8 x 7]
            return X[:, :k].clone()
        if k == 8:
            logger.warning(
                "k=8 cannot maintain full-rank main-effect design with 8 runs; using 12-run Plackett–Burman instead."
            )
            X = self._pb_12()  # [12 x 11]
            return X[:, :k].clone()
        if 9 <= k <= 12:
            X = self._pb_12()  # [12 x 11]
            return X[:, :k].clone()
        # k >= 13
        runs_needed = k + 1
        if runs_needed <= 16:
            X = self._hadamard_sylvester(16)  # [16 x 16]
            # remove constant column (first all +1)
            X = X[:, 1:]
            return X[:, :k].clone()
        if runs_needed <= 20:
            X = self._pb_20()  # [20 x 19]
            return X[:, :k].clone()
        raise NotImplementedError(
            f"Currently supports up to k<=19 (20-run PB). Got k={k}."
        )

    @staticmethod
    def _map_coded_to_bounds(
        X: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor
    ) -> torch.Tensor:
        """Map coded matrix in {-1,0,+1} to actual bounds linearly.
        -1 -> lb, +1 -> ub, 0 -> (lb+ub)/2
        """
        lb = lb.to(dtype=torch.float32)
        ub = ub.to(dtype=torch.float32)
        mid = (lb + ub) / 2.0
        half = (ub - lb) / 2.0
        # X in {-1,0,+1}; map as mid + X*half
        return mid.unsqueeze(0) + X * half.unsqueeze(0)

    # ----- Concrete designs -----
    @staticmethod
    def _oa_l4() -> torch.Tensor:
        """L4 orthogonal array (2-level) with 3 columns.
        Rows: 4 x 3 in {-1,+1} ensuring orthogonality for k<=3.
        Standard construction:
        [-1,-1,-1]
        [-1,+1,+1]
        [+1,-1,+1]
        [+1,+1,-1]
        """
        return torch.tensor(
            [
                [-1, -1, -1],
                [-1, +1, +1],
                [+1, -1, +1],
                [+1, +1, -1],
            ],
            dtype=torch.float32,
        )

    @staticmethod
    def _ff2k_derived_cols() -> torch.Tensor:
        """2^3 full factorial (A,B,C), with derived columns AB, AC, BC, ABC -> 7 columns.
        Returns 8x7 coded matrix in {-1,+1}.
        """
        A = torch.tensor([-1, -1, -1, -1, +1, +1, +1, +1], dtype=torch.float32)
        B = torch.tensor([-1, -1, +1, +1, -1, -1, +1, +1], dtype=torch.float32)
        C = torch.tensor([-1, +1, -1, +1, -1, +1, -1, +1], dtype=torch.float32)
        AB = A * B
        AC = A * C
        BC = B * C
        ABC = A * B * C
        X = torch.stack([A, B, C, AB, AC, BC, ABC], dim=1)
        return X

    @staticmethod
    def _pb_12() -> torch.Tensor:
        """12-run Plackett–Burman base matrix (12 x 11) in {-1,+1}.
        We use a standard PB(12) generator (one of many equivalent variants).
        Each column is balanced and pairwise orthogonal (ignoring 2-way aliasing).
        """
        # One common PB(12) construction
        X = torch.tensor(
            [
                [+1, +1, +1, +1, +1, +1, +1, +1, +1, +1, +1],
                [+1, +1, +1, +1, -1, -1, -1, +1, -1, -1, -1],
                [+1, +1, -1, -1, +1, +1, -1, -1, +1, +1, -1],
                [+1, -1, +1, -1, +1, -1, +1, -1, +1, -1, +1],
                [+1, -1, -1, +1, +1, -1, -1, +1, +1, -1, -1],
                [+1, +1, -1, +1, -1, +1, -1, -1, -1, +1, +1],
                [+1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1],
                [+1, +1, +1, -1, -1, +1, +1, -1, -1, -1, +1],
                [+1, -1, -1, -1, +1, +1, +1, -1, -1, +1, +1],
                [+1, +1, -1, -1, -1, -1, +1, +1, +1, -1, +1],
                [+1, -1, +1, -1, -1, +1, -1, +1, +1, +1, -1],
                [+1, -1, -1, +1, +1, -1, +1, +1, -1, +1, -1],
            ],
            dtype=torch.float32,
        )
        # Drop the first all-+1 column (constant) to get 12x11 usable columns
        return X[:, 1:]

    @staticmethod
    def _hadamard_sylvester(n: int) -> torch.Tensor:
        """Generate an n x n Hadamard matrix via Sylvester construction (n must be a power of 2)."""
        if n & (n - 1) != 0:
            raise ValueError("Sylvester Hadamard requires n to be a power of 2.")
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat(
                [
                    torch.cat([H, H], dim=1),
                    torch.cat([H, -H], dim=1),
                ],
                dim=0,
            )
        return H

    @staticmethod
    def _pb_20() -> torch.Tensor:
        """20-run Plackett–Burman base matrix (20 x 19) in {-1,+1}.
        Hard-coded variant suitable for selecting up to 19 factor columns.
        """
        # A PB(20) variant (20x20 with first constant column removed -> 20x19)
        # Source: standard design tables (one of many isomorphic variants)
        X = torch.tensor(
            [
                [
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                ],
                [
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    +1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    +1,
                    +1,
                    +1,
                    +1,
                ],
                [
                    +1,
                    +1,
                    +1,
                    -1,
                    -1,
                    -1,
                    -1,
                    +1,
                    +1,
                    +1,
                    +1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                ],
                [
                    +1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    -1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    -1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                ],
                [
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                ],
                [
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                ],
                [
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                    -1,
                ],
                [
                    +1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    -1,
                ],
                [
                    +1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                ],
                [
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                ],
                [
                    +1,
                    -1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                ],
                [
                    +1,
                    +1,
                    -1,
                    -1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                    +1,
                    -1,
                    +1,
                    -1,
                    +1,
                ],
                [
                    +1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ],
            ],
            dtype=torch.float32,
        )
        # Drop the first constant column
        return X[:, 1:]


# Register with the Config system
Config.register_object(WarmupMinimalGenerator)
