#!/usr/bin/env python3
"""
WarmupMixedPoolGenerator: pool-based, high-information-density warmup selector for mixed-type variables.

Goal
- From a candidate pool, select a small subset that maximizes main-effect information under a tight budget.
- Prioritize main effects (full rank), allow 2-way interaction aliasing; optional center point preference.
- Mixed variable types supported via main-effect encodings.

Approach
- Build a main-effect model matrix for all pool candidates using encodings per variable type.
- Greedy D-optimal subset selection (logdet gain) to pick n_run points from the pool.
- If add_center=True, reserve one slot for the candidate closest to geometric center (in normalized space).

Schema format (list of dicts, length = dim):
- type: one of {"continuous", "integer", "ordinal", "categorical", "boolean"}
- For continuous/integer/ordinal: provide lb, ub (or levels for ordinal)
- For categorical: provide levels (list of distinct values) or n_levels; pool values should be integers or strings in that set
- For boolean: pool values are {0,1} or {False, True}

Notes
- This generator is model-agnostic (_requires_model = False) and returns points from the pool.
- Output points are exactly rows from the input pool; no synthesis outside the pool.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence
import sys
import os

# Make bundled AEPsych visible
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "temp_aepsych")
)

import math
import torch
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.utils_logging import logger


def _normalize_cont_like(
    x: torch.Tensor, lb: float | torch.Tensor, ub: float | torch.Tensor
) -> torch.Tensor:
    lb_t = torch.as_tensor(lb, dtype=torch.float32)
    ub_t = torch.as_tensor(ub, dtype=torch.float32)
    return (2.0 * (x - lb_t) / (ub_t - lb_t)) - 1.0


def _boolean_to_pm1(x: torch.Tensor) -> torch.Tensor:
    # Accept {0,1} or {False, True}
    x = x.to(torch.float32)
    return x * 2.0 - 1.0


class WarmupMixedPoolGenerator(AEPsychGenerator):
    _requires_model = False

    def __init__(
        self,
        pool_points: torch.Tensor,
        schema: Sequence[dict[str, Any]],
        n_runs: Optional[int] = None,
        prefer_rule: bool = True,
        add_center: bool = True,
        ridge: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the mixed-type, pool-based warmup generator.

        Args:
            pool_points: [N x d] candidate pool (raw values; numbers or string-encoded categories).
            schema: list of per-dimension specs (type, levels/lb/ub, etc.).
            n_runs: desired warmup size; if None, computed from rule(dim) and columns.
            prefer_rule: if True, try to use rule(dim); then ensure >= #columns.
            add_center: if True, include the candidate closest to center in normalized space.
            ridge: numerical stability term for info matrix.
            seed: optional torch manual seed (unused in deterministic greedy path but stored).
        """
        super().__init__()
        assert isinstance(pool_points, torch.Tensor) and pool_points.dim() == 2
        self.pool_points = pool_points.clone()
        self.N, self.dim = self.pool_points.shape
        self.schema = list(schema)
        if len(self.schema) != self.dim:
            raise ValueError("Schema length must match pool dimensionality.")
        self.seed = seed
        self.add_center = bool(add_center)
        self.ridge = float(ridge)
        self.prefer_rule = bool(prefer_rule)

        # Build model matrix for main effects
        self._X_all, self._enc_meta = self._build_main_effect_matrix(
            self.pool_points, self.schema
        )
        p = self._X_all.shape[1]

        # Determine target n_runs
        default_rule = self._rule_suggested_runs(self.dim)
        if n_runs is None:
            n_runs = default_rule if self.prefer_rule else p + 1
        n_runs = max(n_runs, p + 1)  # ensure full-rank possible
        if n_runs > self.N:
            logger.warning(
                f"Requested n_runs={n_runs} exceeds pool size N={self.N}; clamping to N."
            )
            n_runs = self.N
        self.n_runs_target = n_runs

        # Optionally reserve a center candidate index
        reserved_idx: Optional[int] = None
        if self.add_center:
            reserved_idx = self._pick_center_candidate(self.pool_points, self.schema)

        # Perform greedy D-optimal subset selection on remaining candidates
        self._selected_indices = self._greedy_d_optimal_select(
            self._X_all, self.n_runs_target, reserved_idx=reserved_idx
        )

        self._cursor = 0
        self.max_asks = len(self._selected_indices)

    # -------------------- Public API --------------------
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
        if self._cursor >= len(self._selected_indices):
            raise RuntimeError("WarmupMixedPoolGenerator exhausted.")
        end = min(self._cursor + num_points, len(self._selected_indices))
        idx = self._selected_indices[self._cursor : end]
        self._cursor = end
        return self.pool_points[idx]

    @property
    def finished(self) -> bool:
        return self._cursor >= len(self._selected_indices)

    def reset(self) -> None:
        self._cursor = 0
        logger.info("WarmupMixedPoolGenerator reset.")

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        options = super().get_config_options(config=config, name=name, options=options)
        return options

    # -------------------- Internals --------------------
    def _rule_suggested_runs(self, k: int) -> int:
        if k <= 3:
            return 4
        if 4 <= k <= 8:
            return 8 if k <= 7 else 12
        if 9 <= k <= 12:
            return k + 1  # PB12-like spirit
        # k >= 13
        return 16 if (k + 1) <= 16 else 20

    def _build_main_effect_matrix(
        self, P: torch.Tensor, schema: Sequence[dict[str, Any]]
    ):
        cols = []
        enc_meta: list[dict[str, Any]] = []
        N, d = P.shape
        for j in range(d):
            col = P[:, j]
            typ = schema[j].get("type", "continuous").lower()
            if typ in ("continuous", "integer", "ordinal"):
                if "lb" in schema[j] and "ub" in schema[j]:
                    lb, ub = schema[j]["lb"], schema[j]["ub"]
                elif "levels" in schema[j]:
                    levels = schema[j]["levels"]
                    lb, ub = float(min(levels)), float(max(levels))
                else:
                    # Fallback: infer from pool
                    lb, ub = float(col.min().item()), float(col.max().item())
                z = _normalize_cont_like(col.to(torch.float32), lb, ub).reshape(N, 1)
                cols.append(z)
                enc_meta.append({"type": typ, "mode": "scaled", "lb": lb, "ub": ub})
            elif typ == "boolean":
                z = _boolean_to_pm1(col.to(torch.float32)).reshape(N, 1)
                cols.append(z)
                enc_meta.append({"type": typ, "mode": "pm1"})
            elif typ == "categorical":
                # Map values to indices 0..L-1 using provided levels or inferred unique
                levels = schema[j].get("levels")
                if levels is None:
                    # infer from pool (sorted unique)
                    uniq = torch.unique(col)
                    # Convert tensor uniques to a list
                    levels = [u.item() if torch.is_tensor(u) else u for u in uniq]
                level_to_idx = {val: i for i, val in enumerate(levels)}
                idxs = torch.tensor(
                    [
                        level_to_idx.get(v.item() if torch.is_tensor(v) else v, 0)
                        for v in col
                    ],
                    dtype=torch.long,
                )
                L = len(levels)
                if L <= 1:
                    # degenerate
                    z = torch.zeros(N, 1)
                    cols.append(z)
                    enc_meta.append(
                        {"type": typ, "mode": "degenerate", "levels": levels}
                    )
                else:
                    # Treatment coding: L-1 columns
                    Z = torch.zeros(N, L - 1, dtype=torch.float32)
                    for i in range(N):
                        kidx = int(idxs[i].item())
                        if kidx > 0:
                            Z[i, kidx - 1] = 1.0
                    cols.append(Z)
                    enc_meta.append(
                        {"type": typ, "mode": "treatment", "levels": levels}
                    )
            else:
                raise ValueError(f"Unsupported variable type: {typ}")
        # Add intercept column for stability
        intercept = torch.ones(N, 1, dtype=torch.float32)
        X = torch.cat([intercept] + cols, dim=1)
        return X, enc_meta

    def _pick_center_candidate(
        self, P: torch.Tensor, schema: Sequence[dict[str, Any]]
    ) -> Optional[int]:
        # Compute normalized representation for distance
        Ns, d = P.shape
        Z = []
        for j in range(d):
            col = P[:, j].to(torch.float32)
            typ = schema[j].get("type", "continuous").lower()
            if typ in ("continuous", "integer", "ordinal"):
                lb = schema[j].get("lb", float(col.min().item()))
                ub = schema[j].get("ub", float(col.max().item()))
                z = _normalize_cont_like(col, lb, ub)
            elif typ == "boolean":
                z = _boolean_to_pm1(col)
            elif typ == "categorical":
                # map to 0..1 by level index / (L-1)
                levels = schema[j].get("levels")
                if levels is None:
                    uniq = torch.unique(col)
                    levels = [u.item() if torch.is_tensor(u) else u for u in uniq]
                level_to_idx = {val: i for i, val in enumerate(levels)}
                idxs = torch.tensor(
                    [
                        level_to_idx.get(v.item() if torch.is_tensor(v) else v, 0)
                        for v in col
                    ],
                    dtype=torch.float32,
                )
                L = max(1, len(levels) - 1)
                z = (idxs / L) * 2.0 - 1.0
            else:
                raise ValueError(
                    f"Unsupported variable type in center selection: {typ}"
                )
            Z.append(z.reshape(Ns, 1))
        Z = torch.cat(Z, dim=1)
        center = torch.zeros(1, d, dtype=torch.float32)
        dists = torch.norm(Z - center, dim=1)
        idx = int(torch.argmin(dists).item())
        return idx

    def _greedy_d_optimal_select(
        self, X: torch.Tensor, n_runs: int, reserved_idx: Optional[int] = None
    ) -> torch.Tensor:
        N, p = X.shape
        remaining = torch.ones(N, dtype=torch.bool)
        selected: list[int] = []

        # Reserve center if requested
        if reserved_idx is not None:
            selected.append(int(reserved_idx))
            remaining[reserved_idx] = False

        # Initialize info matrix V = ridge * I
        V = self.ridge * torch.eye(p, dtype=torch.float32)
        if selected:
            Xsel = X[selected, :]
            V = V + Xsel.T @ Xsel

        # Greedy additions
        while len(selected) < n_runs:
            # Compute gain for each candidate i: log(1 + x_i^T V^{-1} x_i)
            try:
                Vinv = torch.linalg.inv(V)
            except Exception:
                # add extra ridge and invert
                V = V + (10 * self.ridge) * torch.eye(p, dtype=torch.float32)
                Vinv = torch.linalg.inv(V)

            gains = torch.full((N,), -math.inf, dtype=torch.float32)
            cand_idx = torch.nonzero(remaining, as_tuple=False).flatten()
            if cand_idx.numel() == 0:
                break
            Xi = X[cand_idx, :]
            # compute q_i = diag(X_i * V^{-1} * X_i^T) efficiently
            # q = sum((Xi @ Vinv) * Xi, dim=1)
            temp = Xi @ Vinv
            q = torch.sum(temp * Xi, dim=1)
            gains[cand_idx] = torch.log1p(torch.clamp(q, min=0.0))

            i_star = int(torch.argmax(gains).item())
            if not remaining[i_star]:
                # safety fallback
                cand_list = cand_idx.tolist()
                if not cand_list:
                    break
                i_star = cand_list[0]
            # Update
            selected.append(i_star)
            remaining[i_star] = False
            x = X[i_star : i_star + 1, :]
            V = V + x.T @ x

            if len(selected) >= N:
                break
        return torch.tensor(selected, dtype=torch.long)


# Register with Config
Config.register_object(WarmupMixedPoolGenerator)
