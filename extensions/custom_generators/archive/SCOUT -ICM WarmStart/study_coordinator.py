"""
SCOUT Study Coordinator

Global coordinator for multi-subject Phase-1 warmup studies in AEPsych.
Handles cross-subject planning, budget allocation, batch design, and bridge subjects.

Module Positioning:
- Scope: Multi-subject, multi-batch study coordination
- Input: design_df + study parameters
- Output: Subject plans (quotas + constraints) for each subject
- Does NOT generate actual trial points (delegates to WarmupSubjectGenerator)
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any
import itertools
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONTINUOUS_THRESHOLD = 10
# Note: N_BINS_CONTINUOUS is now managed dynamically by WarmupAEPsychGenerator
# based on dimensionality. See fit_initial_plan() and summarize_global() for details.
DEFAULT_SEED = 42
MIN_COUNT_PER_LEVEL = 3  # Minimum count per level for single subject


class StudyCoordinator:
    """
    Global coordinator for multi-subject warmup studies.

    Responsibilities:
    - Detect dimensionality and emit warnings
    - Generate global Core-1 candidate set
    - Build interaction candidate pairs
    - Build boundary library
    - Allocate budgets to subjects using maximum remainder method
    - Plan bridge subjects across batches
    - Validate global constraints (e.g., Core-1 repeat policy)
    """

    def __init__(
        self,
        design_df: pd.DataFrame,
        n_subjects: int = 10,
        total_budget: int = 350,
        n_batches: int = 3,
        seed: Optional[int] = None,
    ):
        """
        Initialize the study coordinator.

        Args:
            design_df: Design DataFrame with f1...fd columns
            n_subjects: Number of subjects in study
            total_budget: Total trial budget across all subjects
            n_batches: Number of batches
            seed: Random seed
        """
        self.design_df = design_df.copy()
        self.n_subjects = n_subjects
        self.total_budget = total_budget
        self.n_batches = n_batches
        self.seed = seed if seed is not None else DEFAULT_SEED

        # Modern RNG initialization
        self.rng = np.random.default_rng(self.seed)
        np.random.seed(self.seed)  # Fallback for compatibility with existing code

        # Factor detection
        self.factor_names: List[str] = []
        self.factor_types: Dict[str, str] = {}
        self.discretized_factors: Dict[str, np.ndarray] = {}
        self.d: int = 0

        # Global planning components
        self.global_core1_candidates: Optional[pd.DataFrame] = None
        self.interaction_pairs: List[Tuple[int, int]] = []
        self.boundary_library: List[Dict[str, Any]] = []
        self.budget_split: Dict[str, int] = {}
        self.subject_budgets: Dict[int, int] = {}  # Per-subject budget cache
        self.bridge_plan: Dict[str, Any] = {}

        # Warnings
        self.warnings: List[str] = []

        logger.info(
            f"Initialized StudyCoordinator: {n_subjects} subjects, {n_batches} batches, budget={total_budget}"
        )

    def fit_initial_plan(self) -> "StudyCoordinator":
        """
        Fit the initial global plan.

        Creates:
        - Global Core-1 candidate set
        - Interaction candidate pairs
        - Boundary library
        - Budget split percentages
        - Bridge subject plan

        Returns:
            Self for chaining
        """
        logger.info("Fitting initial global plan...")

        # 1. Detect factors and types
        self._detect_factors()

        # 2. Emit dimensionality warnings
        self._emit_dimensionality_warnings()

        # 3. Generate global Core-1 candidates
        self.global_core1_candidates = self._generate_global_core1()
        logger.info(
            f"Generated {len(self.global_core1_candidates)} global Core-1 candidates"
        )

        # 4. Build interaction pairs
        self.interaction_pairs = self._build_interaction_pairs_heuristic()
        logger.info(f"Selected {len(self.interaction_pairs)} interaction pairs")

        # 5. Build boundary library
        self.boundary_library = self._build_boundary_library()
        logger.info(f"Built boundary library with {len(self.boundary_library)} points")

        # 6. Compute global budget split
        self.budget_split = self._compute_global_budget_split()
        logger.info(f"Budget split: {self.budget_split}")

        # 7. Allocate per-subject budgets using maximum remainder method
        self._allocate_per_subject_budgets()
        s = sum(self.subject_budgets.values())
        if s != self.total_budget:
            raise ValueError(
                f"Per-subject budgets sum {s} != total_budget {self.total_budget}"
            )
        logger.info(f"Per-subject budgets: {self.subject_budgets}")

        # 8. Plan bridge subjects
        self.bridge_plan = self._plan_bridge_subjects()
        logger.info(f"Bridge plan: {self.bridge_plan}")

        logger.info("Global plan fitting completed")
        return self

    def allocate_subject_plan(
        self,
        subject_id: int,
        batch_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Allocate a subject plan with quotas and constraints.

        Args:
            subject_id: Subject identifier
            batch_id: Batch assignment (optional, auto-assigned if None)

        Returns:
            Subject plan dictionary compatible with WarmupSubjectGenerator
        """
        if self.d == 0:
            raise ValueError("Must call fit_initial_plan() first")
        if subject_id < 0 or subject_id >= self.n_subjects:
            raise ValueError(
                f"subject_id {subject_id} out of range [0, {self.n_subjects - 1}]"
            )

        # Auto-assign batch if not provided
        if batch_id is None:
            batch_id = subject_id % self.n_batches

        # Determine if this is a bridge subject
        is_bridge = self._is_bridge_subject(subject_id, batch_id)

        # Allocate quotas using cached per-subject budget (respects maximum remainder method)
        per_subject_budget = self.subject_budgets.get(
            subject_id, self.total_budget // self.n_subjects
        )  # Fallback for safety
        quotas = self._allocate_subject_quotas(per_subject_budget)

        # Generate constraints
        constraints = self._generate_subject_constraints(
            subject_id, batch_id, is_bridge
        )

        # Attach metadata for Warmup endpoint (non-breaking additions)
        constraints.update(
            {
                "schema_version": "1.0",
                "seed": self.seed + subject_id,  # Per-subject seed
                "quota_recipe": self._quota_recipe(),
                # Bridge-related fields for explicit handling
                "bridge": {
                    "is_bridge": bool(is_bridge),
                    "repeat_fraction": self.bridge_plan.get(
                        "core1_repeat_threshold", 0.5
                    ),
                    "repeat_cap": int(
                        np.ceil(quotas.get("core1", 10) * 0.5)
                    ),  # 50% hard cap on core1
                    "repeat_priority": "core1-first",
                    "last_batch_core1_ids": self.bridge_plan.get(
                        "core1_last_batch_ids", []
                    ),
                    "fallback_policy": "core1_pool_indices > core1_points > all_design",
                },
                # Distance metric recommendations
                "distance_metric": "mixed_gower",
                "approximate_match_tolerance": 1e-9,
            }
        )

        subject_plan = {
            "subject_id": subject_id,
            "batch_id": batch_id,
            "is_bridge": is_bridge,
            "quotas": quotas,
            "constraints": constraints,
        }

        logger.debug(
            f"Allocated plan for subject {subject_id} (batch {batch_id}, bridge={is_bridge})"
        )

        return subject_plan

    def summarize_global(self) -> Dict[str, Any]:
        """
        Summarize global study plan (before execution).

        Returns:
            Global summary dictionary with all configuration and state
        """
        if self.d == 0:
            return {"error": "Must call fit_initial_plan() first"}

        # Compute adaptive n_bins based on dimensionality (mirrors WarmupAEPsychGenerator logic)
        if self.d <= 4:
            n_bins_adaptive = 2
        elif 5 <= self.d <= 8:
            n_bins_adaptive = 3
        elif 9 <= self.d <= 12:
            n_bins_adaptive = 4
        else:  # d > 12
            n_bins_adaptive = 5

        summary = {
            "study_parameters": {
                "n_subjects": self.n_subjects,
                "total_budget": self.total_budget,
                "n_batches": self.n_batches,
                "n_factors": self.d,
                "seed": self.seed,
                "rng": "numpy.random.default_rng",  # Modern RNG
            },
            "factor_info": {
                "factor_names": self.factor_names,
                "factor_types": self.factor_types,
            },
            "binning_config": {
                "n_bins_continuous": n_bins_adaptive,
                "continuous_threshold": CONTINUOUS_THRESHOLD,
                "note": "n_bins_continuous auto-adapts based on dimensionality (2-5 bins)",
            },
            "global_components": {
                "n_core1_candidates": (
                    len(self.global_core1_candidates)
                    if self.global_core1_candidates is not None
                    else 0
                ),
                "n_interaction_pairs": len(self.interaction_pairs),
                "n_boundary_points": len(self.boundary_library),
            },
            "budget_allocation": self.budget_split,
            "quota_recipe": self._quota_recipe(),  # New: quota allocation recipe
            "per_subject_budgets": self.subject_budgets,  # Per-subject budget allocation
            "per_subject_budget_total": sum(
                self.subject_budgets.values()
            ),  # Verify sum equals total
            "bridge_plan": self.bridge_plan,
            "expected_coverage": self.compute_global_coverage_expectation(),
            "warnings": self.warnings,
        }

        return summary

    # ========== Private Methods ==========

    def _detect_factors(self):
        """Detect factor names, types, and discretization."""
        self.factor_names = [
            col for col in self.design_df.columns if col.startswith("f")
        ]
        self.d = len(self.factor_names)

        # Detect types
        for factor in self.factor_names:
            unique_vals = self.design_df[factor].nunique()
            # Try to convert to numeric to determine type
            numeric_check = pd.to_numeric(self.design_df[factor], errors="coerce")
            if numeric_check.notna().sum() > 0:  # Has at least some numeric values
                if unique_vals > CONTINUOUS_THRESHOLD:
                    self.factor_types[factor] = "continuous"
                else:
                    self.factor_types[factor] = "discrete"
            else:
                # Categorical/string factor
                self.factor_types[factor] = "discrete"

        # Discretize
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                self.discretized_factors[factor] = self.design_df[factor].unique()
            else:
                # For continuous, compute percentiles on numeric values only
                numeric_vals = pd.to_numeric(
                    self.design_df[factor], errors="coerce"
                ).dropna()
                if len(numeric_vals) > 0:
                    self.discretized_factors[factor] = np.percentile(
                        numeric_vals, [10, 50, 90]
                    )
                else:
                    # Fallback: use all unique values
                    self.discretized_factors[factor] = self.design_df[factor].unique()

        logger.info(f"Detected {self.d} factors: {self.factor_names}")

    def _emit_dimensionality_warnings(self):
        """Emit warnings for high dimensionality."""
        if self.d > 10:
            msg = (
                "Warning: d>10. Phase-1 warm-up becomes sample-hungry; "
                "tightening Core-2 and boundary share; slightly relaxing early GP targets. Proceeding."
            )
            self.warnings.append(msg)
            warnings.warn(msg)
            logger.warning(msg)

        if self.d > 12:
            msg = (
                "Strong Warning: d>12. Efficiency degrades; consider factor blocking "
                "and sparse DOE skeleton. Proceeding with reduced interaction pairs and higher boundary share."
            )
            self.warnings.append(msg)
            warnings.warn(msg)
            logger.warning(msg)

    def _generate_global_core1(self) -> pd.DataFrame:
        """
        Generate global Core-1 candidate set using corners + centers + maximin.

        Returns:
            DataFrame with Core-1 candidates
        """
        n_core1 = self._compute_core1_size()

        candidates = self.design_df.copy()
        core1_points = []

        # Get representative values
        factor_reps = {}
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                # Handle both numeric and string discrete factors
                try:
                    levels = sorted(self.discretized_factors[factor])
                except TypeError:
                    # Can't sort (mixed types), just use first and last
                    levels = list(self.discretized_factors[factor])
                factor_reps[factor] = {"low": levels[0], "high": levels[-1]}
            else:
                # For continuous, use numeric values only
                numeric_vals = pd.to_numeric(
                    candidates[factor], errors="coerce"
                ).dropna()
                if len(numeric_vals) > 0:
                    p10 = np.percentile(numeric_vals, 10)
                    p90 = np.percentile(numeric_vals, 90)
                    factor_reps[factor] = {"low": p10, "high": p90}
                else:
                    # Fallback for non-numeric continuous (shouldn't happen)
                    levels = sorted(set(candidates[factor]))
                    factor_reps[factor] = {"low": levels[0], "high": levels[-1]}

        # 1. All-low corner
        all_low_conditions = {f: factor_reps[f]["low"] for f in self.factor_names}
        all_low_idx = self._find_closest_point(candidates, all_low_conditions)
        core1_points.append(candidates.iloc[all_low_idx : all_low_idx + 1])

        # 2. All-high corner
        all_high_conditions = {f: factor_reps[f]["high"] for f in self.factor_names}
        all_high_idx = self._find_closest_point(candidates, all_high_conditions)
        core1_points.append(candidates.iloc[all_high_idx : all_high_idx + 1])

        # 3. Center point
        median_conditions = {}
        for f in self.factor_names:
            if self.factor_types[f] == "continuous":
                numeric_vals = pd.to_numeric(candidates[f], errors="coerce")
                median_conditions[f] = numeric_vals.median()
            else:
                # For discrete, use first value as representative
                median_conditions[f] = candidates[f].iloc[0]
        center_idx = self._find_closest_point(candidates, median_conditions)
        core1_points.append(candidates.iloc[center_idx : center_idx + 1])

        # Concatenate and deduplicate
        core1_df = pd.concat(core1_points, ignore_index=True).drop_duplicates()

        # Fill remaining with maximin
        if len(core1_df) < n_core1:
            additional = self._maximin_select(
                candidates.drop(core1_df.index, errors="ignore"),
                k=n_core1 - len(core1_df),
                exclude=core1_df,
            )
            core1_df = pd.concat([core1_df, additional], ignore_index=True)

        return core1_df.iloc[:n_core1].reset_index(drop=True)

    def _find_closest_point(
        self, candidates: pd.DataFrame, target: Dict[str, float]
    ) -> int:
        """Find index of point closest to target conditions."""
        scores = np.zeros(len(candidates))
        for factor in self.factor_names:
            # Try to convert to numeric for distance calculation
            numeric_vals = pd.to_numeric(candidates[factor], errors="coerce")
            if numeric_vals.notna().any():
                # If at least some values are numeric, use numeric distance
                target_val = target[factor]
                try:
                    target_num = float(target_val)
                    scores += np.abs(numeric_vals.fillna(target_num) - target_num)
                except (ValueError, TypeError):
                    # If target is not numeric, use exact match penalty
                    scores += (candidates[factor] != str(target_val)).astype(
                        float
                    ) * 1.0
            else:
                # All values are categorical, use exact match penalty
                scores += (candidates[factor] != str(target[factor])).astype(
                    float
                ) * 1.0
        return int(np.argmin(scores))

    def _maximin_select(
        self, candidates: pd.DataFrame, k: int, exclude: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Select k points using maximin criterion.

        Args:
            candidates: Candidate points
            k: Number to select
            exclude: Points to maintain distance from

        Returns:
            Selected points
        """
        if k >= len(candidates):
            return candidates

        candidate_vals = candidates[self.factor_names].values
        selected_indices = []

        # Initialize with random point
        selected_indices.append(np.random.randint(len(candidate_vals)))

        # Greedy selection
        for _ in range(k - 1):
            max_min_dist = -1
            best_idx = 0

            for i in range(len(candidate_vals)):
                if i in selected_indices:
                    continue

                # Compute min distance to selected + excluded
                min_dist = float("inf")

                # Distance to selected
                for j in selected_indices:
                    dist = np.sqrt(np.sum((candidate_vals[i] - candidate_vals[j]) ** 2))
                    min_dist = min(min_dist, dist)

                # Distance to excluded
                if exclude is not None and len(exclude) > 0:
                    exclude_vals = exclude[self.factor_names].values
                    for ex_val in exclude_vals:
                        dist = np.sqrt(np.sum((candidate_vals[i] - ex_val) ** 2))
                        min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            selected_indices.append(best_idx)

        return candidates.iloc[selected_indices].reset_index(drop=True)

    def _build_interaction_pairs_heuristic(self) -> List[Tuple[int, int]]:
        """
        Build interaction pairs using variance-based heuristic.

        Returns:
            List of (i, j) factor index pairs
        """
        # Determine K based on dimensionality
        if self.d <= 8:
            K = min(12, self.d * (self.d - 1) // 2)
        elif 9 <= self.d <= 10:
            K = 10
        elif 11 <= self.d <= 12:
            K = 9
        else:  # d > 12
            K = min(8, max(6, self.d // 2))

        # All possible pairs
        all_pairs = list(itertools.combinations(range(self.d), 2))

        if len(all_pairs) <= K:
            return all_pairs

        # Prioritize by variance
        factor_variances = self.design_df[self.factor_names].var()
        sorted_indices = factor_variances.argsort()[::-1].values

        # Select pairs involving high-variance factors
        prioritized = []
        for i in range(min(len(sorted_indices), self.d)):
            for j in range(i + 1, min(len(sorted_indices), self.d)):
                prioritized.append((int(sorted_indices[i]), int(sorted_indices[j])))
                if len(prioritized) >= K:
                    return prioritized[:K]

        return prioritized

    def _build_boundary_library(self) -> List[Dict[str, Any]]:
        """
        Build boundary library (extreme points).

        Returns:
            List of boundary point dictionaries
        """
        boundary_points = []

        # Get representative values
        factor_reps = {}
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                levels = sorted(self.discretized_factors[factor])
                factor_reps[factor] = {"low": levels[0], "high": levels[-1]}
            else:
                p10 = np.percentile(self.design_df[factor], 10)
                p90 = np.percentile(self.design_df[factor], 90)
                factor_reps[factor] = {"low": p10, "high": p90}

        # All-low
        all_low = {f: factor_reps[f]["low"] for f in self.factor_names}
        all_low["type"] = "all_low"
        boundary_points.append(all_low)

        # All-high
        all_high = {f: factor_reps[f]["high"] for f in self.factor_names}
        all_high["type"] = "all_high"
        boundary_points.append(all_high)

        # Uni-factor extremes
        for i, factor in enumerate(self.factor_names):
            medians = {f: self.design_df[f].median() for f in self.factor_names}

            low_ext = medians.copy()
            low_ext[factor] = factor_reps[factor]["low"]
            low_ext["type"] = f"uni_low_{i}"
            boundary_points.append(low_ext)

            high_ext = medians.copy()
            high_ext[factor] = factor_reps[factor]["high"]
            high_ext["type"] = f"uni_high_{i}"
            boundary_points.append(high_ext)

        return boundary_points

    def _compute_global_budget_split(self) -> Dict[str, float]:
        """
        Compute global budget split percentages.

        Returns:
            Dictionary with percentage allocations
        """
        # Budget percentages by dimensionality
        if self.d <= 8:
            core1_pct = 0.22
            core2_pct = 0.44
            individual_pct = 0.34
            boundary_pct = 0.12 + (0.15 - 0.12) * (self.d - 1) / 7
        elif 9 <= self.d <= 10:
            core1_pct = 0.24
            core2_pct = 0.47
            individual_pct = 0.29
            boundary_pct = 0.15 + (0.18 - 0.15) * (self.d - 9)
        elif 11 <= self.d <= 12:
            core1_pct = 0.25
            core2_pct = 0.48
            individual_pct = 0.27
            boundary_pct = 0.18 + (0.22 - 0.18) * (self.d - 11)
        else:  # d > 12
            core1_pct = 0.25
            core2_pct = 0.48
            individual_pct = 0.27
            boundary_pct = min(0.25, 0.22 + (self.d - 12) * 0.01)

        return {
            "core1": core1_pct,
            "core2": core2_pct,
            "main": core2_pct * 0.6,
            "inter": core2_pct * 0.4,
            "individual": individual_pct,
            "boundary": individual_pct * boundary_pct,
            "lhs": individual_pct * (1 - boundary_pct),
        }

    def _allocate_per_subject_budgets(
        self, weights: Optional[List[float]] = None
    ) -> None:
        """
        Allocate per-subject budgets using maximum remainder method (Hamilton's method).

        Distributes total_budget to n_subjects with optional weights, ensuring:
        - sum(budgets) == total_budget (exact)
        - Proportional to weights if provided
        - Deterministic and reproducible (uses seed for tie-breaking)
        - Fair (no subject systematically disadvantaged)

        Args:
            weights: Optional list of weights for each subject (default: uniform)
                     Will be normalized to sum=1
        """
        n = self.n_subjects
        T = self.total_budget

        if n <= 0:
            raise ValueError("n_subjects must be positive.")
        if T < n:
            msg = f"Total budget {T} < n_subjects {n}; some subjects may get 0."
            self.warnings.append(msg)
            logger.warning(msg)

        # Use uniform weights if not provided
        if weights is None:
            weights = [1.0] * n
        if len(weights) != n:
            raise ValueError("weights length must equal n_subjects.")

        w_sum = float(sum(weights))
        if w_sum <= 0:
            raise ValueError("weights sum must be positive.")

        # Compute ideal float values and floor
        ideal = [T * (w / w_sum) for w in weights]
        floored = [int(np.floor(x)) for x in ideal]
        remainder_total = T - sum(floored)

        # Extract fractional parts
        frac = [x - f for x, f in zip(ideal, floored)]

        # Create allocation order: sort by fractional part (descending), then by index
        order = list(range(n))
        order.sort(key=lambda i: (-frac[i], i))

        # For subjects with identical fractional parts (within tolerance),
        # apply seed-based shuffle for fairness and reproducibility
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(frac[order[j]] - frac[order[i]]) < 1e-12:
                j += 1
            if j - i > 1:  # Found a tie block
                block = order[i:j]
                self.rng.shuffle(block)
                order[i:j] = block
            i = j

        # Allocate base budgets and remainder
        budgets = {i: floored[i] for i in range(n)}
        for k in range(remainder_total):
            budgets[order[k]] += 1

        self.subject_budgets = budgets

        logger.info(
            f"Allocated per-subject budgets (weights={weights}): {self.subject_budgets} "
            f"(sum={sum(self.subject_budgets.values())}, total={self.total_budget})"
        )

    def _quota_recipe(self) -> Dict[str, float]:
        """
        Return the per-subject quota recipe (proportion allocation).

        This recipe is applied to each subject's budget to determine
        quota breakdown. Can be overridden by subclasses or strategy.

        Returns:
            Dict with keys: core1, main, inter, boundary, lhs
        """
        # Dynamic recipe based on dimensionality (from budget_split)
        return {
            "core1": self.budget_split.get("core1", 0.22),
            "main": self.budget_split.get("main", 0.26),
            "inter": self.budget_split.get("inter", 0.18),
            "boundary": self.budget_split.get("boundary", 0.17),
            "lhs": self.budget_split.get("lhs", 0.17),
        }

    def _allocate_subject_quotas(self, per_subject_budget: int) -> Dict[str, int]:
        """
        Allocate integer quotas for a subject using maximum remainder method.

        Applies per-subject quota recipe with deterministic remainder allocation.
        Ensures sum(quotas) == per_subject_budget exactly.

        Args:
            per_subject_budget: Budget for this subject

        Returns:
            Dictionary of integer quotas (keys: core1, main, inter, boundary, lhs)
        """
        if per_subject_budget <= 0:
            return {k: 0 for k in ["core1", "main", "inter", "boundary", "lhs"]}

        recipe = self._quota_recipe()

        # Compute float values
        raw = {k: per_subject_budget * v for k, v in recipe.items()}
        quotas = {k: int(np.floor(x)) for k, x in raw.items()}

        # Compute remainder
        remainder = per_subject_budget - sum(quotas.values())

        # Distribute remainder with fixed priority for determinism
        priority = ["core1", "inter", "main", "lhs", "boundary"]
        idx = 0
        while remainder > 0:
            quota_key = priority[idx % len(priority)]
            if quota_key in quotas:
                quotas[quota_key] += 1
            remainder -= 1
            idx += 1

        return quotas

    def _plan_bridge_subjects(self) -> Dict[str, Any]:
        """
        Plan bridge subjects across batches.

        Returns:
            Bridge plan dictionary
        """
        # Number of bridge subjects per bridge based on dimensionality
        if self.d <= 8:
            bridge_per_bridge = 2
        elif 9 <= self.d <= 10:
            bridge_per_bridge = 3
        else:
            bridge_per_bridge = 3

        # Total bridge subjects needed
        n_bridges = self.n_batches - 1
        total_bridge_subjects = bridge_per_bridge * n_bridges

        if total_bridge_subjects > self.n_subjects:
            logger.warning("Not enough subjects for recommended bridge design")

        return {
            "n_batches": self.n_batches,
            "bridge_subjects_per_bridge": bridge_per_bridge,
            "total_bridge_subjects": min(total_bridge_subjects, self.n_subjects),
            "core1_repeat_threshold": 0.5,
        }

    def _is_bridge_subject(self, subject_id: int, batch_id: int) -> bool:
        """
        Determine if a subject is a bridge subject.

        Args:
            subject_id: Subject ID
            batch_id: Batch ID

        Returns:
            True if bridge subject
        """
        total_bridge = self.bridge_plan.get("total_bridge_subjects", 0)
        return subject_id < total_bridge

    def _generate_subject_constraints(
        self, subject_id: int, batch_id: int, is_bridge: bool
    ) -> Dict[str, Any]:
        """
        Generate constraints for a subject.

        Args:
            subject_id: Subject ID
            batch_id: Batch ID
            is_bridge: Whether bridge subject

        Returns:
            Constraints dictionary
        """
        constraints = {}

        # Must-include Core-1 points
        if self.global_core1_candidates is not None:
            # Sample subset of Core-1 points
            n_core1_subset = min(5, len(self.global_core1_candidates))
            core1_subset = self.global_core1_candidates.sample(
                n=n_core1_subset, random_state=self.seed + subject_id
            )
            constraints["must_include_design_ids"] = core1_subset.index.tolist()
        else:
            constraints["must_include_design_ids"] = []

        # Per-factor minimum counts (simplified)
        constraints["per_factor_min_counts"] = {}
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                L_i = len(self.discretized_factors[factor])
                min_count = max(3, 20 // L_i)  # Reduced for single subject
                constraints["per_factor_min_counts"][factor] = min_count

        # Interaction specifications
        constraints["interactions"] = []
        for pair in self.interaction_pairs[:3]:  # Limit to first 3 pairs per subject
            constraints["interactions"].append(
                {
                    "pair": list(pair),
                    "quadrants": 3,
                    "strategy": "balanced",
                }
            )

        return constraints

    def _compute_core1_size(self) -> int:
        """Compute Core-1 size based on dimensionality."""
        if self.d <= 8:
            return max(8, min(10, self.d + 2))
        elif 9 <= self.d <= 10:
            return max(10, min(12, self.d + 1))
        else:
            return 12

    def compute_global_coverage_expectation(self) -> float:
        """
        Compute expected coverage rate based on allocated quotas.

        Returns:
            Expected coverage rate (0-1)
        """
        # Simplified: assume uniform coverage
        per_subject_budget = self.total_budget // max(1, self.n_subjects)
        total_unique_expected = per_subject_budget * self.n_subjects * 0.7  # 70% unique
        total_design_space = len(self.design_df)

        return (
            min(1.0, total_unique_expected / total_design_space)
            if total_design_space > 0
            else 0.0
        )

    # ========== 跨进程状态管理 JSON 持久化 ==========

    def load_run_state(self, study_id: str, runs_dir: str = "runs") -> Dict[str, Any]:
        """
        Load mutable run state from JSON file (cross-process checkpoint).

        File path: {runs_dir}/{study_id}/run_state.json

        Schema:
        {
            "study_id": str,
            "current_batch": int (1-indexed),
            "next_subject_id": int (1-indexed),
            "n_batches": int,
            "n_subjects_total": int,
            "base_seed": int,
            "core1_last_batch_ids": [design_row_id],
            "bridge_subjects": { "1": [subj_ids], "2": [...], ... },
            "history": [ { batch_id, subject_ids, coverage, gini, core1_repeat_rate, seed_span, timestamp } ]
        }

        Args:
            study_id: Study identifier
            runs_dir: Directory containing runs

        Returns:
            Run state dictionary, or initial state if file doesn't exist
        """
        import json
        from pathlib import Path

        runs_path = Path(runs_dir) / study_id / "run_state.json"

        if runs_path.exists():
            with open(runs_path, "r") as f:
                state = json.load(f)
            logger.info(
                f"Loaded run_state from {runs_path} at batch {state['current_batch']}"
            )
            return state
        else:
            # Initialize new run state
            state = {
                "study_id": study_id,
                "current_batch": 1,
                "next_subject_id": 1,
                "n_batches": self.n_batches,
                "n_subjects_total": self.n_subjects,
                "base_seed": self.seed,
                "core1_last_batch_ids": [],
                "bridge_subjects": {},  # { "1": [subj_ids], "2": [...], ... }
                "history": [],
            }
            logger.info(f"Initialized new run_state for study {study_id}")
            return state

    def save_run_state(
        self, study_id: str, run_state: Dict[str, Any], runs_dir: str = "runs"
    ) -> None:
        """
        Save mutable run state to JSON file (cross-process checkpoint).

        Args:
            study_id: Study identifier
            run_state: Run state dictionary to save
            runs_dir: Directory to contain runs
        """
        import json
        from pathlib import Path

        runs_path = Path(runs_dir) / study_id
        runs_path.mkdir(parents=True, exist_ok=True)

        state_file = runs_path / "run_state.json"
        with open(state_file, "w") as f:
            json.dump(run_state, f, indent=2, default=str)

        logger.info(
            f"Saved run_state to {state_file} at batch {run_state['current_batch']}"
        )

    def _apply_high_dim_quotas(self, quotas: Dict[str, int], d: int) -> Dict[str, int]:
        """
        Apply high-dimensional quotas adjustments:
        - d > 10: interaction ≤15%, boundary+lhs ≥35%
        - d > 12: interaction ≤8%, boundary+lhs ≥45%

        Args:
            quotas: Base quota dictionary
            d: Number of dimensions

        Returns:
            Adjusted quota dictionary
        """
        if d > 12:
            # Ultra-high dimension: reduce interaction, increase boundary+lhs
            logger.warning(
                f"d={d} > 12: applying ultra-high-dim quotas "
                f"(interaction ≤8%, boundary+lhs ≥45%)"
            )
            total = sum(quotas.values())
            quotas["inter"] = max(1, int(total * 0.08))
            remaining = total - quotas["core1"] - quotas["main"]
            quotas["boundary"] = int(remaining * 0.45 / (remaining - quotas["lhs"]))
            quotas["lhs"] = remaining - quotas["boundary"]
        elif d > 10:
            # High dimension: moderate interaction, increase boundary+lhs
            logger.warning(
                f"d={d} > 10: applying high-dim quotas "
                f"(interaction ≤15%, boundary+lhs ≥35%)"
            )
            total = sum(quotas.values())
            quotas["inter"] = max(1, int(total * 0.15))
            remaining = total - quotas["core1"] - quotas["main"]
            quotas["boundary"] = int(remaining * 0.35 / (remaining - quotas["lhs"]))
            quotas["lhs"] = remaining - quotas["boundary"]
        return quotas

    def _apply_strategy_adjustment(
        self, quotas: Dict[str, int], strategy_adj: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Apply strategy adjustment based on previous batch performance.
        Triggered if coverage < 0.6 or gini > 0.6.

        Args:
            quotas: Base quota dictionary
            strategy_adj: Strategy adjustment dict with lhs_increase_pct, boundary_increase_pct

        Returns:
            Adjusted quota dictionary
        """
        lhs_increase = strategy_adj.get("lhs_increase_pct", 0) / 100.0
        boundary_increase = strategy_adj.get("boundary_increase_pct", 0) / 100.0

        if lhs_increase > 0 or boundary_increase > 0:
            logger.info(
                f"Applying strategy adjustment: lhs +{lhs_increase*100:.0f}%, "
                f"boundary +{boundary_increase*100:.0f}%"
            )
            quotas["lhs"] = max(1, int(quotas.get("lhs", 1) * (1 + lhs_increase)))
            quotas["boundary"] = max(
                1, int(quotas.get("boundary", 1) * (1 + boundary_increase))
            )
        return quotas

    def make_subject_plan(
        self,
        subject_id: int,
        batch_id: int,
        run_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate subject plan for a single subject in a specific batch.

        Input: subject_id, batch_id, run_state (contains core1_last_batch_ids, bridge_subjects)
        Output: plan with quotas, constraints, seed

        CRITICAL:
        - core1_repeat_indices strictly capped at 50% of core1 quota
        - Bridge repeat applied only if batch_id > 1 and is_bridge=True
        - High-dimensional adjustments applied if d > 10 or d > 12

        Args:
            subject_id: Subject identifier
            batch_id: Batch number (1-indexed)
            run_state: Current run state with cross-batch info

        Returns:
            Subject plan dict compatible with WarmupAEPsychGenerator.apply_plan()
        """
        # Determine if bridge subject
        bridge_subjects_this_batch = run_state.get("bridge_subjects", {}).get(
            str(batch_id), []
        )
        is_bridge = subject_id in bridge_subjects_this_batch

        # Allocate quotas for this subject using cached per-subject budget
        per_subject_budget = self.subject_budgets.get(
            subject_id, self.total_budget // max(1, self.n_subjects)
        )
        quotas = self._allocate_subject_quotas(per_subject_budget)

        # Apply high-dimensional adjustments if needed
        if self.d is not None:
            quotas = self._apply_high_dim_quotas(quotas, self.d)

        # Apply strategy adjustment from previous batch if coverage/gini out of target
        if run_state.get("strategy_adjustment"):
            quotas = self._apply_strategy_adjustment(
                quotas, run_state["strategy_adjustment"]
            )

        # Generate constraints with bridge repeat indices (with 50% cap)
        core1_quota = quotas.get("core1", 10)
        core1_repeat_max = int(np.ceil(core1_quota * 0.5))  # Hard cap: 50%

        core1_repeat_indices_raw = (
            run_state.get("core1_last_batch_ids", [])
            if is_bridge and batch_id > 1
            else []
        )
        # Enforce 50% cap on repeat indices passed to generator
        core1_repeat_indices = core1_repeat_indices_raw[:core1_repeat_max]

        constraints = {
            "core1_pool_indices": (
                self.global_core1_candidates.index.tolist()
                if self.global_core1_candidates is not None
                else []
            ),
            "core1_repeat_indices": core1_repeat_indices,  # CRITICAL: capped at 50%
            "core1_repeat_cap": 0.5,  # Repeat fraction (hard constraint)
            "core1_repeat_priority": "core1-first",  # Repeat priority strategy
            "interaction_pairs": self.interaction_pairs,
            "boundary_library": self.boundary_library,
            "fallback_policy": "core1_pool_indices > core1_points > all_design",  # Pool selection fallback order
            "policy": {
                "core1_repeat_ratio": 0.5,
                "coverage_min": 0.10,
                "gini_max": 0.40,
            },
        }

        # Per-subject RNG seed (deterministic: base_seed + subject_id)
        seed = self.seed + subject_id

        plan = {
            "subject_id": subject_id,
            "batch_id": batch_id,
            "is_bridge": is_bridge,
            "quotas": quotas,
            "constraints": constraints,
            "seed": seed,
        }

        logger.info(
            f"Generated plan for subject {subject_id} batch {batch_id} "
            f"(bridge={is_bridge}, core1_quota={core1_quota}, core1_repeat_cap={core1_repeat_max}, seed={seed})"
        )

        return plan

    def update_after_batch(
        self,
        run_state: Dict[str, Any],
        batch_id: int,
        all_trials_df: pd.DataFrame,
        all_summaries: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Update run state after batch execution.

        Responsibilities:
        - Extract core1_last_batch_ids from trials (design_row_id of actual Core-1 used)
        - Record coverage, gini, core1_repeat_rate metrics
        - Update bridge_subjects for next batch if needed
        - Check if coverage/gini met targets; if not, adjust LHS/boundary for next batch
        - Increment current_batch
        - Append to history

        Args:
            run_state: Current run state
            batch_id: Batch number that just completed
            all_trials_df: All trials from this batch (all subjects combined)
            all_summaries: Optional list of summaries from each subject's WarmupGenerator

        Returns:
            Updated run state
        """
        from datetime import datetime

        # Extract Core-1 points actually used
        core1_trials = all_trials_df[all_trials_df["block_type"] == "core1"]
        if len(core1_trials) > 0:
            actual_core1_ids = core1_trials["design_row_id"].unique().tolist()
            run_state["core1_last_batch_ids"] = actual_core1_ids
        else:
            logger.warning(f"No Core-1 trials found in batch {batch_id}")

        # Calculate aggregate metrics
        coverage = (
            self._compute_aggregate_coverage(all_trials_df)
            if len(all_trials_df) > 0
            else 0.0
        )
        gini = (
            self._compute_aggregate_gini(all_trials_df)
            if len(all_trials_df) > 0
            else 0.3
        )

        # Count Core-1 repeats
        if "is_core1_repeat" in all_trials_df.columns and len(core1_trials) > 0:
            repeats = core1_trials[core1_trials["is_core1_repeat"] == True]
            core1_repeat_rate = len(repeats) / len(core1_trials)
        else:
            core1_repeat_rate = 0.0

        # Count trials by block
        block_counts = {
            "core1": len(all_trials_df[all_trials_df["block_type"] == "core1"]),
            "main": len(all_trials_df[all_trials_df["block_type"] == "core2"]),
            "interaction": len(
                all_trials_df[all_trials_df["block_type"] == "interaction"]
            ),
            "boundary": len(all_trials_df[all_trials_df["block_type"] == "boundary"]),
            "lhs": len(all_trials_df[all_trials_df["block_type"] == "lhs"]),
        }

        # Append to history
        subject_ids = all_trials_df["subject_id"].unique().tolist()
        history_entry = {
            "batch_id": batch_id,
            "subject_ids": subject_ids,
            "n_subjects": len(subject_ids),
            "coverage": coverage,
            "gini": gini,
            "core1_repeat_rate": core1_repeat_rate,
            "block_counts": block_counts,
            "timestamp": datetime.now().isoformat(),
        }
        run_state["history"].append(history_entry)

        # Strategy adjustment: if coverage < 0.6 or gini > 0.6, increase LHS/boundary by 10%
        if coverage < 0.6 or gini > 0.6:
            logger.warning(
                f"Coverage {coverage:.2f} or Gini {gini:.2f} below target. "
                f"Will increase LHS/boundary by 10% for next batch."
            )
            # This adjustment should be reflected in budget_split for next batch
            # Note: Implementation in WarmupGenerator to apply dynamic adjustments
            run_state["strategy_adjustment"] = {
                "batch_triggered": batch_id,
                "lhs_increase_pct": 10,
                "boundary_increase_pct": 10,
            }

        # Increment batch counter
        run_state["current_batch"] = batch_id + 1

        # Check if study complete
        if run_state["current_batch"] > run_state["n_batches"]:
            logger.info(f"Study {run_state['study_id']} completed!")
            run_state["status"] = "completed"
        else:
            run_state["status"] = "in_progress"

        logger.info(
            f"Updated run_state after batch {batch_id}: "
            f"coverage={coverage:.3f}, gini={gini:.3f}, repeat_rate={core1_repeat_rate:.3f}"
        )

        return run_state

    def _compute_aggregate_coverage(self, trials_df: pd.DataFrame) -> float:
        """Compute coverage rate from trials (requires importing Generator)."""
        try:
            from scout_warmup_generator import WarmupAEPsychGenerator

            dummy_gen = WarmupAEPsychGenerator(self.design_df, seed=self.seed)
            dummy_gen.fit_planning()
            return dummy_gen.compute_coverage_rate(trials_df)
        except Exception as e:
            logger.warning(f"Could not compute coverage: {e}")
            return 0.0

    def _compute_aggregate_gini(self, trials_df: pd.DataFrame) -> float:
        """Compute Gini coefficient from trials."""
        try:
            from scout_warmup_generator import WarmupAEPsychGenerator

            dummy_gen = WarmupAEPsychGenerator(self.design_df, seed=self.seed)
            dummy_gen.fit_planning()
            return dummy_gen.compute_gini(trials_df)
        except Exception as e:
            logger.warning(f"Could not compute Gini: {e}")
            return 0.3

    def save_global_plan(self, path: str) -> None:
        """
        Save the immutable global plan (libraries, budgets, strategy).

        Args:
            path: File path to save global_plan.json
        """
        import json
        from datetime import datetime

        plan = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_subjects": self.n_subjects,
                "n_batches": self.n_batches,
                "total_budget": self.total_budget,
                "seed": self.seed,
                "schema_version": "1.0",
            },
            "factors": {
                "names": self.factor_names,
                "types": self.factor_types,
                "discretized": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in self.discretized_factors.items()
                },
                "d": self.d,
            },
            "global_components": {
                "core1_candidates_indices": (
                    self.global_core1_candidates.index.tolist()
                    if self.global_core1_candidates is not None
                    else []
                ),
                "interaction_pairs": self.interaction_pairs,
                "n_boundary_points": len(self.boundary_library),
            },
            "budget_split": self.budget_split,
            "bridge_plan": self.bridge_plan,
            "warnings": self.warnings,
        }

        with open(path, "w") as f:
            json.dump(plan, f, indent=2, default=str)

        logger.info(f"Saved global plan to {path}")

    def export_subject_plan(self, subject_id: int, path: str) -> None:
        """
        Export a subject's complete plan for use by WarmupAEPsychGenerator.

        Standardized plan schema includes:
        - quotas: {core1, core2_main, interaction, boundary, lhs}
        - constraints: {core1_pool_indices, core1_repeat_indices, interaction_pairs,
                       boundary_library, policy}
        - seed: Per-subject RNG seed

        Args:
            subject_id: Subject ID
            path: File path to save subject_plan.json
        """
        import json

        plan = self.allocate_subject_plan(subject_id)

        # Get Core-1 indices from global candidates
        core1_pool_indices = None
        if self.global_core1_candidates is not None:
            if "design_row_id" in self.global_core1_candidates.columns:
                core1_pool_indices = self.global_core1_candidates[
                    "design_row_id"
                ].tolist()
            else:
                core1_pool_indices = self.global_core1_candidates.index.tolist()

        # Standardized plan with normalized constraint fields
        plan_export = {
            "subject_id": subject_id,
            "batch_id": plan.get("batch_id", 0),
            "is_bridge": plan.get("is_bridge", False),
            "quotas": {
                "core1": plan["quotas"].get("core1", 0),
                "core2_main": plan["quotas"].get("main", 0),
                "interaction": plan["quotas"].get("inter", 0),
                "boundary": plan["quotas"].get("boundary", 0),
                "lhs": plan["quotas"].get("lhs", 0),
            },
            "constraints": {
                "core1_pool_indices": core1_pool_indices,
                "core1_repeat_indices": plan["constraints"].get(
                    "core1_repeat_indices", []
                ),
                "interaction_pairs": self.interaction_pairs,
                "boundary_library": self.boundary_library,
                "policy": {
                    "core1_repeat_ratio": 0.5,
                    "coverage_min": 0.10,
                    "gini_max": 0.40,
                },
            },
            "seed": self.seed + subject_id,  # Per-subject RNG seed
            "schema_version": "2.0",
        }

        with open(path, "w") as f:
            json.dump(plan_export, f, indent=2, default=str)

        logger.info(f"Exported subject plan for subject {subject_id} to {path}")

    def validate_global_constraints(self, trials_all: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate global constraints after batch execution.

        Args:
            trials_all: Combined trials DataFrame from all subjects/batches

        Returns:
            Validation results dictionary
        """
        result = {
            "core1_repeat_ratio": 0.0,
            "bridge_coverage": {},
            "coverage_rate": 0.0,
            "gini_coefficient": 0.0,
            "warnings": [],
        }

        # Core-1 repeat ratio check
        if (
            self.global_core1_candidates is not None
            and len(self.global_core1_candidates) > 0
        ):
            core1_trials = trials_all[trials_all["block_type"] == "core1"]
            if len(core1_trials) > 0:
                core1_indices = core1_trials["design_row_id"].unique()

                # Count how many Core-1 points appear in multiple batches
                batch_appearances = {}
                for idx in core1_indices:
                    batches = core1_trials[core1_trials["design_row_id"] == idx][
                        "batch_id"
                    ].unique()
                    batch_appearances[idx] = len(batches)

                repeat_count = sum(1 for b in batch_appearances.values() if b > 1)
                result["core1_repeat_ratio"] = (
                    repeat_count / len(core1_indices) if len(core1_indices) > 0 else 0
                )

                if result["core1_repeat_ratio"] < 0.5:
                    result["warnings"].append(
                        f"⚠️ Core-1 repeat ratio {result['core1_repeat_ratio']:.1%} < 50% target"
                    )

        # Bridge subject coverage check
        if "total_bridge_subjects" in self.bridge_plan:
            bridge_subjects = trials_all[trials_all["is_bridge"] == True]
            for subject_id in range(self.bridge_plan.get("total_bridge_subjects", 0)):
                if subject_id in bridge_subjects["subject_id"].values:
                    subject_batches = bridge_subjects[
                        bridge_subjects["subject_id"] == subject_id
                    ]["batch_id"].unique()
                    if len(subject_batches) >= 2:
                        result["bridge_coverage"][subject_id] = "✓"
                    else:
                        result["bridge_coverage"][subject_id] = "✗"
                        result["warnings"].append(
                            f"⚠️ Bridge subject {subject_id} does not span ≥2 batches"
                        )

        # Global coverage and Gini (requires importing Generator)
        try:
            from scout_warmup_generator import WarmupAEPsychGenerator

            dummy_gen = WarmupAEPsychGenerator(self.design_df, seed=self.seed)
            dummy_gen.fit_planning()
            result["coverage_rate"] = dummy_gen.compute_coverage_rate(trials_all)
            result["gini_coefficient"] = dummy_gen.compute_gini(trials_all)
        except ImportError:
            logger.warning(
                "Could not import WarmupAEPsychGenerator for coverage/Gini computation"
            )
            result["coverage_rate"] = None
            result["gini_coefficient"] = None

        return result

    def check_core1_repeat_policy(self) -> Dict[str, Any]:
        """
        Static check for Core-1 repeat policy feasibility.

        Returns:
            Dictionary with feasibility check results
        """
        n_core1 = (
            len(self.global_core1_candidates)
            if self.global_core1_candidates is not None
            else 0
        )
        threshold = self.bridge_plan.get("core1_repeat_threshold", 0.5)

        # Check if we have enough Core-1 points for repeats
        min_repeats_needed = int(n_core1 * threshold)

        result = {
            "n_core1_global": n_core1,
            "repeat_threshold": threshold,
            "min_repeats_needed": min_repeats_needed,
            "feasible": n_core1 >= min_repeats_needed,
        }

        if not result["feasible"]:
            logger.warning(f"Core-1 repeat policy may not be feasible: {result}")

        return result


# ========== Self-Test Entry Point ==========

if __name__ == "__main__":
    print("=" * 60)
    print("SCOUT Study Coordinator - Self Test")
    print("=" * 60)

    # Generate synthetic design_df
    np.random.seed(42)
    n_candidates = 300
    d = 5

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_candidates)
    design_data["metadata"] = [f"stim_{i}" for i in range(n_candidates)]

    design_df = pd.DataFrame(design_data)
    print(f"\nCreated design_df: {len(design_df)} candidates, {d} factors")

    # Create coordinator
    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42
    )

    # Fit initial plan
    coordinator.fit_initial_plan()

    # Generate subject plans for first 3 subjects
    print("\n" + "-" * 60)
    print("Subject Plans:")
    print("-" * 60)

    for subject_id in range(3):
        subject_plan = coordinator.allocate_subject_plan(subject_id)

        print(f"\nSubject {subject_id}:")
        print(f"  Batch: {subject_plan['batch_id']}")
        print(f"  Bridge: {subject_plan['is_bridge']}")
        print(f"  Quotas: {subject_plan['quotas']}")
        print(
            f"  Must-include: {len(subject_plan['constraints']['must_include_design_ids'])} points"
        )
        print(
            f"  Interactions: {len(subject_plan['constraints']['interactions'])} pairs"
        )

    # Global summary
    print("\n" + "-" * 60)
    print("Global Summary:")
    print("-" * 60)

    summary = coordinator.summarize_global()
    print(f"Study parameters: {summary['study_parameters']}")
    print(f"Global components: {summary['global_components']}")
    print(f"Expected coverage: {summary['expected_coverage']:.3f}")
    print(f"Warnings: {len(summary['warnings'])}")

    # Core-1 repeat policy check
    repeat_check = coordinator.check_core1_repeat_policy()
    print(f"\nCore-1 repeat policy check:")
    print(f"  Feasible: {repeat_check['feasible']}")
    print(f"  Global Core-1 points: {repeat_check['n_core1_global']}")
    print(f"  Min repeats needed: {repeat_check['min_repeats_needed']}")

    print("\n" + "=" * 60)
    print("Self test completed successfully!")
    print("=" * 60)
