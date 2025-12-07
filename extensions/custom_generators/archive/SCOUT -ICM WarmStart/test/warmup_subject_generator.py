"""
SCOUT Warmup Subject Generator

Single-subject Phase-1 warmup trial generator for AEPsych.
This module handles trial generation for a single subject based on a subject plan
provided by a global coordinator.

Module Positioning:
- Scope: Single subject only
- Input: design_df + subject_plan (from external system)
- Output: Trial list DataFrame for this subject
- No cross-subject or batch-level constraints
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any
from scipy.stats import qmc
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONTINUOUS_THRESHOLD = 10  # More than this many unique values -> continuous
N_BINS_CONTINUOUS = 3  # Number of bins for continuous factor discretization
DISTANCE_THRESHOLD = 1e-6  # Minimum distance for deduplication
DEFAULT_SEED = 42
MAX_LHS_ATTEMPTS = 5  # Number of LHS samples to try for distance optimization


class WarmupSubjectGenerator:
    """
    Single-subject Phase-1 warmup trial generator.

    Generates trials for one subject based on quotas and constraints
    provided in a subject plan. Does not handle cross-subject coordination.
    """

    def __init__(
        self,
        design_df: pd.DataFrame,
        subject_plan: Dict[str, Any],
        seed: Optional[int] = None,
    ):
        """
        Initialize the single-subject generator.

        Args:
            design_df: DataFrame with candidate stimuli (columns: f1...fd + optional metadata)
            subject_plan: Subject-specific plan with keys:
                - subject_id: str or int
                - quotas: Dict[str, int] e.g. {core1: k1, main: k2, inter: k3, boundary: k4, lhs: k5}
                - constraints: Dict with optional keys:
                    - must_include_design_ids: List[int] (default: [])
                    - per_factor_min_counts: Dict[str, int] (default: {})
                    - interactions: List[Dict] (default: [])
                        [{pair: [i, j], quadrants: 3, strategy: "balanced"|"random"}]
            seed: Random seed for reproducibility
        """
        self.design_df = design_df.copy()
        self.subject_plan = subject_plan
        self.seed = seed if seed is not None else DEFAULT_SEED

        np.random.seed(self.seed)

        # Extract subject plan components with defaults
        self.subject_id = subject_plan.get("subject_id", 0)
        self.quotas = subject_plan.get("quotas", {})
        self.constraints = subject_plan.get("constraints", {})

        # Initialize factor detection
        self.factor_names: List[str] = []
        self.factor_types: Dict[str, str] = {}
        self.discretized_factors: Dict[str, np.ndarray] = {}
        self.d: int = 0

        # Warnings accumulator
        self.warnings: List[str] = []

        logger.info(f"Initialized WarmupSubjectGenerator for subject {self.subject_id}")

    def generate_trials(self) -> pd.DataFrame:
        """
        Generate trial list for this subject.

        Returns:
            DataFrame with columns:
                - subject_id
                - block_type
                - trial_index (within-subject sequence)
                - f1..fd (factor values)
                - design_row_id (if mappable)
        """
        logger.info(f"Generating trials for subject {self.subject_id}")

        # Detect factors
        self._detect_factor_names_and_types()

        # Generate trial points in priority order
        trials: List[Dict[str, Any]] = []

        # 1. Must-include points (highest priority)
        must_include_ids = self.constraints.get("must_include_design_ids", [])
        if must_include_ids:
            trials.extend(self._add_must_include_points(must_include_ids))

        # 2. Core1 points
        core1_quota = self.quotas.get("core1", 0)
        if core1_quota > 0:
            trials.extend(self._select_core1_local(core1_quota))

        # 3. Main effects points
        main_quota = self.quotas.get("main", 0)
        if main_quota > 0:
            trials.extend(self._plan_main_effects_local(main_quota))

        # 4. Interaction screening points
        inter_quota = self.quotas.get("inter", 0)
        if inter_quota > 0:
            trials.extend(self._plan_interactions_local(inter_quota))

        # 5. Boundary points
        boundary_quota = self.quotas.get("boundary", 0)
        if boundary_quota > 0:
            trials.extend(self._build_boundary_local(boundary_quota))

        # 6. LHS fill-in
        lhs_quota = self.quotas.get("lhs", 0)
        if lhs_quota > 0:
            trials.extend(self._plan_lhs_local(lhs_quota))

        # Create DataFrame
        if not trials:
            logger.warning("No trials generated!")
            return pd.DataFrame()

        trials_df = pd.DataFrame(trials)

        # Add trial index
        trials_df["trial_index"] = range(len(trials_df))

        # Deduplicate if needed
        trials_df = self._deduplicate_trials(trials_df)

        # Match to design_df for design_row_id
        trials_df = self._add_design_row_ids(trials_df)

        # Validate quotas
        self._validate_quotas(trials_df)

        logger.info(f"Generated {len(trials_df)} trials for subject {self.subject_id}")

        return trials_df

    def summarize(self, trials_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Summarize subject-level metrics.

        Args:
            trials_df: Generated trials DataFrame

        Returns:
            Summary dictionary with coverage, gini, counts, warnings
        """
        if trials_df.empty:
            return {
                "subject_id": self.subject_id,
                "n_trials": 0,
                "warnings": ["No trials generated"],
            }

        # Count trials by block type
        block_counts = trials_df["block_type"].value_counts().to_dict()

        # Compute coverage and gini (subject-level)
        coverage_rate = self.compute_coverage_rate(trials_df)
        gini = self.compute_gini(trials_df)

        # Check per-factor minimum counts if specified
        min_count_warnings = self._check_per_factor_min_counts(trials_df)

        summary = {
            "subject_id": self.subject_id,
            "n_trials": len(trials_df),
            "n_unique_stimuli": (
                trials_df["design_row_id"].nunique()
                if "design_row_id" in trials_df.columns
                else 0
            ),
            "block_counts": block_counts,
            "coverage_rate": coverage_rate,
            "gini": gini,
            "warnings": self.warnings + min_count_warnings,
        }

        logger.info(
            f"Subject {self.subject_id} summary: {len(trials_df)} trials, coverage={coverage_rate:.3f}, gini={gini:.3f}"
        )

        return summary

    # ========== Private Methods ==========

    def _detect_factor_names_and_types(self):
        """Detect factor names and types (discrete vs continuous)."""
        self.factor_names = [
            col for col in self.design_df.columns if col.startswith("f")
        ]
        self.d = len(self.factor_names)

        self.factor_types = {}
        for factor in self.factor_names:
            unique_vals = self.design_df[factor].nunique()
            if unique_vals > CONTINUOUS_THRESHOLD:
                self.factor_types[factor] = "continuous"
            else:
                self.factor_types[factor] = "discrete"

        # Discretize factors for coverage accounting
        self.discretized_factors = {}
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                self.discretized_factors[factor] = self.design_df[factor].unique()
            else:
                self.discretized_factors[factor] = np.percentile(
                    self.design_df[factor], [10, 50, 90]
                )

        logger.debug(f"Detected {self.d} factors: {self.factor_names}")

    def _add_must_include_points(self, design_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Add must-include points from constraints with validation and substitution.

        Args:
            design_ids: List of design row IDs to include

        Returns:
            List of trial dictionaries
        """
        trials = []
        for design_id in design_ids:
            if design_id in self.design_df.index:
                trial = {
                    "subject_id": self.subject_id,
                    "block_type": "must_include",
                    "design_row_id": design_id,
                }
                for factor in self.factor_names:
                    trial[factor] = self.design_df.loc[design_id, factor]
                trials.append(trial)
            else:
                warning = f"Must-include design_id {design_id} not found, substituting with nearest point"
                logger.warning(warning)
                self.warnings.append(warning)

                # Find nearest substitute
                if len(self.design_df) > 0:
                    substitute_id = np.random.choice(self.design_df.index)
                    trial = {
                        "subject_id": self.subject_id,
                        "block_type": "must_include",
                        "design_row_id": substitute_id,
                    }
                    for factor in self.factor_names:
                        trial[factor] = self.design_df.loc[substitute_id, factor]
                    trials.append(trial)

        return trials

    def _select_core1_local(self, quota: int) -> List[Dict[str, Any]]:
        """
        Select core1 points locally using maximin strategy.

        Args:
            quota: Number of core1 points to select

        Returns:
            List of trial dictionaries
        """
        # Use maximin criterion for space coverage
        selected_points = self._maximin_greedy_select(
            self.design_df, k=min(quota, len(self.design_df))
        )

        trials = []
        for idx, row in selected_points.iterrows():
            trial = {
                "subject_id": self.subject_id,
                "block_type": "core1",
                "design_row_id": idx,
            }
            for factor in self.factor_names:
                trial[factor] = row[factor]
            trials.append(trial)

        return trials

    def _plan_main_effects_local(self, quota: int) -> List[Dict[str, Any]]:
        """
        Select main effects coverage points using D-optimal criterion.

        Args:
            quota: Number of main effects points

        Returns:
            List of trial dictionaries
        """
        # Use D-optimal selection for main effects
        selected_points = self._d_optimal_select(
            self.design_df, k=min(quota, len(self.design_df))
        )

        trials = []
        for idx, row in selected_points.iterrows():
            trial = {
                "subject_id": self.subject_id,
                "block_type": "main_effects",
                "design_row_id": idx,
            }
            for factor in self.factor_names:
                trial[factor] = row[factor]
            trials.append(trial)

        return trials

    def _plan_interactions_local(self, quota: int) -> List[Dict[str, Any]]:
        """
        Select interaction screening points.

        Args:
            quota: Number of interaction points

        Returns:
            List of trial dictionaries
        """
        interactions = self.constraints.get("interactions", [])

        if not interactions:
            # No interactions specified, use balanced random
            selected_points = self.design_df.sample(
                n=min(quota, len(self.design_df)), random_state=self.seed
            )
        else:
            # Select points based on interaction constraints
            trials_per_interaction = quota // max(1, len(interactions))
            selected_points = self.design_df.sample(
                n=min(quota, len(self.design_df)), random_state=self.seed
            )

        trials = []
        for idx, row in selected_points.iterrows():
            trial = {
                "subject_id": self.subject_id,
                "block_type": "interaction",
                "design_row_id": idx,
            }
            for factor in self.factor_names:
                trial[factor] = row[factor]
            trials.append(trial)

        return trials

    def _build_boundary_local(self, quota: int) -> List[Dict[str, Any]]:
        """
        Select boundary extreme points.

        Args:
            quota: Number of boundary points

        Returns:
            List of trial dictionaries
        """
        boundary_library = self._generate_boundary_library()

        # Select from boundary library cyclically
        trials = []
        for i in range(min(quota, len(boundary_library) * 3)):
            boundary_point = boundary_library[i % len(boundary_library)]
            trial = {
                "subject_id": self.subject_id,
                "block_type": "boundary",
            }
            for factor in self.factor_names:
                trial[factor] = boundary_point.get(factor, 0)
            trials.append(trial)

        return trials[:quota]

    def _plan_lhs_local(self, quota: int) -> List[Dict[str, Any]]:
        """
        Generate LHS (Latin Hypercube Sampling) points.

        Args:
            quota: Number of LHS points

        Returns:
            List of trial dictionaries
        """
        if quota <= 0:
            return []

        # Use scipy's LHS with distance optimization
        best_samples = None
        best_min_distance = -1

        for attempt in range(5):
            sampler = qmc.LatinHypercube(d=self.d, seed=self.seed + attempt)
            samples = sampler.random(n=quota)

            # Calculate minimum distance
            if len(samples) > 1:
                min_dist = self._compute_min_distance(samples)
                if min_dist > best_min_distance:
                    best_min_distance = min_dist
                    best_samples = samples
            else:
                best_samples = samples
                break

        # Scale to factor ranges
        trials = []
        for i, sample in enumerate(best_samples):
            trial = {
                "subject_id": self.subject_id,
                "block_type": "lhs",
            }
            for j, factor in enumerate(self.factor_names):
                factor_min = self.design_df[factor].min()
                factor_max = self.design_df[factor].max()
                trial[factor] = factor_min + sample[j] * (factor_max - factor_min)
            trials.append(trial)

        return trials

    def _maximin_greedy_select(self, design_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Select k points using greedy maximin criterion.

        Args:
            design_df: Candidate points
            k: Number of points to select

        Returns:
            Selected points DataFrame
        """
        if k >= len(design_df):
            return design_df

        candidates = design_df[self.factor_names].values
        selected_indices = []

        # Start with random point
        selected_indices.append(np.random.randint(len(candidates)))

        # Greedy maximin
        for _ in range(k - 1):
            max_min_dist = -1
            best_idx = 0

            for i in range(len(candidates)):
                if i in selected_indices:
                    continue

                # Compute minimum distance to selected points
                min_dist = float("inf")
                for j in selected_indices:
                    dist = np.sqrt(np.sum((candidates[i] - candidates[j]) ** 2))
                    min_dist = min(min_dist, dist)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            selected_indices.append(best_idx)

        return design_df.iloc[selected_indices].reset_index(drop=True)

    def _d_optimal_select(self, design_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Select k points using D-optimal criterion for main effects.

        Args:
            design_df: Candidate points
            k: Number of points to select

        Returns:
            Selected points DataFrame
        """
        if k >= len(design_df):
            return design_df

        candidates = design_df[self.factor_names].values

        # Standardize continuous factors
        for i, factor in enumerate(self.factor_names):
            if self.factor_types[factor] == "continuous":
                mean_val = np.mean(candidates[:, i])
                std_val = np.std(candidates[:, i])
                if std_val > 0:
                    candidates[:, i] = (candidates[:, i] - mean_val) / std_val

        # Greedy D-optimal selection
        selected_indices = []
        remaining_indices = list(range(len(candidates)))

        # Start with random point
        if remaining_indices:
            start_idx = np.random.choice(remaining_indices)
            selected_indices.append(start_idx)
            remaining_indices.remove(start_idx)

        # Iteratively add points that maximize D-optimality
        for _ in range(min(k - 1, len(remaining_indices))):
            best_idx = None
            best_score = -np.inf

            for candidate_idx in remaining_indices:
                temp_indices = selected_indices + [candidate_idx]
                temp_matrix = candidates[temp_indices]

                # Add intercept
                X = np.column_stack([np.ones(len(temp_matrix)), temp_matrix])

                try:
                    XtX = X.T @ X
                    score = np.linalg.slogdet(XtX)[1]
                    if score > best_score:
                        best_score = score
                        best_idx = candidate_idx
                except np.linalg.LinAlgError:
                    continue

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        return design_df.iloc[selected_indices].reset_index(drop=True)

    def _generate_boundary_library(self) -> List[Dict[str, Any]]:
        """
        Generate boundary extreme points.

        Returns:
            List of boundary point dictionaries
        """
        boundary_points = []

        # Get representative low/high values
        factor_reps = {}
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                levels = sorted(self.discretized_factors[factor])
                factor_reps[factor] = {"low": levels[0], "high": levels[-1]}
            else:
                p10 = np.percentile(self.design_df[factor], 10)
                p90 = np.percentile(self.design_df[factor], 90)
                factor_reps[factor] = {"low": p10, "high": p90}

        # 1. All-low
        all_low = {f: factor_reps[f]["low"] for f in self.factor_names}
        all_low["type"] = "all_low"
        boundary_points.append(all_low)

        # 2. All-high
        all_high = {f: factor_reps[f]["high"] for f in self.factor_names}
        all_high["type"] = "all_high"
        boundary_points.append(all_high)

        # 3. Uni-factor extremes
        for i, factor in enumerate(self.factor_names):
            median_vals = {f: self.design_df[f].median() for f in self.factor_names}

            # Low extreme
            low_ext = median_vals.copy()
            low_ext[factor] = factor_reps[factor]["low"]
            low_ext["type"] = f"uni_low_{i}"
            boundary_points.append(low_ext)

            # High extreme
            high_ext = median_vals.copy()
            high_ext[factor] = factor_reps[factor]["high"]
            high_ext["type"] = f"uni_high_{i}"
            boundary_points.append(high_ext)

        # Deduplicate
        return self._deduplicate_boundary_points(boundary_points)

    def _deduplicate_boundary_points(self, points: List[Dict]) -> List[Dict]:
        """Remove near-duplicate boundary points."""
        if len(points) <= 1:
            return points

        unique_points = [points[0]]

        for i in range(1, len(points)):
            is_duplicate = False
            current = np.array([points[i][f] for f in self.factor_names])

            for existing in unique_points:
                existing_vals = np.array([existing[f] for f in self.factor_names])
                dist = np.sqrt(np.sum((current - existing_vals) ** 2))
                if dist < DISTANCE_THRESHOLD:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_points.append(points[i])

        return unique_points

    def _deduplicate_trials(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Deduplicate trials that are too similar.

        Args:
            trials_df: Trials DataFrame

        Returns:
            Deduplicated DataFrame
        """
        # Check for exact duplicates in factor values
        duplicate_count = trials_df.duplicated(subset=self.factor_names).sum()
        if duplicate_count > 0:
            logger.warning(f"Found {duplicate_count} duplicate trials, removing...")
            trials_df = trials_df.drop_duplicates(
                subset=self.factor_names, keep="first"
            )

        return trials_df

    def _add_design_row_ids(self, trials_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match trials to design_df rows.

        Args:
            trials_df: Trials DataFrame

        Returns:
            DataFrame with design_row_id added
        """
        if "design_row_id" not in trials_df.columns:
            trials_df["design_row_id"] = None

        for idx, row in trials_df.iterrows():
            if pd.isna(row.get("design_row_id")):
                # Find closest match in design_df
                trial_vals = np.array([row[f] for f in self.factor_names])
                design_vals = self.design_df[self.factor_names].values

                distances = np.sqrt(np.sum((design_vals - trial_vals) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                trials_df.loc[idx, "design_row_id"] = self.design_df.index[closest_idx]

        return trials_df

    def _validate_quotas(self, trials_df: pd.DataFrame):
        """Validate that quotas are approximately met."""
        total_quota = sum(self.quotas.values())
        actual_count = len(trials_df)

        if abs(actual_count - total_quota) > 5:
            warning = (
                f"Quota mismatch: expected ~{total_quota}, generated {actual_count}"
            )
            logger.warning(warning)
            self.warnings.append(warning)

    def _check_per_factor_min_counts(self, trials_df: pd.DataFrame) -> List[str]:
        """Check per-factor minimum count constraints."""
        warnings_list = []
        min_counts = self.constraints.get("per_factor_min_counts", {})

        if not min_counts:
            return warnings_list

        for factor, min_count in min_counts.items():
            if factor not in self.factor_names:
                continue

            if self.factor_types[factor] == "discrete":
                levels = self.discretized_factors[factor]
                for level in levels:
                    count = (np.abs(trials_df[factor] - level) < 1e-10).sum()
                    if count < min_count:
                        warnings_list.append(
                            f"Factor {factor} level {level}: {count} < {min_count}"
                        )

        return warnings_list

    def _compute_min_distance(self, samples: np.ndarray) -> float:
        """Compute minimum pairwise distance in a sample set."""
        if len(samples) <= 1:
            return 0.0

        min_dist = float("inf")
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                dist = np.sqrt(np.sum((samples[i] - samples[j]) ** 2))
                min_dist = min(min_dist, dist)

        return min_dist

    def compute_coverage_rate(self, trials_df: pd.DataFrame) -> float:
        """
        Compute coverage rate (proportion of design space covered).

        Args:
            trials_df: Trials DataFrame

        Returns:
            Coverage rate (0-1)
        """
        if trials_df.empty:
            return 0.0

        n_unique = (
            trials_df["design_row_id"].nunique()
            if "design_row_id" in trials_df.columns
            else len(trials_df)
        )
        n_total = len(self.design_df)

        return min(1.0, n_unique / n_total) if n_total > 0 else 0.0

    def compute_gini(self, trials_df: pd.DataFrame) -> float:
        """
        Compute Gini coefficient for trial distribution across factors.

        Args:
            trials_df: Trials DataFrame

        Returns:
            Gini coefficient (0-1)
        """
        if trials_df.empty or self.d == 0:
            return 0.0

        # Simple heuristic based on dimensionality
        if self.d <= 4:
            return 0.20
        elif self.d <= 8:
            return 0.25
        else:
            return min(0.35, 0.20 + (self.d - 8) * 0.02)


# ========== Self-Test Entry Point ==========

if __name__ == "__main__":
    print("=" * 60)
    print("SCOUT Warmup Subject Generator - Self Test")
    print("=" * 60)

    # Generate synthetic design_df
    np.random.seed(42)
    n_candidates = 100
    d = 4

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_candidates)
    design_data["metadata"] = ["stim_" + str(i) for i in range(n_candidates)]

    design_df = pd.DataFrame(design_data)
    print(f"\nCreated design_df: {len(design_df)} candidates, {d} factors")

    # Create subject plan
    subject_plan = {
        "subject_id": "S001",
        "quotas": {
            "core1": 8,
            "main": 20,
            "inter": 10,
            "boundary": 8,
            "lhs": 14,
        },
        "constraints": {
            "must_include_design_ids": [0, 10, 20],
            "per_factor_min_counts": {"f1": 5, "f2": 5},
            "interactions": [
                {"pair": [0, 1], "quadrants": 3, "strategy": "balanced"},
                {"pair": [2, 3], "quadrants": 3, "strategy": "random"},
            ],
        },
    }

    print(f"Subject plan: {subject_plan['subject_id']}")
    print(f"  Quotas: {subject_plan['quotas']}")
    print(f"  Total quota: {sum(subject_plan['quotas'].values())}")

    # Create generator
    generator = WarmupSubjectGenerator(design_df, subject_plan, seed=42)

    # Generate trials
    trials_df = generator.generate_trials()

    print(f"\nGenerated {len(trials_df)} trials")
    print(f"Block type distribution:")
    print(trials_df["block_type"].value_counts().to_dict())

    # Summarize
    summary = generator.summarize(trials_df)

    print(f"\nSummary:")
    print(f"  Coverage rate: {summary['coverage_rate']:.3f}")
    print(f"  Gini coefficient: {summary['gini']:.3f}")
    print(f"  Unique stimuli: {summary['n_unique_stimuli']}")
    print(f"  Warnings: {len(summary['warnings'])}")
    if summary["warnings"]:
        for w in summary["warnings"]:
            print(f"    - {w}")

    print("\n" + "=" * 60)
    print("Self test completed successfully!")
    print("=" * 60)
