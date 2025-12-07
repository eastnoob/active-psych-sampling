"""
SCOUT Warm-up Generator for AEPsych

A Phase-1 "Warm-up" Sampler that automatically chooses sampling parameters
and produces a Phase-1 design plan and sampled trial list.

Targets: measurement calibration (ICC, batch effects, test-retest),
coarse main effects estimation, initial GP model training, space coverage.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import warnings
import json
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import itertools
from scipy.stats import qmc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WarmupAEPsychGenerator:
    """
    A Phase-1 warm-up sampler for AEPsych that automatically generates a
    design plan and sampled trial list based on a design grid.
    """

    # Constants for discretization
    # Default for all dimensions; will be overridden in fit_planning() based on d
    N_BINS_CONTINUOUS = 3  # Number of bins for continuous factors

    def __init__(
        self,
        design_df: pd.DataFrame,
        n_subjects: int = 10,
        total_budget: int = 350,
        n_batches: int = 3,
        seed: Optional[int] = None,
        candidate_interactions: Optional[List[Tuple[int, int]]] = None,
        n_bins_continuous: Optional[int] = None,
    ):
        """
        Initialize the WarmupAEPsychGenerator.

        Args:
            design_df: A pandas DataFrame containing all candidate stimuli with
                      columns for factors (e.g., f1,...,fd) and optional metadata.
            n_subjects: Number of subjects (default=10)
            total_budget: Total number of trials (default=350)
            n_batches: Number of batches (default=3)
            seed: Random seed for reproducibility
            candidate_interactions: Optional list of (i,j) interaction pairs
            n_bins_continuous: Optional override for number of bins for continuous factors.
                             If None, will auto-adapt based on d in fit_planning().
                             Recommended: 2-3 for low-d, 3-5 for mid-d, 5+ for high-d.
        """
        self.design_df = design_df.copy()
        self.n_subjects = n_subjects
        self.total_budget = total_budget
        self.n_batches = n_batches
        self.seed = seed
        self.candidate_interactions = candidate_interactions
        self.n_bins_continuous_override = (
            n_bins_continuous  # User override, if provided
        )

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Initialize planning attributes
        self.d = None  # Number of factors
        self.factor_names = None
        self.factor_types = None
        self.discretized_factors = None
        self.budget_split = None
        self.n_core1_points = None
        self.core1_points = None
        self.main_effects_plan = None
        self.interaction_plan = None
        self.boundary_set = None
        self.lhs_plan = None
        self.batch_plan = None
        self.warnings = []
        self._n_bins_adaptive = (
            self.N_BINS_CONTINUOUS
        )  # Adaptive value, set in fit_planning()

    def fit_planning(self):
        """
        Analyze factors, discretize continuous factors to bins for coverage accounting,
        compute d, emit warnings, and compute all planning components.
        """
        # Detect factor types and get factor names
        self.factor_names = [
            col for col in self.design_df.columns if col.startswith("f")
        ]
        self.d = len(self.factor_names)

        # Adaptively set N_BINS_CONTINUOUS based on d and per-subject budget
        # User override takes precedence
        if self.n_bins_continuous_override is not None:
            self._n_bins_adaptive = self.n_bins_continuous_override
            logger.info(
                f"Using user-provided n_bins_continuous={self._n_bins_adaptive}"
            )
        else:
            # Auto-adapt based on dimensionality
            # Lower d → fewer bins (more coverage per bin, stable Gini)
            # Higher d → more bins (finer granularity, but harder to fill all bins)
            if self.d <= 4:
                self._n_bins_adaptive = 2  # Very coarse: focus on extreme coverage
            elif 5 <= self.d <= 8:
                self._n_bins_adaptive = 3  # Default: balanced coverage-gini
            elif 9 <= self.d <= 12:
                self._n_bins_adaptive = 4  # Mid-high-d: finer granularity
            else:  # d > 12
                self._n_bins_adaptive = 5  # Ultra-high-d: maximize spatial coverage
            logger.info(
                f"Auto-adapted n_bins_continuous={self._n_bins_adaptive} for d={self.d}"
            )

        # Emit warnings for high dimensionality
        if self.d > 10:
            warning_msg = (
                "Warning: d>10. Phase-1 warm-up becomes sample-hungry; "
                "tightening Core-2 and boundary share; slightly relaxing early GP targets. Proceeding."
            )
            self.warnings.append(warning_msg)
            warnings.warn(warning_msg)

        if self.d > 12:
            warning_msg = (
                "Strong Warning: d>12. Efficiency degrades; consider factor blocking "
                "and sparse DOE skeleton. Proceeding with reduced interaction pairs and higher boundary share."
            )
            self.warnings.append(warning_msg)
            warnings.warn(warning_msg)

        # Detect factor types and discretize continuous factors
        self.factor_types = self.detect_factor_types()
        self.discretized_factors = self.get_levels_or_bins()

        # Compute budget split based on dimensionality
        self.budget_split = self._compute_budget_split()

        # Compute core-1 points
        self.n_core1_points = self._compute_core1_size()
        self.core1_points = self.select_core1_points()

        # Compute main effects coverage plan
        self.main_effects_plan = self._plan_main_effects_coverage()

        # Compute interaction screening plan
        self.interaction_plan = self._plan_interaction_screening()

        # Compute boundary set
        self.boundary_set = self.build_boundary_library()

        # Compute LHS plan
        self.lhs_plan = self._plan_stratified_lhs()

        # Compute batch plan with bridges
        self.batch_plan = self.plan_batches_and_bridges()

        return self

    def apply_plan(self, plan: Dict) -> "WarmupAEPsychGenerator":
        """
        Apply Coordinator's subject plan to this generator.

        Completely overrides quotas, constraints, and RNG seed from the plan.
        This is the primary interface for multi-subject, multi-batch coordination.

        Plan schema from Coordinator.make_subject_plan():
        {
            "subject_id": int,
            "batch_id": int,
            "is_bridge": bool,
            "quotas": {
                "core1": int,
                "main": int,
                "inter": int,
                "boundary": int,
                "lhs": int
            },
            "constraints": {
                "core1_pool_indices": [design_row_id],
                "core1_repeat_indices": [design_row_id],  # From prev batch
                "interaction_pairs": [(i,j), ...],
                "boundary_library": [point_dict, ...]
            },
            "seed": int
        }

        Args:
            plan: Subject plan dict as specified above

        Returns:
            Self for method chaining
        """
        # Store subject/batch metadata
        self.subject_id = plan.get("subject_id", 0)
        self.batch_id = plan.get("batch_id", 0)
        self.is_bridge = plan.get("is_bridge", False)

        # Extract and cache quotas
        if "quotas" in plan:
            quotas = plan["quotas"]
            # Override global quotas with subject-level allocations
            self.n_core1_points = quotas.get("core1", 10)
            self._quota_overrides = {
                "core1": quotas.get("core1", 10),
                "core2_main": quotas.get("main", 15),
                "core2_inter": quotas.get("inter", 6),
                "boundary": quotas.get("boundary", 3),
                "lhs": quotas.get("lhs", 3),
            }
        else:
            self._quota_overrides = {}

        # Extract and cache constraints
        if "constraints" in plan:
            constraints = plan["constraints"]

            # Core-1 pool and repeat indices (critical for bridge continuity)
            self.core1_pool_indices = constraints.get("core1_pool_indices", None)
            self.core1_repeat_indices = constraints.get("core1_repeat_indices", None)

            # Boundary library override
            if "boundary_library" in constraints and constraints["boundary_library"]:
                self.boundary_set = constraints["boundary_library"]

            # Interaction pairs override
            if "interaction_pairs" in constraints and constraints["interaction_pairs"]:
                self.interaction_pairs = constraints["interaction_pairs"]
        else:
            self.core1_pool_indices = None
            self.core1_repeat_indices = None

        # Set per-subject RNG seed (critical for reproducibility)
        # IMPORTANT: This MUST be done before ANY random operations
        if "seed" in plan:
            np.random.seed(plan["seed"])
            self.seed = plan["seed"]
            logger.debug(f"Forced RNG seed to {plan['seed']} from plan")
        else:
            logger.warning(
                "No seed provided in plan; RNG state may be non-deterministic"
            )

        logger.info(
            f"Applied plan for subject {self.subject_id} batch {self.batch_id} "
            f"(bridge={self.is_bridge}, core1_quota={self.n_core1_points}, seed={self.seed})"
        )

        return self

    def generate_trials(self, save_to: Optional[str] = None) -> pd.DataFrame:
        """
        Generate the trial schedule DataFrame.

        Args:
            save_to: Optional file path to save trials (.parquet or .csv)

        Returns:
            A DataFrame with trial schedule including subject_id, batch_id,
            is_bridge, block_type, and factor values.
        """
        if self.d is None:
            raise ValueError("Must call fit_planning() before generate_trials()")

        # Generate all trial points
        trial_points = []

        # Add Core-1 points (rated by all subjects)
        core1_trials = self._generate_core1_trials()
        trial_points.extend(core1_trials)

        # Add Core-2 points
        core2_trials = self._generate_core2_trials()
        trial_points.extend(core2_trials)

        # Add Individual points
        individual_trials = self._generate_individual_trials()
        trial_points.extend(individual_trials)

        # Create DataFrame
        trial_schedule_df = pd.DataFrame(trial_points)

        # Add design_row_id referencing the row in design_df
        trial_schedule_df = self._add_design_row_ids(trial_schedule_df)

        # Add seed column for reproducibility tracking
        if "seed" not in trial_schedule_df.columns:
            trial_schedule_df["seed"] = self.seed

        # Store for later use in summarize()
        self.trial_schedule_df = trial_schedule_df

        # Run validation hooks (single call)
        self._run_validation_hooks(trial_schedule_df)

        # Save to file if requested
        if save_to:
            if save_to.endswith(".parquet"):
                try:
                    trial_schedule_df.to_parquet(save_to)
                except ImportError:
                    # Fallback to CSV if parquet not available
                    csv_path = save_to.replace(".parquet", ".csv")
                    trial_schedule_df.to_csv(csv_path, index=False)
            elif save_to.endswith(".csv"):
                trial_schedule_df.to_csv(save_to, index=False)
            else:
                # Default to CSV
                trial_schedule_df.to_csv(save_to + ".csv", index=False)

        return trial_schedule_df

    def _run_validation_hooks(self, trial_schedule_df: pd.DataFrame):
        """
        Run validation hooks to ensure the generated trials meet requirements.

        Args:
            trial_schedule_df: DataFrame with trial schedule
        """
        # Assert that per-level/bin minimum counts are met before LHS
        self._validate_marginal_coverage(trial_schedule_df)

        # Assert that bridge subjects are assigned across batches
        self._validate_bridge_subjects(trial_schedule_df)

        # Assert that ≥50% of Core-1 points repeat across adjacent batches
        self._validate_core1_repeats(trial_schedule_df)

        # Emit a coverage and Gini report (passing trials_df)
        self._emit_coverage_report(trial_schedule_df)

    def _validate_marginal_coverage(self, trial_schedule_df: pd.DataFrame):
        """
        Validate that per-level/bin minimum counts are met by actually counting occurrences.

        Uses consistent half-open interval [low, high) for all bins, with last bin containing right endpoint.
        This matches compute_gini() logic for consistency.

        Args:
            trial_schedule_df: DataFrame with trial schedule
        """
        min_counts = self.compute_marginal_min_counts()

        # Count actual occurrences for each factor
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                # For discrete factors, count each level
                levels = self.discretized_factors[factor]
                for level in levels:
                    # Count how many times this level appears in trials
                    level_count = 0
                    for _, trial in trial_schedule_df.iterrows():
                        if (
                            abs(trial[factor] - level) < 1e-10
                        ):  # Account for floating point precision
                            level_count += 1

                    required_count = min_counts[factor]
                    if level_count < required_count:
                        msg = (
                            f"Factor {factor} level {level} does not meet minimum count requirement: "
                            f"required {required_count}, actual {level_count}"
                        )
                        warnings.warn(msg)
                        self.warnings.append(msg)
            else:
                # For continuous factors, count bins using consistent [low, high) logic
                bins = self.discretized_factors[factor]
                # Create bin boundaries from discretized factors
                bin_boundaries = bins  # These are already the bin edges

                for i in range(len(bin_boundaries) - 1):
                    lower_bound = bin_boundaries[i]
                    upper_bound = bin_boundaries[i + 1]

                    # Count trials in this bin using [low, high) logic
                    # Last bin includes right endpoint
                    if i == len(bin_boundaries) - 2:  # Last bin
                        bin_count = len(
                            trial_schedule_df[
                                (trial_schedule_df[factor] >= lower_bound)
                                & (trial_schedule_df[factor] <= upper_bound)
                            ]
                        )
                    else:
                        bin_count = len(
                            trial_schedule_df[
                                (trial_schedule_df[factor] >= lower_bound)
                                & (trial_schedule_df[factor] < upper_bound)
                            ]
                        )

                    required_count = min_counts[factor]
                    if bin_count < required_count:
                        msg = (
                            f"Factor {factor} bin {i} does not meet minimum count requirement: "
                            f"required {required_count}, actual {bin_count}"
                        )
                        warnings.warn(msg)
                        self.warnings.append(msg)

    def _validate_bridge_subjects(self, trial_schedule_df: pd.DataFrame):
        """
        Validate bridge subject assignment (local consistency check).

        Checks if the generator correctly honored the is_bridge flag from apply_plan.
        Global cross-batch bridge continuity is validated in Coordinator.validate_global_constraints().

        Args:
            trial_schedule_df: DataFrame with trial schedule
        """
        # Local check: verify is_bridge column exists and has consistent values
        if "is_bridge" not in trial_schedule_df.columns:
            warnings.warn("is_bridge column missing from trial schedule")
            return

        # Check if bridge flag matches plan (if apply_plan was called)
        if hasattr(self, "is_bridge"):
            expected_is_bridge = self.is_bridge
            actual_is_bridge = (
                trial_schedule_df["is_bridge"].iloc[0]
                if len(trial_schedule_df) > 0
                else expected_is_bridge
            )
            if actual_is_bridge != expected_is_bridge:
                warnings.warn(
                    f"Bridge subject flag mismatch: expected {expected_is_bridge}, got {actual_is_bridge}"
                )

    def _validate_core1_repeats(self, trial_schedule_df: pd.DataFrame):
        """
        Validate Core-1 repeat implementation (local consistency check).

        Checks if the generator correctly placed core1_repeat_indices when apply_plan provided them.
        Global cross-batch repeat coverage is validated in Coordinator.validate_global_constraints().

        Args:
            trial_schedule_df: DataFrame with trial schedule
        """
        # Local check: verify is_core1_repeat column exists
        if "is_core1_repeat" not in trial_schedule_df.columns:
            return

        # Check if repeat indices were honored (if apply_plan provided them)
        if hasattr(self, "core1_repeat_indices") and self.core1_repeat_indices:
            core1_trials = trial_schedule_df[trial_schedule_df["block_type"] == "core1"]
            if len(core1_trials) == 0:
                return

            # Verify that core1_repeat_indices were included in trials
            repeat_design_ids = set(self.core1_repeat_indices)
            actual_repeat_ids = set(
                core1_trials[core1_trials["is_core1_repeat"] == True][
                    "design_row_id"
                ].unique()
            )

            missing_repeats = repeat_design_ids - actual_repeat_ids
            if missing_repeats:
                warnings.warn(
                    f"Some core1_repeat_indices were not included: {missing_repeats}"
                )

    def _emit_coverage_report(self, trials_df: pd.DataFrame):
        """
        Emit a detailed coverage and Gini report based on generated trials.

        Args:
            trials_df: DataFrame with generated trials
        """
        coverage_rate = self.compute_coverage_rate(trials_df)
        gini_coeff = self.compute_gini(trials_df)

        print(f"Coverage Report:")
        print(f"  Coverage Rate: {coverage_rate:.3f} (target >0.10)")
        print(f"  Gini Coefficient: {gini_coeff:.3f} (target <0.40)")
        print(f"  Target Metrics:")
        print(f"    - ICC target: ≥0.45 (min ≥0.30)")
        print(f"    - Batch effect magnitude target: <0.20 (min <0.30)")
        print(f"    - Test–retest: ≥0.80 (min ≥0.70)")
        print(f"    - Main effect SE target: <0.12 (min <0.15)")
        print(f"    - GP CV-RMSE target: <0.85 (min <1.0)")

        # Check against targets
        if coverage_rate < 0.08:
            warnings.warn("Coverage rate below minimum threshold (0.08)")
        if gini_coeff > 0.50:
            warnings.warn("Gini coefficient above maximum threshold (0.50)")

    def export_metadata(self, path: str) -> None:
        """
        Export audit trail metadata (subject, batch, seed, coverage, Gini, etc.).

        Args:
            path: File path to save metadata (.json)
        """
        from datetime import datetime

        metadata = {
            "subject_id": getattr(self, "subject_id", None),
            "batch_id": getattr(self, "batch_id", None),
            "is_bridge": getattr(self, "is_bridge", False),
            "timestamp": datetime.now().isoformat(),
            "seed": self.seed,
            "design_shape": self.design_df.shape,
            "factors": {
                "names": self.factor_names,
                "types": self.factor_types,
                "d": self.d,
            },
            "budget": {
                "n_core1": getattr(self, "n_core1_points", 0),
                "n_core2": getattr(self, "n_core2_points", 0),
                "n_individual": getattr(self, "n_individual_points", 0),
                "n_boundary": getattr(self, "n_boundary_points", 0),
                "total": (
                    getattr(self, "n_core1_points", 0)
                    + getattr(self, "n_core2_points", 0)
                    + getattr(self, "n_individual_points", 0)
                    + getattr(self, "n_boundary_points", 0)
                ),
            },
            "schema_version": "1.0",
        }

        # Add coverage and Gini if available
        if hasattr(self, "trial_schedule_df"):
            metadata["coverage_rate"] = self.compute_coverage_rate(
                self.trial_schedule_df
            )
            metadata["gini_coefficient"] = self.compute_gini(self.trial_schedule_df)
        else:
            metadata["coverage_rate"] = None
            metadata["gini_coefficient"] = None

        # Add warnings if any
        if hasattr(self, "warnings") and self.warnings:
            metadata["warnings"] = self.warnings

        with open(path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def summarize(self, save_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Produce complete summary for Coordinator's update_after_batch().

        CRITICAL: This summary is consumed by Coordinator.update_after_batch() to:
        1. Track coverage and Gini metrics
        2. Decide if strategy adjustment (LHS/boundary +10%) is needed
        3. Record subject performance in history

        Args:
            save_to: Optional file path to save summary as JSON

        Returns:
            Dictionary with keys:
            {
                "metadata": {coverage, gini, core1_repeat_rate, seed, block_counts, ...},
                "measurement_quality": {...},
                "spatial_coverage": {...},
                ...
            }
        """
        if self.d is None:
            raise ValueError("Must call fit_planning() before summarize()")

        # Calculate unique stimuli count
        n_unique_stimuli = len(self.design_df)

        # Compute metrics from generated trials
        if hasattr(self, "trial_schedule_df") and self.trial_schedule_df is not None:
            coverage_rate = self.compute_coverage_rate(self.trial_schedule_df)
            gini_coeff = self.compute_gini(self.trial_schedule_df)

            # Count Core-1 repeats **precisely**
            if "is_core1_repeat" in self.trial_schedule_df.columns:
                core1_trials = self.trial_schedule_df[
                    self.trial_schedule_df["block_type"] == "core1"
                ]
                if len(core1_trials) > 0:
                    core1_repeats = core1_trials[
                        core1_trials["is_core1_repeat"] == True
                    ]
                    core1_repeat_rate = len(core1_repeats) / len(core1_trials)
                else:
                    core1_repeat_rate = 0.0
            else:
                core1_repeat_rate = 0.0

            # Count trials by block type
            block_counts = {}
            # Core types
            block_counts["core1"] = len(
                self.trial_schedule_df[self.trial_schedule_df["block_type"] == "core1"]
            )
            block_counts["core2"] = len(
                self.trial_schedule_df[self.trial_schedule_df["block_type"] == "core2"]
            )
            block_counts["boundary"] = len(
                self.trial_schedule_df[
                    self.trial_schedule_df["block_type"] == "boundary"
                ]
            )
            block_counts["lhs"] = len(
                self.trial_schedule_df[self.trial_schedule_df["block_type"] == "lhs"]
            )
            # Interaction count: core2 trials with non-null interaction_pair_id
            if "interaction_pair_id" in self.trial_schedule_df.columns:
                block_counts["interaction"] = len(
                    self.trial_schedule_df[
                        (self.trial_schedule_df["block_type"] == "core2")
                        & (self.trial_schedule_df["interaction_pair_id"].notna())
                    ]
                )
            else:
                block_counts["interaction"] = 0

            # Extract design_row_id list for Core-1 (for Coordinator.update_after_batch)
            core1_ids = (
                self.trial_schedule_df[self.trial_schedule_df["block_type"] == "core1"][
                    "design_row_id"
                ].tolist()
                if "design_row_id" in self.trial_schedule_df.columns
                else []
            )
        else:
            # Fallback if trials not generated
            coverage_rate = 0.0
            gini_coeff = 0.30
            core1_repeat_rate = 0.0
            block_counts = {
                "core1": 0,
                "core2": 0,
                "interaction": 0,
                "boundary": 0,
                "lhs": 0,
            }
            core1_ids = []

        # **CRITICAL**: This metadata is used directly by Coordinator
        summary = {
            "metadata": {
                "phase": 1,
                "subject_id": getattr(self, "subject_id", None),
                "batch_id": getattr(self, "batch_id", None),
                "is_bridge": getattr(self, "is_bridge", False),
                "seed": self.seed,
                "n_factors": self.d,
                "n_unique_stimuli": n_unique_stimuli,
                "coverage_rate": float(coverage_rate),  # **For strategy check**
                "gini": float(gini_coeff),  # **For strategy check**
                "core1_repeat_rate": float(
                    core1_repeat_rate
                ),  # **For bridge validation**
                "core1_ids_used": core1_ids,  # **For next batch repeat setup**
                "block_counts": block_counts,
                "warnings": self.warnings,
            },
            "measurement_quality": {
                "icc_target": "≥0.45 (min ≥0.30)",
                "batch_effect_target": "<0.20 (min <0.30)",
                "test_retest_target": "≥0.80 (min ≥0.70)",
                "planned_icc_replicates": "To be determined in later stages",
                "placeholder_values": True,
            },
            "factor_effects": {
                "main_effects": "Planned for coverage",
                "lambda_main": "To be determined",
                "min_counts_per_factor": (
                    self.main_effects_plan["min_counts_per_factor"]
                    if self.main_effects_plan
                    else {}
                ),
            },
            "interaction_screening": {
                "candidate_pairs": (
                    self.interaction_plan["pairs"] if self.interaction_plan else []
                ),
                "quadrant_planning_count": "3-4 per pair",
                "lambda_interaction": "To be determined",
                "total_tasting_budget": (
                    self.interaction_plan["total_tasting_budget"]
                    if self.interaction_plan
                    else 0
                ),
            },
            "gp_model": {
                "kernel": "Matérn 5/2",
                "ARD": True,
                "parameters": "To be estimated in later stages",
                "cv_rmse": "To be calculated after training",
            },
            "spatial_coverage": {
                "high_uncertainty_regions": "Sparse coverage or placeholder",
                "coverage_rate": float(coverage_rate),
                "gini_coefficient": float(gini_coeff),
            },
            "budget_allocation": self.budget_split,
            "block_counts": block_counts,
            "core1_info": {
                "n_points": self.n_core1_points,
                "points": (
                    self.core1_points.index.tolist()
                    if self.core1_points is not None
                    else None
                ),
            },
        }

        # Save to file if requested
        if save_to:
            import os

            os.makedirs(os.path.dirname(save_to) or ".", exist_ok=True)
            with open(save_to, "w") as f:
                json.dump(summary, f, indent=2, default=str)

        logger.info(
            f"Subject {self.subject_id} batch {self.batch_id}: summarize() "
            f"coverage={coverage_rate:.3f}, gini={gini_coeff:.3f}, repeat_rate={core1_repeat_rate:.3f}"
        )

        return summary

    def detect_factor_types(self) -> Dict[str, str]:
        """
        Detect whether factors are discrete or continuous.

        Returns:
            A dictionary mapping factor names to their types ('discrete' or 'continuous')
        """
        factor_types = {}
        for factor in self.factor_names:
            unique_vals = self.design_df[factor].nunique()
            # Heuristic: if more than 10 unique values, treat as continuous
            if unique_vals > 10:
                factor_types[factor] = "continuous"
            else:
                factor_types[factor] = "discrete"
        return factor_types

    def get_levels_or_bins(self) -> Dict[str, np.ndarray]:
        """
        For each factor, get levels (if discrete) or discretize to bins (if continuous).

        For continuous factors, returns bin edges (not percentile values).
        Bin edges are computed using the adaptive n_bins value from fit_planning().

        Returns:
            A dictionary mapping factor names to:
            - For discrete factors: unique level values
            - For continuous factors: bin edge values (length = n_bins + 1)
        """
        discretized = {}
        n_bins = getattr(self, "_n_bins_adaptive", self.N_BINS_CONTINUOUS)

        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                discretized[factor] = self.design_df[factor].unique()
            else:
                # Discretize continuous factors using adaptive n_bins
                # Generate percentiles from 0 to 100 with (n_bins + 1) points
                percentiles = np.linspace(0, 100, n_bins + 1)
                bin_edges = np.percentile(self.design_df[factor], percentiles)
                discretized[factor] = bin_edges  # Store bin edges, not just points
        return discretized

    def get_bin_centers(self, factor: str) -> np.ndarray:
        """
        Get the center values of bins for a continuous factor.

        Args:
            factor: Factor name

        Returns:
            Array of bin center values (length = N_BINS_CONTINUOUS)
        """
        if self.factor_types.get(factor) == "continuous":
            edges = self.discretized_factors[factor]
            return (edges[:-1] + edges[1:]) / 2
        else:
            # For discrete factors, return unique levels
            return self.discretized_factors[factor]

    def compute_marginal_min_counts(self) -> Dict[str, int]:
        """
        Compute minimum counts per level/bin for main effects coverage.

        For discrete factors: uses actual number of levels
        For continuous factors: uses adaptive n_bins (from fit_planning)

        Returns:
            A dictionary mapping factor names to their minimum counts
        """
        min_counts = {}
        n_bins = getattr(self, "_n_bins_adaptive", self.N_BINS_CONTINUOUS)

        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                L_i = len(self.discretized_factors[factor])
            else:
                # For continuous factors, use adaptive n_bins
                L_i = n_bins
            # max(8, ceil(70 / L_i)) per level, with hard floor of 7
            min_count = max(8, int(np.ceil(70 / L_i)))
            min_counts[factor] = max(min_count, 7)  # Hard floor of 7
        return min_counts

    def d_optimal_select(
        self, design_df: pd.DataFrame, k: int, model_terms: str = "main effects"
    ) -> pd.DataFrame:
        """
        Select k points from design_df using D-optimal approximation with incremental updates.

        Uses incremental XtX updates instead of full recomputation for efficiency.
        Complexity: O(k × n × d) instead of O(k × n × d²)

        Args:
            design_df: Candidate set DataFrame
            k: Number of points to select
            model_terms: Model terms ("main effects" or other)

        Returns:
            Selected points DataFrame
        """
        if k >= len(design_df):
            return design_df

        if model_terms == "main effects":
            # For main effects, we want to maximize the determinant of X^T X
            # where X is the design matrix
            factor_names = [col for col in design_df.columns if col.startswith("f")]
            candidates = design_df[factor_names].values

            # Standardize continuous factors
            for i, factor in enumerate(factor_names):
                if self.factor_types.get(factor, "continuous") == "continuous":
                    mean_val = np.mean(candidates[:, i])
                    std_val = np.std(candidates[:, i])
                    if std_val > 0:
                        candidates[:, i] = (candidates[:, i] - mean_val) / std_val

            # Greedy D-optimal selection with incremental updates
            selected_indices = []
            remaining_indices = set(range(len(candidates)))

            # Start with a randomly selected point
            if remaining_indices:
                start_idx = np.random.choice(list(remaining_indices))
                selected_indices.append(start_idx)
                remaining_indices.remove(start_idx)

                # Initialize XtX with first point
                x_init = np.append(1, candidates[start_idx])  # [1, f1, f2, ...]
                XtX = np.outer(x_init, x_init)
            else:
                return design_df.iloc[:0]

            # Iteratively add points that maximize D-optimality
            for iteration in range(min(k - 1, len(remaining_indices))):
                best_idx = None
                best_score = -np.inf
                best_XtX_new = None

                # For each remaining candidate, evaluate D-optimality score
                # Using incremental update: XtX_new = XtX_old + outer(x_new, x_new)
                for candidate_idx in remaining_indices:
                    x_cand = np.append(1, candidates[candidate_idx])

                    # Incremental update: compute new XtX
                    XtX_new = XtX + np.outer(x_cand, x_cand)

                    # Calculate D-optimality score (log determinant of X^T X)
                    try:
                        score = np.linalg.slogdet(XtX_new)[1]  # Log determinant
                        if score > best_score:
                            best_score = score
                            best_idx = candidate_idx
                            best_XtX_new = XtX_new
                    except np.linalg.LinAlgError:
                        # Singular matrix, skip this candidate
                        continue

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                    XtX = best_XtX_new  # Update for next iteration
                else:
                    # If no valid candidate found, randomly select one
                    if remaining_indices:
                        random_idx = np.random.choice(list(remaining_indices))
                        selected_indices.append(random_idx)
                        remaining_indices.remove(random_idx)
                        # Still update XtX for consistency
                        x_rand = np.append(1, candidates[random_idx])
                        XtX = XtX + np.outer(x_rand, x_rand)

            return design_df.iloc[selected_indices].reset_index(drop=True)
        else:
            # For other model terms, use simple random sampling as fallback
            return design_df.sample(n=min(k, len(design_df)), random_state=self.seed)

    def select_core1_points(self, strategy: str = "corners+centers") -> pd.DataFrame:
        """
        Select Core-1 points (global skeleton) sampled by all subjects.

        Args:
            strategy: Selection strategy (default: "corners+centers")

        Returns:
            A DataFrame with selected Core-1 points
        """
        if strategy == "corners+centers":
            candidates = self.design_df.copy()

            # Generate low/high representative values for each factor
            factor_representatives = {}
            for factor in self.factor_names:
                if self.factor_types[factor] == "discrete":
                    # For discrete factors, use min/max levels
                    levels = sorted(self.discretized_factors[factor])
                    factor_representatives[factor] = {
                        "low": levels[0],
                        "high": levels[-1],
                    }
                else:
                    # For continuous factors, use p10/p90 percentiles
                    p10 = np.percentile(candidates[factor], 10)
                    p90 = np.percentile(candidates[factor], 90)
                    # Find closest candidate rows to these percentiles
                    candidates_p10 = candidates.iloc[
                        (candidates[factor] - p10).abs().argsort()[:3]
                    ]
                    candidates_p90 = candidates.iloc[
                        (candidates[factor] - p90).abs().argsort()[:3]
                    ]
                    factor_representatives[factor] = {
                        "low": candidates_p10[factor].iloc[0],
                        "high": candidates_p90[factor].iloc[0],
                    }

            # 1. All-low point
            all_low_conditions = {
                factor: factor_representatives[factor]["low"]
                for factor in self.factor_names
            }
            # Find closest candidate to all-low conditions
            all_low_scores = np.zeros(len(candidates))
            for factor in self.factor_names:
                all_low_scores += np.abs(
                    candidates[factor] - all_low_conditions[factor]
                )
            all_low_idx = np.argmin(all_low_scores)
            all_low_point = candidates.iloc[all_low_idx : all_low_idx + 1]

            # 2. All-high point
            all_high_conditions = {
                factor: factor_representatives[factor]["high"]
                for factor in self.factor_names
            }
            # Find closest candidate to all-high conditions
            all_high_scores = np.zeros(len(candidates))
            for factor in self.factor_names:
                all_high_scores += np.abs(
                    candidates[factor] - all_high_conditions[factor]
                )
            all_high_idx = np.argmin(all_high_scores)
            all_high_point = candidates.iloc[all_high_idx : all_high_idx + 1]

            # 3. Center points (p50 or closest to median)
            median_values = candidates[self.factor_names].median()
            candidates_values = candidates[self.factor_names].values
            median_array = np.array(median_values)
            distances_to_median = np.sqrt(
                ((candidates_values - median_array) ** 2).sum(axis=1)
            )
            center_idx = np.argmin(distances_to_median)
            center_point = candidates.iloc[center_idx : center_idx + 1]

            # Add a "style variant" center point with slight bias
            # Bias toward higher values for variety
            biased_median = median_values.copy()
            for factor in self.factor_names:
                factor_range = candidates[factor].max() - candidates[factor].min()
                biased_median[factor] = min(
                    candidates[factor].max(),
                    max(
                        candidates[factor].min(),
                        biased_median[factor] + 0.1 * factor_range,
                    ),
                )
            biased_candidates_values = candidates[self.factor_names].values
            biased_median_array = np.array(biased_median)
            distances_to_biased_median = np.sqrt(
                ((biased_candidates_values - biased_median_array) ** 2).sum(axis=1)
            )
            biased_center_idx = np.argmin(distances_to_biased_median)
            biased_center_point = candidates.iloc[
                biased_center_idx : biased_center_idx + 1
            ]

            # 4. Semantic extremes (if possible)
            # Create semantic extremes by grouping factors by variance
            semantic_points = []
            if len(self.factor_names) >= 2:
                # Group factors by variance (high variance vs low variance)
                factor_variances = candidates[self.factor_names].var()
                sorted_factors = factor_variances.sort_values(ascending=False)

                # Split factors into high variance and low variance groups
                n_factors = len(self.factor_names)
                split_point = n_factors // 2

                if (
                    split_point > 0 and split_point < n_factors
                ):  # Ensure both groups have factors
                    high_variance_factors = sorted_factors.index[:split_point].tolist()
                    low_variance_factors = sorted_factors.index[split_point:].tolist()

                    # Create semantic extremes: high variance factors high, low variance factors low
                    semantic_conditions_1 = {}
                    semantic_conditions_2 = {}

                    # First semantic extreme: high variance factors high, low variance factors low
                    for factor in high_variance_factors:
                        semantic_conditions_1[factor] = factor_representatives[factor][
                            "high"
                        ]
                    for factor in low_variance_factors:
                        semantic_conditions_1[factor] = factor_representatives[factor][
                            "low"
                        ]

                    # Second semantic extreme: high variance factors low, low variance factors high
                    for factor in high_variance_factors:
                        semantic_conditions_2[factor] = factor_representatives[factor][
                            "low"
                        ]
                    for factor in low_variance_factors:
                        semantic_conditions_2[factor] = factor_representatives[factor][
                            "high"
                        ]

                    # Find closest candidates to these semantic conditions
                    semantic_scores_1 = np.zeros(len(candidates))
                    semantic_scores_2 = np.zeros(len(candidates))
                    for factor in self.factor_names:
                        semantic_scores_1 += np.abs(
                            candidates[factor] - semantic_conditions_1[factor]
                        )
                        semantic_scores_2 += np.abs(
                            candidates[factor] - semantic_conditions_2[factor]
                        )

                    semantic_idx_1 = np.argmin(semantic_scores_1)
                    semantic_idx_2 = np.argmin(semantic_scores_2)
                    semantic_point_1 = candidates.iloc[
                        semantic_idx_1 : semantic_idx_1 + 1
                    ]
                    semantic_point_2 = candidates.iloc[
                        semantic_idx_2 : semantic_idx_2 + 1
                    ]

                    semantic_points = [semantic_point_1, semantic_point_2]

            # Combine required points
            core1_points_list = [
                all_low_point,
                all_high_point,
                center_point,
                biased_center_point,
            ]
            core1_points_list.extend(semantic_points)

            # Concatenate all required points
            core1_points = pd.concat(core1_points_list, ignore_index=True)

            # Remove duplicates
            core1_points = core1_points.drop_duplicates()

            # If we need more points, add maximin from remaining candidates
            if len(core1_points) < self.n_core1_points:
                additional_needed = self.n_core1_points - len(core1_points)
                remaining_candidates = candidates.drop(
                    core1_points.index, errors="ignore"
                )

                if len(remaining_candidates) > 0:
                    # Use maximin criterion to select additional points
                    # Maximize the minimum distance to already selected points
                    # OPTIMIZED: Use distance matrix caching instead of nested loops
                    from scipy.spatial.distance import cdist

                    core1_values = core1_points[self.factor_names].values
                    remaining_values = remaining_candidates[self.factor_names].values
                    candidate_indices = remaining_candidates.index.tolist()

                    # Compute distance matrix: remaining candidates -> existing core1 points
                    # Shape: (n_remaining, n_existing)
                    dist_to_core1 = cdist(
                        remaining_values, core1_values, metric="euclidean"
                    )

                    # Initialize min_distance for each candidate (distance to nearest core1)
                    min_distance_to_selected = dist_to_core1.min(axis=1)

                    selected_indices = []

                    # Add points one by one using maximin
                    for _ in range(min(additional_needed, len(candidate_indices))):
                        # Find candidate with maximum minimum distance
                        if len(selected_indices) == 0:
                            # For first additional point, use distance to existing core1 points
                            best_position = np.argmax(min_distance_to_selected)
                        else:
                            # For subsequent points, also consider distance to newly selected points
                            selected_values = remaining_values[
                                [
                                    candidate_indices.index(idx)
                                    for idx in selected_indices
                                ]
                            ]

                            # Compute distance to newly selected points
                            # Shape: (n_remaining, n_selected)
                            dist_to_selected = cdist(
                                remaining_values, selected_values, metric="euclidean"
                            )
                            min_dist_to_new = dist_to_selected.min(axis=1)

                            # Minimum distance to either core1 or newly selected points
                            min_dist_overall = np.minimum(
                                min_distance_to_selected, min_dist_to_new
                            )
                            best_position = np.argmax(min_dist_overall)

                        selected_indices.append(candidate_indices[best_position])
                        # Mark this candidate as selected by setting its min_distance to -inf
                        # (so it won't be selected again)
                        min_distance_to_selected[best_position] = -np.inf

                    # Add selected points to core1_points
                    if selected_indices:
                        additional_points = remaining_candidates.loc[selected_indices]
                        core1_points = pd.concat(
                            [core1_points, additional_points], ignore_index=True
                        )

            # Ensure we don't exceed the requested number of points
            if len(core1_points) > self.n_core1_points:
                core1_points = core1_points.iloc[: self.n_core1_points]

            return core1_points.reset_index(drop=True)
        else:
            # Default to random selection
            return self.design_df.sample(n=self.n_core1_points, random_state=self.seed)

    def build_interaction_pairs(
        self, K: Optional[int] = None, heuristic: bool = True
    ) -> List[Tuple[int, int]]:
        """
        Automatically generate interaction pairs with heuristic prioritization.

        Args:
            K: Number of pairs to generate (if None, use default based on d)
            heuristic: Whether to use heuristic prioritization

        Returns:
            A list of (i,j) factor index pairs
        """
        if self.candidate_interactions is not None:
            return self.candidate_interactions

        # Default K based on dimensionality with strategy degradation for d>12
        if K is None:
            if self.d <= 8:
                K = min(12, self.d * (self.d - 1) // 2)
            elif 9 <= self.d <= 10:
                K = 10
            elif 11 <= self.d <= 12:
                K = 9
            else:  # d > 12 (strong warning path)
                # Strategy degradation: reduce K to 6-8 pairs
                K = min(8, max(6, self.d // 2))

        # Generate all possible pairs
        all_pairs = list(itertools.combinations(range(self.d), 2))

        # If we have fewer pairs than K, return all
        if len(all_pairs) <= K:
            return all_pairs

        # For heuristic prioritization, we could implement domain-specific logic
        # For now, we'll just randomly select K pairs
        if heuristic and self.d > 2:
            # Simple heuristic: prioritize pairs with higher variance factors
            factor_variances = self.design_df[self.factor_names].var()
            # Rank factors by variance (argsort returns indices in ascending order, [::-1] reverses to descending)
            # ⚠️ CRITICAL: Keep integer indices from argsort(). Never switch to string column names, which breaks mapping.
            factor_indices_by_var = np.argsort(factor_variances.values)[
                ::-1
            ]  # Indices of factors by descending variance

            # Prioritize pairs involving high-variance factors (use integer indices directly)
            prioritized_pairs = []
            for i in range(min(len(factor_indices_by_var), self.d)):
                for j in range(i + 1, min(len(factor_indices_by_var), self.d)):
                    # Convert back to factor index pairs (these are already 0..d-1 indices)
                    pair = tuple(
                        sorted([factor_indices_by_var[i], factor_indices_by_var[j]])
                    )
                    if pair not in prioritized_pairs:  # Avoid duplicates
                        prioritized_pairs.append(pair)
                    if len(prioritized_pairs) >= K:
                        return prioritized_pairs[:K]

            # Fill remaining with random pairs if needed
            remaining_needed = K - len(prioritized_pairs)
            if remaining_needed > 0:
                remaining_pairs = [p for p in all_pairs if p not in prioritized_pairs]
                if len(remaining_pairs) >= remaining_needed:
                    selected_indices = np.random.choice(
                        len(remaining_pairs), remaining_needed, replace=False
                    )
                    prioritized_pairs.extend(
                        [remaining_pairs[i] for i in selected_indices]
                    )
            return prioritized_pairs[:K]
        else:
            # Random selection
            selected_indices = np.random.choice(len(all_pairs), K, replace=False)
            return [all_pairs[i] for i in selected_indices]

    def build_boundary_library(self) -> List[Dict[str, Any]]:
        """
        Build boundary library including all-low, all-high, uni-factor extremes,
        and domain-plausible compound extremes with deduplication.

        Returns:
            A list of boundary point dictionaries
        """
        boundary_points = []
        candidate_points = self.design_df.copy()

        # Generate representative values for each factor
        factor_representatives = {}
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                # For discrete factors, use min/max levels
                levels = sorted(self.discretized_factors[factor])
                factor_representatives[factor] = {"low": levels[0], "high": levels[-1]}
            else:
                # For continuous factors, use p10/p90 percentiles
                p10 = np.percentile(candidate_points[factor], 10)
                p90 = np.percentile(candidate_points[factor], 90)
                # Find closest candidate rows to these percentiles
                candidates_p10 = candidate_points.iloc[
                    (candidate_points[factor] - p10).abs().argsort()[:3]
                ]
                candidates_p90 = candidate_points.iloc[
                    (candidate_points[factor] - p90).abs().argsort()[:3]
                ]
                factor_representatives[factor] = {
                    "low": candidates_p10[factor].iloc[0],
                    "high": candidates_p90[factor].iloc[0],
                }

        # 1. All-low point
        all_low_conditions = {
            factor: factor_representatives[factor]["low"]
            for factor in self.factor_names
        }
        # Find closest candidate to all-low conditions
        all_low_scores = np.zeros(len(candidate_points))
        for factor in self.factor_names:
            all_low_scores += np.abs(
                candidate_points[factor] - all_low_conditions[factor]
            )
        all_low_idx = np.argmin(all_low_scores)
        all_low_point = candidate_points.iloc[all_low_idx].to_dict()
        all_low_point["type"] = "all_low"
        boundary_points.append(all_low_point)

        # 2. All-high point
        all_high_conditions = {
            factor: factor_representatives[factor]["high"]
            for factor in self.factor_names
        }
        # Find closest candidate to all-high conditions
        all_high_scores = np.zeros(len(candidate_points))
        for factor in self.factor_names:
            all_high_scores += np.abs(
                candidate_points[factor] - all_high_conditions[factor]
            )
        all_high_idx = np.argmin(all_high_scores)
        all_high_point = candidate_points.iloc[all_high_idx].to_dict()
        all_high_point["type"] = "all_high"
        boundary_points.append(all_high_point)

        # 3. Uni-factor extremes
        for i, factor in enumerate(self.factor_names):
            # Low for this factor, median for others
            low_extreme_conditions = {
                f: candidate_points[f].median() for f in self.factor_names
            }
            low_extreme_conditions[factor] = factor_representatives[factor]["low"]
            # Find closest candidate
            low_extreme_scores = np.zeros(len(candidate_points))
            for f in self.factor_names:
                low_extreme_scores += np.abs(
                    candidate_points[f] - low_extreme_conditions[f]
                )
            low_extreme_idx = np.argmin(low_extreme_scores)
            low_extreme_point = candidate_points.iloc[low_extreme_idx].to_dict()
            low_extreme_point["type"] = f"uni_low_{i}"
            boundary_points.append(low_extreme_point)

            # High for this factor, median for others
            high_extreme_conditions = {
                f: candidate_points[f].median() for f in self.factor_names
            }
            high_extreme_conditions[factor] = factor_representatives[factor]["high"]
            # Find closest candidate
            high_extreme_scores = np.zeros(len(candidate_points))
            for f in self.factor_names:
                high_extreme_scores += np.abs(
                    candidate_points[f] - high_extreme_conditions[f]
                )
            high_extreme_idx = np.argmin(high_extreme_scores)
            high_extreme_point = candidate_points.iloc[high_extreme_idx].to_dict()
            high_extreme_point["type"] = f"uni_high_{i}"
            boundary_points.append(high_extreme_point)

        # 4. Domain-plausible compound extremes (simple approach)
        # Create some combinations of high-low patterns
        if len(self.factor_names) >= 2:
            # First half high, second half low
            half_point = len(self.factor_names) // 2
            compound_conditions_1 = {}
            compound_conditions_2 = {}

            for i, factor in enumerate(self.factor_names):
                if i < half_point:
                    compound_conditions_1[factor] = factor_representatives[factor][
                        "high"
                    ]
                    compound_conditions_2[factor] = factor_representatives[factor][
                        "low"
                    ]
                else:
                    compound_conditions_1[factor] = factor_representatives[factor][
                        "low"
                    ]
                    compound_conditions_2[factor] = factor_representatives[factor][
                        "high"
                    ]

            # Find closest candidates
            compound_scores_1 = np.zeros(len(candidate_points))
            compound_scores_2 = np.zeros(len(candidate_points))
            for factor in self.factor_names:
                compound_scores_1 += np.abs(
                    candidate_points[factor] - compound_conditions_1[factor]
                )
                compound_scores_2 += np.abs(
                    candidate_points[factor] - compound_conditions_2[factor]
                )

            compound_idx_1 = np.argmin(compound_scores_1)
            compound_idx_2 = np.argmin(compound_scores_2)
            compound_point_1 = candidate_points.iloc[compound_idx_1].to_dict()
            compound_point_2 = candidate_points.iloc[compound_idx_2].to_dict()
            compound_point_1["type"] = "compound_1"
            compound_point_2["type"] = "compound_2"
            boundary_points.append(compound_point_1)
            boundary_points.append(compound_point_2)

        # 5. Deduplicate boundary points using pairwise distances with normalized thresholds
        if len(boundary_points) > 1:
            # Convert to array for distance calculation
            boundary_array = np.array(
                [[bp[f] for f in self.factor_names] for bp in boundary_points]
            )

            # Normalize distances to [0,1] range
            min_vals = boundary_array.min(axis=0)
            max_vals = boundary_array.max(axis=0)
            ranges = np.maximum(max_vals - min_vals, 1e-10)  # Avoid division by zero
            normalized_array = (boundary_array - min_vals) / ranges

            # Calculate pairwise distances on normalized array
            unique_points = [boundary_points[0]]  # Always keep first point
            unique_indices = [0]

            # Dynamic threshold based on factor types:
            # - For continuous factors: 1% of normalized range (0.01)
            # - For discrete factors: Use stricter threshold (0.001)
            has_continuous = any(
                self.factor_types[f] == "continuous" for f in self.factor_names
            )
            threshold = 0.01 if has_continuous else 0.001

            for i in range(1, len(boundary_points)):
                # Calculate minimum distance to already selected points
                min_dist = float("inf")
                current_point = normalized_array[i]

                for j in unique_indices:
                    existing_point = normalized_array[j]
                    dist = np.sqrt(((current_point - existing_point) ** 2).sum())
                    min_dist = min(min_dist, dist)

                # Only add if distance is above threshold (avoid near-duplicates)
                if min_dist > threshold:
                    unique_points.append(boundary_points[i])
                    unique_indices.append(i)

            boundary_points = unique_points

        return boundary_points

    def plan_batches_and_bridges(self) -> Dict[str, Any]:
        """
        Plan batch allocation with bridge subjects across batches.

        Returns:
            A dictionary with batch plan information
        """
        # Determine bridge subjects per bridge based on dimensionality
        if self.d <= 8:
            bridge_subjects_per_bridge = 2
        elif 9 <= self.d <= 10:
            bridge_subjects_per_bridge = 3
        else:  # d >= 11
            bridge_subjects_per_bridge = 3

        # Ensure we have enough subjects
        if self.n_subjects < bridge_subjects_per_bridge * (self.n_batches - 1):
            warnings.warn("Not enough subjects for recommended bridge design")

        return {
            "n_batches": self.n_batches,
            "bridge_subjects_per_bridge": bridge_subjects_per_bridge,
            "core1_repeat_threshold": 0.5,  # ≥50% of Core-1 points repeated across adjacent batches
        }

    def compute_coverage_rate(self, trials_df: pd.DataFrame) -> float:
        """
        Compute coverage rate based on how well generated trials cover the discretized design space.

        Uses grid-based coverage calculation:
        - For each factor, discretize into bins using adaptive n_bins (from fit_planning)
        - Count how many grid cells are covered by at least one trial
        - Coverage = covered_cells / total_cells

        Args:
            trials_df: DataFrame with generated trials (not design_df)

        Returns:
            Coverage rate as a float between 0 and 1
        """
        if self.d is None or self.discretized_factors is None or len(trials_df) == 0:
            return 0.15  # Default value if not initialized

        # Build grid cell definitions
        grid_structure = {}
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                grid_structure[factor] = sorted(self.discretized_factors[factor])
            else:
                # For continuous factors, use bin edges
                bin_edges = self.discretized_factors[factor]
                grid_structure[factor] = bin_edges

        # Count total possible grid cells
        total_cells = 1
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                total_cells *= len(grid_structure[factor])
            else:
                # Number of bins = number of edges - 1
                total_cells *= len(grid_structure[factor]) - 1

        # For very large grids, use distance-based coverage instead
        if total_cells > 100000:
            # Distance-based coverage: what fraction of design space is within threshold distance of trials
            # Sample design space points and check distance to nearest trial
            sample_size = min(1000, len(self.design_df))
            sample_indices = np.random.choice(
                len(self.design_df), size=sample_size, replace=False
            )
            sample_points = self.design_df.iloc[sample_indices][
                self.factor_names
            ].values

            # Compute distance threshold as 5% of the design space diagonal
            design_ranges = (
                self.design_df[self.factor_names].max()
                - self.design_df[self.factor_names].min()
            )
            diagonal = np.sqrt(np.sum(design_ranges**2))
            threshold = 0.05 * diagonal

            # Compute coverage as fraction of sampled points within threshold of ANY trial
            trial_points = trials_df[self.factor_names].values
            distances = pairwise_distances(sample_points, trial_points).min(axis=1)
            covered = (distances <= threshold).sum()

            return min(1.0, covered / len(sample_indices))

        # Grid-based coverage for small grids
        covered_cells = set()

        # Map TRIALS (not design_df) to grid cells
        for _, trial_row in trials_df.iterrows():
            # Map trial to grid cell
            cell_id = []
            for factor in self.factor_names:
                trial_val = trial_row[factor]

                if self.factor_types[factor] == "discrete":
                    levels = grid_structure[factor]
                    cell_idx = np.argmin(np.abs(np.array(levels) - trial_val))
                    cell_id.append(cell_idx)
                else:
                    # Find which bin this value falls into
                    bin_edges = grid_structure[factor]
                    bin_idx = np.searchsorted(bin_edges, trial_val) - 1
                    bin_idx = np.clip(bin_idx, 0, len(bin_edges) - 2)
                    cell_id.append(bin_idx)

            covered_cells.add(tuple(cell_id))

        # Calculate coverage rate
        coverage = len(covered_cells) / max(1, total_cells)
        return min(1.0, max(0.0, coverage))

    def compute_gini(self, trials_df: pd.DataFrame) -> float:
        """
        Compute Gini coefficient based on trial distribution inequality.

        Gini = (2 * Σ(i * count_i)) / (n * Σ(count_i)) - (n+1)/n
        where count_i are frequency counts sorted in ascending order

        Args:
            trials_df: DataFrame with generated trials (not design_df)

        Returns:
            Gini coefficient as a float between 0 (perfect equality) and 1 (perfect inequality)
        """
        if self.d is None or len(trials_df) == 0:
            return 0.30  # Default value if not initialized

        # Calculate Gini for each factor's distribution across TRIALS
        gini_values = []

        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                # For discrete factors, count occurrences of each level in TRIALS
                level_counts = trials_df[factor].value_counts().values
            else:
                # For continuous factors, count occurrences in each bin for TRIALS
                bin_edges = self.discretized_factors[factor]
                level_counts = []
                for i in range(len(bin_edges) - 1):
                    count = len(
                        trials_df[
                            (trials_df[factor] >= bin_edges[i])
                            & (trials_df[factor] < bin_edges[i + 1])
                        ]
                    )
                    level_counts.append(count)

                # Handle values exactly at max edge
                last_count = len(trials_df[trials_df[factor] == bin_edges[-1]])
                if last_count > 0 and len(level_counts) > 0:
                    level_counts[-1] += last_count

            if len(level_counts) > 0:
                # Calculate Gini coefficient for this factor
                counts_sorted = np.sort(level_counts)
                n = len(counts_sorted)
                total_count = np.sum(counts_sorted)

                if total_count > 0:
                    cumsum_idx = np.arange(1, n + 1)
                    gini_factor = (2 * np.sum(cumsum_idx * counts_sorted)) / (
                        n * total_count
                    ) - (n + 1) / n
                    gini_values.append(max(0.0, min(1.0, gini_factor)))

        # Return average Gini across factors
        if gini_values:
            return np.mean(gini_values)
        else:
            return 0.30  # Default value

    def _compute_budget_split(self) -> Dict[str, int]:
        """
        Compute budget split based on dimensionality with maximum remainder method for integer allocation.

        Returns:
            A dictionary with budget allocation for core1, core2, and individual points
        """
        # Budget allocation by dimensionality
        if self.d <= 8:
            core1_pct = 0.22
            core2_pct = 0.44
            individual_pct = 0.34
            # Boundary share: 12-15%, linearly increasing with d
            boundary_base = 0.12
            boundary_increase = (
                (0.15 - 0.12) * (self.d - 1) / 7
            )  # Linear from d=1 to d=8
            boundary_pct = boundary_base + boundary_increase
        elif 9 <= self.d <= 10:
            core1_pct = 0.24
            core2_pct = 0.47
            individual_pct = 0.29
            # Boundary share: 15-18%
            boundary_pct = 0.15 + (0.18 - 0.15) * (self.d - 9) / 1
        elif 11 <= self.d <= 12:
            core1_pct = 0.25
            core2_pct = 0.48
            individual_pct = 0.27
            # Boundary share: 18-22%
            boundary_pct = 0.18 + (0.22 - 0.18) * (self.d - 11) / 1
        else:  # d > 12 (strong warning path)
            core1_pct = 0.25  # Cap at 25%
            core2_pct = 0.48
            individual_pct = 0.27
            # Boundary share: 22-25%
            boundary_pct = min(
                0.25, 0.22 + (self.d - 12) * 0.01
            )  # Increase with d but cap at 25%

        # Integer budget allocation using maximum remainder method
        core1_budget_float = self.total_budget * core1_pct
        core2_budget_float = self.total_budget * core2_pct
        individual_budget_float = self.total_budget * individual_pct

        # Initial integer allocation
        core1_budget = int(core1_budget_float)
        core2_budget = int(core2_budget_float)
        individual_budget = int(individual_budget_float)

        # Distribute remainder using maximum remainder method
        remainders = [
            (core1_budget_float - core1_budget, "core1"),
            (core2_budget_float - core2_budget, "core2"),
            (individual_budget_float - individual_budget, "individual"),
        ]
        remainders.sort(reverse=True)  # Sort by remainder in descending order

        # Distribute the remaining budget (total may not equal total_budget due to rounding)
        remaining_budget = self.total_budget - (
            core1_budget + core2_budget + individual_budget
        )
        for i in range(min(remaining_budget, len(remainders))):
            if remainders[i][1] == "core1":
                core1_budget += 1
            elif remainders[i][1] == "core2":
                core2_budget += 1
            elif remainders[i][1] == "individual":
                individual_budget += 1

        # Core-2 split: main effects (60%) and interactions (40%) with maximum remainder method
        core2_main_effects_float = core2_budget * 0.6
        core2_interactions_float = core2_budget * 0.4

        core2_main_effects = int(core2_main_effects_float)
        core2_interactions = int(core2_interactions_float)

        # Distribute remainder for Core-2 split
        core2_remainder = core2_budget - (core2_main_effects + core2_interactions)
        if core2_remainder > 0:
            # Distribute to the component with the largest remainder
            if (core2_main_effects_float - core2_main_effects) > (
                core2_interactions_float - core2_interactions
            ):
                core2_main_effects += core2_remainder
            else:
                core2_interactions += core2_remainder

        # Individual split: boundary and LHS with maximum remainder method
        individual_boundary_float = individual_budget * boundary_pct
        individual_lhs_float = individual_budget * (1 - boundary_pct)

        individual_boundary = int(individual_boundary_float)
        individual_lhs = int(individual_lhs_float)

        # Distribute remainder for individual split
        individual_remainder = individual_budget - (
            individual_boundary + individual_lhs
        )
        if individual_remainder > 0:
            # Distribute to the component with the largest remainder
            if (individual_boundary_float - individual_boundary) > (
                individual_lhs_float - individual_lhs
            ):
                individual_boundary += individual_remainder
            else:
                individual_lhs += individual_remainder

        return {
            "core1": core1_budget,
            "core2": core2_budget,
            "individual": individual_budget,
            "core2_main_effects": core2_main_effects,
            "core2_interactions": core2_interactions,
            "individual_boundary": individual_boundary,
            "individual_lhs": individual_lhs,
        }

    def _compute_core1_size(self) -> int:
        """
        Compute Core-1 size (number of unique stimuli) based on dimensionality.

        Returns:
            Number of Core-1 points
        """
        if self.d <= 8:
            return max(8, min(10, self.d + 2))
        elif 9 <= self.d <= 10:
            return max(10, min(12, self.d + 1))
        else:  # d >= 11
            return 12

    def _plan_main_effects_coverage(self) -> Dict[str, Any]:
        """
        Plan main effects coverage with per-level minimum counts using greedy selection.

        Returns:
            A dictionary with main effects coverage plan
        """
        min_counts = self.compute_marginal_min_counts()

        # Check global feasibility against total_budget
        total_required = sum(min_counts.values())
        if total_required > self.budget_split["core2_main_effects"]:
            # Reduce uniformly but keep >=7 as hard floor
            reduction_factor = self.budget_split["core2_main_effects"] / total_required
            for factor in min_counts:
                min_counts[factor] = max(7, int(min_counts[factor] * reduction_factor))
            total_required = sum(min_counts.values())

        return {
            "min_counts_per_factor": min_counts,
            "total_required": total_required,
        }

    def _maximin_select_subset(
        self, points_df: pd.DataFrame, k: int, factors: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Select k points from points_df using maximin criterion (maximize minimum pairwise distance).

        Uses pairwise distance computation for efficiency.

        Args:
            points_df: DataFrame with candidate points
            k: Number of points to select
            factors: Factor columns to use for distance calculation (default: self.factor_names)

        Returns:
            DataFrame with selected points (indices preserved)
        """
        if factors is None:
            factors = self.factor_names

        if k >= len(points_df):
            return points_df

        if k <= 0:
            return points_df.iloc[:0]

        # Use precomputed pairwise distances for efficiency
        candidates_array = points_df[factors].values
        dist_matrix = pairwise_distances(candidates_array, metric="euclidean")

        # Greedy maximin selection
        selected_indices = [0]  # Start with first point
        remaining_indices = list(range(1, len(points_df)))

        while len(selected_indices) < k and remaining_indices:
            # For each remaining point, find its min distance to selected points
            best_min_dist = -np.inf
            best_idx_in_remaining = 0
            best_global_idx = remaining_indices[0]

            for i, global_idx in enumerate(remaining_indices):
                # Min distance to any selected point
                min_dist = min(
                    dist_matrix[global_idx, sel_idx] for sel_idx in selected_indices
                )

                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx_in_remaining = i
                    best_global_idx = global_idx

            # Add the point with maximum min-distance
            selected_indices.append(best_global_idx)
            remaining_indices.pop(best_idx_in_remaining)

        # Return in original index order
        selected_df = points_df.iloc[selected_indices]
        return selected_df

    def _sample_boundary_region(
        self,
        f1_idx: int,
        f2_idx: int,
        n_per_quadrant: int = 1,
        strategy: str = "balanced",
    ) -> List[int]:
        """
        Sample points from quadrants defined by two factors.

        For continuous×continuous factors: quadrants defined by median
        For discrete×discrete factors: all level combinations (cartesian product)
        For mixed types: discrete levels × continuous quartiles

        Args:
            f1_idx: Index of first factor
            f2_idx: Index of second factor
            n_per_quadrant: Target points per quadrant
            strategy: "balanced" (equal distribution) or "random" (at least 1 per quadrant)

        Returns:
            List of design_df indices sampled from interaction space
        """
        f1_name = self.factor_names[f1_idx]
        f2_name = self.factor_names[f2_idx]

        selected_indices = []

        # Get factor types
        f1_discrete = self.factor_types[f1_name] == "discrete"
        f2_discrete = self.factor_types[f2_name] == "discrete"

        if f1_discrete and f2_discrete:
            # Cartesian product of levels
            f1_levels = sorted(self.discretized_factors[f1_name])
            f2_levels = sorted(self.discretized_factors[f2_name])

            for f1_val in f1_levels:
                for f2_val in f2_levels:
                    # Find points matching this combination
                    mask = (np.abs(self.design_df[f1_name] - f1_val) < 1e-6) & (
                        np.abs(self.design_df[f2_name] - f2_val) < 1e-6
                    )
                    matching_indices = self.design_df[mask].index.tolist()

                    if matching_indices:
                        if strategy == "balanced":
                            selected_indices.extend(matching_indices[:n_per_quadrant])
                        else:  # random
                            selected_indices.append(np.random.choice(matching_indices))
        else:
            # Continuous × continuous or mixed: use median/quartile divisions
            if f1_discrete:
                f1_levels = sorted(self.discretized_factors[f1_name])
                f1_dividers = f1_levels
            else:
                # Get median as divider for continuous
                f1_median = self.design_df[f1_name].median()
                f1_dividers = [f1_median]

            if f2_discrete:
                f2_levels = sorted(self.discretized_factors[f2_name])
                f2_dividers = f2_levels
            else:
                # Get median as divider for continuous
                f2_median = self.design_df[f2_name].median()
                f2_dividers = [f2_median]

            # Create quadrant conditions
            f1_ranges = (
                [(-np.inf, f1_dividers[0])]
                + [
                    (f1_dividers[i], f1_dividers[i + 1])
                    for i in range(len(f1_dividers) - 1)
                ]
                + [(f1_dividers[-1], np.inf)]
            )
            f2_ranges = (
                [(-np.inf, f2_dividers[0])]
                + [
                    (f2_dividers[i], f2_dividers[i + 1])
                    for i in range(len(f2_dividers) - 1)
                ]
                + [(f2_dividers[-1], np.inf)]
            )

            # Sample from each quadrant
            for f1_range in f1_ranges:
                for f2_range in f2_ranges:
                    mask = (
                        (self.design_df[f1_name] >= f1_range[0])
                        & (self.design_df[f1_name] <= f1_range[1])
                        & (self.design_df[f2_name] >= f2_range[0])
                        & (self.design_df[f2_name] <= f2_range[1])
                    )

                    matching_indices = self.design_df[mask].index.tolist()

                    if matching_indices:
                        if strategy == "balanced":
                            selected_indices.extend(matching_indices[:n_per_quadrant])
                        else:  # random
                            selected_indices.append(np.random.choice(matching_indices))

        return selected_indices

    def _plan_interaction_screening(self) -> Dict[str, Any]:
        """
        Plan interaction screening with selected pairs and tasting allocations.

        Returns:
            A dictionary with interaction screening plan
        """
        # Determine number of interaction pairs based on dimensionality
        if self.d <= 8:
            K = min(12, max(10, self.d * 2))
        elif 9 <= self.d <= 10:
            K = 10
        elif 11 <= self.d <= 12:
            K = 9
        else:  # d > 12 (strong warning path)
            K = min(8, max(6, self.d // 2))  # 6-8 pairs for d>12

        interaction_pairs = self.build_interaction_pairs(K=K)

        # For each pair, plan quadrant tasting (4 quadrants with 3-4 repeats)
        tasting_per_pair = 4  # 3-4 total repeats per pair
        total_tasting_budget = len(interaction_pairs) * tasting_per_pair

        return {
            "pairs": interaction_pairs,
            "tasting_per_pair": tasting_per_pair,
            "total_tasting_budget": total_tasting_budget,
        }

    def _plan_stratified_lhs(self) -> Dict[str, Any]:
        """
        Plan stratified LHS with constraints and distance penalties.

        Returns:
            A dictionary with LHS plan
        """
        return {
            "method": "maximin_penalized",
            "constraints_satisfied_first": True,
            "distance_penalty": True,
        }

    def _generate_core1_trials(self) -> List[Dict[str, Any]]:
        """
        Generate Core-1 trials with strict priority:
        1. Place all points from core1_repeat_indices (cross-batch continuity)
        2. Fill remaining quota from core1_pool_indices with non-repeat markers
        3. Enforce: repeats ≤ 50% of core1 quota

        CRITICAL REQUIREMENTS:
        - Mark repeat indices as is_core1_repeat=True
        - Mark new points as is_core1_repeat=False
        - Include design_row_id for all trials
        - Never exceed quota
        - Log repeat placement with diagnostics

        Returns:
            A list of trial dictionaries with explicit is_core1_repeat marker
        """
        trials = []
        quota = int(self.n_core1_points or 10)

        # Ensure design_df has design_row_id for cross-process traceability
        if "design_row_id" not in self.design_df.columns:
            df = self.design_df.reset_index(drop=True).copy()
            df["design_row_id"] = df.index
        else:
            df = self.design_df.copy()

        # Track used indices to avoid duplicates
        used_ids = set()

        # STEP 1: Place repeat indices with priority (cross-batch continuity)
        repeat_quota = quota  # All repeats count toward quota
        repeat_max = int(np.ceil(quota * 0.5))  # Hard cap: ≤ 50% repeats

        if (
            getattr(self, "core1_repeat_indices", None)
            and len(self.core1_repeat_indices) > 0
        ):
            repeat_indices = self.core1_repeat_indices[:repeat_max]  # Cap to 50%
            repeat_df = df[df["design_row_id"].isin(repeat_indices)].drop_duplicates(
                subset=["design_row_id"]
            )

            for idx, row in repeat_df.iterrows():
                if len(trials) >= quota:
                    logger.warning(
                        f"Core-1 quota {quota} reached, stopping repeat placement"
                    )
                    break

                design_row_id = int(row["design_row_id"])
                trial = {
                    "subject_id": getattr(self, "subject_id", 0),
                    "batch_id": getattr(self, "batch_id", 0),
                    "is_bridge": getattr(self, "is_bridge", False),
                    "block_type": "core1",
                    "is_core1": True,
                    "is_core2": False,
                    "is_individual": False,
                    "is_boundary": False,
                    "is_lhs": False,
                    "is_core1_repeat": True,  # **CRITICAL: Mark as repeat**
                    "interaction_pair_id": None,
                    "design_row_id": design_row_id,
                }

                # Add factor values
                for factor in self.factor_names:
                    if factor in row:
                        trial[factor] = row[factor]

                trials.append(trial)
                used_ids.add(design_row_id)

            logger.info(
                f"Subject {self.subject_id} batch {self.batch_id}: placed {len(trials)} Core-1 repeats "
                f"(max allowed: {repeat_max})"
            )

        # STEP 2: Fill remaining quota from pool with non-repeat markers
        remaining_quota = quota - len(trials)

        if remaining_quota > 0:
            # Use provided pool or fallback to pre-selected core1_points
            if (
                getattr(self, "core1_pool_indices", None)
                and len(self.core1_pool_indices) > 0
            ):
                pool_df = df[df["design_row_id"].isin(self.core1_pool_indices)]
            elif self.core1_points is not None and len(self.core1_points) > 0:
                pool_df = self.core1_points.copy()
                if "design_row_id" not in pool_df.columns:
                    pool_df["design_row_id"] = pool_df.index
            else:
                # Fallback to all design points
                pool_df = df.copy()

            # Exclude already-used points
            pool_df = pool_df[~pool_df["design_row_id"].isin(used_ids)].drop_duplicates(
                subset=["design_row_id"]
            )

            # Sample remaining_quota points
            if len(pool_df) > remaining_quota:
                pool_df = pool_df.sample(n=remaining_quota, random_state=self.seed)
            elif len(pool_df) < remaining_quota:
                logger.warning(
                    f"Insufficient pool points: needed {remaining_quota}, got {len(pool_df)}"
                )

            for idx, row in pool_df.iterrows():
                design_row_id = int(row["design_row_id"])
                trial = {
                    "subject_id": getattr(self, "subject_id", 0),
                    "batch_id": getattr(self, "batch_id", 0),
                    "is_bridge": getattr(self, "is_bridge", False),
                    "block_type": "core1",
                    "is_core1": True,
                    "is_core2": False,
                    "is_individual": False,
                    "is_boundary": False,
                    "is_lhs": False,
                    "is_core1_repeat": False,  # **CRITICAL: Mark as new**
                    "interaction_pair_id": None,
                    "design_row_id": design_row_id,
                }

                # Add factor values
                for factor in self.factor_names:
                    if factor in row:
                        trial[factor] = row[factor]

                trials.append(trial)
                used_ids.add(design_row_id)

        logger.info(
            f"Subject {self.subject_id} batch {self.batch_id}: Core-1 trials generated: "
            f"{len(trials)} total ({len([t for t in trials if t['is_core1_repeat']])} repeats, "
            f"{len([t for t in trials if not t['is_core1_repeat']])} new)"
        )

        return trials

    def _generate_core2_trials(self) -> List[Dict[str, Any]]:
        """
        Generate Core-2 trials (main effects coverage + interaction screening).

        Implements proper interaction screening with:
        - Quadrant sampling for continuous×continuous pairs
        - Cartesian product coverage for discrete×discrete pairs
        - Mixed-type combinations for heterogeneous factor pairs

        Returns:
            A list of trial dictionaries
        """
        trials = []

        # Main effects coverage trials
        main_effects_budget = self.budget_split["core2_main_effects"]
        # Use D-optimal selection for main effects coverage
        main_effects_trials = self.d_optimal_select(
            self.design_df,
            min(main_effects_budget, len(self.design_df)),
            "main effects",
        )

        for i, (design_row_id, row) in enumerate(main_effects_trials.iterrows()):
            if i >= main_effects_budget:
                break
            # Distribute across subjects and batches
            subject_id = i % self.n_subjects
            batch_id = (i // self.n_subjects) % self.n_batches

            trial = {
                "subject_id": subject_id,
                "batch_id": batch_id,
                "is_bridge": False,  # Will be set later
                "block_type": "core2",
                "is_core1": False,
                "is_core2": True,
                "is_individual": False,
                "is_boundary": False,
                "is_lhs": False,
                "interaction_pair_id": None,
                "design_row_id": design_row_id,
            }
            # Add factor values
            for factor in self.factor_names:
                trial[factor] = row[factor]
            trials.append(trial)

        # Interaction screening trials - use proper quadrant sampling
        # DESIGN NOTE: Interaction trials are marked as block_type="core2" with interaction_pair_id set,
        # rather than block_type="interaction". This design treats interactions as Core-2 sub-types,
        # enabling flexible grouping while maintaining logical coherence.
        # In summarize(), interaction count = core2 trials with non-null interaction_pair_id.
        interaction_budget = self.budget_split["core2_interactions"]
        interaction_plan = self.interaction_plan
        pairs = interaction_plan["pairs"]
        tasting_per_pair = interaction_plan["tasting_per_pair"]

        # Strategy: "balanced" for even quadrant distribution, "random" otherwise
        interaction_strategy = "balanced" if tasting_per_pair >= 4 else "random"

        # Collect all interaction points
        all_interaction_indices = []
        for pair_idx, (f1_idx, f2_idx) in enumerate(pairs):
            # Sample from quadrants defined by this pair
            quadrant_indices = self._sample_interaction_quadrants(
                f1_idx,
                f2_idx,
                n_per_quadrant=max(1, tasting_per_pair // 4),
                strategy=interaction_strategy,
            )
            all_interaction_indices.extend(
                [(idx, pair_idx) for idx in quadrant_indices]
            )

        # Shuffle and select top interaction_budget points
        if all_interaction_indices:
            np.random.shuffle(all_interaction_indices)
            all_interaction_indices = all_interaction_indices[:interaction_budget]

        # Generate trials from selected indices
        for trial_offset, (design_row_id, pair_idx) in enumerate(
            all_interaction_indices
        ):
            subject_id = trial_offset % self.n_subjects
            batch_id = (trial_offset // self.n_subjects) % self.n_batches

            trial = {
                "subject_id": subject_id,
                "batch_id": batch_id,
                "is_bridge": False,
                "block_type": "core2",
                "is_core1": False,
                "is_core2": True,
                "is_individual": False,
                "is_boundary": False,
                "is_lhs": False,
                "interaction_pair_id": pair_idx,
                "design_row_id": design_row_id,
            }
            # Add factor values
            for factor in self.factor_names:
                trial[factor] = self.design_df.loc[design_row_id, factor]
            trials.append(trial)

        return trials

    def _sample_interaction_quadrants(
        self,
        f1_idx: int,
        f2_idx: int,
        n_per_quadrant: int = 1,
        strategy: str = "balanced",
    ) -> List[int]:
        """
        Sample design points from quadrants defined by a factor pair interaction.

        Supports:
        - Continuous × Continuous: 4 quadrants (low/high, low/high)
        - Discrete × Discrete: Cartesian product of levels (all combinations)
        - Mixed types: Appropriate gridding for each type

        Args:
            f1_idx: Index of first factor in pair
            f2_idx: Index of second factor in pair
            n_per_quadrant: Points to sample per quadrant
            strategy: "balanced" (even per quadrant) or "random" (random per quadrant)

        Returns:
            List of design_df indices for selected points
        """
        f1 = self.factor_names[f1_idx]
        f2 = self.factor_names[f2_idx]
        f1_type = self.factor_types[f1]
        f2_type = self.factor_types[f2]

        selected_indices = []

        if f1_type == "continuous" and f2_type == "continuous":
            # Continuous × Continuous: 4 quadrants based on medians
            f1_median = self.design_df[f1].median()
            f2_median = self.design_df[f2].median()

            quadrants = [
                # (f1_condition, f2_condition, name)
                (
                    self.design_df[f1] <= f1_median,
                    self.design_df[f2] <= f2_median,
                    "low_low",
                ),
                (
                    self.design_df[f1] <= f1_median,
                    self.design_df[f2] > f2_median,
                    "low_high",
                ),
                (
                    self.design_df[f1] > f1_median,
                    self.design_df[f2] <= f2_median,
                    "high_low",
                ),
                (
                    self.design_df[f1] > f1_median,
                    self.design_df[f2] > f2_median,
                    "high_high",
                ),
            ]

            for f1_cond, f2_cond, name in quadrants:
                quadrant_df = self.design_df[f1_cond & f2_cond]
                if len(quadrant_df) > 0:
                    if strategy == "balanced":
                        # Select evenly distributed points
                        indices = np.linspace(
                            0,
                            len(quadrant_df) - 1,
                            min(n_per_quadrant, len(quadrant_df)),
                        ).astype(int)
                        selected_indices.extend(
                            quadrant_df.iloc[indices].index.tolist()
                        )
                    else:  # "random"
                        sample_size = min(n_per_quadrant, len(quadrant_df))
                        selected_indices.extend(
                            quadrant_df.sample(
                                n=sample_size, random_state=self.seed
                            ).index.tolist()
                        )

        elif f1_type == "discrete" and f2_type == "discrete":
            # Discrete × Discrete: Cartesian product
            f1_levels = sorted(self.discretized_factors[f1])
            f2_levels = sorted(self.discretized_factors[f2])

            for f1_val in f1_levels:
                for f2_val in f2_levels:
                    # Find points matching this combination
                    cell_df = self.design_df[
                        (
                            np.abs(self.design_df[f1] - f1_val) < 1e-9
                        )  # Account for float precision
                        & (np.abs(self.design_df[f2] - f2_val) < 1e-9)
                    ]
                    if len(cell_df) > 0:
                        if strategy == "balanced":
                            sample_size = min(n_per_quadrant, len(cell_df))
                            indices = np.linspace(
                                0, len(cell_df) - 1, sample_size
                            ).astype(int)
                            selected_indices.extend(
                                cell_df.iloc[indices].index.tolist()
                            )
                        else:  # "random"
                            sample_size = min(n_per_quadrant, len(cell_df))
                            selected_indices.extend(
                                cell_df.sample(
                                    n=sample_size, random_state=self.seed
                                ).index.tolist()
                            )

        else:
            # Mixed: continuous × discrete or discrete × continuous
            if f1_type == "continuous":
                cont_factor, disc_factor = f1, f2
                cont_median = self.design_df[cont_factor].median()
                disc_levels = sorted(self.discretized_factors[disc_factor])
                cont_parts = [
                    self.design_df[cont_factor] <= cont_median,
                    self.design_df[cont_factor] > cont_median,
                ]
            else:
                disc_factor, cont_factor = f1, f2
                cont_median = self.design_df[cont_factor].median()
                disc_levels = sorted(self.discretized_factors[disc_factor])
                cont_parts = [
                    self.design_df[cont_factor] <= cont_median,
                    self.design_df[cont_factor] > cont_median,
                ]

            # Sample from each (discrete level, continuous part) combination
            for disc_val in disc_levels:
                for cont_part in cont_parts:
                    cell_df = self.design_df[
                        (np.abs(self.design_df[disc_factor] - disc_val) < 1e-9)
                        & cont_part
                    ]
                    if len(cell_df) > 0:
                        if strategy == "balanced":
                            sample_size = min(n_per_quadrant, len(cell_df))
                            indices = np.linspace(
                                0, len(cell_df) - 1, sample_size
                            ).astype(int)
                            selected_indices.extend(
                                cell_df.iloc[indices].index.tolist()
                            )
                        else:  # "random"
                            sample_size = min(n_per_quadrant, len(cell_df))
                            selected_indices.extend(
                                cell_df.sample(
                                    n=sample_size, random_state=self.seed
                                ).index.tolist()
                            )

        return selected_indices

    def _generate_individual_trials(self) -> List[Dict[str, Any]]:
        """
        Generate Individual trials (boundary extremes + stratified LHS fill-in).

        Uses maximin selection for boundary points instead of simple cycling.

        Returns:
            A list of trial dictionaries
        """
        trials = []

        # Boundary trials - use maximin selection to avoid duplicates
        boundary_budget = self.budget_split["individual_boundary"]
        boundary_library = self.boundary_set

        # Convert boundary_library (list of dicts) to DataFrame for maximin selection
        if boundary_library:
            boundary_df = pd.DataFrame(boundary_library)

            # Use maximin to select diverse subset
            if boundary_budget <= len(boundary_library):
                selected_boundary_df = self._maximin_select_subset(
                    boundary_df, boundary_budget, factors=self.factor_names
                )
            else:
                # Use all boundary points + sample from design space boundary
                selected_boundary_df = boundary_df.copy()
                shortage = boundary_budget - len(boundary_df)

                # Sample additional points from outer percentiles of design space
                if shortage > 0:
                    # Create supplementary boundary points from extreme percentiles
                    supplement_conditions = []
                    for factor in self.factor_names:
                        p1 = np.percentile(self.design_df[factor], 1)
                        p99 = np.percentile(self.design_df[factor], 99)

                        # Find design points near these extremes
                        near_p1_idx = (self.design_df[factor] - p1).abs().idxmin()
                        near_p99_idx = (self.design_df[factor] - p99).abs().idxmin()

                        supplement_conditions.append(self.design_df.loc[near_p1_idx])
                        supplement_conditions.append(self.design_df.loc[near_p99_idx])

                    supplement_df = pd.DataFrame(supplement_conditions)
                    supplement_df = supplement_df.drop_duplicates()

                    if len(supplement_df) > shortage:
                        supplement_df = self._maximin_select_subset(
                            supplement_df, shortage, factors=self.factor_names
                        )

                    # Append to boundary
                    selected_boundary_df = pd.concat(
                        [selected_boundary_df, supplement_df], ignore_index=True
                    ).drop_duplicates()

            # Convert back to list of dicts and generate trials
            boundary_list = selected_boundary_df.to_dict("records")[:boundary_budget]

            for i, boundary_point in enumerate(boundary_list):
                subject_id = i % self.n_subjects
                batch_id = (i // self.n_subjects) % self.n_batches

                trial = {
                    "subject_id": subject_id,
                    "batch_id": batch_id,
                    "is_bridge": False,
                    "block_type": "individual",
                    "is_core1": False,
                    "is_core2": False,
                    "is_individual": True,
                    "is_boundary": True,
                    "is_lhs": False,
                    "interaction_pair_id": None,
                    "design_row_id": None,  # Will be matched later
                }
                # Add factor values
                for factor in self.factor_names:
                    if factor in boundary_point:
                        trial[factor] = boundary_point[factor]
                trials.append(trial)

        # LHS trials with optimized sampling using scipy's built-in optimization
        lhs_budget = self.budget_split["individual_lhs"]

        if lhs_budget > 0:
            # Use scipy's built-in optimization instead of manual iteration
            sampler = qmc.LatinHypercube(
                d=self.d,
                seed=self.seed,
                optimization="random-cd",  # Use Centered Discrepancy optimization
            )
            lhs_samples = sampler.random(n=lhs_budget)

            # Scale samples to factor ranges
            for i, sample in enumerate(lhs_samples):
                subject_id = i % self.n_subjects
                batch_id = (i // self.n_subjects) % self.n_batches

                trial = {
                    "subject_id": subject_id,
                    "batch_id": batch_id,
                    "is_bridge": False,
                    "block_type": "individual",
                    "is_core1": False,
                    "is_core2": False,
                    "is_individual": True,
                    "is_boundary": False,
                    "is_lhs": True,
                    "interaction_pair_id": None,
                    "design_row_id": None,  # Will be matched later
                }

                # Map LHS sample to actual factor values
                for j, factor in enumerate(self.factor_names):
                    factor_min = self.design_df[factor].min()
                    factor_max = self.design_df[factor].max()
                    trial[factor] = factor_min + sample[j] * (factor_max - factor_min)

                trials.append(trial)

        return trials

    def _add_design_row_ids(self, trial_schedule_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add design_row_id referencing the row in design_df by batch distance computation.

        Uses cdist for efficient vectorized distance calculation instead of row-by-row iteration.
        Complexity: O(n_trials × n_design × d) single pass, vs O(n_trials × d × n_design) iterative

        Args:
            trial_schedule_df: DataFrame with trial schedule

        Returns:
            DataFrame with design_row_id added
        """
        # Identify rows that need design_row_id matching
        needs_matching = (
            trial_schedule_df["design_row_id"].isna()
            | trial_schedule_df["design_row_id"].isnull()
        )

        if not needs_matching.any():
            # All rows already have design_row_id
            return trial_schedule_df

        # Extract feature matrices
        trial_features = trial_schedule_df.loc[needs_matching, self.factor_names].values
        design_features = self.design_df[self.factor_names].values

        # Use distance-based matching with different metrics for different factor types
        # For now, use Euclidean distance which works well for mixed types
        distances = pairwise_distances(
            trial_features, design_features, metric="euclidean"
        )

        # Find closest design point for each trial
        closest_indices = distances.argmin(axis=1)
        matched_design_ids = self.design_df.index[closest_indices].tolist()

        # Assign the matched IDs back to trial_schedule_df
        trial_schedule_df.loc[needs_matching, "design_row_id"] = matched_design_ids

        return trial_schedule_df

    def _get_marginal_coverage_table(self) -> Dict[str, Dict[Any, int]]:
        """
        Get marginal coverage table: count per factor-level/bin.

        Returns:
            A dictionary with coverage counts per factor
        """
        coverage_table = {}
        # This is a placeholder - in a full implementation, this would track actual coverage
        for factor in self.factor_names:
            if self.factor_types[factor] == "discrete":
                coverage_table[factor] = {
                    level: 0 for level in self.discretized_factors[factor]
                }
            else:
                # For continuous factors, use discretized bins
                coverage_table[factor] = {
                    f"bin_{i}": 0 for i in range(len(self.discretized_factors[factor]))
                }
        return coverage_table


# Utility functions
def detect_factor_types(design_df: pd.DataFrame) -> Dict[str, str]:
    """Wrapper for factor type detection."""
    generator = WarmupAEPsychGenerator(design_df)
    generator.factor_names = [col for col in design_df.columns if col.startswith("f")]
    return generator.detect_factor_types()


def get_levels_or_bins(design_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Wrapper for factor level/bin detection."""
    generator = WarmupAEPsychGenerator(design_df)
    generator.factor_names = [col for col in design_df.columns if col.startswith("f")]
    generator.factor_types = generator.detect_factor_types()
    return generator.get_levels_or_bins()


def compute_marginal_min_counts(design_df: pd.DataFrame) -> Dict[str, int]:
    """Wrapper for computing marginal minimum counts."""
    generator = WarmupAEPsychGenerator(design_df)
    generator.factor_names = [col for col in design_df.columns if col.startswith("f")]
    generator.factor_types = generator.detect_factor_types()
    generator.discretized_factors = generator.get_levels_or_bins()
    return generator.compute_marginal_min_counts()


def select_core1_points(
    design_df: pd.DataFrame, d: int, strategy: str = "corners+centers"
) -> pd.DataFrame:
    """Wrapper for Core-1 point selection."""
    generator = WarmupAEPsychGenerator(design_df)
    generator.d = d
    generator.factor_names = [col for col in design_df.columns if col.startswith("f")]
    generator.n_core1_points = generator._compute_core1_size()
    return generator.select_core1_points(strategy)


def build_interaction_pairs(
    factors: List[str], K: Optional[int] = None, heuristic: bool = True
) -> List[Tuple[int, int]]:
    """Wrapper for interaction pair building."""
    # Create a dummy design_df for initialization
    dummy_data = {f: np.random.rand(10) for f in factors}
    design_df = pd.DataFrame(dummy_data)
    generator = WarmupAEPsychGenerator(design_df)
    generator.d = len(factors)
    generator.factor_names = factors
    return generator.build_interaction_pairs(K, heuristic)


def build_boundary_library(design_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Wrapper for boundary library building."""
    generator = WarmupAEPsychGenerator(design_df)
    generator.factor_names = [col for col in design_df.columns if col.startswith("f")]
    return generator.build_boundary_library()


def compute_coverage_rate() -> float:
    """Wrapper for coverage rate computation."""
    # This is just a placeholder
    return 0.15


def compute_gini() -> float:
    """Wrapper for Gini coefficient computation."""
    # This is just a placeholder
    return 0.35


# Example usage (commented out)
"""
# Example usage:
# Create a sample design DataFrame
sample_data = {
    'f1': np.random.rand(100),
    'f2': np.random.rand(100),
    'f3': np.random.rand(100),
    'metadata': ['info'] * 100
}
design_df = pd.DataFrame(sample_data)

# Create generator
gen = WarmupAEPsychGenerator(design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42)
gen.fit_planning()
trials = gen.generate_trials()
plan = gen.summarize()
"""
