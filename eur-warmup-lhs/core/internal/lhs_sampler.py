import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


def is_categorical(dtype) -> bool:
    return (
        isinstance(dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    )


def gower_distance(x1: pd.Series, x2: pd.Series, df: pd.DataFrame) -> float:
    distances = []

    for col in x1.index:
        val1 = x1[col]
        val2 = x2[col]

        if pd.isna(val1) or pd.isna(val2):
            distances.append(1.0)
            continue

        col_dtype = df[col].dtype

        if is_categorical(col_dtype) or pd.api.types.is_bool_dtype(col_dtype):
            distances.append(0.0 if val1 == val2 else 1.0)
        else:
            col_range = df[col].max() - df[col].min()
            if col_range == 0:
                distances.append(0.0)
            else:
                distances.append(abs(val1 - val2) / col_range)

    return float(np.mean(distances)) if distances else 0.0


class LhsSampler:
    """Pure LHS warmup sampler with optional shared points."""

    def __init__(self, design_csv_path: str):
        self.design_csv_path = design_csv_path
        self.design_df = pd.read_csv(design_csv_path)
        self.factor_names = list(self.design_df.columns)

        logger.info(f"Loaded design space: {design_csv_path}")
        logger.info(f"  Total configurations: {len(self.design_df)}")
        logger.info(f"  Number of factors: {len(self.factor_names)}")
        logger.info(f"  Factor names: {', '.join(self.factor_names)}")

    def describe_design_space(self) -> None:
        logger.info("=" * 70)
        logger.info("Design Space Analysis")
        logger.info("=" * 70)
        logger.info(f"Total configurations: {len(self.design_df)}")
        logger.info(f"Number of factors: {len(self.factor_names)}")

        for factor in self.factor_names:
            values = self.design_df[factor]
            unique_vals = values.unique()
            if is_categorical(values.dtype) or pd.api.types.is_bool_dtype(values.dtype):
                logger.info(
                    f"  {factor}: categorical, {len(unique_vals)} levels, sample={list(unique_vals)[:8]}"
                )
            else:
                logger.info(
                    f"  {factor}: numeric, range=[{values.min():.3f}, {values.max():.3f}], unique={len(unique_vals)}"
                )

    def evaluate_plan(
        self,
        n_subjects: int,
        trials_per_subject: int,
        shared_points: int = 0,
        allocation_mode: str = "disjoint",
    ) -> Dict[str, int]:
        if shared_points < 0:
            raise ValueError("shared_points must be >= 0")
        if shared_points > trials_per_subject:
            raise ValueError("shared_points cannot exceed trials_per_subject")
        if allocation_mode not in {"disjoint", "shared"}:
            raise ValueError("allocation_mode must be 'disjoint' or 'shared'")

        if allocation_mode == "shared":
            requested_unique = trials_per_subject
            unique_per_subject = trials_per_subject
        else:
            unique_per_subject = trials_per_subject - shared_points
            requested_unique = shared_points + n_subjects * unique_per_subject

        available_unique = len(self.design_df)
        actual_unique = min(requested_unique, available_unique)
        duplicate_pressure = max(0, requested_unique - available_unique)

        return {
            "n_subjects": n_subjects,
            "trials_per_subject": trials_per_subject,
            "shared_points": shared_points,
            "allocation_mode": allocation_mode,
            "requested_unique": requested_unique,
            "actual_unique": actual_unique,
            "duplicate_pressure": duplicate_pressure,
            "total_samples": n_subjects * trials_per_subject,
            "unique_per_subject": unique_per_subject,
        }

    def print_plan_report(self, plan: Dict[str, int], interaction_pairs: Optional[List[Tuple[int, int]]] = None) -> None:
        logger.info("=" * 70)
        logger.info("LHS Sampling Plan")
        logger.info("=" * 70)
        logger.info(f"Subjects: {plan['n_subjects']}")
        logger.info(f"Trials per subject: {plan['trials_per_subject']}")
        logger.info(f"Allocation mode: {plan['allocation_mode']}")
        logger.info(f"Shared points: {plan['shared_points']}")
        logger.info(f"Requested unique configurations: {plan['requested_unique']}")
        logger.info(f"Available unique configurations: {len(self.design_df)}")
        logger.info(f"Actual unique configurations used: {plan['actual_unique']}")
        logger.info(f"Total sample records: {plan['total_samples']}")

        if plan["duplicate_pressure"] > 0:
            logger.warning(
                f"Unique design space is smaller than requested by {plan['duplicate_pressure']} points; duplicates may occur."
            )

        if interaction_pairs:
            logger.info(
                f"Interaction settings preserved for downstream steps: {len(interaction_pairs)} specified pairs"
            )

    def generate_samples(
        self,
        n_subjects: int,
        trials_per_subject: int,
        output_dir: str,
        merge: bool = False,
        subject_col_name: str = "subject_id",
        shared_points: int = 0,
        allocation_mode: str = "disjoint",
        lhs_oversample_factor: int = 5,
        random_seed: int = 42,
        repeat_one_point: bool = False,
        interaction_pairs_to_explore: Optional[List[Tuple[int, int]]] = None,
    ) -> List[str]:
        if allocation_mode not in {"disjoint", "shared"}:
            raise ValueError("allocation_mode must be 'disjoint' or 'shared'")

        rng = np.random.default_rng(random_seed)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        plan = self.evaluate_plan(
            n_subjects=n_subjects,
            trials_per_subject=trials_per_subject,
            shared_points=shared_points,
            allocation_mode=allocation_mode,
        )
        self.print_plan_report(plan, interaction_pairs=interaction_pairs_to_explore)

        shared_indices: List[int] = []
        if allocation_mode == "disjoint" and shared_points > 0:
            shared_indices = self._select_maximin_points(
                n_points=shared_points,
                used_indices=set(),
                rng=rng,
            )

        used_indices = set(shared_indices)

        if allocation_mode == "shared":
            shared_indices = self._select_lhs_points(
                n_points=trials_per_subject,
                used_indices=set(),
                oversample_factor=lhs_oversample_factor,
                rng=rng,
            )
            subject_point_sets = [list(shared_indices) for _ in range(n_subjects)]
        else:
            subject_unique = trials_per_subject - shared_points
            total_unique_needed = n_subjects * subject_unique
            global_indices = self._select_lhs_points(
                n_points=total_unique_needed,
                used_indices=used_indices,
                oversample_factor=lhs_oversample_factor,
                rng=rng,
            )

            subject_point_sets = []
            offset = 0
            for _ in range(n_subjects):
                own_indices = global_indices[offset:offset + subject_unique]
                offset += subject_unique
                subject_point_sets.append(shared_indices + own_indices)

        exported_files: List[str] = []
        all_samples = []

        for subject_id, indices in enumerate(subject_point_sets, start=1):
            subject_df = self.design_df.loc[indices].copy().reset_index(drop=True)
            subject_df[subject_col_name] = subject_id

            if repeat_one_point and len(subject_df) >= 2:
                source_idx = int(rng.integers(0, len(subject_df)))
                target_idx = (source_idx + 1) % len(subject_df)
                subject_df.iloc[target_idx] = subject_df.iloc[source_idx]

            subject_df = subject_df.sample(frac=1, random_state=random_seed + subject_id).reset_index(drop=True)
            all_samples.append(subject_df)

        if merge:
            merged_df = pd.concat(all_samples, ignore_index=True)
            merged_path = output_path / "warmup_samples_all.csv"
            merged_df.to_csv(merged_path, index=False)
            exported_files.append(str(merged_path))
        else:
            for subject_id, subject_df in enumerate(all_samples, start=1):
                file_path = output_path / f"subject_{subject_id}.csv"
                subject_df.drop(columns=[subject_col_name]).to_csv(file_path, index=False)
                exported_files.append(str(file_path))

        self._generate_readme(
            readme_path=output_path / "README_sampling.txt",
            plan=plan,
            merged=merge,
            lhs_oversample_factor=lhs_oversample_factor,
            interaction_pairs_to_explore=interaction_pairs_to_explore,
        )

        return exported_files

    def _select_lhs_points(
        self,
        n_points: int,
        used_indices: set,
        oversample_factor: int,
        rng: np.random.Generator,
    ) -> List[int]:
        available = [idx for idx in self.design_df.index if idx not in used_indices]
        target = min(n_points, len(available))
        if target <= 0:
            return []

        selected: List[int] = []

        try:
            from scipy.stats import qmc

            sampler = qmc.LatinHypercube(d=len(self.factor_names), seed=int(rng.integers(0, 1_000_000)))
            lhs_samples = sampler.random(n=max(target * max(1, oversample_factor), target))

            for sample in lhs_samples:
                if len(selected) >= target:
                    break
                target_config = self._lhs_sample_to_target(sample)
                best_idx = self._nearest_available_index(target_config, used_indices | set(selected))
                if best_idx is not None:
                    selected.append(best_idx)
        except ImportError:
            logger.warning("scipy not installed, falling back to maximin fill")

        if len(selected) < target:
            fill = self._select_maximin_points(
                n_points=target - len(selected),
                used_indices=used_indices | set(selected),
                rng=rng,
            )
            selected.extend(fill)

        return selected

    def _lhs_sample_to_target(self, sample: np.ndarray) -> pd.Series:
        target_config = pd.Series(index=self.design_df.columns, dtype=object)

        for idx, col in enumerate(self.design_df.columns):
            col_data = self.design_df[col]
            val = sample[idx]
            if is_categorical(col_data.dtype):
                if isinstance(col_data.dtype, pd.CategoricalDtype):
                    levels = col_data.cat.categories.tolist()
                else:
                    levels = sorted(col_data.unique())
                level_idx = min(int(math.floor(val * len(levels))), len(levels) - 1)
                target_config[col] = levels[level_idx]
            elif pd.api.types.is_bool_dtype(col_data.dtype):
                target_config[col] = bool(val > 0.5)
            else:
                col_min = col_data.min()
                col_max = col_data.max()
                target_config[col] = col_min + val * (col_max - col_min)

        return target_config

    def _nearest_available_index(self, target_config: pd.Series, blocked_indices: set) -> Optional[int]:
        available = [idx for idx in self.design_df.index if idx not in blocked_indices]
        if not available:
            return None

        distances = {
            idx: gower_distance(target_config, self.design_df.loc[idx], self.design_df)
            for idx in available
        }
        return min(distances, key=distances.get)

    def _select_maximin_points(
        self,
        n_points: int,
        used_indices: set,
        rng: np.random.Generator,
    ) -> List[int]:
        available = [idx for idx in self.design_df.index if idx not in used_indices]
        if not available or n_points <= 0:
            return []

        selected: List[int] = []
        first_idx = int(rng.choice(available))
        selected.append(first_idx)

        while len(selected) < min(n_points, len(available)):
            best_idx = None
            best_min_dist = -1.0

            for idx in available:
                if idx in selected:
                    continue
                min_dist = min(
                    gower_distance(self.design_df.loc[idx], self.design_df.loc[chosen], self.design_df)
                    for chosen in selected
                )
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = idx

            if best_idx is None:
                break
            selected.append(best_idx)

        return selected[:n_points]

    def _generate_readme(
        self,
        readme_path: Path,
        plan: Dict[str, int],
        merged: bool,
        lhs_oversample_factor: int,
        interaction_pairs_to_explore: Optional[List[Tuple[int, int]]],
    ) -> None:
        with open(readme_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("=" * 80 + "\n")
            file_obj.write("Warmup Phase Sampling Instructions (LHS Version)\n")
            file_obj.write("=" * 80 + "\n\n")

            file_obj.write("1. Experimental Design\n")
            file_obj.write("-" * 80 + "\n")
            file_obj.write(f"Design Space: {self.design_csv_path}\n")
            file_obj.write(f"Number of Subjects: {plan['n_subjects']}\n")
            file_obj.write(f"Trials per Subject: {plan['trials_per_subject']}\n")
            file_obj.write(f"Total Sample Records: {plan['total_samples']}\n")
            file_obj.write(f"Actual Unique Configurations: {plan['actual_unique']}\n\n")

            file_obj.write("2. Sampling Strategy\n")
            file_obj.write("-" * 80 + "\n")
            file_obj.write("Sampling method: global Latin Hypercube Sampling (mapped to discrete design space)\n")
            file_obj.write(f"Allocation mode: {plan['allocation_mode']}\n")
            file_obj.write(f"Shared points: {plan['shared_points']}\n")
            file_obj.write(f"LHS oversample factor: {lhs_oversample_factor}\n")
            if interaction_pairs_to_explore:
                file_obj.write(f"Downstream interaction pairs preserved: {interaction_pairs_to_explore}\n")
            file_obj.write("\n")

            file_obj.write("3. Data Collection Guide\n")
            file_obj.write("-" * 80 + "\n")
            if merged:
                file_obj.write("- Use file: warmup_samples_all.csv\n")
                file_obj.write("- subject_id column identifies subject number\n")
            else:
                file_obj.write("- One file per subject: subject_1.csv ~ subject_N.csv\n")
                file_obj.write("- Test sequentially according to the row order in the file\n")

            file_obj.write("- Record response values in a new response column, e.g. y\n")
