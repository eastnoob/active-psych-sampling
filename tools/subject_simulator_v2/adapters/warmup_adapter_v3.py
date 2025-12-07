#!/usr/bin/env python3
"""
Warmup Adapter V3 - Interaction-as-Features Method

This is a simplified adapter that implements the "interaction-as-features" approach
for optimal distribution (Config #1 from testing).

Key differences from V2:
- Interactions are treated as explicit features (appended to design space)
- Main effects: N(0.0, 0.3)
- Interaction x3*x4: 0.12 (fixed, strong, detectable)
- Interaction x0*x1: -0.02 (fixed, weak, for balance)

This produces excellent distribution:
  Mean: ~3.0, Max ratio: ~29%, All 5 levels covered

Usage:
    from subject_simulator_v2.adapters.warmup_adapter_v3 import run

    run(
        input_dir="path/to/sampling/plan",
        seed=99,
        design_space_csv="path/to/full_design_space.csv",
        # Other standard parameters...
    )
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional

# Add subject_simulator_v2 to path
_adapter_dir = Path(__file__).resolve().parent
_v2_root = _adapter_dir.parent
sys.path.insert(0, str(_v2_root.parent))

from subject_simulator_v2 import LinearSubject


def run(
    input_dir: Path,
    seed: int = 42,
    output_mode: str = "individual",
    clean: bool = False,
    # Interaction parameters (Config #1)
    interaction_x3x4_weight: float = 0.12,  # Strong categorical interaction
    interaction_x0x1_weight: float = -0.02,  # Weak continuous interaction (balance)
    # Model parameters
    output_type: str = "likert",
    likert_levels: int = 5,
    likert_sensitivity: float = 2.0,
    population_mean: float = 0.0,
    population_std: float = 0.3,
    individual_std_percent: float = 0.3,
    noise_std: float = 0.0,
    design_space_csv: Optional[str] = None,
    # Output control
    print_model: bool = False,
    save_model_summary: bool = False,
):
    """
    Warmup Adapter V3 with interaction-as-features method

    Produces optimal distribution with detectable interactions.
    """
    input_dir = Path(input_dir)
    result_dir = input_dir / "result"

    if clean and result_dir.exists():
        import shutil

        shutil.rmtree(result_dir)

    result_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Warmup Adapter V3 - Interaction-as-Features Method")
    print("=" * 80)
    print()

    # Find subject CSVs
    subject_csvs = sorted(input_dir.glob("subject_*.csv"))
    n_subjects = len(subject_csvs)

    if n_subjects == 0:
        raise FileNotFoundError(f"No subject_*.csv files found in {input_dir}")

    print(f"Found {n_subjects} subject CSV files")

    # Read first CSV to get feature info
    df_sample = pd.read_csv(subject_csvs[0])

    # Convert categorical variables
    categorical_mappings = {}
    categorical_cols = [
        "x3_OuterFurniture",
        "x4_VisualBoundary",
        "x5_PhysicalBoundary",
        "x6_InnerFurniture",
    ]

    for col in categorical_cols:
        if col in df_sample.columns:
            unique_vals = sorted(df_sample[col].unique())
            mapping = {val: i for i, val in enumerate(unique_vals)}
            categorical_mappings[col] = mapping
            print(f"  Categorical mapping for {col}: {mapping}")

    # Load full design space if provided
    if design_space_csv:
        df_full = pd.read_csv(design_space_csv)
        # Apply categorical mappings
        for col, mapping in categorical_mappings.items():
            if col in df_full.columns:
                df_full[col] = df_full[col].map(mapping)
    else:
        print(
            f"[Warning] No design_space_csv provided, using first subject CSV as design space"
        )
        df_full = df_sample.copy()
        for col, mapping in categorical_mappings.items():
            if col in df_full.columns:
                df_full[col] = df_full[col].map(mapping)

    # Extract base features (6 features)
    feature_cols = [
        "x1_CeilingHeight",
        "x2_GridModule",
        "x3_OuterFurniture",
        "x4_VisualBoundary",
        "x5_PhysicalBoundary",
        "x6_InnerFurniture",
    ]
    X_full_base = df_full[feature_cols].values

    # Create interaction features
    interaction_x3x4 = X_full_base[:, 2] * X_full_base[:, 3]  # x3 * x4
    interaction_x0x1 = X_full_base[:, 0] * X_full_base[:, 1]  # x0 * x1

    # Extended design space (8 features)
    X_full_extended = np.column_stack([X_full_base, interaction_x3x4, interaction_x0x1])

    print(
        f"Sampling design space: {X_full_base.shape[0]} points, {X_full_base.shape[1]} base features"
    )
    print(
        f"Extended design space: {X_full_extended.shape[0]} points, {X_full_extended.shape[1]} features (6 main + 2 interactions)"
    )
    print()

    # Sample population weights
    np.random.seed(seed)

    # Main effects (6 features): N(population_mean, population_std)
    main_weights = np.random.normal(population_mean, population_std, size=6)

    # Interaction features (2 features): Fixed weights
    interaction_weights = np.array([interaction_x3x4_weight, interaction_x0x1_weight])

    # Combined population weights (8 features)
    population_weights_extended = np.concatenate([main_weights, interaction_weights])

    print(f"[Population Weights]")
    print(f"  Main effects (6): {main_weights}")
    print(f"  Interaction x3*x4: {interaction_weights[0]:.3f}")
    print(f"  Interaction x0*x1: {interaction_weights[1]:.3f}")
    print()

    # Calculate bias
    continuous_output = X_full_extended @ population_weights_extended
    auto_bias = -continuous_output.mean()

    print(f"[Auto Bias Calculation]")
    print(f"  Design space: {X_full_extended.shape[0]} points")
    print(
        f"  Continuous output range: [{continuous_output.min():.2f}, {continuous_output.max():.2f}]"
    )
    print(f"  Continuous output mean: {continuous_output.mean():.2f}")
    print(f"  Auto-calculated bias: {auto_bias:.2f}")
    print()

    # Generate subject weights (extended)
    individual_std = population_std * individual_std_percent
    all_subject_weights = []

    print(f"[Generating Subject Weights]")
    print(f"  Population mean: {population_mean}")
    print(f"  Population std: {population_std}")
    print(
        f"  Individual std: {individual_std} ({individual_std_percent} * {population_std})"
    )
    print(f"  Noise std: {noise_std}")
    print()

    for subject_id in range(1, n_subjects + 1):
        # Sample individual weights (main effects only, interactions stay fixed)
        if n_subjects == 1:
            # When testing with single subject, use population weights directly
            individual_main_weights = main_weights.copy()
        else:
            # For multiple subjects, add individual deviation
            np.random.seed(seed + subject_id)
            deviation = np.random.normal(0, individual_std, size=6)
            individual_main_weights = main_weights + deviation

        individual_weights_extended = np.concatenate(
            [individual_main_weights, interaction_weights]
        )
        all_subject_weights.append(individual_weights_extended)

        print(f"  [OK] Subject {subject_id} weights generated")

    print()

    # Generate responses
    print(f"[Generating Responses]")
    all_results = []

    for subject_id, subject_csv in enumerate(subject_csvs, 1):
        # Read subject CSV
        df_subject = pd.read_csv(subject_csv)

        # Apply categorical mappings
        for col, mapping in categorical_mappings.items():
            if col in df_subject.columns:
                df_subject[col] = df_subject[col].map(mapping)

        # Extract base features
        X_subject_base = df_subject[feature_cols].values

        # Create interaction features
        interact_x3x4 = X_subject_base[:, 2] * X_subject_base[:, 3]
        interact_x0x1 = X_subject_base[:, 0] * X_subject_base[:, 1]

        # Extended features
        X_subject_extended = np.column_stack(
            [X_subject_base, interact_x3x4, interact_x0x1]
        )

        # Generate responses manually (without LinearSubject)
        subject_weights = all_subject_weights[subject_id - 1]

        # Calculate continuous output: y = X @ weights + bias + noise
        continuous_output = X_subject_extended @ subject_weights + auto_bias

        # Add noise if needed
        if noise_std > 0:
            np.random.seed(seed + subject_id + 1000)
            continuous_output += np.random.normal(
                0, noise_std, size=len(continuous_output)
            )

        # Convert to Likert using tanh transformation
        tanh_output = np.tanh(likert_sensitivity * continuous_output)
        responses = np.clip(
            np.round((tanh_output + 1) * 2 + 1), 1, likert_levels
        ).astype(int)

        # Save individual result
        # 只保留自变量列（x1-x6）和响应列，排除元数据列（如 Condition_ID）
        df_result = df_subject[feature_cols].copy()
        df_result["y"] = responses

        result_csv = result_dir / f"subject_{subject_id}.csv"
        df_result.to_csv(result_csv, index=False)

        all_results.append(df_result)
        print(f"  Subject {subject_id}: {len(responses)} responses generated")

    # Combined results
    if output_mode in ["combined", "both"]:
        df_combined = pd.concat(all_results, ignore_index=True)
        combined_csv = result_dir / "combined_results.csv"
        df_combined.to_csv(combined_csv, index=False)
        print(f"\n  Combined results: {len(df_combined)} total responses")

    # Save fixed_weights_auto.json
    fixed_weights_data = {
        "global": [main_weights.tolist()],
        "interactions": {
            "3,4": float(interaction_weights[0]),
            "0,1": float(interaction_weights[1]),
        },
        "bias": float(auto_bias),
        "method": "interaction_as_features_v3",
    }

    fixed_weights_path = result_dir / "fixed_weights_auto.json"
    with open(fixed_weights_path, "w") as f:
        json.dump(fixed_weights_data, f, indent=2)

    print()
    print(f"[OK] Simulation completed!")
    print(f"  Output directory: {result_dir}")
    print(f"  Files:")
    if output_mode in ["combined", "both"]:
        print(f"    - combined_results.csv")
    print(f"    - subject_1.csv ... subject_{n_subjects}.csv")
    print(f"    - fixed_weights_auto.json")
    print()

    # Print distribution stats
    if output_mode in ["combined", "both"]:
        from collections import Counter

        counter = Counter(df_combined["y"])
        print(f"[Combined Distribution]")
        for level in range(1, likert_levels + 1):
            count = counter.get(level, 0)
            pct = count / len(df_combined) * 100
            bar = "#" * int(pct / 5)
            print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

        mean = df_combined["y"].mean()
        print(f"\n  Mean: {mean:.2f}")
        print()

    print("=" * 80)

    return {
        "subject_weights": all_subject_weights,
        "results": all_results,
        "fixed_weights": fixed_weights_data,
    }


if __name__ == "__main__":
    print("Warmup Adapter V3 - Interaction-as-Features Method")
    print("This adapter implements Config #1 for optimal distribution")
    print()
    print("To use in quick_start.py:")
    print("  from subject_simulator_v2.adapters.warmup_adapter_v3 import run")
