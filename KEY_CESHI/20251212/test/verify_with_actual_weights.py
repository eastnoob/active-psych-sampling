#!/usr/bin/env python3
"""
Verify using actual weights from subject_X_model.md
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json

os.environ['PYTHONIOENCODING'] = 'utf-8'

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "modules"))

from single_output_subject import SingleOutputLatentSubject

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def transform_design_point(row):
    """Transform a single design point to canonical format"""
    x = np.zeros(6, dtype=float)

    height_map = {2.8: 0, 4.0: 1, 8.5: 2}
    x[0] = height_map[row['x1_CeilingHeight']]

    grid_map = {6.5: 0, 8.0: 1}
    x[1] = grid_map[row['x2_GridModule']]

    furniture_map = {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
    x[2] = furniture_map[row['x3_OuterFurniture']]

    visual_map = {'Solid': 0, 'Translucent': 1, 'Color': 2}
    x[3] = visual_map[row['x4_VisualBoundary']]

    physical_map = {'Closed': 0, 'Open': 1}
    x[4] = physical_map[row['x5_PhysicalBoundary']]

    x[5] = furniture_map[row['x6_InnerFurniture']]

    return x


def create_oracle(fixed_weights, item_biases, item_noises, interaction_weights):
    """Create oracle with fixed weights and item parameters"""
    oracle_kwargs = {
        "seed": 42,  # Seed doesn't matter for deterministic weights
        "bias": 0.0,
        "noise_std": 0.0,  # Deterministic: no trial-to-trial noise
        "interaction_pairs": [],  # Don't auto-generate, we'll set manually
        "interaction_scale": 0.25,
        "likert_levels": 5,
        "likert_sensitivity": 2.0,
        "use_latent": False,
        "fixed_weights": fixed_weights,
        "num_features": 6,
        "item_biases": item_biases,
        "item_noises": np.zeros_like(item_noises),  # Set to zero for deterministic output
    }
    oracle = SingleOutputLatentSubject(**oracle_kwargs)
    # Manually set interaction weights to the pre-computed values
    oracle.interaction_weights = interaction_weights
    return oracle


def main():
    """Main verification"""
    print("\\nVerifying with actual weights from subject_X_model.md\\n")

    # Load combined_results.csv
    combined_path = (
        PROJECT_ROOT
        / "extensions/warmup_budget_check/sample/202511301011/result/combined_results.csv"
    )
    combined_df = pd.read_csv(combined_path)

    # Load item parameters
    item_params_path = Path(__file__).parent / "item_params.json"
    with open(item_params_path, 'r') as f:
        item_params = json.load(f)

    # Actual weights from subject_X_model.md
    subject_weights = {
        "subject_1": np.array([[-0.09457, 0.34170, 0.18657, 0.03447, -0.24101, -0.39357]]),
        "subject_2": np.array([[-0.21553, 0.60868, 0.38153, -0.09393, -0.52016, -0.54981]]),
        "subject_3": np.array([[-0.12229, 0.48195, 0.18458, 0.07414, -0.49658, -0.65563]]),
        "subject_4": np.array([[-0.05527, 0.59846, 0.33062, 0.00275, -0.29453, -0.36514]]),
        "subject_5": np.array([[-0.22722, 0.60742, 0.34290, 0.17551, -0.47055, -0.12827]]),
    }

    total_matches = 0
    total_samples = 0

    for subject_name, weights in subject_weights.items():
        # Get item parameters for this subject
        subj_params = item_params[subject_name]
        item_biases = np.array(subj_params['item_biases'])
        item_noises = np.array(subj_params['item_noises'])

        # Convert interaction_weights from "i,j": weight format to (i,j): weight
        interaction_weights = {
            tuple(map(int, k.split(','))): v
            for k, v in subj_params['interaction_weights'].items()
        }

        oracle = create_oracle(weights, item_biases, item_noises, interaction_weights)

        subject_df = combined_df[combined_df['subject'] == subject_name].copy()

        predicted_responses = []
        for idx, row in subject_df.iterrows():
            x = transform_design_point(row)
            y_pred = oracle(x)
            predicted_responses.append(y_pred)

        subject_df['y_predicted'] = predicted_responses

        matches = (subject_df['y'] == subject_df['y_predicted']).sum()
        total = len(subject_df)
        match_rate = matches / total * 100

        total_matches += matches
        total_samples += total

        print(f"{subject_name}:")
        print(f"  Match rate: {matches}/{total} ({match_rate:.1f}%)")
        print(f"  Original: {dict(subject_df['y'].value_counts().sort_index())}")
        print(f"  Predicted: {dict(subject_df['y_predicted'].value_counts().sort_index())}")

        if match_rate < 100:
            print(f"  Mismatches:")
            mismatches = subject_df[subject_df['y'] != subject_df['y_predicted']]
            for idx, row in mismatches.head(3).iterrows():
                print(f"    Orig={row['y']}, Pred={row['y_predicted']}")
        print()

    overall_rate = total_matches / total_samples * 100
    print(f"{'='*60}")
    print(f"Overall match rate: {total_matches}/{total_samples} ({overall_rate:.1f}%)")
    if overall_rate > 95:
        print("SUCCESS: Configuration is correct!")
    else:
        print("FAILED: Configuration is still incorrect.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
