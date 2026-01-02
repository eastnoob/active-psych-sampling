#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test individual subject weights from 202511301011

Purpose:
1. Compare response distributions across different subject configurations
2. Test population average vs individual subjects (1-5)
3. Identify which subject weight provides best response diversity
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Set UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Add module paths
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "modules"))

from single_output_subject import SingleOutputLatentSubject
import json

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def load_design_space_inline(design_space_path: Path):
    """Load and transform design space"""
    df_design = pd.read_csv(design_space_path)
    X_canonical = np.zeros((len(df_design), 6), dtype=float)

    # CeilingHeight: 0,1,2
    height_map = {2.8: 0, 4.0: 1, 8.5: 2}
    X_canonical[:, 0] = df_design.iloc[:, 0].map(height_map).values

    # GridModule: 0,1
    grid_map = {6.5: 0, 8.0: 1}
    X_canonical[:, 1] = df_design.iloc[:, 1].map(grid_map).values

    # Categorical variables
    for col_idx in [2, 3, 5]:
        col_name = df_design.columns[col_idx]
        unique_vals = df_design[col_name].unique()
        val_to_idx = {val: float(idx) for idx, val in enumerate(sorted(unique_vals))}
        X_canonical[:, col_idx] = df_design[col_name].map(val_to_idx).values

    # PhysicalBoundary: 0,1
    boundary_vals = df_design.iloc[:, 4].unique()
    boundary_map = {val: float(idx) for idx, val in enumerate(sorted(boundary_vals))}
    X_canonical[:, 4] = df_design.iloc[:, 4].map(boundary_map).values

    return df_design, X_canonical


def create_oracle_with_weights(weights: np.ndarray, subject_name: str):
    """
    Create Oracle with specified weights

    Args:
        weights: Weight matrix (1, 6)
        subject_name: Name for identification
    """
    oracle_kwargs = {
        "seed": 42,
        "bias": 0.0,
        "noise_std": 0.1,
        "interaction_pairs": [(3, 4), (0, 1)],
        "interaction_scale": 0.25,
        "likert_levels": 5,
        "likert_sensitivity": 2.0,
        "use_latent": False,
        "fixed_weights": weights,
        "num_features": weights.shape[1],
    }

    oracle = SingleOutputLatentSubject(**oracle_kwargs)
    print(f"Created Oracle: {subject_name}")
    print(f"  Weights: {weights[0]}")

    return oracle


def sample_design_space(oracle, design_space_np: np.ndarray):
    """Sample entire design space with given oracle"""
    N = design_space_np.shape[0]
    responses = np.zeros(N, dtype=int)

    for i in range(N):
        responses[i] = oracle(design_space_np[i])

    return responses


def analyze_subject_distribution(responses: np.ndarray, subject_name: str):
    """Analyze and print distribution for a subject"""
    counter = Counter(responses)
    total = len(responses)

    print(f"\n{subject_name} Response Distribution:")
    print(f"  Likert=1: {counter.get(1, 0):3d} points ({counter.get(1, 0)/total*100:5.1f}%)")
    print(f"  Likert=2: {counter.get(2, 0):3d} points ({counter.get(2, 0)/total*100:5.1f}%)")
    print(f"  Likert=3: {counter.get(3, 0):3d} points ({counter.get(3, 0)/total*100:5.1f}%)")
    print(f"  Likert=4: {counter.get(4, 0):3d} points ({counter.get(4, 0)/total*100:5.1f}%)")
    print(f"  Likert=5: {counter.get(5, 0):3d} points ({counter.get(5, 0)/total*100:5.1f}%)")
    print(f"  Mean: {np.mean(responses):.2f}, Std: {np.std(responses):.2f}")
    print(f"  Range: [{np.min(responses)}, {np.max(responses)}]")

    return counter


def plot_comparison(distributions: dict, save_dir: Path):
    """
    Plot comparison of all subject distributions

    Args:
        distributions: Dict of {subject_name: Counter}
        save_dir: Save directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    likert_values = list(range(1, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 5))

    for idx, (subject_name, counter) in enumerate(distributions.items()):
        ax = axes[idx]
        counts = [counter.get(v, 0) for v in likert_values]
        total = sum(counts)

        bars = ax.bar(likert_values, counts, color=colors, alpha=0.8, edgecolor='black')
        ax.set_xlabel('Likert Response', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f'{subject_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(likert_values)
        ax.set_ylim(0, 324)
        ax.grid(axis='y', alpha=0.3)

        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{count}\n({count/total*100:.1f}%)',
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    fig_path = save_dir / "subjects_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved: {fig_path}")
    plt.close()


def recommend_subject(distributions: dict):
    """Recommend best subject based on response diversity"""
    print(f"\n{'='*60}")
    print(f"Subject Recommendation Analysis")
    print(f"{'='*60}")

    best_score = -1
    best_subject = None

    for subject_name, counter in distributions.items():
        if subject_name == "Population Average":
            continue

        # Calculate diversity metrics
        counts = [counter.get(v, 0) for v in range(1, 6)]
        total = sum(counts)

        # Diversity score: entropy-like measure
        probs = [c/total for c in counts if c > 0]
        entropy = -sum(p * np.log(p) for p in probs)

        # Coverage: number of Likert levels with >0 samples
        coverage = sum(1 for c in counts if c > 0)

        # Balance: inverse of std (lower std = more balanced)
        balance = 1 / (np.std(counts) + 1)

        # Combined score (you can adjust weights)
        score = entropy * 0.5 + coverage * 0.3 + balance * 0.2

        print(f"\n{subject_name}:")
        print(f"  Entropy: {entropy:.3f}")
        print(f"  Coverage: {coverage}/5 levels")
        print(f"  Balance: {balance:.3f}")
        print(f"  Combined Score: {score:.3f}")

        if score > best_score:
            best_score = score
            best_subject = subject_name

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {best_subject}")
    print(f"{'='*60}\n")

    return best_subject


def main():
    """Main function"""
    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Individual Subject Weights Testing")
    print(f"{'='*60}\n")

    # Load design space
    print("Step 1: Loading design space")
    design_space_csv = (
        PROJECT_ROOT
        / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    )
    design_space_df, design_space_np = load_design_space_inline(design_space_csv)
    print(f"Design space: {design_space_np.shape}")

    # Define all subject weights (from 202511301011/result)
    subjects_weights = {
        "Population Average": np.array([[-0.12546, 0.45071, 0.23199, 0.09866, -0.34398, -0.34401]]),
        "Subject 1": np.array([[-0.09457, 0.34170, 0.18657, 0.03447, -0.24101, -0.39357]]),
        "Subject 2": np.array([[-0.21553, 0.60868, 0.38153, -0.09393, -0.52016, -0.54981]]),
        "Subject 3": np.array([[-0.12229, 0.48195, 0.18458, 0.07414, -0.49658, -0.65563]]),
        "Subject 4": np.array([[-0.05527, 0.59846, 0.33062, 0.00275, -0.29453, -0.36514]]),
        "Subject 5": np.array([[-0.22722, 0.60742, 0.34290, 0.17551, -0.47055, -0.12827]]),
    }

    # Test each subject
    print("\nStep 2: Testing all subjects")
    distributions = {}

    for subject_name, weights in subjects_weights.items():
        print(f"\n{'-'*60}")
        oracle = create_oracle_with_weights(weights, subject_name)
        responses = sample_design_space(oracle, design_space_np)
        counter = analyze_subject_distribution(responses, subject_name)
        distributions[subject_name] = counter

        # Save individual results
        result_df = design_space_df.copy()
        result_df['y_likert'] = responses
        csv_path = save_dir / f"{subject_name.lower().replace(' ', '_')}_results.csv"
        result_df.to_csv(csv_path, index=False)

    # Plot comparison
    print("\nStep 3: Plotting comparison")
    plot_comparison(distributions, save_dir)

    # Recommend best subject
    print("\nStep 4: Analyzing diversity")
    best_subject = recommend_subject(distributions)

    # Save recommendation
    recommendation = {
        "best_subject": best_subject,
        "weights": subjects_weights[best_subject].tolist(),
        "distributions": {
            name: {str(k): v for k, v in counter.items()}
            for name, counter in distributions.items()
        }
    }

    import json
    rec_path = save_dir / "recommendation.json"
    with open(rec_path, 'w') as f:
        json.dump(recommendation, f, indent=2)
    print(f"Recommendation saved: {rec_path}")

    print(f"\n{'='*60}")
    print(f"Test completed! Results saved in: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
