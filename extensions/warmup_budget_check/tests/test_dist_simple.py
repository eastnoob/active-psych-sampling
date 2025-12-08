#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的响应分布测试 - 无emoji版本
测试模拟被试在实际设计空间上的响应分布特征
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from collections import Counter

# 添加项目路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "tools"))

try:
    from subject_simulator_v2 import LinearSubject
except ImportError as e:
    print(f"[ERROR] Cannot import subject_simulator_v2: {e}")
    sys.exit(1)


def load_design_space(csv_path):
    """Load design space and extract factor columns"""
    df = pd.read_csv(csv_path)
    print(f"\n[LOAD] Design space: {Path(csv_path).name}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Smart filter: only x1, x2, ... xN columns
    import re
    factor_cols = [c for c in df.columns if re.match(r'^x\d+', c)]

    if not factor_cols:
        print("[WARN] No x1, x2... columns found, using all columns")
        factor_cols = list(df.columns)
    else:
        excluded = set(df.columns) - set(factor_cols)
        if excluded:
            print(f"  [FILTER] Using {len(factor_cols)} factor columns")
            print(f"  [FILTER] Excluded: {', '.join(sorted(excluded))}")

    return df, factor_cols


def encode_design_space(df, factor_cols):
    """Encode categorical variables to numeric"""
    df_factors = df[factor_cols].copy()
    encodings = {}

    for col in df_factors.columns:
        if df_factors[col].dtype == 'object':
            unique_vals = sorted(df_factors[col].dropna().unique())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            encodings[col] = mapping
            df_factors[col] = df_factors[col].map(mapping)
        elif df_factors[col].dtype == 'bool':
            df_factors[col] = df_factors[col].astype(int)

    X = df_factors.values.astype(float)
    return X, encodings


def simulate_responses(X, n_subjects=20, seed=42, config=None):
    """Simulate responses using LinearSubject"""
    if config is None:
        config = {}

    n_trials, n_features = X.shape
    pop_mean = config.get('population_mean', 0.0)
    pop_std = config.get('population_std', 0.25)
    ind_std_pct = config.get('individual_std_percent', 0.5)
    likert_levels = config.get('likert_levels', 5)
    likert_sensitivity = config.get('likert_sensitivity', 2.0)
    interaction_pairs = config.get('interaction_pairs', None)
    interaction_scale = config.get('interaction_scale', 0.25)

    print(f"\n[SIMULATE] {n_subjects} subjects, {n_trials} trials, {n_features} features")
    print(f"  Population: N({pop_mean}, {pop_std}^2)")
    print(f"  Individual std: {ind_std_pct} x {pop_std} = {ind_std_pct * pop_std}")
    print(f"  Likert: {likert_levels} levels, sensitivity={likert_sensitivity}")

    # Build interaction weights dict
    interaction_weights_dict = None
    if interaction_pairs:
        np.random.seed(seed)
        n_inter = len(interaction_pairs)
        weights_array = np.random.normal(0.0, interaction_scale, size=(n_inter,))
        interaction_weights_dict = {
            tuple(pair): float(w) for pair, w in zip(interaction_pairs, weights_array)
        }
        print(f"  Interactions: {len(interaction_pairs)} pairs, scale={interaction_scale}")

    # Generate subjects
    responses = np.zeros((n_subjects, n_trials))
    subjects = []

    for i in range(n_subjects):
        np.random.seed(seed + i)
        # Individual weights: population + individual variation
        ind_weights = np.random.normal(
            pop_mean,
            np.sqrt(pop_std**2 + (pop_std * ind_std_pct)**2),
            size=(n_features,)
        )

        subject = LinearSubject(
            weights=ind_weights,
            interaction_weights=interaction_weights_dict,
            bias=0.0,
            noise_std=0.0,
            likert_levels=likert_levels,
            likert_sensitivity=likert_sensitivity,
            seed=seed + i,
        )
        subjects.append(subject)

        # Simulate trials
        for j in range(n_trials):
            responses[i, j] = subject(X[j, :])

    print("  [OK] Simulation complete")
    return responses, subjects


def analyze_distribution(responses, likert_levels=5):
    """Analyze response distribution statistics"""
    all_resp = responses.flatten()

    # Basic stats
    stats = {
        "mean": float(np.mean(all_resp)),
        "std": float(np.std(all_resp)),
        "median": float(np.median(all_resp)),
        "min": float(np.min(all_resp)),
        "max": float(np.max(all_resp)),
    }

    # Skewness & Kurtosis
    x = all_resp - np.mean(all_resp)
    m2 = np.mean(x**2)
    m3 = np.mean(x**3)
    m4 = np.mean(x**4)
    stats["skewness"] = float(m3 / (m2**1.5 + 1e-10))
    stats["kurtosis"] = float(m4 / (m2**2 + 1e-10) - 3)

    # Likert distribution
    if likert_levels > 0:
        counter = Counter(all_resp)
        likert_dist = {int(k): int(v) for k, v in sorted(counter.items())}
        stats["likert_distribution"] = likert_dist

        total = len(all_resp)
        likert_pct = {k: round(v/total*100, 2) for k, v in likert_dist.items()}
        stats["likert_percent"] = likert_pct

    # Between-subject variance
    subj_means = np.mean(responses, axis=1)
    stats["between_subject_std"] = float(np.std(subj_means))
    stats["between_subject_range"] = float(np.max(subj_means) - np.min(subj_means))

    return stats


def print_report(stats, output_dir=None):
    """Print analysis report"""
    print("\n" + "="*80)
    print("[REPORT] Response Distribution Analysis")
    print("="*80)

    print("\n[BASIC STATS]")
    print(f"  Mean:   {stats['mean']:.3f}")
    print(f"  Std:    {stats['std']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  Range:  [{stats['min']:.1f}, {stats['max']:.1f}]")

    print("\n[DISTRIBUTION SHAPE]")
    print(f"  Skewness: {stats['skewness']:.3f}")
    skew_label = "left-skewed" if stats['skewness'] < -0.5 else ("right-skewed" if stats['skewness'] > 0.5 else "symmetric")
    print(f"    -> {skew_label}")

    print(f"  Kurtosis: {stats['kurtosis']:.3f}")
    kurt_label = "heavy-tailed" if stats['kurtosis'] > 0 else "light-tailed"
    print(f"    -> {kurt_label}")

    if "likert_distribution" in stats:
        print("\n[LIKERT DISTRIBUTION]")
        for level in sorted(stats["likert_distribution"].keys()):
            count = stats["likert_distribution"][level]
            pct = stats["likert_percent"][level]
            bar = "#" * int(pct / 2)
            print(f"  {level}: {bar} {pct:.1f}% (n={count})")

    print("\n[BETWEEN-SUBJECT VARIANCE]")
    print(f"  Std of subject means: {stats['between_subject_std']:.3f}")
    print(f"  Range of subject means: {stats['between_subject_range']:.3f}")

    print("\n[ASSESSMENT]")
    is_symmetric = abs(stats['skewness']) < 0.5
    is_normal_kurt = abs(stats['kurtosis']) < 1.0
    print(f"  Symmetry: {'PASS' if is_symmetric else 'FAIL (skewed)'}")
    print(f"  Normal kurtosis: {'PASS' if is_normal_kurt else 'FAIL (abnormal)'}")

    if "likert_percent" in stats:
        middle = stats["likert_percent"].get(3, 0)
        extremes = stats["likert_percent"].get(1, 0) + stats["likert_percent"].get(5, 0)
        is_reasonable = middle > extremes
        print(f"  Likert reasonableness: {'PASS (middle > extremes)' if is_reasonable else 'FAIL (too many extremes)'}")

    print("\n" + "="*80)

    # Save JSON
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / "distribution_stats.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] Stats saved to: {json_path}")


def main():
    # Configuration
    design_csv = project_root / "data" / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    output_dir = Path(__file__).parent / "output" / "dist_test"

    config = {
        "population_mean": 0.0,
        "population_std": 0.25,
        "individual_std_percent": 0.5,
        "likert_levels": 5,
        "likert_sensitivity": 0.3,  # 进一步降低以测试更均衡的分布
        "interaction_pairs": [(3, 4), (0, 1)],
        "interaction_scale": 0.25,
    }

    print("\n" + "="*80)
    print("[TEST] Response Distribution Test")
    print("="*80)

    # 1. Load design space
    df, factor_cols = load_design_space(str(design_csv))

    # 2. Encode
    X, encodings = encode_design_space(df, factor_cols)
    if encodings:
        print(f"\n[ENCODE] Encoded columns: {list(encodings.keys())}")

    # 3. Simulate
    responses, subjects = simulate_responses(X, n_subjects=20, seed=42, config=config)

    # 4. Analyze
    stats = analyze_distribution(responses, config["likert_levels"])

    # 5. Report
    print_report(stats, output_dir)

    # 6. Save sample
    sample_df = df[factor_cols].head(10).copy()
    for i in range(min(5, len(subjects))):
        sample_df[f"subject_{i+1}"] = responses[i, :10]

    sample_csv = output_dir / "sample_responses.csv"
    sample_df.to_csv(sample_csv, index=False)
    print(f"[SAVE] Sample responses: {sample_csv}")

    print("\n[DONE] Test complete!\n")


if __name__ == "__main__":
    main()
