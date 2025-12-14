#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import json
import time
from scipy import stats
import pandas as pd
import shutil

# Add tools directory to path
tools_path = Path(__file__).parent.parent.parent.parent / "tools"
sys.path.insert(0, str(tools_path))

sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from subject_simulator_v2.adapters.warmup_adapter import run as simulate_responses
from warmup_sampler import WarmupSampler

def test_likert_distribution():
    # Load parameters from quick_start.py ALL_CONFIG
    quick_start_path = Path(__file__).parent.parent / "quick_start.py"

    # Import ALL_CONFIG from quick_start.py
    import importlib.util
    spec = importlib.util.spec_from_file_location("quick_start", quick_start_path)
    quick_start = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quick_start)

    # Extract parameters from ALL_CONFIG
    all_config = quick_start.ALL_CONFIG

    config = {
        "seed": all_config.get("seed", 42),
        "population_mean": all_config.get("population_mean", 0.0),
        "population_std": all_config.get("population_std", 0.25),
        "individual_std_percent": all_config.get("individual_std_percent", 0.5),
        "individual_corr": all_config.get("individual_corr", 0.0),
        "likert_levels": all_config.get("likert_levels", 5),
        "likert_mode": all_config.get("likert_mode", "tanh"),
        "likert_sensitivity": all_config.get("likert_sensitivity", 2.0),
        "interaction_pairs": all_config.get("interaction_pairs", [(3, 4), (0, 1)]),
        "num_interactions": all_config.get("num_interactions", 0),
        "interaction_scale": all_config.get("interaction_scale", 0.25),
        "output_type": "likert",
        "design_space_csv": all_config.get("design_csv", str(
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
        )),
    }
    
    print("="*80)
    print("Testing Likert Distribution with quick_start.py ALL_CONFIG parameters")
    print("="*80)
    print(f"\nLoaded parameters from: {quick_start_path}")
    print(f"  seed: {config['seed']}")
    print(f"  population_mean: {config['population_mean']}")
    print(f"  population_std: {config['population_std']}")
    print(f"  individual_std_percent: {config['individual_std_percent']}")
    print(f"  likert_levels: {config['likert_levels']}")
    print(f"  likert_mode: {config['likert_mode']}")
    print(f"  likert_sensitivity: {config['likert_sensitivity']}")
    print()

    design_space_csv = config["design_space_csv"]
    sampler = WarmupSampler(design_space_csv)
    
    step1_dir = Path(__file__).parent / f"temp_test_step1_{int(time.time())}"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sampling plan
    adequacy, budget = sampler.evaluate_budget(
        n_subjects=5,
        trials_per_subject=30,
        skip_interaction=False,
    )
    
    sampler.generate_samples(
        budget=budget,
        output_dir=str(step1_dir),
        merge=False,
    )
    print(f"\nStep 1 DONE: Sampling plan generated")
    
    # Simulate responses
    print("Step 2: Simulating responses with loaded parameters...")
    print(f"  - population_mean: {config['population_mean']}")
    print(f"  - population_std: {config['population_std']}")
    print(f"  - individual_std_percent: {config['individual_std_percent']}")
    print(f"  - likert_sensitivity: {config['likert_sensitivity']}")
    print(f"  - likert_mode: {config['likert_mode']}")
    print()
    
    simulate_responses(input_dir=step1_dir, **config)
    result_dir = step1_dir / "result"
    
    # Analyze distribution
    print("\nStep 3: Analyzing distribution...")
    
    all_responses = []
    for subject_file in sorted(result_dir.glob("subject_*.csv")):
        df = pd.read_csv(subject_file)
        if "y" in df.columns:
            all_responses.extend(df["y"].values)
    
    all_responses = np.array(all_responses)
    
    mean = np.mean(all_responses)
    std = np.std(all_responses)
    skewness = stats.skew(all_responses)
    kurtosis = stats.kurtosis(all_responses)
    
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Total responses: {len(all_responses)}")
    print(f"\nDescriptive Statistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Skewness: {skewness:.4f}")
    print(f"  Kurtosis: {kurtosis:.4f}")
    
    print(f"\nLikert Scale Distribution:")
    for level in range(1, 6):
        count = np.sum(all_responses == level)
        if count > 0:
            proportion = count / len(all_responses)
            bar = "#" * int(proportion * 50)
            print(f"  Level {level}: {count:4d} ({proportion:6.2%}) {bar}")
        else:
            print(f"  Level {level}:    0 (  0.00%)")
    
    print(f"\nNormality Assessment:")
    if abs(skewness) < 0.5:
        verdict = "GOOD: Nearly symmetric"
    elif abs(skewness) < 1.0:
        verdict = "ACCEPTABLE: Moderately skewed"
    else:
        verdict = "POOR: Highly skewed"
    print(f"  Skewness: {skewness:.4f} -> {verdict}")
    
    # Shapiro-Wilk test
    _, p_value = stats.shapiro(all_responses[:min(5000, len(all_responses))])
    print(f"\nShapiro-Wilk Normality Test:")
    print(f"  P-value: {p_value:.6f}")
    if p_value > 0.05:
        print(f"  Result: PASS - Distribution is approximately normal")
    else:
        print(f"  Result: FAIL - Distribution deviates from normal")
    
    # Clean up
    shutil.rmtree(step1_dir)
    print(f"\nTemporary files cleaned up.")
    
if __name__ == "__main__":
    test_likert_distribution()
