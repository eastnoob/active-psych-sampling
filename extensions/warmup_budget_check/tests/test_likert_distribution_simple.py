#!/usr/bin/env python
"""
Test script for Likert distribution analysis
Tests the simulated subject responses using current quick_start.py ALL_CONFIG parameters.
"""

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
    """Test the Likert distribution with current quick_start.py config"""
    
    # Current quick_start.py ALL_CONFIG parameters
    config = {
        "seed": 42,
        "population_mean": 0.0,
        "population_std": 0.25,
        "individual_std_percent": 0.5,
        "individual_corr": 0.0,
        "likert_levels": 5,
        "likert_mode": "tanh",
        "likert_sensitivity": 0.3,
        "interaction_pairs": [(3, 4), (0, 1)],
        "num_interactions": 0,
        "interaction_scale": 0.25,
        "output_type": "likert",
        "design_space_csv": str(
            Path(__file__).parent.parent.parent.parent
            / "data"
            / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
        ),
    }
    
    print("="*80)
    print("Step 1: Generate sampling plan")
    print("="*80)
    
    design_space_csv = config["design_space_csv"]
    sampler = WarmupSampler(design_space_csv)
    
    step1_dir = Path(__file__).parent / f"temp_test_step1_{int(time.time())}"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    # Use evaluate_budget like quick_start.py does
    adequacy, budget = sampler.evaluate_budget(
        n_subjects=5,
        trials_per_subject=30,
        skip_interaction=False,
    )
    print(f"Budget adequacy: {adequacy}")
    print(f"Budget: {budget}")
    
    sampler.generate_samples(
        budget=budget,
        output_dir=str(step1_dir),
        merge=False,
    )
    print(f"Done: {step1_dir}")
    
    print("\n" + "="*80)
    print("Step 2: Simulate responses")
    print("="*80)
    print(f"Config parameters:")
    print(f"  - population_mean: {config['population_mean']}")
    print(f"  - population_std: {config['population_std']}")
    print(f"  - likert_sensitivity: {config['likert_sensitivity']}")
    print(f"  - likert_mode: {config['likert_mode']}")
    
    simulate_responses(input_dir=step1_dir, **config)
    result_dir = step1_dir / "result"
    
    print("\n" + "="*80)
    print("Step 3: Analyze distribution")
    print("="*80)
    
    all_responses = []
    for subject_file in sorted(result_dir.glob("subject_*.csv")):
        df = pd.read_csv(subject_file)
        if "y" in df.columns:
            all_responses.extend(df["y"].values)
    
    all_responses = np.array(all_responses)
    print(f"Total responses: {len(all_responses)}")
    
    mean = np.mean(all_responses)
    std = np.std(all_responses)
    skewness = stats.skew(all_responses)
    kurtosis = stats.kurtosis(all_responses)
    
    print(f"\nStatistics:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Skewness: {skewness:.4f}")
    print(f"  Kurtosis: {kurtosis:.4f}")
    
    print(f"\nLikert scale distribution:")
    for level in range(1, 6):
        count = np.sum(all_responses == level)
        if count > 0:
            proportion = count / len(all_responses)
            bar = "█" * int(proportion * 50)
            print(f"  Level {level}: {count:4d} ({proportion:6.2%}) {bar}")
    
    print(f"\nNormality assessment:")
    if abs(skewness) < 0.5:
        print(f"  ✓ EXCELLENT: Skewness {skewness:.4f} (nearly symmetric)")
    elif abs(skewness) < 1.0:
        print(f"  ✓ GOOD: Skewness {skewness:.4f} (moderately skewed)")
    else:
        print(f"  ✗ POOR: Skewness {skewness:.4f} (highly skewed)")
    
    # Shapiro-Wilk normality test
    _, p_value = stats.shapiro(all_responses[:min(5000, len(all_responses))])
    print(f"\nShapiro-Wilk test (p-value): {p_value:.6f}")
    if p_value > 0.05:
        print(f"  ✓ Distribution is approximately normal (p > 0.05)")
    else:
        print(f"  ✗ Distribution deviates from normal (p < 0.05)")
    
    # Clean up
    print(f"\nCleaning up temporary files...")
    shutil.rmtree(step1_dir)
    
if __name__ == "__main__":
    test_likert_distribution()
