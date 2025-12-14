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

# Add tools directory to path
tools_path = Path(__file__).parent.parent.parent.parent / "tools"
sys.path.insert(0, str(tools_path))

from subject_simulator_v2.adapters.warmup_adapter import run as simulate_responses

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
    }
    
    design_space_csv = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    )
    
    print("="*80)
    print("Step 1: Generate sampling plan")
    print("="*80)
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "extensions" / "warmup_budget_check" / "core"))
    from warmup_sampler import WarmupSampler
    
    sampler = WarmupSampler(str(design_space_csv))
    step1_dir = Path(__file__).parent / f"temp_test_step1_{int(time.time())}"
    step1_dir.mkdir(parents=True, exist_ok=True)
    
    budget = {
        'main_effects': 120,
        'interactions': 30,
        'boundary_points': 10,
        'global_lhs': 0
    }
    
    sampler.generate_samples(
        budget=budget,
        output_dir=str(step1_dir),
        merge=False,
    )
    print(f"Done: {step1_dir}")
    
    print("\n" + "="*80)
    print("Step 2: Simulate responses")
    print("="*80)
    print(f"Config: {json.dumps(config, indent=2)}")
    
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
    print(f"Total: {len(all_responses)} responses")
    
    mean = np.mean(all_responses)
    std = np.std(all_responses)
    skewness = stats.skew(all_responses)
    
    print(f"\nMean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"Skewness: {skewness:.4f}")
    
    print("\nDistribution:")
    for level in sorted(np.unique(all_responses).astype(int)):
        count = np.sum(all_responses == level)
        proportion = count / len(all_responses)
        bar = "█" * int(proportion * 40)
        print(f"Level {level}: {count:4d} ({proportion:6.2%}) {bar}")
    
    print("\nAssessment:")
    if abs(skewness) < 0.5:
        print("✓ GOOD: Near-normal distribution")
    elif abs(skewness) < 1.0:
        print("⚠ MODERATE: Some skewness, acceptable")
    else:
        print(f"✗ POOR: Too skewed ({skewness:.4f})")

if __name__ == "__main__":
    test_likert_distribution()
