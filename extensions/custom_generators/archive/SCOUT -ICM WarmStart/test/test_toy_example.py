"""
Test script for SCOUT Warm-up Generator with toy examples
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add the parent directory to the path so we can import the generator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scout_warmup_generator import WarmupAEPsychGenerator


def create_toy_design(d=6, levels_per_factor=3, seed=42):
    """
    Create a toy design DataFrame for testing.

    Args:
        d: Number of factors
        levels_per_factor: Number of levels per factor
        seed: Random seed

    Returns:
        A DataFrame with toy design
    """
    np.random.seed(seed)

    # Create factors with discrete levels
    factor_data = {}
    for i in range(d):
        factor_name = f"f{i+1}"
        # Create discrete levels
        levels = np.linspace(0, 1, levels_per_factor)
        # Repeat levels to fill the dataset
        factor_data[factor_name] = np.random.choice(levels, 100)

    # Add some metadata
    factor_data["metadata"] = ["info"] * 100

    return pd.DataFrame(factor_data)


def test_small_toy_example():
    """Test with a small toy example (d=6)."""
    print("Testing small toy example (d=6)...")

    # Create toy design
    design_df = create_toy_design(d=6, levels_per_factor=3, seed=42)
    print(f"  Created design with {len(design_df)} candidates and {6} factors")

    # Create generator
    gen = WarmupAEPsychGenerator(
        design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42
    )

    # Test fit_planning
    gen.fit_planning()
    print("  fit_planning() completed successfully")

    # Test generate_trials
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trials = gen.generate_trials()
    print(f"  generate_trials() produced {len(trials)} trials")

    # Test summarize
    plan = gen.summarize()
    print(f"  summarize() produced plan with keys: {list(plan.keys())}")

    # Print some validation information
    print(f"  Core-1 points: {plan['core1_info']['n_points']}")
    print(f"  Budget allocation: {plan['budget_allocation']}")

    print("Small toy example test completed.\n")


def test_high_dimensionality_degradation():
    """Test dimensionality degradation for d=11 and d=13."""
    print("Testing dimensionality degradation...")

    # Test with d=11
    print("  Testing d=11:")
    design_df_11 = create_toy_design(d=11, levels_per_factor=3, seed=42)
    gen_11 = WarmupAEPsychGenerator(
        design_df_11, n_subjects=10, total_budget=350, n_batches=3, seed=42
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gen_11.fit_planning()

        # Check if warnings were emitted
        warning_messages = [
            warning.message.args[0] if warning.message.args else str(warning.message)
            for warning in w
        ]

        d10_warning_found = any("d>10" in msg for msg in warning_messages)
        if d10_warning_found:
            print("    d>10 warning correctly emitted")
        else:
            print("    WARNING: d>10 warning not found")

    print(f"    Core-1 points: {gen_11.n_core1_points}")
    print(f"    Interaction pairs: {len(gen_11.interaction_plan['pairs'])}")

    # Test with d=13
    print("  Testing d=13:")
    design_df_13 = create_toy_design(d=13, levels_per_factor=3, seed=42)
    gen_13 = WarmupAEPsychGenerator(
        design_df_13, n_subjects=10, total_budget=350, n_batches=3, seed=42
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gen_13.fit_planning()

        # Check if warnings were emitted
        warning_messages = [
            warning.message.args[0] if warning.message.args else str(warning.message)
            for warning in w
        ]

        d12_warning_found = any("d>12" in msg for msg in warning_messages)
        if d12_warning_found:
            print("    d>12 warning correctly emitted")
        else:
            print("    WARNING: d>12 warning not found")

    print(f"    Core-1 points: {gen_13.n_core1_points}")
    print(f"    Interaction pairs: {len(gen_13.interaction_plan['pairs'])}")
    print(f"    Budget allocation: {gen_13.budget_split}")

    print("Dimensionality degradation test completed.\n")


def test_validation_and_reporting():
    """Test validation and reporting features."""
    print("Testing validation and reporting...")

    # Create toy design
    design_df = create_toy_design(d=6, levels_per_factor=3, seed=42)

    # Create generator
    gen = WarmupAEPsychGenerator(
        design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42
    )

    # Run full pipeline
    gen.fit_planning()
    trials = gen.generate_trials()

    # Run validation hooks explicitly
    gen._run_validation_hooks(trials)

    # Check coverage report
    gen._emit_coverage_report()

    print("Validation and reporting test completed.\n")


def main():
    """Run all toy example tests."""
    print("Running SCOUT Warm-up Generator Toy Example Tests\n")

    try:
        test_small_toy_example()
        test_high_dimensionality_degradation()
        test_validation_and_reporting()

        print("All toy example tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
