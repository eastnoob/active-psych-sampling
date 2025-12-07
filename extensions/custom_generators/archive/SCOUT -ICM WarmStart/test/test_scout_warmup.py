"""
Test script for SCOUT Warm-up Generator
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings

# Add the parent directory to the path so we can import the generator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scout_warmup_generator import WarmupAEPsychGenerator


def test_basic_functionality():
    """Test basic functionality of the WarmupAEPsychGenerator."""
    print("Testing basic functionality...")

    # Create a sample design DataFrame
    np.random.seed(42)
    sample_data = {
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100),
        "metadata": ["info"] * 100,
    }
    design_df = pd.DataFrame(sample_data)

    # Create generator
    gen = WarmupAEPsychGenerator(
        design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42
    )

    # Test fit_planning
    gen.fit_planning()
    print("  fit_planning() completed successfully")

    # Test generate_trials
    trials = gen.generate_trials()
    print(f"  generate_trials() produced {len(trials)} trials")

    # Check that we have the expected columns
    expected_columns = [
        "subject_id",
        "batch_id",
        "is_bridge",
        "block_type",
        "is_core1",
        "is_core2",
        "is_individual",
        "is_boundary",
        "is_lhs",
        "interaction_pair_id",
        "design_row_id",
        "f1",
        "f2",
        "f3",
    ]
    missing_columns = [col for col in expected_columns if col not in trials.columns]
    if missing_columns:
        print(f"  WARNING: Missing columns: {missing_columns}")
    else:
        print("  All expected columns present")

    # Test summarize
    plan = gen.summarize()
    print(f"  summarize() produced plan with keys: {list(plan.keys())}")

    print("Basic functionality test completed.\n")


def test_high_dimensionality_warnings():
    """Test that warnings are emitted for high dimensionality."""
    print("Testing high dimensionality warnings...")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create a high-dimensional design DataFrame
        np.random.seed(42)
        sample_data = {
            f"f{i}": np.random.rand(50) for i in range(1, 14)
        }  # 13 dimensions
        sample_data["metadata"] = ["info"] * 50
        design_df = pd.DataFrame(sample_data)

        # Create generator
        gen = WarmupAEPsychGenerator(
            design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42
        )
        gen.fit_planning()

        # Check if warnings were emitted
        warning_messages = [
            warning.message.args[0] if warning.message.args else str(warning.message)
            for warning in w
        ]

        d10_warning_found = any("d>10" in msg for msg in warning_messages)
        d12_warning_found = any("d>12" in msg for msg in warning_messages)

        if d10_warning_found:
            print("  d>10 warning correctly emitted")
        else:
            print("  WARNING: d>10 warning not found")

        if d12_warning_found:
            print("  d>12 warning correctly emitted")
        else:
            print("  WARNING: d>12 warning not found")

    print("High dimensionality warnings test completed.\n")


def test_validation_hooks():
    """Test that validation hooks work correctly."""
    print("Testing validation hooks...")

    # Capture warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create a sample design DataFrame
        np.random.seed(42)
        sample_data = {
            "f1": np.random.rand(100),
            "f2": np.random.rand(100),
            "f3": np.random.rand(100),
            "metadata": ["info"] * 100,
        }
        design_df = pd.DataFrame(sample_data)

        # Create generator
        gen = WarmupAEPsychGenerator(
            design_df, n_subjects=10, total_budget=350, n_batches=3, seed=42
        )
        gen.fit_planning()
        trials = gen.generate_trials()

        # Check if validation warnings were emitted
        warning_messages = [
            warning.message.args[0] if warning.message.args else str(warning.message)
            for warning in w
        ]

        print(f"  Validation produced {len(warning_messages)} warnings:")
        for msg in warning_messages:
            print(f"    - {msg}")

    print("Validation hooks test completed.\n")


def test_utility_functions():
    """Test utility functions."""
    print("Testing utility functions...")

    # Create a sample design DataFrame
    np.random.seed(42)
    sample_data = {
        "f1": np.random.rand(100),
        "f2": np.random.rand(100),
        "f3": np.random.rand(100),
        "metadata": ["info"] * 100,
    }
    design_df = pd.DataFrame(sample_data)

    # Test detect_factor_types
    from scout_warmup_generator import detect_factor_types

    factor_types = detect_factor_types(design_df)
    print(f"  detect_factor_types: {factor_types}")

    # Test get_levels_or_bins
    from scout_warmup_generator import get_levels_or_bins

    levels_or_bins = get_levels_or_bins(design_df)
    print(f"  get_levels_or_bins produced {len(levels_or_bins)} factors")

    # Test compute_marginal_min_counts
    from scout_warmup_generator import compute_marginal_min_counts

    min_counts = compute_marginal_min_counts(design_df)
    print(f"  compute_marginal_min_counts: {min_counts}")

    # Test build_interaction_pairs
    from scout_warmup_generator import build_interaction_pairs

    interaction_pairs = build_interaction_pairs(["f1", "f2", "f3"])
    print(f"  build_interaction_pairs produced {len(interaction_pairs)} pairs")

    print("Utility functions test completed.\n")


def main():
    """Run all tests."""
    print("Running SCOUT Warm-up Generator Tests\n")

    try:
        test_basic_functionality()
        test_high_dimensionality_warnings()
        test_validation_hooks()
        test_utility_functions()

        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
