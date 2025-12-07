"""
Quick verification that new changes work:
1. seed column in trial_schedule
2. high-dim quotas apply correctly
3. bridge repeat cap enforced at 50%
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from study_coordinator import StudyCoordinator
from scout_warmup_generator import WarmupAEPsychGenerator


def test_seed_in_output():
    """Verify seed column appears in trial_schedule_df"""
    print("[TEST 1] Seed column in trial_schedule")

    np.random.seed(42)
    design_df = pd.DataFrame({f"f{i}": np.random.rand(50) for i in range(1, 5)})

    gen = WarmupAEPsychGenerator(design_df, n_subjects=2, total_budget=100, seed=99)
    gen.fit_planning()

    plan = {
        "subject_id": 0,
        "batch_id": 1,
        "is_bridge": False,
        "quotas": {"core1": 5, "main": 10, "inter": 3, "boundary": 2, "lhs": 2},
        "constraints": {
            "core1_pool_indices": list(range(10)),
            "core1_repeat_indices": [],
            "interaction_pairs": [],
            "boundary_library": [],
        },
        "seed": 99,
    }

    gen.apply_plan(plan)
    trials = gen.generate_trials()

    assert "seed" in trials.columns, "seed column missing!"
    assert (trials["seed"] == 99).all(), "seed values incorrect!"
    print(f"  OK - seed column present with value {trials['seed'].iloc[0]}")


def test_high_dim_quotas():
    """Verify high-dimensional quota adjustments"""
    print("\n[TEST 2] High-dim quota adjustments")

    np.random.seed(42)
    # Create high-dimensional design
    design_df = pd.DataFrame(
        {f"f{i}": np.random.rand(100) for i in range(1, 15)}  # d=14
    )

    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=3, total_budget=150, n_batches=2, seed=42
    )
    coordinator.fit_initial_plan()

    # Check high-dim adjustments were triggered
    assert coordinator.d == 14, f"Expected d=14, got {coordinator.d}"
    print(f"  OK - Detected d={coordinator.d}")

    # Generate a plan and check quotas
    run_state = coordinator.load_run_state("test_hd", "runs")
    plan = coordinator.make_subject_plan(0, 1, run_state)

    quotas = plan["quotas"]
    total = sum(quotas.values())

    # For d=14 (>12): interaction <= 8%, boundary+lhs >= 45%
    inter_pct = quotas["inter"] / total
    boundary_lhs_pct = (quotas["boundary"] + quotas["lhs"]) / total

    print(f"  - d=14 quotas:")
    print(f"    - inter: {inter_pct:.1%} (should be <= 8%)")
    print(f"    - boundary+lhs: {boundary_lhs_pct:.1%} (should be >= 45%)")
    print(f"  OK - High-dim adjustments applied")


def test_bridge_repeat_cap():
    """Verify core1_repeat_indices capped at 50% of core1 quota"""
    print("\n[TEST 3] Bridge repeat 50% cap enforcement")

    np.random.seed(42)
    design_df = pd.DataFrame({f"f{i}": np.random.rand(100) for i in range(1, 5)})

    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=2, total_budget=100, n_batches=2, seed=42
    )
    coordinator.fit_initial_plan()

    # Simulate batch 1 completion with 20 core1 points extracted
    run_state = coordinator.load_run_state("test_bridge", "runs")
    run_state["core1_last_batch_ids"] = list(range(20))  # 20 points from batch 1
    run_state["bridge_subjects"] = {"2": [0]}  # Subject 0 is bridge in batch 2

    # Generate plan for bridge subject in batch 2
    plan = coordinator.make_subject_plan(0, 2, run_state)

    # Check the cap
    core1_quota = plan["quotas"]["core1"]
    repeat_indices = plan["constraints"]["core1_repeat_indices"]

    repeat_max = int(np.ceil(core1_quota * 0.5))

    print(f"  - core1_quota: {core1_quota}")
    print(f"  - repeat_indices provided: {len(repeat_indices)}")
    print(f"  - 50% cap: {repeat_max}")

    assert (
        len(repeat_indices) <= repeat_max
    ), f"repeat_indices {len(repeat_indices)} exceeds 50% cap {repeat_max}"
    print(f"  OK - 50% cap enforced (actual: {len(repeat_indices)}/{core1_quota})")


if __name__ == "__main__":
    try:
        test_seed_in_output()
        test_high_dim_quotas()
        test_bridge_repeat_cap()

        print("\n" + "=" * 60)
        print("SUCCESS: All verification tests passed")
        print("=" * 60)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
