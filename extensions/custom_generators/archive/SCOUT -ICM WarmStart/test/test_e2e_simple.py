"""
End-to-End Integration Test: Multi-Batch, Multi-Subject Phase-1 Warmup Study
ASCII-safe version for Windows PowerShell compatibility
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from study_coordinator import StudyCoordinator
from scout_warmup_generator import WarmupAEPsychGenerator


def create_test_design_df(n_rows=200, n_factors=4):
    """Create a test design DataFrame."""
    np.random.seed(42)
    data = {f"f{i}": np.random.rand(n_rows) for i in range(1, n_factors + 1)}
    df = pd.DataFrame(data)
    df["design_row_id"] = df.index
    return df


def test_multi_batch_workflow():
    """Test complete multi-batch workflow with state persistence."""
    print("\n" + "=" * 80)
    print("TEST: Multi-Batch Warmup Study E2E Workflow")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        study_id = "TEST_STUDY_001"
        runs_dir = os.path.join(tmpdir, "runs")

        design_df = create_test_design_df(n_rows=200, n_factors=4)
        n_subjects = 6
        total_budget = 300
        n_batches = 3

        # Initialize coordinator
        print("\n[STEP 1] Initialize Coordinator")
        coordinator = StudyCoordinator(
            design_df=design_df,
            n_subjects=n_subjects,
            total_budget=total_budget,
            n_batches=n_batches,
            seed=42,
        )
        coordinator.fit_initial_plan()
        print(
            f"OK - Coordinator initialized: {coordinator.d} factors, {len(coordinator.global_core1_candidates)} Core-1 candidates"
        )

        # Batch 1: Initialize
        print("\n[STEP 2] Batch 1: Initialize and Plan")
        run_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        batch_1_id = run_state["current_batch"]
        print(
            f"OK - Run state loaded: batch={batch_1_id}, base_seed={run_state['base_seed']}"
        )

        all_plans = []
        for subject_id in range(n_subjects):
            plan = coordinator.make_subject_plan(subject_id, batch_1_id, run_state)
            all_plans.append(plan)

        # Batch 1: Generate trials
        print("\n[STEP 3] Batch 1: Generate Trials")
        batch_trials = []
        batch_summaries = []

        for subject_plan in all_plans:
            gen = WarmupAEPsychGenerator(
                design_df=design_df,
                n_subjects=n_subjects,
                total_budget=total_budget,
                n_batches=n_batches,
                seed=42,
            )
            gen.apply_plan(subject_plan)
            gen.fit_planning()
            trials_df = gen.generate_trials()
            summary = gen.summarize()

            batch_trials.append(trials_df)
            batch_summaries.append(summary)

        batch1_all = pd.concat(batch_trials, ignore_index=True)
        print(f"OK - Generated {len(batch1_all)} trials for Batch 1")

        # Batch 1: Update state
        print("\n[STEP 4] Batch 1: Update State")
        run_state = coordinator.update_after_batch(
            run_state, batch_1_id, batch1_all, batch_summaries
        )
        coordinator.save_run_state(study_id, run_state, runs_dir=runs_dir)

        coverage_b1 = run_state["history"][-1]["coverage"]
        gini_b1 = run_state["history"][-1]["gini"]
        repeat_b1 = run_state["history"][-1]["core1_repeat_rate"]
        print(
            f"OK - Batch 1 metrics: coverage={coverage_b1:.3f}, gini={gini_b1:.3f}, repeat_rate={repeat_b1:.3f}"
        )
        print(
            f"OK - Core-1 IDs for next batch: {len(run_state['core1_last_batch_ids'])} points"
        )

        # Batch 2: Load and plan with Core-1 repeats
        print("\n[STEP 5] Batch 2: Load State with Core-1 Repeats")
        run_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        batch_2_id = run_state["current_batch"]

        all_plans = []
        for subject_id in range(n_subjects):
            plan = coordinator.make_subject_plan(subject_id, batch_2_id, run_state)
            all_plans.append(plan)
            repeat_count = len(plan["constraints"]["core1_repeat_indices"])
            if repeat_count > 0:
                print(
                    f"  Subject {subject_id}: {repeat_count} Core-1 repeats from prev batch"
                )

        # Batch 2: Generate with repeats
        print("\n[STEP 6] Batch 2: Generate Trials with Repeats")
        batch_trials = []
        batch_summaries = []

        for subject_plan in all_plans:
            gen = WarmupAEPsychGenerator(
                design_df=design_df,
                n_subjects=n_subjects,
                total_budget=total_budget,
                n_batches=n_batches,
                seed=42,
            )
            gen.apply_plan(subject_plan)
            gen.fit_planning()
            trials_df = gen.generate_trials()
            summary = gen.summarize()

            batch_trials.append(trials_df)
            batch_summaries.append(summary)

            # Verify repeats
            if "is_core1_repeat" in trials_df.columns:
                core1_trials = trials_df[trials_df["block_type"] == "core1"]
                marked = len(core1_trials[core1_trials["is_core1_repeat"] == True])
                total = len(core1_trials)
                print(
                    f"  Subject {subject_plan['subject_id']}: {marked}/{total} Core-1 marked as repeats"
                )

        batch2_all = pd.concat(batch_trials, ignore_index=True)
        print(f"OK - Generated {len(batch2_all)} trials for Batch 2")

        # Batch 2: Update state
        print("\n[STEP 7] Batch 2: Update State")
        run_state = coordinator.update_after_batch(
            run_state, batch_2_id, batch2_all, batch_summaries
        )
        coordinator.save_run_state(study_id, run_state, runs_dir=runs_dir)

        coverage_b2 = run_state["history"][-1]["coverage"]
        gini_b2 = run_state["history"][-1]["gini"]
        repeat_b2 = run_state["history"][-1]["core1_repeat_rate"]
        print(
            f"OK - Batch 2 metrics: coverage={coverage_b2:.3f}, gini={gini_b2:.3f}, repeat_rate={repeat_b2:.3f}"
        )

        # Batch 3: Final batch
        print("\n[STEP 8] Batch 3: Final Batch")
        run_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        batch_3_id = run_state["current_batch"]

        all_plans = []
        for subject_id in range(n_subjects):
            plan = coordinator.make_subject_plan(subject_id, batch_3_id, run_state)
            all_plans.append(plan)

        batch_trials = []
        batch_summaries = []
        for subject_plan in all_plans:
            gen = WarmupAEPsychGenerator(
                design_df=design_df,
                n_subjects=n_subjects,
                total_budget=total_budget,
                n_batches=n_batches,
                seed=42,
            )
            gen.apply_plan(subject_plan)
            gen.fit_planning()
            trials_df = gen.generate_trials()
            summary = gen.summarize()

            batch_trials.append(trials_df)
            batch_summaries.append(summary)

        batch3_all = pd.concat(batch_trials, ignore_index=True)

        # Final update
        run_state = coordinator.update_after_batch(
            run_state, batch_3_id, batch3_all, batch_summaries
        )
        coordinator.save_run_state(study_id, run_state, runs_dir=runs_dir)

        print(f"OK - Batch 3 completed. Study status: {run_state.get('status', '?')}")

        # Validation
        print("\n[STEP 9] Final Validation")
        all_trials = pd.concat([batch1_all, batch2_all, batch3_all], ignore_index=True)
        validation = coordinator.validate_global_constraints(all_trials)

        print(f"OK - Global constraints validation:")
        print(f"  - Core-1 repeat ratio: {validation['core1_repeat_ratio']:.2%}")
        print(f"  - Coverage rate: {validation['coverage_rate']:.3f}")
        print(f"  - Gini coefficient: {validation['gini_coefficient']:.3f}")

        # History summary
        print("\n[STEP 10] Study History")
        for entry in run_state["history"]:
            print(
                f"  Batch {entry['batch_id']}: coverage={entry['coverage']:.3f}, gini={entry['gini']:.3f}"
            )

        # Final state verification
        print("\n[STEP 11] State Persistence Verification")
        final_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        assert final_state["current_batch"] == n_batches + 1
        assert final_state["status"] == "completed"
        assert len(final_state["history"]) == n_batches
        print(
            f"OK - Final state verified: batch={final_state['current_batch']}, status={final_state['status']}"
        )

        print("\n" + "=" * 80)
        print("SUCCESS: All tests passed")
        print("=" * 80)
        return True


if __name__ == "__main__":
    try:
        success = test_multi_batch_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
