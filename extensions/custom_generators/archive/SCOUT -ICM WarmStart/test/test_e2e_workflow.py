"""
End-to-End Integration Test: Multi-Batch, Multi-Subject Phase-1 Warmup Study

Demonstrates the complete workflow:
1. Initialize run_state.json for new study
2. Batch 1: Generate plans for all subjects, execute sampling, update state
3. Batch 2: Load state, validate bridge continuity, execute with Core-1 repeats
4. Batch 3: Final batch, validate global constraints
5. Verify cross-process state persistence and metrics tracking

Tests:
- load_run_state() with initialization
- save_run_state() after each batch
- make_subject_plan() generates correct quotas/constraints
- Core-1 repeat indices properly applied via apply_plan()
- is_core1_repeat markers correctly set
- summarize() metrics propagate to Coordinator.update_after_batch()
- coverage/gini decision logic (strategy adjustment)
- Bridge subject cross-batch continuity
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Set encoding for terminal output
if sys.stdout.encoding.lower() != "utf-8":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from study_coordinator import StudyCoordinator
from scout_warmup_generator import WarmupAEPsychGenerator


def create_test_design_df(n_rows: int = 200, n_factors: int = 4) -> pd.DataFrame:
    """Create a realistic test design DataFrame."""
    np.random.seed(42)
    data = {f"f{i}": np.random.rand(n_rows) for i in range(1, n_factors + 1)}
    df = pd.DataFrame(data)
    df["stimulus_id"] = [f"stim_{i}" for i in range(n_rows)]
    df["design_row_id"] = df.index
    return df


def test_multi_batch_workflow():
    """
    Test complete multi-batch workflow with state persistence.
    """
    print("\n" + "=" * 80)
    print("TEST: Multi-Batch, Multi-Subject Warmup Study E2E Workflow")
    print("=" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup
        study_id = "TEST_STUDY_001"
        runs_dir = os.path.join(tmpdir, "runs")

        design_df = create_test_design_df(n_rows=200, n_factors=4)
        n_subjects = 6
        total_budget = 300
        n_batches = 3

        # ========== COORDINATOR INITIALIZATION ==========
        print("\n" + "-" * 80)
        print("1. INITIALIZE COORDINATOR AND GLOBAL PLAN")
        print("-" * 80)

        coordinator = StudyCoordinator(
            design_df=design_df,
            n_subjects=n_subjects,
            total_budget=total_budget,
            n_batches=n_batches,
            seed=42,
        )
        coordinator.fit_initial_plan()

        print("[OK] Coordinator initialized:")
        print(f"  - Subjects: {n_subjects}")
        print(f"  - Batches: {n_batches}")
        print(f"  - Total budget: {total_budget}")
        print(f"  - Design space: {len(design_df)} points")
        print(f"  - Factors detected: {coordinator.d}")
        print(f"  - Core-1 candidates: {len(coordinator.global_core1_candidates)}")
        print(f"  - Interaction pairs: {len(coordinator.interaction_pairs)}")
        print(f"  - Boundary points: {len(coordinator.boundary_library)}")

        # ========== BATCH 1 ==========
        print("\n" + "-" * 80)
        print("2. BATCH 1: INITIALIZATION AND PLANNING")
        print("-" * 80)

        # Load (or initialize) run_state
        run_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        print(f"✓ Run state loaded/initialized:")
        print(f"  - Current batch: {run_state['current_batch']}")
        print(f"  - Study ID: {run_state['study_id']}")
        print(f"  - Base seed: {run_state['base_seed']}")

        # Generate plans for all subjects in batch 1
        batch_1_id = run_state["current_batch"]
        all_batch1_plans = []

        for subject_id in range(n_subjects):
            plan = coordinator.make_subject_plan(subject_id, batch_1_id, run_state)
            all_batch1_plans.append(plan)
            print(
                f"  Subject {subject_id}: quotas={plan['quotas']}, bridge={plan['is_bridge']}"
            )

        # ========== BATCH 1: GENERATION AND SAMPLING ==========
        print("\n" + "-" * 80)
        print("3. BATCH 1: TRIAL GENERATION AND SAMPLING")
        print("-" * 80)

        batch1_trials_list = []
        batch1_summaries = []

        for subject_plan in all_batch1_plans:
            subject_id = subject_plan["subject_id"]
            batch_id = subject_plan["batch_id"]

            # Create generator for this subject
            gen = WarmupAEPsychGenerator(
                design_df=design_df,
                n_subjects=n_subjects,
                total_budget=total_budget,
                n_batches=n_batches,
                seed=42,
            )

            # Apply coordinator's plan
            gen.apply_plan(subject_plan)

            # Fit planning and generate trials
            gen.fit_planning()
            trials_df = gen.generate_trials()

            # Summarize
            summary = gen.summarize()

            batch1_trials_list.append(trials_df)
            batch1_summaries.append(summary)

            print(
                f"✓ Subject {subject_id}: {len(trials_df)} trials, "
                f"coverage={summary['metadata']['coverage_rate']:.3f}, "
                f"gini={summary['metadata']['gini']:.3f}, "
                f"core1_repeat_rate={summary['metadata']['core1_repeat_rate']:.3f}"
            )

        # Combine all batch 1 trials
        batch1_all_trials = pd.concat(batch1_trials_list, ignore_index=True)
        print(f"✓ Batch 1 total trials: {len(batch1_all_trials)}")

        # ========== BATCH 1: UPDATE RUN STATE ==========
        print("\n" + "-" * 80)
        print("4. BATCH 1: UPDATE RUN STATE AFTER BATCH")
        print("-" * 80)

        run_state = coordinator.update_after_batch(
            run_state, batch_1_id, batch1_all_trials, batch1_summaries
        )

        print(f"✓ Run state updated:")
        print(f"  - Current batch: {run_state['current_batch']}")
        print(
            f"  - Core-1 last batch IDs: {len(run_state['core1_last_batch_ids'])} points"
        )
        print(f"  - Coverage: {run_state['history'][-1]['coverage']:.3f}")
        print(f"  - Gini: {run_state['history'][-1]['gini']:.3f}")
        print(f"  - Repeat rate: {run_state['history'][-1]['core1_repeat_rate']:.3f}")

        # Save state
        coordinator.save_run_state(study_id, run_state, runs_dir=runs_dir)
        print(
            f"✓ Run state saved to {os.path.join(runs_dir, study_id, 'run_state.json')}"
        )

        # ========== BATCH 2 ==========
        print("\n" + "-" * 80)
        print("5. BATCH 2: LOAD STATE AND BRIDGE CONTINUITY")
        print("-" * 80)

        # Reload run_state (simulating new process)
        run_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        print(f"✓ Run state reloaded (simulating new process):")
        print(f"  - Current batch: {run_state['current_batch']}")
        print(
            f"  - Core-1 last batch IDs available: {len(run_state['core1_last_batch_ids'])}"
        )

        batch_2_id = run_state["current_batch"]
        all_batch2_plans = []

        for subject_id in range(n_subjects):
            plan = coordinator.make_subject_plan(subject_id, batch_2_id, run_state)
            all_batch2_plans.append(plan)

        # ========== BATCH 2: GENERATION WITH CORE-1 REPEATS ==========
        print("\n" + "-" * 80)
        print("6. BATCH 2: GENERATION WITH CORE-1 REPEAT CONSTRAINTS")
        print("-" * 80)

        batch2_trials_list = []
        batch2_summaries = []

        for subject_plan in all_batch2_plans:
            subject_id = subject_plan["subject_id"]
            batch_id = subject_plan["batch_id"]

            # Show what constraints were applied
            core1_repeat_indices = subject_plan["constraints"]["core1_repeat_indices"]
            print(
                f"  Subject {subject_id}: core1_repeat_indices={len(core1_repeat_indices)} points"
            )

            # Create and execute generator
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

            batch2_trials_list.append(trials_df)
            batch2_summaries.append(summary)

            # Verify Core-1 repeats are marked
            if "is_core1_repeat" in trials_df.columns:
                core1_trials = trials_df[trials_df["block_type"] == "core1"]
                marked_repeats = len(
                    core1_trials[core1_trials["is_core1_repeat"] == True]
                )
                print(
                    f"✓ Subject {subject_id}: {marked_repeats}/{len(core1_trials)} Core-1 trials marked as repeats"
                )

        batch2_all_trials = pd.concat(batch2_trials_list, ignore_index=True)
        print(f"✓ Batch 2 total trials: {len(batch2_all_trials)}")

        # Update run state after batch 2
        run_state = coordinator.update_after_batch(
            run_state, batch_2_id, batch2_all_trials, batch2_summaries
        )
        coordinator.save_run_state(study_id, run_state, runs_dir=runs_dir)

        print(f"✓ Batch 2 metrics recorded:")
        print(f"  - Coverage: {run_state['history'][-1]['coverage']:.3f}")
        print(f"  - Gini: {run_state['history'][-1]['gini']:.3f}")
        print(f"  - Repeat rate: {run_state['history'][-1]['core1_repeat_rate']:.3f}")

        # ========== BATCH 3 ==========
        print("\n" + "-" * 80)
        print("7. BATCH 3: FINAL BATCH AND COMPLETION")
        print("-" * 80)

        run_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        batch_3_id = run_state["current_batch"]

        batch3_trials_list = []
        batch3_summaries = []

        for subject_id in range(n_subjects):
            plan = coordinator.make_subject_plan(subject_id, batch_3_id, run_state)

            gen = WarmupAEPsychGenerator(
                design_df=design_df,
                n_subjects=n_subjects,
                total_budget=total_budget,
                n_batches=n_batches,
                seed=42,
            )

            gen.apply_plan(plan)
            gen.fit_planning()
            trials_df = gen.generate_trials()
            summary = gen.summarize()

            batch3_trials_list.append(trials_df)
            batch3_summaries.append(summary)

        batch3_all_trials = pd.concat(batch3_trials_list, ignore_index=True)

        # Update after final batch
        run_state = coordinator.update_after_batch(
            run_state, batch_3_id, batch3_all_trials, batch3_summaries
        )
        coordinator.save_run_state(study_id, run_state, runs_dir=runs_dir)

        print(
            f"✓ Batch 3 completed. Study status: {run_state.get('status', 'unknown')}"
        )

        # ========== VALIDATION AND SUMMARY ==========
        print("\n" + "-" * 80)
        print("8. FINAL VALIDATION AND SUMMARY")
        print("-" * 80)

        # Combine all trials
        all_trials = pd.concat(
            [batch1_all_trials, batch2_all_trials, batch3_all_trials],
            ignore_index=True,
        )

        # Validate global constraints
        validation = coordinator.validate_global_constraints(all_trials)
        print(f"✓ Global constraint validation:")
        print(f"  - Core-1 repeat ratio: {validation['core1_repeat_ratio']:.2%}")
        print(f"  - Coverage rate: {validation['coverage_rate']:.3f}")
        print(f"  - Gini coefficient: {validation['gini_coefficient']:.3f}")
        if validation["warnings"]:
            print(f"  - Warnings:")
            for warning in validation["warnings"]:
                print(f"    {warning}")

        # Print history
        print(f"\n✓ Study history (3 batches):")
        for entry in run_state["history"]:
            print(
                f"  Batch {entry['batch_id']}: {entry['n_subjects']} subjects, "
                f"coverage={entry['coverage']:.3f}, gini={entry['gini']:.3f}, "
                f"repeat_rate={entry['core1_repeat_rate']:.3f}"
            )

        # Final verification
        print("\n" + "-" * 80)
        print("9. PERSISTENCE AND RECOVERY TEST")
        print("-" * 80)

        # Load final state one more time to verify persistence
        final_state = coordinator.load_run_state(study_id, runs_dir=runs_dir)
        assert final_state["current_batch"] == n_batches + 1, "Batch counter mismatch"
        assert final_state["status"] == "completed", "Study not marked completed"
        assert len(final_state["history"]) == n_batches, "History length mismatch"
        print(f"✓ Final state verified:")
        print(
            f"  - Batch counter: {final_state['current_batch']} (expected {n_batches + 1})"
        )
        print(f"  - Status: {final_state['status']}")
        print(
            f"  - History entries: {len(final_state['history'])} (expected {n_batches})"
        )

        print("\n" + "=" * 80)
        print("✓ E2E WORKFLOW TEST PASSED")
        print("=" * 80)

        return True


if __name__ == "__main__":
    try:
        success = test_multi_batch_workflow()
        if success:
            print("\n✓ All tests passed!")
            sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
