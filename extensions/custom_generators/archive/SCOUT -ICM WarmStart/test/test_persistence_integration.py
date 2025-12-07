"""
Integration tests for Coordinator and Generator persistence layer.

Tests:
1. save_global_plan() and load integrity
2. save_run_state() and load_run_state() checkpoint round-trip
3. export_subject_plan() and apply_plan() injection
4. validate_global_constraints() with mock trials
5. generate_trials(save_to=path) direct file saving
6. export_metadata() audit trail export
7. End-to-end workflow: plan -> save -> load -> execute -> validate
"""

import os
import sys
import json
import pickle
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from study_coordinator import StudyCoordinator
from scout_warmup_generator import WarmupAEPsychGenerator


def create_test_design_df(n_rows: int = 100, n_factors: int = 5) -> pd.DataFrame:
    """Create a simple test design DataFrame."""
    np.random.seed(42)
    data = {f"f{i}": np.random.rand(n_rows) for i in range(n_factors)}
    return pd.DataFrame(data)


def test_save_global_plan():
    """Test save_global_plan() serialization."""
    print("\n=== Test 1: save_global_plan ===")

    design_df = create_test_design_df(n_factors=4)
    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=3, n_batches=2, total_budget=300, seed=42
    )
    coordinator.fit_initial_plan()

    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = os.path.join(tmpdir, "global_plan.json")
        coordinator.save_global_plan(plan_path)

        assert os.path.exists(plan_path), "Plan file not created"

        with open(plan_path, "r") as f:
            plan = json.load(f)

        assert "metadata" in plan, "Missing metadata"
        assert plan["metadata"]["n_subjects"] == 3, "Wrong n_subjects"
        assert plan["metadata"]["n_batches"] == 2, "Wrong n_batches"
        assert "factors" in plan, "Missing factors"
        assert "budget_split" in plan, "Missing budget_split"

        print("[OK] Global plan saved with {} keys".format(len(plan)))
        print("  - Metadata: {}".format(plan["metadata"]["timestamp"]))
        print("  - Budget split keys: {}".format(list(plan["budget_split"].keys())))


def test_save_load_run_state():
    """Test save_run_state() and load_run_state() round-trip."""
    print("\n=== Test 2: save/load_run_state ===")

    design_df = create_test_design_df(n_factors=4)
    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=3, n_batches=2, total_budget=300, seed=42
    )
    coordinator.fit_initial_plan()

    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "run_state.pkl")

        # Save state
        coordinator.save_run_state(state_path, batch_k=0)
        assert os.path.exists(state_path), "State file not created"

        # Load state
        loaded_state = coordinator.load_run_state(state_path)

        assert "checkpoint" in loaded_state, "Missing checkpoint"
        assert loaded_state["checkpoint"]["current_batch"] == 0, "Wrong batch"
        assert "subject_states" in loaded_state, "Missing subject_states"
        assert len(loaded_state["subject_states"]) == 3, "Wrong number of subjects"

        print("[OK] Run state saved and loaded successfully")
        print(
            "  - Checkpoint batch: {}".format(
                loaded_state["checkpoint"]["current_batch"]
            )
        )
        print("  - Subjects tracked: {}".format(len(loaded_state["subject_states"])))


def test_export_subject_plan():
    """Test export_subject_plan() for per-subject plan export."""
    print("\n=== Test 3: export_subject_plan ===")

    design_df = create_test_design_df(n_factors=4)
    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=3, n_batches=2, total_budget=300, seed=42
    )
    coordinator.fit_initial_plan()

    with tempfile.TemporaryDirectory() as tmpdir:
        plan_path = os.path.join(tmpdir, "subject_0.plan.json")
        coordinator.export_subject_plan(subject_id=0, path=plan_path)

        assert os.path.exists(plan_path), "Subject plan file not created"

        with open(plan_path, "r") as f:
            plan = json.load(f)

        assert plan["subject_id"] == 0, "Wrong subject ID"
        assert "quotas" in plan, "Missing quotas"
        assert "constraints" in plan, "Missing constraints"
        assert "seed" in plan, "Missing seed"

        print("[OK] Subject plan exported successfully")
        print("  - Subject ID: {}".format(plan["subject_id"]))
        print("  - Quotas: {}".format(plan["quotas"]))


def test_apply_plan():
    """Test apply_plan() injection of Coordinator's plan into Generator."""
    print("\n=== Test 4: apply_plan ===")

    design_df = create_test_design_df(n_factors=4)

    # Create Coordinator and extract subject plan
    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=3, n_batches=2, total_budget=300, seed=42
    )
    coordinator.fit_initial_plan()

    # Get subject plan
    subject_plan = coordinator.allocate_subject_plan(subject_id=0)

    # Create Generator and apply plan
    generator = WarmupAEPsychGenerator(
        design_df=design_df, n_subjects=3, total_budget=300, n_batches=2, seed=42
    )

    # Apply plan before fit_planning
    generator.apply_plan(subject_plan)

    assert generator.subject_id == 0, "Subject ID not applied"
    assert generator.batch_id == subject_plan["batch_id"], "Batch ID not applied"

    print("[OK] Plan applied to Generator successfully")
    print("  - Subject ID: {}".format(generator.subject_id))
    print("  - Batch ID: {}".format(generator.batch_id))
    print("  - Core-1 quota overridden: {}".format(generator.n_core1_points))


def test_generate_trials_with_save():
    """Test generate_trials(save_to=path) direct file saving."""
    print("\n=== Test 5: generate_trials(save_to=path) ===")

    design_df = create_test_design_df(n_factors=4)
    generator = WarmupAEPsychGenerator(
        design_df=design_df, n_subjects=3, total_budget=300, n_batches=2, seed=42
    )
    generator.fit_planning()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test CSV saving
        csv_path = os.path.join(tmpdir, "trials.csv")
        trials_df = generator.generate_trials(save_to=csv_path)
        assert os.path.exists(csv_path), "CSV file not created"
        loaded_trials = pd.read_csv(csv_path)
        assert len(loaded_trials) == len(trials_df), "Loaded trials shape mismatch"

        # Test default saving (should create .csv due to fallback)
        default_path = os.path.join(tmpdir, "trials2")
        trials_df = generator.generate_trials(save_to=default_path)
        # Check if CSV was created
        csv_default = default_path + ".csv"
        assert os.path.exists(csv_default), "Default file not created at {}".format(
            csv_default
        )

        print("[OK] Trials saved to CSV successfully")
        print("  - CSV size: {} rows".format(len(loaded_trials)))
        print("  - Default format: CSV (fallback from parquet)")


def test_export_metadata():
    """Test export_metadata() audit trail export."""
    print("\n=== Test 6: export_metadata ===")

    design_df = create_test_design_df(n_factors=4)
    generator = WarmupAEPsychGenerator(
        design_df=design_df, n_subjects=3, total_budget=300, n_batches=2, seed=42
    )
    generator.subject_id = 0
    generator.batch_id = 1
    generator.fit_planning()
    generator.generate_trials()

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = os.path.join(tmpdir, "metadata.json")
        generator.export_metadata(metadata_path)

        assert os.path.exists(metadata_path), "Metadata file not created"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["subject_id"] == 0, "Wrong subject_id in metadata"
        assert metadata["batch_id"] == 1, "Wrong batch_id in metadata"
        assert "coverage_rate" in metadata, "Missing coverage_rate"
        assert "gini_coefficient" in metadata, "Missing gini_coefficient"
        assert metadata["schema_version"] == "1.0", "Wrong schema_version"

        print("[OK] Metadata exported successfully")
        print(
            "  - Subject: {}, Batch: {}".format(
                metadata["subject_id"], metadata["batch_id"]
            )
        )
        print("  - Coverage: {:.3f}".format(metadata["coverage_rate"]))
        print("  - Gini: {:.3f}".format(metadata["gini_coefficient"]))


def test_validate_global_constraints():
    """Test validate_global_constraints() with mock trials."""
    print("\n=== Test 7: validate_global_constraints ===")

    design_df = create_test_design_df(n_factors=4)
    coordinator = StudyCoordinator(
        design_df=design_df, n_subjects=3, n_batches=2, total_budget=300, seed=42
    )
    coordinator.fit_initial_plan()

    # Generate mock trials
    mock_trials = pd.DataFrame(
        {
            "f0": np.random.rand(30),
            "f1": np.random.rand(30),
            "f2": np.random.rand(30),
            "f3": np.random.rand(30),
            "subject_id": [0] * 10 + [1] * 10 + [2] * 10,
            "batch_id": [0] * 15 + [1] * 15,
            "block_type": ["core1"] * 10 + ["core2"] * 10 + ["individual"] * 10,
            "design_row_id": np.arange(30),
            "is_bridge": [False] * 30,
        }
    )

    result = coordinator.validate_global_constraints(mock_trials)

    assert "core1_repeat_ratio" in result, "Missing core1_repeat_ratio"
    assert "warnings" in result, "Missing warnings list"
    assert isinstance(result["warnings"], list), "Warnings should be a list"

    print("[OK] Global constraints validated successfully")
    print("  - Core-1 repeat ratio: {:.1%}".format(result["core1_repeat_ratio"]))
    print("  - Warnings: {}".format(len(result["warnings"])))


def test_end_to_end_workflow():
    """Test end-to-end workflow: plan -> save -> load -> execute -> validate."""
    print("\n=== Test 8: End-to-end workflow ===")

    design_df = create_test_design_df(n_factors=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Phase 1: Global planning
        print("  Phase 1: Global planning...")
        coordinator = StudyCoordinator(
            design_df=design_df, n_subjects=3, n_batches=2, total_budget=300, seed=42
        )
        coordinator.fit_initial_plan()

        # Phase 2: Save global state
        print("  Phase 2: Saving global state...")
        global_plan_path = os.path.join(tmpdir, "global_plan.json")
        run_state_path = os.path.join(tmpdir, "run_state.pkl")
        coordinator.save_global_plan(global_plan_path)
        coordinator.save_run_state(run_state_path, batch_k=0)

        # Phase 3: Per-subject planning (simulate batch execution)
        print("  Phase 3: Per-subject batch execution...")
        all_trials = []
        for subject_id in range(3):
            subject_plan_path = os.path.join(
                tmpdir, "subject_{}.plan.json".format(subject_id)
            )
            coordinator.export_subject_plan(subject_id, subject_plan_path)

            # Create generator for this subject
            generator = WarmupAEPsychGenerator(
                design_df=design_df,
                n_subjects=3,
                total_budget=300,
                n_batches=2,
                seed=42,
            )

            # Load subject plan and apply
            with open(subject_plan_path, "r") as f:
                plan = json.load(f)
            generator.apply_plan(plan)

            # Generate trials
            generator.fit_planning()
            trials_path = os.path.join(
                tmpdir, "subject_{}_batch_0.csv".format(subject_id)
            )
            trials = generator.generate_trials(save_to=trials_path)
            all_trials.append(trials)

            # Export metadata
            metadata_path = os.path.join(
                tmpdir, "subject_{}_metadata.json".format(subject_id)
            )
            generator.export_metadata(metadata_path)

        # Phase 4: Validate global constraints
        print("  Phase 4: Aggregate validation...")
        combined_trials = pd.concat(all_trials, ignore_index=True)
        validation_result = coordinator.validate_global_constraints(combined_trials)

        assert validation_result["core1_repeat_ratio"] >= 0, "Invalid repeat ratio"

        print("[OK] End-to-end workflow completed successfully")
        print(
            "  - Generated {} total trials across {} subjects".format(
                len(combined_trials), len(all_trials)
            )
        )
        print(
            "  - Global validation result: {} warnings".format(
                len(validation_result["warnings"])
            )
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Persistence Layer Integration")
    print("=" * 60)

    try:
        test_save_global_plan()
        test_save_load_run_state()
        test_export_subject_plan()
        test_apply_plan()
        test_generate_trials_with_save()
        test_export_metadata()
        test_validate_global_constraints()
        test_end_to_end_workflow()

        print("\n" + "=" * 60)
        print("[SUCCESS] All 8 persistence tests PASSED!")
        print("=" * 60)
    except Exception as e:
        print("\n[FAILURE] Test failed with error: {}".format(e))
        import traceback

        traceback.print_exc()
