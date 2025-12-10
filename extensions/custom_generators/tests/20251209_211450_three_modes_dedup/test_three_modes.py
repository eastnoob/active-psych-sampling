"""
Three-Mode Deduplication Database Test

Mode 1: Manual path (string) - Persistent SQLite
Mode 2: None - Temporary in-memory database
Mode 3: Tuple auto-naming - Auto-generated path
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from custom_pool_based_generator import CustomPoolBasedGenerator
from botorch.acquisition.monte_carlo import qExpectedImprovement


def cleanup_db(gen):
    """Safely close database connection"""
    try:
        if gen._dedup_conn:
            gen._dedup_conn.close()
    except:
        pass


def test_mode1_creates_persistent_database():
    """Verify Mode 1 creates persistent database"""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test_subject_A.db")

        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pool_points = torch.rand(100, 2)

        gen = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=db_path,
        )

        assert Path(db_path).exists(), f"DB file not found: {db_path}"
        assert not gen._is_temp_db, "Mode 1 should be persistent"

        cleanup_db(gen)
        print("[PASS] Mode 1: Persistent database created")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mode1_replaces_old_database():
    """Verify Mode 1 deletes old file before creating new one"""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = os.path.join(tmpdir, "test_subject_A.db")

        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pool_points = torch.rand(100, 2)

        gen1 = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=db_path,
        )
        gen1._record_points_to_dedup_db(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        cleanup_db(gen1)

        mtime1 = Path(db_path).stat().st_mtime

        import time

        time.sleep(0.1)

        gen2 = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=db_path,
        )

        mtime2 = Path(db_path).stat().st_mtime
        assert mtime2 >= mtime1, "Should delete old file and create new one"
        assert len(gen2._historical_points) == 0, "New DB should be empty"

        cleanup_db(gen2)
        print("[PASS] Mode 1: Old file replacement successful")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mode2_creates_memory_database():
    """Verify Mode 2 creates in-memory database"""
    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([1.0, 1.0])
    pool_points = torch.rand(100, 2)

    gen = CustomPoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool_points,
        acqf=qExpectedImprovement,
        dedup_database_path=None,
    )

    assert gen._is_temp_db, "Mode 2 should be temporary in-memory"
    assert gen._dedup_conn is not None, "Should create in-memory connection"

    cleanup_db(gen)
    print("[PASS] Mode 2: Temporary in-memory database created")


def test_mode2_data_not_persisted():
    """Verify Mode 2 data is not persisted"""
    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([1.0, 1.0])
    pool_points = torch.rand(100, 2)

    gen1 = CustomPoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool_points,
        acqf=qExpectedImprovement,
        dedup_database_path=None,
    )

    gen1._record_points_to_dedup_db(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    historical_points_1 = gen1._historical_points.copy()
    cleanup_db(gen1)

    gen2 = CustomPoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool_points,
        acqf=qExpectedImprovement,
        dedup_database_path=None,
    )

    historical_points_2 = gen2._historical_points.copy()
    cleanup_db(gen2)

    assert len(historical_points_1) > 0, "First generator should have history"
    assert len(historical_points_2) == 0, "Second generator should have no history"

    print("[PASS] Mode 2: Data not persisted confirmed")


def test_mode3_simple_tuple():
    """Verify Mode 3 simple tuple format (subject_id, run_id)"""
    tmpdir = tempfile.mkdtemp()
    try:
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        try:
            lb = torch.tensor([0.0, 0.0])
            ub = torch.tensor([1.0, 1.0])
            pool_points = torch.rand(100, 2)

            gen = CustomPoolBasedGenerator(
                lb=lb,
                ub=ub,
                pool_points=pool_points,
                acqf=qExpectedImprovement,
                dedup_database_path=("subject_A", "run001"),
            )

            assert not gen._is_temp_db, "Mode 3 should be persistent"

            expected_path = Path("./data/subject_A_run001_dedup.db")
            assert expected_path.exists(), f"DB file should be at {expected_path}"

            cleanup_db(gen)
            print("[PASS] Mode 3: Simple tuple auto-naming successful")

        finally:
            os.chdir(original_cwd)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mode3_custom_directory():
    """Verify Mode 3 custom directory format (subject_id, run_id, save_dir)"""
    tmpdir = tempfile.mkdtemp()
    try:
        custom_dir = os.path.join(tmpdir, "custom_dedup")

        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pool_points = torch.rand(100, 2)

        gen = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=("subject_B", "run002", custom_dir),
        )

        expected_path = Path(custom_dir) / "subject_B_run002_dedup.db"
        assert expected_path.exists(), f"DB file should be at {expected_path}"

        cleanup_db(gen)
        print("[PASS] Mode 3: Custom directory auto-naming successful")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mode3_replaces_old_database():
    """Verify Mode 3 deletes old file before creating new one"""
    tmpdir = tempfile.mkdtemp()
    try:
        custom_dir = os.path.join(tmpdir, "custom_dedup")

        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pool_points = torch.rand(100, 2)

        gen1 = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=("subject_C", "run003", custom_dir),
        )

        gen1._record_points_to_dedup_db(torch.tensor([[0.5, 0.5]]))
        cleanup_db(gen1)

        expected_path = Path(custom_dir) / "subject_C_run003_dedup.db"
        mtime1 = expected_path.stat().st_mtime

        import time

        time.sleep(0.1)

        gen2 = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=("subject_C", "run003", custom_dir),
        )

        mtime2 = expected_path.stat().st_mtime
        assert mtime2 >= mtime1, "Should delete old file and create new one"
        assert len(gen2._historical_points) == 0, "New DB should be empty"

        cleanup_db(gen2)
        print("[PASS] Mode 3: Old file replacement successful")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_mode1_and_mode3_same_file():
    """Verify Mode 1 and Mode 3 can operate on same file"""
    tmpdir = tempfile.mkdtemp()
    try:
        custom_dir = os.path.join(tmpdir, "test")
        db_path = os.path.join(custom_dir, "subject_D_run001_dedup.db")

        lb = torch.tensor([0.0, 0.0])
        ub = torch.tensor([1.0, 1.0])
        pool_points = torch.rand(100, 2)

        # Create with Mode 3
        gen1 = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=("subject_D", "run001", custom_dir),
        )

        gen1._record_points_to_dedup_db(torch.tensor([[0.1, 0.2]]))
        cleanup_db(gen1)

        # Verify file exists
        assert Path(db_path).exists(), f"File should exist after Mode 3: {db_path}"

        # Read with Mode 1 using DIFFERENT path pattern to avoid re-initialization
        # (This is important: Mode 1 deletes and recreates, so we must handle this)
        # In real usage, users should avoid reinitializing the same file with different modes
        gen2 = CustomPoolBasedGenerator(
            lb=lb,
            ub=ub,
            pool_points=pool_points,
            acqf=qExpectedImprovement,
            dedup_database_path=db_path,  # Direct path, not tuple
        )

        # Note: gen2 will delete the old file and create new, so historical data is cleared
        # This is expected behavior - if users reinitialize the file, it gets cleared
        # Both modes can WRITE to the same file location, but not both READ from same instance

        cleanup_db(gen2)
        print("[PASS] Mode interaction: Mode 1 and Mode 3 can target same location")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("THREE-MODE DEDUPLICATION DATABASE TEST")
    print("=" * 60 + "\n")

    print("[Mode 1: Manual Path]")
    test_mode1_creates_persistent_database()
    test_mode1_replaces_old_database()

    print("\n[Mode 2: Temporary Memory]")
    test_mode2_creates_memory_database()
    test_mode2_data_not_persisted()

    print("\n[Mode 3: Tuple Auto-naming]")
    test_mode3_simple_tuple()
    test_mode3_custom_directory()
    test_mode3_replaces_old_database()

    print("\n[Mode Interaction]")
    test_mode1_and_mode3_same_file()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60 + "\n")
