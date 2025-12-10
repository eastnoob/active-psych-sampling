#!/usr/bin/env python3
"""
测试历史点排除机制的一致性
验证 _used_indices 和 _historical_points 的同步更新
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
project_root = Path(__file__).resolve().parents[3]
custom_gen_path = project_root / "extensions" / "custom_generators"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(custom_gen_path))

import torch
from aepsych.config import Config
from botorch.acquisition import qUpperConfidenceBound

# Import from extensions.custom_generators
sys.path.insert(0, str(project_root / "extensions"))
from custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator


def test_history_exclusion_sync():
    """测试历史点排除时 _used_indices 和 _historical_points 的同步"""
    print("\n" + "=" * 70)
    print("测试: 历史点排除的双重机制同步")
    print("=" * 70)

    # Create simple pool
    pool_points = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ],
        dtype=torch.float32,
    )

    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([10.0, 10.0])

    # Initialize generator with temporary database (Mode 2)
    generator = CustomPoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool_points,
        acqf=qUpperConfidenceBound,
        acqf_kwargs={"beta": 0.1},
        allow_resampling=False,
        shuffle=False,  # Disable shuffle for predictable testing
        dedup_database_path=None,  # Temporary database
    )

    print(f"\n✓ 初始化完成: pool_size={len(pool_points)}")
    print(f"  _used_indices: {generator._used_indices}")
    print(f"  _historical_points: {generator._historical_points}")

    # Simulate historical points from server
    sampling_history = torch.tensor(
        [
            [1.0, 2.0],  # Matches pool[0]
            [5.0, 6.0],  # Matches pool[2]
        ],
        dtype=torch.float32,
    )

    print(f"\n模拟服务器历史: {sampling_history.shape[0]} 个点")
    for i, point in enumerate(sampling_history):
        print(f"  History[{i}]: {point.tolist()}")

    # Test exclusion
    excluded_count = generator._exclude_historical_points_from_history(sampling_history)

    print(f"\n排除结果: {excluded_count} 个新历史点")
    print(f"  _used_indices: {generator._used_indices}")
    print(f"  _historical_points: {len(generator._historical_points)} 个点")

    # Verify synchronization
    assert excluded_count == 2, f"Expected 2 excluded, got {excluded_count}"
    assert (
        len(generator._used_indices) == 2
    ), f"Expected 2 in _used_indices, got {len(generator._used_indices)}"
    assert (
        len(generator._historical_points) == 2
    ), f"Expected 2 in _historical_points, got {len(generator._historical_points)}"

    # Verify specific indices (note: actual indices depend on pool order, so just check count)
    print(f"  实际排除的索引: {generator._used_indices}")

    # Verify specific tuples
    expected_tuples = {tuple([1.0, 2.0]), tuple([5.0, 6.0])}
    assert (
        generator._historical_points == expected_tuples
    ), f"Historical points mismatch"

    print("\n✓ 双重机制同步正确:")
    print("  ✓ _used_indices 包含2个pool索引")
    print("  ✓ _historical_points 包含正确的点tuple")

    # Test get_available_indices
    available = generator._get_available_indices()
    print(f"\n可用索引: {available.tolist()}")

    # Should have 3 available points (5 total - 2 excluded)
    assert len(available) == 3, f"Expected 3 available, got {len(available)}"

    print(f"✓ 可用索引数量正确: 3 个 (总5个 - 已用2个)")

    print("\n" + "=" * 70)
    print("✅ 测试通过: 历史点排除双重机制同步正确")
    print("=" * 70)


def test_repeated_history_loading():
    """测试重复加载历史点不会重复计数"""
    print("\n" + "=" * 70)
    print("测试: 重复加载历史点的幂等性")
    print("=" * 70)

    pool_points = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )

    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([10.0, 10.0])

    generator = CustomPoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool_points,
        acqf=qUpperConfidenceBound,
        acqf_kwargs={"beta": 0.1},
        allow_resampling=False,
        dedup_database_path=None,
    )

    sampling_history = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    # First load
    count1 = generator._exclude_historical_points_from_history(sampling_history)
    used_count1 = len(generator._used_indices)
    hist_count1 = len(generator._historical_points)

    print(f"\n第一次加载: excluded={count1}, used={used_count1}, hist={hist_count1}")

    # Second load (same history)
    count2 = generator._exclude_historical_points_from_history(sampling_history)
    used_count2 = len(generator._used_indices)
    hist_count2 = len(generator._historical_points)

    print(f"第二次加载: excluded={count2}, used={used_count2}, hist={hist_count2}")

    # Should be idempotent
    assert count1 == 1, f"First load should exclude 1 point, got {count1}"
    assert count2 == 0, f"Second load should exclude 0 new points, got {count2}"
    assert used_count1 == used_count2 == 1, f"Used indices should stay at 1"
    assert hist_count1 == hist_count2 == 1, f"Historical points should stay at 1"

    print("\n✓ 幂等性验证通过:")
    print("  ✓ 第二次加载不重复计数")
    print("  ✓ _used_indices 和 _historical_points 保持一致")

    print("\n" + "=" * 70)
    print("✅ 测试通过: 重复加载历史点的幂等性正确")
    print("=" * 70)


def test_mixed_server_and_local_history():
    """测试从服务器获取的历史和本地选择的点混合场景"""
    print("\n" + "=" * 70)
    print("测试: 服务器历史 + 本地选择混合场景")
    print("=" * 70)

    pool_points = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )

    lb = torch.tensor([0.0, 0.0])
    ub = torch.tensor([10.0, 10.0])

    generator = CustomPoolBasedGenerator(
        lb=lb,
        ub=ub,
        pool_points=pool_points,
        acqf=qUpperConfidenceBound,
        acqf_kwargs={"beta": 0.1},
        allow_resampling=False,
        shuffle=False,  # Disable shuffle for predictable testing
        dedup_database_path=None,
    )

    # Simulate server history (warmup points)
    server_history = torch.tensor(
        [[1.0, 1.0], [2.0, 2.0]],  # Pool[0], Pool[1]
        dtype=torch.float32,
    )

    generator._exclude_historical_points_from_history(server_history)
    print(f"\n服务器历史加载: {len(server_history)} 个点")
    print(f"  _used_indices: {generator._used_indices}")
    print(f"  _historical_points: {len(generator._historical_points)} 个点")

    # Simulate local selection
    local_selected = torch.tensor([[3.0, 3.0]], dtype=torch.float32)  # Pool[2]
    generator._record_points_to_dedup_db(local_selected)
    generator._used_indices.add(2)

    print(f"\n本地选择记录: {len(local_selected)} 个点")
    print(f"  _used_indices: {generator._used_indices}")
    print(f"  _historical_points: {len(generator._historical_points)} 个点")

    # Verify combined exclusion
    available = generator._get_available_indices()
    print(f"\n可用索引: {available.tolist()}")

    # Should have 2 available points (5 total - 3 excluded)
    assert len(available) == 2, f"Expected 2 available, got {len(available)}"

    print(f"✓ 混合历史排除正确: 2个可用 (总5个 - 已用3个)")

    # Verify both mechanisms have all points
    assert (
        len(generator._used_indices) == 3
    ), f"Expected 3 used indices, got {len(generator._used_indices)}"
    assert (
        len(generator._historical_points) == 3
    ), f"Expected 3 historical points, got {len(generator._historical_points)}"

    print("\n✓ 双重机制都包含所有3个点 (2个服务器 + 1个本地)")

    print("\n" + "=" * 70)
    print("✅ 测试通过: 混合历史场景处理正确")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_history_exclusion_sync()
        test_repeated_history_loading()
        test_mixed_server_and_local_history()

        print("\n" + "=" * 70)
        print("🎉 所有测试通过!")
        print("=" * 70)

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 测试错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
