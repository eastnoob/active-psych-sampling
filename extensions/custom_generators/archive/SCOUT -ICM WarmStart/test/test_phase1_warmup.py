"""
测试Phase 1预热采样器
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scout_warmup_251113 import Phase1WarmupSampler


def test_basic_sampling():
    """测试基本采样功能"""
    print("=" * 60)
    print("测试1: 基本采样功能")
    print("=" * 60)

    # 生成模拟设计空间
    np.random.seed(42)
    n_configs = 1200
    d = 5

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_configs)

    design_df = pd.DataFrame(design_data)
    print(f"✓ 设计空间创建成功: {len(design_df)} 个配置, {d} 个因子")

    # 创建采样器
    sampler = Phase1WarmupSampler(
        design_df=design_df, n_subjects=7, trials_per_subject=25, seed=42
    )
    print(f"✓ 采样器初始化成功")

    # 测试Core-1选择
    core1 = sampler.select_core1_points()
    assert len(core1) == 8, f"Core-1应该有8个点，实际有{len(core1)}个"
    print(f"✓ Core-1选择成功: {len(core1)} 个点")

    # 测试Core-2主效应
    main_effects = sampler.select_core2_main_effects()
    assert len(main_effects) == 45, f"主效应应该有45个点，实际有{len(main_effects)}个"
    print(f"✓ Core-2主效应选择成功: {len(main_effects)} 个点")

    # 测试Core-2交互
    interactions = sampler.select_core2_interactions()
    assert len(interactions) == 25, f"交互应该有25个点，实际有{len(interactions)}个"
    print(f"✓ Core-2交互选择成功: {len(interactions)} 个点")

    # 测试边界点
    boundary = sampler.select_boundary_points()
    assert len(boundary) == 20, f"边界应该有20个点，实际有{len(boundary)}个"
    print(f"✓ 边界点选择成功: {len(boundary)} 个点")

    # 测试LHS点
    lhs = sampler.select_lhs_points()
    assert len(lhs) == 29, f"LHS应该有29个点，实际有{len(lhs)}个"
    print(f"✓ LHS点选择成功: {len(lhs)} 个点")

    print("\n✅ 测试1通过！\n")


def test_complete_workflow():
    """测试完整工作流"""
    print("=" * 60)
    print("测试2: 完整工作流")
    print("=" * 60)

    # 生成设计空间
    np.random.seed(42)
    n_configs = 1200
    d = 5

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_configs)

    design_df = pd.DataFrame(design_data)

    # 创建采样器
    sampler = Phase1WarmupSampler(
        design_df=design_df, n_subjects=7, trials_per_subject=25, seed=42
    )

    # 执行完整采样
    results = sampler.run_sampling()

    trials = results["trials"]
    print(f"✓ 生成试验清单: {len(trials)} 个试验")

    # 验证预算
    expected_total = 8 * 7 + 45 + 25 + 20 + 29  # Core-1 + main + inter + boundary + lhs
    assert (
        len(trials) == expected_total
    ), f"总试验数应该是{expected_total}，实际是{len(trials)}"
    print(f"✓ 预算验证通过: {len(trials)} = {expected_total}")

    # 验证每个受试者的试验数
    for subject_id in range(7):
        subject_trials = trials[trials["subject_id"] == subject_id]
        print(f"  受试者 {subject_id}: {len(subject_trials)} 个试验")

    # 验证块类型分布
    block_counts = trials["block_type"].value_counts()
    print(f"\n✓ 块类型分布:")
    for block, count in block_counts.items():
        print(f"  {block}: {count}")

    # 验证Core-1每人都有
    core1_trials = trials[trials["block_type"] == "core1"]
    assert len(core1_trials) == 8 * 7, f"Core-1应该有{8*7}个试验"
    for subject_id in range(7):
        subject_core1 = core1_trials[core1_trials["subject_id"] == subject_id]
        assert len(subject_core1) == 8, f"每个受试者应该有8个Core-1试验"
    print(f"✓ Core-1分配验证通过")

    print("\n✅ 测试2通过！\n")


def test_interaction_selection_methods():
    """测试不同的交互对选择方法"""
    print("=" * 60)
    print("测试3: 交互对选择方法对比")
    print("=" * 60)

    # 生成设计空间
    np.random.seed(42)
    n_configs = 1200
    d = 10

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_configs)
    design_df = pd.DataFrame(design_data)

    methods = ["variance", "correlation", "auto"]
    for method in methods:
        print(f"\n--- {method.upper()} 方法 ---")
        sampler = Phase1WarmupSampler(
            design_df=design_df,
            n_subjects=7,
            trials_per_subject=25,
            interaction_selection=method,
            seed=42,
        )
        pairs = sampler._select_interaction_pairs()
        print(f"选择的交互对: {pairs}")

    print("\n✅ 测试3通过！\n")


def test_with_priority_pairs():
    """测试使用优先交互对"""
    print("=" * 60)
    print("测试4: 优先交互对")
    print("=" * 60)

    # 生成设计空间
    np.random.seed(42)
    n_configs = 1200
    d = 10

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_configs)

    design_df = pd.DataFrame(design_data)

    # 指定优先交互对
    priority_pairs = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

    # 创建采样器
    sampler = Phase1WarmupSampler(
        design_df=design_df,
        n_subjects=7,
        trials_per_subject=25,
        priority_pairs=priority_pairs,
        seed=42,
    )

    print(f"✓ 采样器初始化成功，使用 {len(priority_pairs)} 个优先交互对")

    # 执行采样
    results = sampler.run_sampling()

    # 检查交互点
    inter_samples = results["selected_samples"]["core2_inter"]
    print(f"✓ 生成 {len(inter_samples)} 个交互初筛点")

    # 验证每对有5个点
    pair_counts = {}
    for idx, pair_id in inter_samples:
        pair_name = pair_id.split("_")[1]  # 提取pair_0, pair_1等
        pair_counts[pair_name] = pair_counts.get(pair_name, 0) + 1

    print(f"✓ 交互对分布:")
    for pair, count in sorted(pair_counts.items()):
        print(f"  {pair}: {count} 个点")
        assert count == 5, f"{pair}应该有5个点，实际有{count}个"

    print("\n✅ 测试4通过！\n")


def test_quality_evaluation():
    """测试质量评估功能"""
    print("=" * 60)
    print("测试5: 质量评估")
    print("=" * 60)

    # 生成设计空间
    np.random.seed(42)
    n_configs = 1200
    d = 5

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_configs)
    design_df = pd.DataFrame(design_data)

    # 创建采样器并运行
    sampler = Phase1WarmupSampler(
        design_df=design_df, n_subjects=7, trials_per_subject=25, seed=42
    )

    results = sampler.run_sampling()
    quality = results["quality"]

    print(f"✓ 质量指标:")
    print(f"  覆盖率: {quality['coverage_rate']:.2%}")
    print(f"  唯一配置: {quality['n_unique_configs']}")
    print(f"  最小距离: {quality.get('min_dist', 0):.4f}")
    print(f"  中位距离: {quality.get('median_dist', 0):.4f}")

    # 检查因子平衡
    print(f"\n✓ 因子水平平衡:")
    for f, balance in quality["level_balance"].items():
        print(
            f"  {f}: Gini={balance['gini']:.3f}, "
            f"min_count={balance['min_count']}, max_count={balance['max_count']}"
        )

    # 检查警告
    if "warnings" in quality and quality["warnings"]:
        print(f"\n⚠️ 质量警告:")
        for w in quality["warnings"]:
            print(f"  - {w}")

    print("\n✅ 测试5通过！\n")


def test_export_results():
    """测试结果导出"""
    print("=" * 60)
    print("测试6: 结果导出")
    print("=" * 60)

    # 生成设计空间
    np.random.seed(42)
    n_configs = 1200
    d = 5

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_configs)

    design_df = pd.DataFrame(design_data)

    # 创建采样器并运行
    sampler = Phase1WarmupSampler(
        design_df=design_df, n_subjects=7, trials_per_subject=25, seed=42
    )

    results = sampler.run_sampling()
    sampler.trials = results["trials"]  # 保存以供导出

    # 导出结果
    output_dir = "./test_output"
    sampler.export_results(output_dir=output_dir)

    # 检查文件是否存在
    import os

    assert os.path.exists(f"{output_dir}/phase1_trials.csv"), "试验清单文件未创建"
    assert os.path.exists(f"{output_dir}/phase1_core1_points.csv"), "Core-1点文件未创建"
    assert os.path.exists(f"{output_dir}/phase1_summary.json"), "摘要文件未创建"

    print(f"✓ 所有文件导出成功到 {output_dir}")

    # 读取并验证
    trials_df = pd.read_csv(f"{output_dir}/phase1_trials.csv")
    print(f"✓ 试验清单: {len(trials_df)} 行")

    core1_df = pd.read_csv(f"{output_dir}/phase1_core1_points.csv")
    print(f"✓ Core-1点: {len(core1_df)} 行")

    import json

    with open(f"{output_dir}/phase1_summary.json") as f:
        summary = json.load(f)
    print(f"✓ 摘要: {summary['total_budget']} 总预算")

    print("\n✅ 测试6通过！\n")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始运行Phase 1预热采样器测试套件")
    print("=" * 60 + "\n")

    try:
        test_basic_sampling()
        test_complete_workflow()
        test_interaction_selection_methods()
        test_with_priority_pairs()
        test_quality_evaluation()
        test_export_results()

        print("=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        return True

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        return False
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
