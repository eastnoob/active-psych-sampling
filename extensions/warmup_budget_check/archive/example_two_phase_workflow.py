"""
两阶段实验工作流示例
演示如何使用TwoPhaseExperimentPlanner进行完整的实验规划
"""

import numpy as np
import pandas as pd
from two_phase_planner import TwoPhaseExperimentPlanner


def example_workflow():
    """完整的两阶段实验工作流示例"""

    print("=" * 80)
    print("两阶段实验规划示例")
    print("=" * 80)
    print()

    # ========== 阶段0: 准备设计空间 ==========
    print("Step 0: 准备设计空间CSV")
    print("-" * 80)

    # 创建示例设计空间（实际使用时应该从现有CSV加载）
    design_csv = "example_design_space.csv"

    # 假设有6个因子，每个5个水平
    factors = {
        "density": [1, 2, 3, 4, 5],
        "height": [1, 2, 3, 4, 5],
        "greenery": [1, 2, 3, 4, 5],
        "street_width": [1, 2, 3, 4, 5],
        "landmark": [1, 2, 3, 4, 5],
        "style": [1, 2, 3, 4, 5],
    }

    # 生成全因子设计（5^6 = 15625种配置，实际可能是部分因子设计）
    # 这里简化为1200种配置
    np.random.seed(42)
    n_configs = 1200
    design_df = pd.DataFrame({
        factor: np.random.choice(levels, size=n_configs)
        for factor, levels in factors.items()
    })
    design_df.to_csv(design_csv, index=False)
    print(f"[OK] 设计空间已保存: {design_csv}")
    print(f"  配置总数: {n_configs}")
    print(f"  因子数: {len(factors)}")
    print()

    # ========== 阶段1: 规划Phase 1 ==========
    print("\n" + "=" * 80)
    print("Step 1: 规划Phase 1（预热阶段）")
    print("=" * 80)

    planner = TwoPhaseExperimentPlanner(design_csv)

    # 规划Phase 1
    phase1_plan = planner.plan_phase1(
        n_subjects=14,  # 14名被试
        trials_per_subject=25,  # 每人25次
        skip_interaction=False,
    )

    # ========== 模拟Phase 1数据收集 ==========
    print("\n" + "=" * 80)
    print("Step 2: [模拟] Phase 1 数据收集")
    print("=" * 80)
    print("(实际实验中，这里应该是真实的数据收集过程)")
    print()

    # 模拟Phase 1数据
    n_phase1_samples = phase1_plan["total_budget"]
    n_factors = len(factors)

    X_warmup = np.random.rand(n_phase1_samples, n_factors) * 5  # 模拟因子值
    # 模拟响应：y = 主效应 + 交互效应 + 噪声
    # 假设真实的交互对是 (0,2) 和 (1,4)
    y_warmup = (
        0.3 * X_warmup[:, 0]  # 主效应1
        + 0.2 * X_warmup[:, 1]  # 主效应2
        + 0.25 * X_warmup[:, 2]  # 主效应3
        + 0.15 * X_warmup[:, 3]  # 主效应4
        + 0.1 * X_warmup[:, 4]  # 主效应5
        + 0.2 * X_warmup[:, 5]  # 主效应6
        + 0.4 * X_warmup[:, 0] * X_warmup[:, 2]  # 真实交互1: (0,2)
        + 0.3 * X_warmup[:, 1] * X_warmup[:, 4]  # 真实交互2: (1,4)
        + np.random.randn(n_phase1_samples) * 0.5  # 噪声
    )

    # 模拟被试ID（14个被试，每人约25次）
    subject_ids = np.repeat(np.arange(14), n_phase1_samples // 14 + 1)[
        :n_phase1_samples
    ]

    print(f"[OK] 模拟数据生成完成:")
    print(f"  样本数: {n_phase1_samples}")
    print(f"  被试数: {len(np.unique(subject_ids))}")
    print(f"  真实交互对: (density, greenery), (height, style)")
    print()

    # ========== 阶段2: 分析Phase 1数据 ==========
    print("\n" + "=" * 80)
    print("Step 3: 分析Phase 1数据")
    print("=" * 80)

    phase1_analysis = planner.analyze_phase1_data(
        X_warmup=X_warmup,
        y_warmup=y_warmup,
        subject_ids=subject_ids,
        max_pairs=5,
        min_pairs=3,
        selection_method="elbow",
        verbose=True,
    )

    # ========== 阶段3: 规划Phase 2 ==========
    print("\n" + "=" * 80)
    print("Step 4: 规划Phase 2（主动学习阶段）")
    print("=" * 80)

    phase2_plan = planner.plan_phase2(
        n_subjects=18,  # 18名被试
        trials_per_subject=25,  # 每人25次
        use_phase1_estimates=True,
        lambda_adjustment=1.2,
    )

    # ========== 阶段4: 导出Phase 1输出 ==========
    print("\n" + "=" * 80)
    print("Step 5: 导出Phase 1输出（供Phase 2使用）")
    print("=" * 80)

    exported_files = planner.export_phase1_output(
        output_dir="phase1_outputs", prefix="example_phase1"
    )

    # ========== 总结 ==========
    print("\n" + "=" * 80)
    print("实验规划总结")
    print("=" * 80)

    total_subjects = phase1_plan["n_subjects"] + phase2_plan["n_subjects"]
    total_budget = phase1_plan["total_budget"] + phase2_plan["total_budget"]

    print(f"总被试数: {total_subjects} ({phase1_plan['n_subjects']} + {phase2_plan['n_subjects']})")
    print(f"总采样次数: {total_budget} ({phase1_plan['total_budget']} + {phase2_plan['total_budget']})")
    print(f"设计空间覆盖率: {total_budget / n_configs * 100:.1f}%")
    print()

    print("Phase 1 → Phase 2 衔接:")
    print(f"  筛选出的交互对: {phase2_plan['n_interaction_pairs']}个")
    print(f"  λ参数传递: {phase1_analysis['lambda_init']:.3f} → {phase2_plan['lambda_init']:.3f}")
    print(f"  γ初始值: {phase2_plan['gamma_init']:.3f}")
    print()

    print("Phase 2 动态调整:")
    # 显示几个关键时间点的λ和γ
    lambda_schedule = phase2_plan["lambda_schedule"]
    gamma_schedule = phase2_plan["gamma_schedule"]

    milestones = [0, len(lambda_schedule) // 4, len(lambda_schedule) // 2, -1]
    print("  时间点    λ值    γ值")
    print("-" * 40)
    for idx in milestones:
        t, lambda_t = lambda_schedule[idx]
        _, gamma_t = gamma_schedule[idx]
        print(f"  第{t:4d}次  {lambda_t:.3f}  {gamma_t:.3f}")

    print()
    print("=" * 80)
    print("[SUCCESS] 两阶段实验规划完成！")
    print("=" * 80)
    print()

    print("下一步:")
    print("1. 使用Phase 1规划执行预热数据收集")
    print("2. 分析Phase 1数据（已完成示例）")
    print("3. 使用Phase 2规划执行主动学习（EUR-ANOVA）")
    print("4. 在Phase 2第{}次进行中期诊断".format(phase2_plan["mid_diagnostic_trial"]))


if __name__ == "__main__":
    example_workflow()
