"""
一键式两阶段实验规划脚本
只需修改配置参数，即可完成从预算规划到Phase 2参数生成的全流程

使用方法：
1. 复制 config_template.py 为 config.py
2. 修改 config.py 中的参数
3. 运行: python run_full_pipeline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
from two_phase_planner import TwoPhaseExperimentPlanner

# ============================================================================
# 加载配置
# ============================================================================

def load_config():
    """加载配置文件"""
    try:
        # 尝试加载用户配置
        import config
        print("[配置] 使用 config.py 中的配置")

        CONFIG = {
            "design_csv_path": config.DESIGN_CSV_PATH,
            "phase1": config.PHASE1_CONFIG,
            "phase1_analysis": config.PHASE1_ANALYSIS_CONFIG,
            "phase2": config.PHASE2_CONFIG,
            "output": config.OUTPUT_CONFIG,
            "simulation": config.SIMULATION_CONFIG,
        }

        # 验证配置
        if hasattr(config, 'validate_config'):
            if not config.validate_config():
                sys.exit(1)

        return CONFIG

    except ImportError:
        # 如果没有config.py，使用默认配置
        print("[配置] 未找到 config.py，使用默认配置")
        print("[提示] 复制 config_template.py 为 config.py 来自定义配置")
        print()

        return {
            "design_csv_path": "your_design_space.csv",
            "phase1": {
                "n_subjects": 14,
                "trials_per_subject": 25,
                "skip_interaction": False,
            },
            "phase1_analysis": {
                "max_pairs": 5,
                "min_pairs": 3,
                "selection_method": "elbow",
            },
            "phase2": {
                "n_subjects": 18,
                "trials_per_subject": 25,
                "lambda_adjustment": 1.2,
            },
            "output": {
                "dir": "experiment_outputs",
                "prefix": "my_experiment",
            },
            "simulation": {
                "enabled": False,
                "true_interactions": [(0, 2), (1, 4)],
                "noise_level": 0.5,
            },
        }


CONFIG = load_config()


# ============================================================================
# 主流程 - 无需修改
# ============================================================================

def print_section(title: str, char: str = "="):
    """打印分节标题"""
    print()
    print(char * 80)
    print(title)
    print(char * 80)
    print()


def step1_plan_phase1(planner: TwoPhaseExperimentPlanner):
    """步骤1: 规划Phase 1"""
    print_section("步骤1: 规划Phase 1（预热阶段）")

    phase1_plan = planner.plan_phase1(
        n_subjects=CONFIG["phase1"]["n_subjects"],
        trials_per_subject=CONFIG["phase1"]["trials_per_subject"],
        skip_interaction=CONFIG["phase1"]["skip_interaction"],
    )

    print("[完成] Phase 1规划完成")
    print(f"  总预算: {phase1_plan['total_budget']}次")
    print(f"  被试数: {phase1_plan['n_subjects']}人")
    print()

    return phase1_plan


def step2_collect_phase1_data(phase1_plan):
    """步骤2: 收集Phase 1数据"""
    print_section("步骤2: 收集Phase 1数据")

    if CONFIG["simulation"]["enabled"]:
        print("[模拟模式] 生成模拟数据...")
        X, y, subject_ids = _generate_simulated_data(phase1_plan)
        print(f"[完成] 模拟数据生成完成: {len(X)}样本")
    else:
        print("[实际实验模式] 请执行以下步骤:")
        print("  1. 按照五步采样法收集数据")
        print("  2. 准备以下数据:")
        print("     - X_warmup: 因子值矩阵，形状 (n_samples, n_factors)")
        print("     - y_warmup: 响应变量，形状 (n_samples,)")
        print("     - subject_ids: 被试ID，形状 (n_samples,)")
        print("  3. 将数据保存为 'phase1_data.npz'")
        print()

        # 尝试加载实际数据
        data_path = Path("phase1_data.npz")
        if data_path.exists():
            print(f"[发现] 找到Phase 1数据: {data_path}")
            data = np.load(data_path)
            X = data['X']
            y = data['y']
            subject_ids = data['subject_ids']
            print(f"[加载] 数据加载成功: {len(X)}样本")
        else:
            print("[警告] 未找到 'phase1_data.npz'")
            print("请先收集数据或启用模拟模式（CONFIG['simulation']['enabled'] = True）")
            return None, None, None

    print()
    return X, y, subject_ids


def _generate_simulated_data(phase1_plan):
    """生成模拟数据（仅用于演示）"""
    n_samples = phase1_plan['total_budget']
    n_factors = 6  # 假设6个因子

    # 模拟因子值
    X = np.random.rand(n_samples, n_factors) * 5

    # 模拟响应（包含主效应和交互）
    y = np.zeros(n_samples)

    # 主效应
    main_effects = [0.3, 0.2, 0.25, 0.15, 0.1, 0.2]
    for i, coef in enumerate(main_effects):
        y += coef * X[:, i]

    # 真实交互
    for i, j in CONFIG["simulation"]["true_interactions"]:
        y += 0.4 * X[:, i] * X[:, j]

    # 噪声
    y += np.random.randn(n_samples) * CONFIG["simulation"]["noise_level"]

    # 被试ID
    n_subjects = phase1_plan['n_subjects']
    subject_ids = np.repeat(np.arange(n_subjects), n_samples // n_subjects + 1)[:n_samples]

    return X, y, subject_ids


def step3_analyze_phase1(planner: TwoPhaseExperimentPlanner, X, y, subject_ids):
    """步骤3: 分析Phase 1数据"""
    print_section("步骤3: 分析Phase 1数据")

    if X is None:
        print("[跳过] 无Phase 1数据，跳过分析")
        return None

    phase1_analysis = planner.analyze_phase1_data(
        X_warmup=X,
        y_warmup=y,
        subject_ids=subject_ids,
        max_pairs=CONFIG["phase1_analysis"]["max_pairs"],
        min_pairs=CONFIG["phase1_analysis"]["min_pairs"],
        selection_method=CONFIG["phase1_analysis"]["selection_method"],
        verbose=True,
    )

    print()
    print("[完成] Phase 1分析完成")
    print(f"  筛选出的交互对: {len(phase1_analysis['selected_pairs'])}个")
    print(f"  λ估计: {phase1_analysis['lambda_init']:.3f}")
    print()

    return phase1_analysis


def step4_plan_phase2(planner: TwoPhaseExperimentPlanner):
    """步骤4: 规划Phase 2"""
    print_section("步骤4: 规划Phase 2（主动学习阶段）")

    phase2_plan = planner.plan_phase2(
        n_subjects=CONFIG["phase2"]["n_subjects"],
        trials_per_subject=CONFIG["phase2"]["trials_per_subject"],
        use_phase1_estimates=True,
        lambda_adjustment=CONFIG["phase2"]["lambda_adjustment"],
    )

    print()
    print("[完成] Phase 2规划完成")
    print(f"  总预算: {phase2_plan['total_budget']}次")
    print(f"  λ初始: {phase2_plan['lambda_init']:.3f}")
    print(f"  γ初始: {phase2_plan['gamma_init']:.3f}")
    print()

    return phase2_plan


def step5_export_outputs(planner: TwoPhaseExperimentPlanner):
    """步骤5: 导出所有输出"""
    print_section("步骤5: 导出Phase 1输出")

    exported_files = planner.export_phase1_output(
        output_dir=CONFIG["output"]["dir"],
        prefix=CONFIG["output"]["prefix"],
    )

    print()
    print("[完成] 所有输出已导出")
    print()

    return exported_files


def step6_summary(phase1_plan, phase2_plan, exported_files):
    """步骤6: 显示总结"""
    print_section("实验规划总结", "=")

    total_subjects = phase1_plan["n_subjects"] + phase2_plan["n_subjects"]
    total_budget = phase1_plan["total_budget"] + phase2_plan["total_budget"]

    print("整体统计:")
    print("-" * 80)
    print(f"  总被试数: {total_subjects}人 (Phase 1: {phase1_plan['n_subjects']}, Phase 2: {phase2_plan['n_subjects']})")
    print(f"  总采样数: {total_budget}次 (Phase 1: {phase1_plan['total_budget']}, Phase 2: {phase2_plan['total_budget']})")
    print()

    print("Phase 1 → Phase 2 衔接:")
    print("-" * 80)
    print(f"  筛选出的交互对: {phase2_plan['n_interaction_pairs']}个")
    print(f"  λ传递: {phase2_plan['phase1_lambda']:.3f} → {phase2_plan['lambda_init']:.3f}")
    print(f"  γ初始: {phase2_plan['gamma_init']:.3f}")
    print()

    print("导出的文件:")
    print("-" * 80)
    for key, path in exported_files.items():
        print(f"  {key:10s}: {path}")
    print()

    print("下一步:")
    print("-" * 80)
    print("1. [Phase 1] 按照规划执行预热数据收集")
    print("2. [Phase 1] 分析数据（已完成）")
    print("3. [Phase 2] 使用EUR-ANOVA执行主动学习")
    print(f"4. [Phase 2] 在第{phase2_plan['mid_diagnostic_trial']}次进行中期诊断")
    print()
    print("=" * 80)
    print("[SUCCESS] 两阶段实验规划流程全部完成！")
    print("=" * 80)
    print()


def main():
    """主函数"""
    print()
    print("=" * 80)
    print("两阶段实验规划 - 一键式流程")
    print("=" * 80)
    print()

    # 显示配置
    print("当前配置:")
    print("-" * 80)
    print(f"  设计空间: {CONFIG['design_csv_path']}")
    print(f"  Phase 1被试: {CONFIG['phase1']['n_subjects']}人 × {CONFIG['phase1']['trials_per_subject']}次")
    print(f"  Phase 2被试: {CONFIG['phase2']['n_subjects']}人 × {CONFIG['phase2']['trials_per_subject']}次")
    print(f"  模拟模式: {'是' if CONFIG['simulation']['enabled'] else '否'}")
    print()

    # 检查设计空间文件
    design_path = Path(CONFIG["design_csv_path"])
    if not design_path.exists():
        print(f"[警告] 设计空间文件不存在: {CONFIG['design_csv_path']}")
        print("将创建示例设计空间...")
        _create_example_design_space(design_path)

    try:
        # 初始化规划器
        planner = TwoPhaseExperimentPlanner(str(design_path))

        # 执行流程
        phase1_plan = step1_plan_phase1(planner)
        X, y, subject_ids = step2_collect_phase1_data(phase1_plan)

        if X is not None:
            phase1_analysis = step3_analyze_phase1(planner, X, y, subject_ids)

            if phase1_analysis is not None:
                phase2_plan = step4_plan_phase2(planner)
                exported_files = step5_export_outputs(planner)
                step6_summary(phase1_plan, phase2_plan, exported_files)
            else:
                print("[错误] Phase 1分析失败")
        else:
            print("[错误] 未收集Phase 1数据")
            print("请设置 CONFIG['simulation']['enabled'] = True 或提供实际数据")

    except Exception as e:
        print()
        print(f"[错误] {str(e)}")
        import traceback
        traceback.print_exc()


def _create_example_design_space(path: Path):
    """创建示例设计空间（如果不存在）"""
    factors = {
        "density": [1, 2, 3, 4, 5],
        "height": [1, 2, 3, 4, 5],
        "greenery": [1, 2, 3, 4, 5],
        "street_width": [1, 2, 3, 4, 5],
        "landmark": [1, 2, 3, 4, 5],
        "style": [1, 2, 3, 4, 5],
    }

    np.random.seed(42)
    n_configs = 1200
    design_df = pd.DataFrame({
        factor: np.random.choice(levels, size=n_configs)
        for factor, levels in factors.items()
    })

    path.parent.mkdir(parents=True, exist_ok=True)
    design_df.to_csv(path, index=False)
    print(f"[创建] 示例设计空间已创建: {path}")
    print()


if __name__ == "__main__":
    main()
