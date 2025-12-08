"""
测试独立工作流（非交互式演示）
演示如何使用warmup_sampler.py和analyze_phase1.py完成完整流程
"""

import numpy as np
import pandas as pd
from pathlib import Path

from warmup_sampler import WarmupSampler
from analyze_phase1 import Phase1DataAnalyzer


def create_example_design_space(path: str = "test_design_space.csv"):
    """创建示例设计空间"""
    print("[1/6] 创建示例设计空间...")

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

    design_df.to_csv(path, index=False)
    print(f"  [OK] 设计空间已创建: {path}")
    print(f"  配置数: {len(design_df)}")
    print(f"  因子数: {len(factors)}")
    print()

    return path


def step1_generate_warmup_samples(design_csv: str):
    """步骤1: 生成预热采样方案（非交互式）"""
    print("[2/6] 生成预热采样方案...")

    sampler = WarmupSampler(design_csv)

    # 评估预算
    n_subjects = 14
    trials_per_subject = 25
    skip_interaction = False

    adequacy, budget = sampler.evaluate_budget(
        n_subjects=n_subjects,
        trials_per_subject=trials_per_subject,
        skip_interaction=skip_interaction,
    )

    # 生成采样文件
    exported_files = sampler.generate_samples(
        budget=budget,
        output_dir="test_sample",
        merge=True,  # 合并为单个CSV以便测试
        subject_col_name="subject_id",
    )

    print(f"  [OK] 采样文件已生成")
    print(f"  导出的文件数: {len(exported_files)}")
    print()

    return "test_sample/warmup_samples_all.csv", budget


def step2_simulate_experiments(sample_csv: str):
    """步骤2: 模拟执行实验（实际流程中由用户完成）"""
    print("[3/6] 模拟执行实验...")

    # 加载采样方案
    samples_df = pd.read_csv(sample_csv)

    # 提取因子列
    factor_cols = [col for col in samples_df.columns if col != "subject_id"]
    X = samples_df[factor_cols].values

    # 模拟响应（包含主效应和交互）
    n_samples = len(X)
    y = np.zeros(n_samples)

    # 主效应
    main_effects = [0.3, 0.2, 0.25, 0.15, 0.1, 0.2]
    for i, coef in enumerate(main_effects):
        y += coef * X[:, i]

    # 真实交互: density×greenery, height×style
    y += 0.4 * X[:, 0] * X[:, 2]  # density × greenery
    y += 0.3 * X[:, 1] * X[:, 5]  # height × style

    # 噪声
    y += np.random.randn(n_samples) * 0.5

    # 添加响应列
    samples_df['response'] = y

    # 保存实验数据
    output_path = "test_warmup_data.csv"
    samples_df.to_csv(output_path, index=False)

    print(f"  [OK] 模拟实验完成")
    print(f"  实验数据已保存: {output_path}")
    print(f"  真实交互: (density, greenery), (height, style)")
    print()

    return output_path


def step3_analyze_phase1_data(data_csv: str):
    """步骤3: 分析Phase 1数据（非交互式）"""
    print("[4/6] 分析Phase 1数据...")

    analyzer = Phase1DataAnalyzer(
        data_csv_path=data_csv,
        subject_col="subject_id",
        response_col="response",
    )

    # 执行分析
    analysis = analyzer.analyze(
        max_pairs=5,
        min_pairs=3,
        selection_method="elbow",
        verbose=True,
    )

    print(f"  [OK] Phase 1分析完成")
    print(f"  筛选出的交互对: {len(analysis['selected_pairs'])}个")
    print()

    return analyzer, analysis


def step4_generate_phase2_config(analyzer, analysis):
    """步骤4: 生成Phase 2配置（非交互式）"""
    print("[5/6] 生成Phase 2配置...")

    phase2_config = analyzer.generate_phase2_config(
        n_subjects=18,
        trials_per_subject=25,
        lambda_adjustment=1.2,
    )

    print(f"  [OK] Phase 2配置生成完成")
    print(f"  总预算: {phase2_config['total_budget']}次")
    print(f"  交互对: {phase2_config['interaction_pairs']}")
    print(f"  λ: {phase2_config['lambda_init']:.3f} → {phase2_config['lambda_end']:.3f}")
    print(f"  γ: {phase2_config['gamma_init']:.3f} → {phase2_config['gamma_end']:.3f}")
    print()

    return phase2_config


def step5_export_reports(analyzer, phase2_config):
    """步骤5: 导出报告（非交互式）"""
    print("[6/6] 导出报告...")

    exported_files = analyzer.export_report(
        phase2_config=phase2_config,
        output_dir="test_phase1_analysis_output",
        prefix="test_phase1",
    )

    print(f"  [OK] 报告导出完成")
    print()

    return exported_files


def step6_verify_outputs(exported_files):
    """步骤6: 验证输出文件"""
    print("=" * 80)
    print("验证输出文件")
    print("=" * 80)
    print()

    import json

    # 验证JSON文件
    json_path = exported_files['json_config']
    print(f"[验证] {json_path}")
    with open(json_path) as f:
        config = json.load(f)
    print(f"  交互对: {config['interaction_pairs']}")
    print(f"  λ初始: {config['lambda_init']}")
    print(f"  γ初始: {config['gamma_init']}")
    print(f"  总预算: {config['total_budget']}")
    print()

    # 验证NPZ文件
    npz_path = exported_files['npz_schedules']
    print(f"[验证] {npz_path}")
    data = np.load(npz_path)
    print(f"  lambda_schedule形状: {data['lambda_schedule'].shape}")
    print(f"  gamma_schedule形状: {data['gamma_schedule'].shape}")
    print(f"  interaction_pairs: {data['interaction_pairs'].tolist()}")
    print()

    # 验证文本报告
    report_path = exported_files['txt_report']
    print(f"[验证] {report_path}")
    with open(report_path, encoding='utf-8') as f:
        lines = f.readlines()
    print(f"  报告行数: {len(lines)}")
    print(f"  前3行:")
    for line in lines[:3]:
        print(f"    {line.rstrip()}")
    print()


def main():
    """主测试流程"""
    print()
    print("=" * 80)
    print("独立工作流测试（非交互式演示）")
    print("=" * 80)
    print()
    print("本脚本演示如何使用warmup_sampler.py和analyze_phase1.py")
    print("完成从设计空间到Phase 2参数的完整流程")
    print()

    try:
        # 创建测试数据
        design_csv = create_example_design_space()

        # Step 1: 生成预热采样方案
        sample_csv, budget = step1_generate_warmup_samples(design_csv)

        # Step 2: 模拟执行实验（实际流程中由用户完成）
        data_csv = step2_simulate_experiments(sample_csv)

        # Step 3: 分析Phase 1数据
        analyzer, analysis = step3_analyze_phase1_data(data_csv)

        # Step 4: 生成Phase 2配置
        phase2_config = step4_generate_phase2_config(analyzer, analysis)

        # Step 5: 导出报告
        exported_files = step5_export_reports(analyzer, phase2_config)

        # Step 6: 验证输出
        step6_verify_outputs(exported_files)

        # 总结
        print("=" * 80)
        print("测试完成！")
        print("=" * 80)
        print()
        print("生成的文件:")
        print("  设计空间: test_design_space.csv")
        print("  采样方案: test_sample/warmup_samples_all.csv")
        print("  实验数据: test_warmup_data.csv")
        print("  Phase 2配置: test_phase1_analysis_output/test_phase1_phase2_config.json")
        print("  动态调度: test_phase1_analysis_output/test_phase1_phase2_schedules.npz")
        print("  分析报告: test_phase1_analysis_output/test_phase1_analysis_report.txt")
        print("  使用指南: test_phase1_analysis_output/PHASE2_USAGE_GUIDE.txt")
        print()
        print("下一步:")
        print("  1. 查看生成的文件，了解输出格式")
        print("  2. 参考 README_STANDALONE.md 了解实际使用流程")
        print("  3. 在实际项目中使用 warmup_sampler.py 和 analyze_phase1.py")
        print()

    except Exception as e:
        print()
        print(f"[错误] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
