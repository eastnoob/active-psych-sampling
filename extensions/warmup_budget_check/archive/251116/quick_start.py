"""
快速启动脚本 - 两阶段实验规划
只需修改下方配置参数，即可快速使用预热采样和数据分析功能

使用方法：
1. 修改下方的配置参数（STEP 1或STEP 2）
2. 选择要运行的步骤（MODE）
3. 运行: python quick_start.py
"""

import sys
from pathlib import Path
import time

# ============================================================================
# 配置参数 - 请根据需要修改
# ============================================================================

# 选择运行模式
# "step1" - 生成预热采样方案
# "step2" - 分析Phase 1数据并生成Phase 2参数
# "both"  - 连续运行两步（需要在步骤1和2之间手动执行实验）
MODE = "step1"  # 修改这里选择运行哪个步骤

# ----------------------------------------------------------------------------
# STEP 1 配置：生成预热采样方案
# ----------------------------------------------------------------------------
STEP1_CONFIG = {
    # 设计空间CSV路径（只包含自变量列）
    "design_csv_path": "D:\\WORKSPACE\\python\\aepsych-source\\data\\only_independences\\data\\only_independences\\6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
    # 预算参数
    "n_subjects": 5,  # 被试数量
    "trials_per_subject": 25,  # 每个被试的测试次数
    "skip_interaction": True,  # 是否跳过交互效应探索（False=包含交互）
    # 输出配置
    "output_dir": f"D:\\WORKSPACE\\python\\aepsych-source\\extensions\\warmup_budget_check\\sample\\{time.strftime('%Y%m%d%H%M')}",  # 输出目录（格式：YYYYMMDDhhmm）
    "merge": False,  # 是否合并为单个CSV（False=每个被试一个文件）
    "subject_col_name": "subject_id",  # 被试编号列名（仅在merge=True时使用）
    # 是否自动执行（False会询问确认）
    "auto_confirm": False,
}

# ----------------------------------------------------------------------------
# STEP 2 配置：分析Phase 1数据
# ----------------------------------------------------------------------------
STEP2_CONFIG = {
    # 实验数据CSV路径（包含响应列）
    "data_csv_path": "warmup_data.csv",
    # 列名
    "subject_col": "subject_id",  # 被试编号列名
    "response_col": "response",  # 响应变量列名
    # 分析参数
    "max_pairs": 5,  # 最多选择的交互对数量
    "min_pairs": 3,  # 最少选择的交互对数量
    "selection_method": "elbow",  # 选择方法：elbow/bic_threshold/top_k
    # Phase 2参数
    "phase2_n_subjects": 18,  # Phase 2被试数
    "phase2_trials_per_subject": 25,  # Phase 2每人测试次数
    "lambda_adjustment": 1.2,  # λ调整系数（1.0=不调整，1.2=增强20%交互探索）
    # 输出配置
    "output_dir": "phase1_analysis_output",
    "prefix": "phase1",
}


# ============================================================================
# 主程序 - 无需修改
# ============================================================================


def run_step1():
    """运行步骤1：生成预热采样方案"""
    from warmup_sampler import WarmupSampler

    print("=" * 80)
    print("步骤1：生成预热采样方案")
    print("=" * 80)
    print()

    # 检查设计空间文件
    design_path = Path(STEP1_CONFIG["design_csv_path"])
    if not design_path.exists():
        print(f"[错误] 设计空间文件不存在: {STEP1_CONFIG['design_csv_path']}")
        print()
        print("请确保CSV文件存在，且包含所有因子列（只有自变量，不包含因变量）")
        print("示例格式:")
        print("  density,height,greenery,street_width,landmark,style")
        print("  1,1,1,1,1,1")
        print("  1,1,1,1,1,2")
        print("  ...")
        sys.exit(1)

    # 初始化采样器
    try:
        sampler = WarmupSampler(STEP1_CONFIG["design_csv_path"])
    except Exception as e:
        print(f"[错误] 加载设计空间失败: {e}")
        sys.exit(1)

    # 评估预算
    print("当前配置:")
    print(f"  被试数: {STEP1_CONFIG['n_subjects']}人")
    print(f"  每人trials: {STEP1_CONFIG['trials_per_subject']}次")
    print(
        f"  总预算: {STEP1_CONFIG['n_subjects'] * STEP1_CONFIG['trials_per_subject']}次"
    )
    print(f"  跳过交互: {'是' if STEP1_CONFIG['skip_interaction'] else '否'}")
    print()

    adequacy, budget = sampler.evaluate_budget(
        n_subjects=STEP1_CONFIG["n_subjects"],
        trials_per_subject=STEP1_CONFIG["trials_per_subject"],
        skip_interaction=STEP1_CONFIG["skip_interaction"],
    )

    # 询问确认（如果需要）
    if not STEP1_CONFIG["auto_confirm"]:
        if adequacy in ["预算不足", "严重不足"]:
            print(f"[!] 预算评估为【{adequacy}】，不建议继续")
            confirm = input("是否仍要生成采样方案？(y/N): ").strip().lower()
            if confirm != "y":
                print("[取消] 已退出")
                sys.exit(0)
        else:
            confirm = input("是否生成采样方案？(Y/n): ").strip().lower()
            if confirm == "n":
                print("[取消] 已退出")
                sys.exit(0)

    # 生成采样文件
    try:
        exported_files = sampler.generate_samples(
            budget=budget,
            output_dir=STEP1_CONFIG["output_dir"],
            merge=STEP1_CONFIG["merge"],
            subject_col_name=STEP1_CONFIG["subject_col_name"],
        )

        print("[OK] 采样方案生成成功！")
        print(f"  文件数: {len(exported_files)}")
        print(f"  保存位置: {STEP1_CONFIG['output_dir']}/")
        print()
        print("=" * 80)
        print("下一步：")
        print("  1. 按照生成的CSV文件执行实验")
        print("  2. 收集响应数据（因变量）")
        print("  3. 将响应值添加到CSV中")
        print("  4. 运行 python quick_start.py（设置 MODE='step2'）")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"[错误] 生成采样文件失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_step2():
    """运行步骤2：分析Phase 1数据"""
    from analyze_phase1 import Phase1DataAnalyzer

    print("=" * 80)
    print("步骤2：分析Phase 1数据")
    print("=" * 80)
    print()

    # 检查数据文件
    data_path = Path(STEP2_CONFIG["data_csv_path"])
    if not data_path.exists():
        print(f"[错误] 数据文件不存在: {STEP2_CONFIG['data_csv_path']}")
        print()
        print("请确保CSV文件存在，且包含以下列:")
        print(f"  - 被试编号列: {STEP2_CONFIG['subject_col']}")
        print(f"  - 响应变量列: {STEP2_CONFIG['response_col']}")
        print("  - 所有因子列")
        print()
        print("示例格式:")
        print("  subject_id,density,height,greenery,...,response")
        print("  1,3,2,5,...,7.2")
        print("  1,1,5,3,...,8.1")
        print("  ...")
        sys.exit(1)

    # 初始化分析器
    try:
        analyzer = Phase1DataAnalyzer(
            data_csv_path=STEP2_CONFIG["data_csv_path"],
            subject_col=STEP2_CONFIG["subject_col"],
            response_col=STEP2_CONFIG["response_col"],
        )
    except Exception as e:
        print(f"[错误] 加载数据失败: {e}")
        sys.exit(1)

    # 执行分析
    print("分析参数:")
    print(f"  交互对范围: {STEP2_CONFIG['min_pairs']}-{STEP2_CONFIG['max_pairs']}个")
    print(f"  选择方法: {STEP2_CONFIG['selection_method']}")
    print()

    try:
        analysis = analyzer.analyze(
            max_pairs=STEP2_CONFIG["max_pairs"],
            min_pairs=STEP2_CONFIG["min_pairs"],
            selection_method=STEP2_CONFIG["selection_method"],
            verbose=True,
        )
    except Exception as e:
        print(f"[错误] 分析失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 生成Phase 2配置
    print()
    print("Phase 2配置参数:")
    print(f"  被试数: {STEP2_CONFIG['phase2_n_subjects']}人")
    print(f"  每人trials: {STEP2_CONFIG['phase2_trials_per_subject']}次")
    print(f"  λ调整系数: {STEP2_CONFIG['lambda_adjustment']}")
    print()

    try:
        phase2_config = analyzer.generate_phase2_config(
            n_subjects=STEP2_CONFIG["phase2_n_subjects"],
            trials_per_subject=STEP2_CONFIG["phase2_trials_per_subject"],
            lambda_adjustment=STEP2_CONFIG["lambda_adjustment"],
        )

        print("Phase 2配置:")
        print(f"  总预算: {phase2_config['total_budget']}次")
        print(f"  筛选的交互对: {len(phase2_config['interaction_pairs'])}个")
        print(
            f"  λ: {phase2_config['lambda_init']:.3f} -> {phase2_config['lambda_end']:.3f}"
        )
        print(
            f"  γ: {phase2_config['gamma_init']:.3f} -> {phase2_config['gamma_end']:.3f}"
        )
        print(f"  中期诊断: 第{phase2_config['mid_diagnostic_trial']}次trial")
        print()

    except Exception as e:
        print(f"[错误] 生成Phase 2配置失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 导出报告
    try:
        exported_files = analyzer.export_report(
            phase2_config=phase2_config,
            output_dir=STEP2_CONFIG["output_dir"],
            prefix=STEP2_CONFIG["prefix"],
        )

        print("[OK] 分析完成！")
        print()
        print("=" * 80)
        print("下一步：")
        print("  1. 查看分析报告:")
        print(f"     {exported_files['txt_report']}")
        print("  2. 阅读Phase 2使用指南:")
        print(f"     {exported_files['usage_guide']}")
        print("  3. 在EUR-ANOVA中加载配置:")
        print(f"     - JSON: {exported_files['json_config']}")
        print(f"     - NumPy: {exported_files['npz_schedules']}")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"[错误] 导出报告失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main():
    """主函数"""
    print()
    print("=" * 80)
    print("两阶段实验规划 - 快速启动")
    print("=" * 80)
    print()

    if MODE == "step1":
        run_step1()
    elif MODE == "step2":
        run_step2()
    elif MODE == "both":
        print("[模式] 连续运行两步")
        print()
        run_step1()
        print()
        print("=" * 80)
        print("请先执行实验，收集响应数据后，再继续运行步骤2")
        print("=" * 80)
        print()
        input("按Enter继续运行步骤2...")
        print()
        run_step2()
    else:
        print(f"[错误] 未知的模式: {MODE}")
        print("请设置 MODE 为 'step1', 'step2' 或 'both'")
        sys.exit(1)


if __name__ == "__main__":
    main()
