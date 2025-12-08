"""
预热采样预算估算 - 快速启动脚本

使用方法：
1. 修改下方的配置参数
2. 运行: python quick_start.py
"""

import sys
from pathlib import Path
from warmup_budget_estimator import WarmupBudgetEstimator

# ============================================================================
# 配置参数 - 在这里修改您的参数
# ============================================================================

# 设计空间CSV文件路径（相对于当前脚本的路径，或使用绝对路径）
DESIGN_CSV_PATH = "D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"

# 被试数量
N_SUBJECTS = 5


# 每个被试的测试次数
TRIALS_PER_SUBJECT = 25

# 是否跳过交互效应探索（Core-2b）
# True = 跳过交互效应，减少25次/人
# False = 包含交互效应探索（默认）
SKIP_INTERACTION = False

# 是否显示有/无交互效应的对比分析
SHOW_COMPARISON = True

# ============================================================================
# 以下代码无需修改
# ============================================================================


def main():
    """运行预算估算"""
    print("=" * 70)
    print("预热采样预算估算 - 快速启动")
    print("=" * 70)
    print()

    # 解析CSV路径
    csv_path = Path(DESIGN_CSV_PATH)
    if not csv_path.is_absolute():
        # 相对路径，相对于脚本所在目录
        script_dir = Path(__file__).parent
        csv_path = (script_dir / csv_path).resolve()

    # 检查文件是否存在
    if not csv_path.exists():
        print(f"[X] 错误: 设计空间文件不存在")
        print(f"    路径: {csv_path}")
        print()
        print("请检查 DESIGN_CSV_PATH 参数是否正确")
        sys.exit(1)

    print(f"[OK] 设计空间文件: {csv_path}")
    print()

    try:
        # 创建估算器
        estimator = WarmupBudgetEstimator(str(csv_path))

        # 分析设计空间
        estimator.analyze_design_space()

        # 打印预算报告
        estimator.print_budget_report(
            N_SUBJECTS,
            TRIALS_PER_SUBJECT,
            skip_interaction=SKIP_INTERACTION,
            show_comparison=SHOW_COMPARISON,
        )

        print()
        print("=" * 70)
        print("[OK] 分析完成")
        print("=" * 70)

    except Exception as e:
        print(f"[X] 错误: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
