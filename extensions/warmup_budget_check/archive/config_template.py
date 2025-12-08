"""
实验配置模板
复制此文件为 config.py 并根据你的实验需求修改参数
"""

# ============================================================================
# 实验设计配置
# ============================================================================

# 设计空间CSV文件路径
# 示例: "data/urban_design_space.csv"
DESIGN_CSV_PATH = "your_design_space.csv"


# ============================================================================
# Phase 1配置（预热阶段）
# ============================================================================

PHASE1_CONFIG = {
    # 被试数量
    # 建议: 14人（基于你的计划文档）
    # 范围: 5-20人（太少ICC估计不足，太多边际收益递减）
    "n_subjects": 14,

    # 每个被试的测试次数
    # 建议: 25次
    # 说明: Phase 1总预算 = n_subjects × trials_per_subject
    "trials_per_subject": 25,

    # 是否跳过交互效应探索（Core-2b）
    # False: 包含交互初筛（推荐）
    # True: 只关注主效应（如果你确定没有交互）
    "skip_interaction": False,
}


# ============================================================================
# Phase 1数据分析配置
# ============================================================================

PHASE1_ANALYSIS_CONFIG = {
    # 最多选择的交互对数量
    # 建议: 5个
    # 说明: 实际选择数量会根据数据自动确定（3-5个）
    "max_pairs": 5,

    # 最少选择的交互对数量
    # 建议: 3个
    # 说明: 即使数据显示交互不明显，也至少选3个
    "min_pairs": 3,

    # 交互对选择方法
    # 'elbow':        肘部法则，自动找拐点（推荐）
    # 'bic_threshold': 只选BIC显著的交互对
    # 'top_k':        固定选max_pairs个（不推荐）
    "selection_method": "elbow",
}


# ============================================================================
# Phase 2配置（主动学习阶段）
# ============================================================================

PHASE2_CONFIG = {
    # 被试数量
    # 建议: 18人（基于你的计划文档）
    # 说明: Phase 2被试数 = 总被试数 - Phase 1被试数
    "n_subjects": 18,

    # 每个被试的测试次数
    # 建议: 25次
    # 说明: 与Phase 1保持一致
    "trials_per_subject": 25,

    # λ调整系数
    # 1.0: 不调整，直接使用Phase 1估计
    # 1.2: 增强20%交互探索（推荐）
    # 1.5: 大幅增强交互探索（如果Phase 1显示强交互）
    # 说明: Phase 2初始λ = Phase 1估算λ × lambda_adjustment
    "lambda_adjustment": 1.2,
}


# ============================================================================
# 输出配置
# ============================================================================

OUTPUT_CONFIG = {
    # 输出目录
    "dir": "experiment_outputs",

    # 文件名前缀
    # 生成文件示例:
    #   - my_experiment.json
    #   - my_experiment_full.pkl
    #   - my_experiment_data.npz
    #   - my_experiment_report.txt
    "prefix": "my_experiment",
}


# ============================================================================
# 模拟数据配置（仅用于测试）
# ============================================================================

SIMULATION_CONFIG = {
    # 是否使用模拟数据
    # False: 实际实验模式，需要提供真实数据（推荐）
    # True:  模拟模式，自动生成测试数据
    "enabled": False,

    # 真实的交互对索引（仅用于模拟）
    # 示例: [(0, 2), (1, 4)] 表示 factor_0×factor_2 和 factor_1×factor_4
    "true_interactions": [(0, 2), (1, 4)],

    # 噪声水平（仅用于模拟）
    # 0.1: 低噪声
    # 0.5: 中等噪声（默认）
    # 1.0: 高噪声
    "noise_level": 0.5,
}


# ============================================================================
# 高级配置（一般无需修改）
# ============================================================================

ADVANCED_CONFIG = {
    # 是否显示详细输出
    "verbose": True,

    # Phase 2中期诊断位置（占总预算的比例）
    # 0.67: 在2/3处进行诊断（默认）
    "mid_diagnostic_ratio": 0.67,

    # 随机种子（用于可复现性）
    # None: 每次运行不同
    # 42: 固定种子，结果可复现
    "random_seed": 42,
}


# ============================================================================
# 配置验证
# ============================================================================

def validate_config():
    """验证配置的合理性"""
    errors = []
    warnings = []

    # 检查Phase 1
    if PHASE1_CONFIG["n_subjects"] < 3:
        errors.append("Phase 1被试数不能少于3人（ICC估计需要）")
    elif PHASE1_CONFIG["n_subjects"] < 5:
        warnings.append("Phase 1被试数少于5人，ICC估计可能不准确")

    if PHASE1_CONFIG["trials_per_subject"] < 20:
        warnings.append("每人测试次数少于20次，预算可能不足")

    # 检查Phase 2
    if PHASE2_CONFIG["n_subjects"] < 5:
        errors.append("Phase 2被试数不能少于5人")

    # 检查总预算
    total_budget = (
        PHASE1_CONFIG["n_subjects"] * PHASE1_CONFIG["trials_per_subject"] +
        PHASE2_CONFIG["n_subjects"] * PHASE2_CONFIG["trials_per_subject"]
    )
    if total_budget < 500:
        warnings.append(f"总预算{total_budget}次可能不足，建议≥500次")

    # 检查λ调整系数
    if PHASE2_CONFIG["lambda_adjustment"] < 0.5 or PHASE2_CONFIG["lambda_adjustment"] > 2.0:
        warnings.append("lambda_adjustment超出合理范围[0.5, 2.0]")

    # 显示结果
    if errors:
        print("[错误] 配置验证失败:")
        for e in errors:
            print(f"  - {e}")
        return False

    if warnings:
        print("[警告] 配置存在以下问题:")
        for w in warnings:
            print(f"  - {w}")
        print()

    return True


if __name__ == "__main__":
    print("配置验证:")
    print("-" * 80)
    if validate_config():
        print("[通过] 配置验证通过")
    else:
        print("[失败] 请修正错误后再运行")
