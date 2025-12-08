"""
快速启动脚本 - 两阶段实验规划
只需修改下方配置参数，即可快速使用预热采样和数据分析功能

使用方法：
1. 修改下方的配置参数（STEP 1或STEP 2）
2. 选择要运行的步骤（MODE）
3. 运行: python quick_start.py

🔧 本脚本现已支持新的 API 接口（向后兼容）
   - 内部使用 config_models.Step1Config, Step2Config, Step3Config
   - 保持原有的配置字典格式，无需修改现有代码
   - 可选使用新的流程管理器进行链式调用
"""

import sys
from pathlib import Path
import time
from typing import Dict, Any, Optional

# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent / "core"))

# 新增：导入配置模型和 API
try:
    from core.config_models import Step1Config, Step2Config, Step3Config
    from core.warmup_api import (
        run_step1 as api_run_step1,
        run_step2 as api_run_step2,
        run_step3 as api_run_step3,
        Step1Step2Chain,
        Step1Step2Step3Chain,
    )

    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    print("[警告] 新的 API 模块不可用，将使用传统实现")

# ============================================================================
# 配置参数 - 请根据需要修改
# ============================================================================

# 选择运行模式
# "step1"      - 生成预热采样方案
# "step1.5"    - 模拟被试作答（可选，用于测试）
# "step2"      - 分析Phase 1数据并生成Phase 2参数
# "step3"      - 训练 Base GP 并扫描设计空间
# "step2+3"    - 整合分析：Step 2 + Step 3，生成统一报告（推荐）✨
# "both"       - 步骤1 -> 手动实验 -> 步骤2
# "all"        - 步骤1 -> 步骤1.5(模拟) -> 步骤2 -> 步骤3
# "chain12"    - 使用流程管理器运行步骤1->2（推荐）
# "chain123"   - 使用流程管理器运行步骤1->2->3（推荐）
MODE = "all"  # 运行 Step1 -> Step1.5(模拟) -> Step2 -> Step3

# ----------------------------------------------------------------------------
# ALL 模式专用配置：统一控制所有步骤的参数（推荐使用）
# ----------------------------------------------------------------------------
# ⭐ 在 ALL_CONFIG 中设置的参数会覆盖下方各 STEP 配置，实现统一管理
ALL_CONFIG = {
    # ==================== 流程控制 ====================
    "base_output_dir": str(Path(__file__).parent / "phase1_analysis_output"),
    "run_step1_5": True,  # 是否自动运行模拟（Step1.5）
    "step1_5_use_result_dir_for_step2": True,  # Step2是否使用Step1.5的结果
    # ==================== 设计空间与预算 ====================
    # 设计空间CSV（Step1和Step3共用）
    "design_csv": str(
        Path(__file__).parent.parent.parent
        / "data"
        / "only_independences"
        / "data"
        / "only_independences"
        / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    ),
    # Phase 1 预算
    "n_subjects": 5,  # Phase 1 被试数量
    "trials_per_subject": 30,  # Phase 1 每个被试的测试次数
    "skip_interaction": False,  # 是否跳过交互效应探索
    "auto_confirm": True,  # 是否自动确认（True=不询问）
    # ==================== 模拟被试参数 (Step1.5) ====================
    "seed": 42,  # 随机种子
    "population_mean": 0.0,  # 群体权重均值
    "population_std": 0.25,  # 群体权重标准差 (降低基础方差！)
    "individual_std_percent": 0.5,  # 个体差异比例 (0.5×0.25=0.125)
    "individual_corr": 0.0,  # 特征间相关
    # Likert输出配置
    "likert_levels": 5,  # Likert量表等级数
    "likert_mode": "tanh",  # tanh=拟真分布 / percentile=均匀分布
    "likert_sensitivity": 2.0,  # Likert灵敏度 (推荐: 1.5-2.5)
    # 交互效应
    "interaction_pairs": [(3, 4), (0, 1)],  # 指定交互对 (索引从0开始)
    "num_interactions": 0,  # 额外随机生成的交互项数
    "interaction_scale": 0.25,  # 交互权重尺度 (推荐: 0.2-0.3)
    # ==================== Phase 2 参数 (Step2) ====================
    "max_pairs": 5,  # 最多选择的交互对数量
    "min_pairs": 2,  # 最少选择的交互对数量
    "selection_method": "elbow",  # 交互对选择方法: elbow/bic_threshold/top_k
    "phase2_n_subjects": 20,  # Phase 2 被试数量
    "phase2_trials_per_subject": 25,  # Phase 2 每人测试次数
    "lambda_adjustment": 1.2,  # λ调整系数 (1.0=不调整, >1.0=增强交互探索)
    # ==================== Base GP 参数 (Step3) ====================
    "max_iters": 200,  # GP训练迭代数 (测试用200, 正式用300+)
    "learning_rate": 0.05,  # 学习率
    "use_cuda": False,  # 是否使用GPU
    "ensure_diversity": True,  # 确保采样点多样性
}

# ==================== 高级用户选项 ====================
# 如果需要更细粒度的控制，可以在 ALL_CONFIG 中嵌套子字典:
# ALL_CONFIG["step1"] = {"merge": True}  # 覆盖 Step1 特定参数
# ALL_CONFIG["step1_5"] = {"output_mode": "both"}  # 覆盖 Step1.5 特定参数
# ALL_CONFIG["step2"] = {"report_format": "txt"}  # 覆盖 Step2 特定参数
# ALL_CONFIG["step3"] = {"max_iters": 500}  # 覆盖 Step3 特定参数


# ----------------------------------------------------------------------------
# STEP 1 配置：生成预热采样方案
# ----------------------------------------------------------------------------
STEP1_CONFIG = {
    # 设计空间CSV路径（只包含自变量列）
    "design_csv_path": str(
        Path(__file__).parent.parent.parent
        / "data"
        / "only_independences"
        / "data"
        / "only_independences"
        / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    ),
    # 预算参数
    "n_subjects": 5,  # 被试数量
    "trials_per_subject": 30,  # 每个被试的测试次数
    "skip_interaction": False,  # 是否跳过交互效应探索（False=包含交互）
    # 输出配置
    "output_dir": str(
        Path(__file__).parent / "sample" / time.strftime("%Y%m%d%H%M")
    ),  # 输出目录（格式：YYYYMMDDhhmm）
    "merge": False,  # 是否合并为单个CSV（False=每个被试一个文件）
    "subject_col_name": "subject_id",  # 被试编号列名（仅在merge=True时使用）
    # 是否自动执行（False会询问确认）
    "auto_confirm": False,
}

# ----------------------------------------------------------------------------
# STEP 1.5 配置：模拟被试作答（可选，用于测试流程）
# ----------------------------------------------------------------------------
STEP1_5_CONFIG = {
    # 输入：Step 1生成的采样方案目录
    "input_dir": str(
        Path(__file__).parent / "sample" / "202511302204"
    ),  # Step 1输出目录（默认），在 MODE='all' 时会被覆盖为 Step1 的输出目录
    # 模拟参数
    "seed": 42,  # 随机种子
    "output_mode": "combined",  # individual/combined/both
    "use_latent": False,  # 是否使用潜变量模型
    "output_type": "likert",  # continuous/likert
    "likert_levels": 5,
    "likert_mode": "tanh",  # tanh=真实人类分布（中心偏多）/ percentile=强制均匀
    "likert_sensitivity": 2.0,  # >1使分布更集中于中间值（更拟真）
    # 被试参数 ⭐调整为更拟真的参数
    "population_mean": 0.0,
    "population_std": 0.4,  # 群体权重分布范围
    "individual_std_percent": 0.4,  # 个体差异=40% (0.4×0.4=0.16, 推荐值，降低被试间差异)
    "individual_corr": 0.0,  # 特征间相关
    # 交互效应 ⭐减少交互对数量
    "interaction_pairs": [(3, 4), (0, 1)],  # 减少到2个交互对（更常见）
    "num_interactions": 0,  # 额外随机生成的交互项数
    "interaction_scale": 0.25,  # 降低交互强度（0.4→0.25）
    # 输出配置
    "clean": True,  # 清理之前的结果
    # 模型显示与保存 ⭐新增
    "print_model": True,  # 是否在控制台打印模型规格
    "save_model_summary": True,  # 是否保存模型摘要到单独文件
    "model_summary_format": "txt",  # txt/md/both - 模型摘要格式
}

# ----------------------------------------------------------------------------
# STEP 2 配置：分析Phase 1数据
# ----------------------------------------------------------------------------
STEP2_CONFIG = {
    # ========== 实验数据路径（二选一，注释掉不用的） ==========
    #
    # 【方式1】目录模式 - 自动读取所有 subject_*.csv（推荐）
    #   - 优点: 直接使用 Step 1.5 的 result 目录，无需手动合并
    #   - 理解: 每个 subject_*.csv 文件代表一个被试的数据
    #   - subject列会自动从文件名生成 (subject_1, subject_2, ...)
    "data_csv_path": str(Path(__file__).parent / "sample" / "202511302204" / "result"),
    # 【方式2】文件模式 - 读取单个合并CSV
    #   - 适用: 已经手动合并了所有被试数据
    #   - 要求: CSV中必须包含 subject 列和响应列
    # "data_csv_path": str(Path(__file__).parent / "sample" / "202511302204" / "result" / "combined_results.csv"),
    # 列名配置
    "subject_col": "subject",  # 被试编号列名
    "response_col": "y",  # 响应变量列名
    # 分析参数
    "max_pairs": 5,  # 最多选择的交互对数量
    "min_pairs": 2,  # 最少选择的交互对数量
    "selection_method": "elbow",  # 选择方法：elbow/bic_threshold/top_k
    # Phase 2参数
    "phase2_n_subjects": 20,  # Phase 2被试数
    "phase2_trials_per_subject": 25,  # Phase 2每人测试次数
    "lambda_adjustment": 1.2,  # λ调整系数（1.0=不调整，1.2=增强20%交互探索）
    # 输出配置
    "output_dir": str(Path(__file__).parent / "step2" / time.strftime("%Y%m%d%H%M")),
    "prefix": "phase1",
    "report_format": "md",  # 报告格式：'md'(默认) 或 'txt'
}

# ----------------------------------------------------------------------------
# STEP 3 配置：Base GP (Matern2.5 + ARD) 与设计空间扫描
# ----------------------------------------------------------------------------
STEP3_CONFIG = {
    # ========== Phase1 数据路径（二选一，注释掉不用的） ==========
    #
    # 【方式1】目录模式 - 自动读取所有 subject_*.csv（推荐）
    #   - 优点: 直接使用 Step 1.5 的 result 目录
    #   - 理解: 每个 subject_*.csv 文件代表一个被试的数据
    #   - subject列会自动从文件名生成
    "data_csv_path": str(Path(__file__).parent / "sample" / "202511302204" / "result"),
    # 【方式2】文件模式 - 读取单个合并CSV
    #   - 适用: 已经手动合并了所有被试数据
    #   - 要求: CSV中必须包含 subject 列和响应列
    # "data_csv_path": str(Path(__file__).parent / "sample" / "202511302204" / "result" / "combined_results.csv"),
    # 列名配置
    "subject_col": "subject",  # 被试列
    "response_col": "y",  # 响应列
    # 设计空间 CSV (只含自变量列，与 Phase1 因子同名)
    "design_space_csv": str(
        Path(__file__).parent.parent.parent
        / "data"
        / "only_independences"
        / "data"
        / "only_independences"
        / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    ),
    # 训练参数
    "max_iters": 200,  # 测试快速迭代，可根据需要提高到300
    "learning_rate": 0.05,
    "use_cuda": False,
    # 采样多样性检查：若为True，确保Sample 3与Sample 1/2不重复；若冲突，选Std第二高的点
    "ensure_diversity": True,
    # 输出目录
    "output_dir": str(
        Path(__file__).parent
        / "phase1_analysis_output"
        / time.strftime("%Y%m%d%H%M")
        / "base_gp"
    ),
}


# ============================================================================
# 配置转换辅助函数
# ============================================================================


def _dict_to_step1_config(config_dict: Dict[str, Any]) -> Step1Config:
    """将字典配置转换为 Step1Config 对象"""
    return Step1Config(
        design_csv_path=config_dict["design_csv_path"],
        n_subjects=config_dict["n_subjects"],
        trials_per_subject=config_dict["trials_per_subject"],
        skip_interaction=config_dict.get("skip_interaction", True),
        output_dir=config_dict.get("output_dir", None),
        merge=config_dict.get("merge", False),
        subject_col_name=config_dict.get("subject_col_name", "subject_id"),
        auto_confirm=config_dict.get("auto_confirm", True),
    )


def _dict_to_step2_config(config_dict: Dict[str, Any]) -> Step2Config:
    """将字典配置转换为 Step2Config 对象"""
    return Step2Config(
        data_csv_path=config_dict["data_csv_path"],
        subject_col=config_dict.get("subject_col", "subject"),
        response_col=config_dict.get("response_col", "y"),
        max_pairs=config_dict.get("max_pairs", 5),
        min_pairs=config_dict.get("min_pairs", 1),
        selection_method=config_dict.get("selection_method", "elbow"),
        phase2_n_subjects=config_dict.get("phase2_n_subjects", 20),
        phase2_trials_per_subject=config_dict.get("phase2_trials_per_subject", 25),
        lambda_adjustment=config_dict.get("lambda_adjustment", 1.0),
        output_dir=config_dict.get("output_dir", None),
        prefix=config_dict.get("prefix", "phase1"),
        report_format=config_dict.get("report_format", "md"),
    )


def _dict_to_step3_config(config_dict: Dict[str, Any]) -> Step3Config:
    """将字典配置转换为 Step3Config 对象"""
    return Step3Config(
        data_csv_path=config_dict["data_csv_path"],
        design_space_csv=config_dict["design_space_csv"],
        subject_col=config_dict.get("subject_col", "subject"),
        response_col=config_dict.get("response_col", "y"),
        max_iters=config_dict.get("max_iters", 300),
        learning_rate=config_dict.get("learning_rate", 0.01),
        use_cuda=config_dict.get("use_cuda", False),
        ensure_diversity=config_dict.get("ensure_diversity", True),
        output_dir=config_dict.get("output_dir", None),
    )


def _apply_all_config() -> None:
    """
    将 ALL_CONFIG 中的全局设置合并到各 STEP 配置字典中。

    支持两种覆盖模式：
      1. 嵌套字典模式：ALL_CONFIG['step1'] = {...} 直接覆盖对应配置
      2. 顶级参数模式：ALL_CONFIG['n_subjects'] 自动分发到相关配置
    """
    # ========== 1. 嵌套字典模式：直接合并 ==========
    for key, target in (
        ("step1", STEP1_CONFIG),
        ("step1_5", STEP1_5_CONFIG),
        ("step2", STEP2_CONFIG),
        ("step3", STEP3_CONFIG),
    ):
        if isinstance(ALL_CONFIG.get(key), dict):
            target.update(ALL_CONFIG[key])

    # ========== 2. 顶级参数模式：智能分发 ==========

    # --- Step1 参数 ---
    for param in [
        "n_subjects",
        "trials_per_subject",
        "skip_interaction",
        "auto_confirm",
    ]:
        if param in ALL_CONFIG:
            STEP1_CONFIG[param] = ALL_CONFIG[param]

    # 设计空间文件
    if "design_csv" in ALL_CONFIG:
        STEP1_CONFIG["design_csv_path"] = ALL_CONFIG["design_csv"]
        STEP3_CONFIG["design_space_csv"] = ALL_CONFIG["design_csv"]  # Step3也用同一个

    # --- Step1.5 参数 ---
    step1_5_params = [
        "seed",
        "population_mean",
        "population_std",
        "individual_std_percent",
        "individual_corr",
        "likert_levels",
        "likert_mode",
        "likert_sensitivity",
        "interaction_pairs",
        "num_interactions",
        "interaction_scale",
    ]
    for param in step1_5_params:
        if param in ALL_CONFIG:
            STEP1_5_CONFIG[param] = ALL_CONFIG[param]

    # --- Step2 参数 ---
    step2_params = [
        "max_pairs",
        "min_pairs",
        "selection_method",
        "phase2_n_subjects",
        "phase2_trials_per_subject",
        "lambda_adjustment",
    ]
    for param in step2_params:
        if param in ALL_CONFIG:
            STEP2_CONFIG[param] = ALL_CONFIG[param]

    # --- Step3 参数 ---
    step3_params = ["max_iters", "learning_rate", "use_cuda", "ensure_diversity"]
    for param in step3_params:
        if param in ALL_CONFIG:
            STEP3_CONFIG[param] = ALL_CONFIG[param]

    # --- 特殊覆盖（向后兼容旧版配置） ---
    if "step2_data_csv" in ALL_CONFIG:
        STEP2_CONFIG["data_csv_path"] = ALL_CONFIG["step2_data_csv"]

    if "step3_design_space_csv" in ALL_CONFIG:
        STEP3_CONFIG["design_space_csv"] = ALL_CONFIG["step3_design_space_csv"]


# 立即应用 ALL_CONFIG 的覆盖（如果用户希望把所有配置放到 ALL_CONFIG 中）
_apply_all_config()


# ============================================================================
# 主程序 - 无需修改
# ============================================================================


def run_step1():
    """运行步骤1：生成预热采样方案"""
    print("=" * 80)
    print("步骤1：生成预热采样方案")
    print("=" * 80)
    print()

    # 使用新的 API（如果可用）
    if API_AVAILABLE:
        try:
            config = _dict_to_step1_config(STEP1_CONFIG)
            result = api_run_step1(config)

            print("[OK] 采样方案生成成功！")
            print(f"  文件数: {len(result.exported_files)}")
            print(f"  保存位置: {result.output_dir}/")
            print(f"  预算评估: {result.budget_adequacy}")
            print()
            print("=" * 80)
            print("下一步：")
            print("  1. 按照生成的CSV文件执行实验")
            print("  2. 收集响应数据（因变量）")
            print("  3. 将响应值添加到CSV中")
            print("  4. 运行 python quick_start.py（设置 MODE='step2'）")
            print("=" * 80)
            print()
            return

        except Exception as e:
            print(f"[警告] 新 API 运行失败，回退到传统实现: {e}")

    # 传统实现（向后兼容）
    from core.warmup_sampler import WarmupSampler

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
    print("=" * 80)
    print("步骤2：分析Phase 1数据")
    print("=" * 80)
    print()

    # 使用新的 API（如果可用）
    if API_AVAILABLE:
        try:
            config = _dict_to_step2_config(STEP2_CONFIG)
            result = api_run_step2(config)

            print("[OK] 分析完成！")
            print(f"  筛选的交互对: {len(result['analysis']['selected_pairs'])}个")
            print(f"  总预算: {result['phase2_config']['total_budget']}次")
            print(
                f"  λ: {result['phase2_config']['lambda_init']:.3f} -> {result['phase2_config']['lambda_end']:.3f}"
            )
            print(
                f"  γ: {result['phase2_config']['gamma_init']:.3f} -> {result['phase2_config']['gamma_end']:.3f}"
            )
            print(
                f"  中期诊断: 第{result['phase2_config']['mid_diagnostic_trial']}次trial"
            )
            print()
            print("=" * 80)
            print("下一步：")
            print("  1. 查看分析报告:")
            print(f"     {result['files']['report']}")
            print("  2. 阅读Phase 2使用指南:")
            print(f"     {result['files']['usage_guide']}")
            print("  3. 在EUR-ANOVA中加载配置:")
            print(f"     - JSON: {result['files']['json_config']}")
            print(f"     - NumPy: {result['files']['npz_schedules']}")
            print("=" * 80)
            print()
            return

        except Exception as e:
            print(f"[警告] 新 API 运行失败，回退到传统实现: {e}")

    # 传统实现（向后兼容）
    from core.analyze_phase1 import Phase1DataAnalyzer

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
            report_format=STEP2_CONFIG.get("report_format", "md"),
        )

        print("[OK] 分析完成！")
        print()
        print("=" * 80)
        print("下一步：")
        print("  1. 查看分析报告:")
        print(f"     {exported_files['report']}")
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


def run_step3():
    """运行步骤3：训练 Base GP & 扫描设计空间"""
    print("=" * 80)
    print("步骤3：Base GP 训练与设计空间扫描")
    print("=" * 80)
    print()

    # 使用新的 API（如果可用）
    if API_AVAILABLE:
        try:
            config = _dict_to_step3_config(STEP3_CONFIG)
            result = api_run_step3(config)

            print("[OK] Base GP 训练与扫描完成")
            print(f"  输出目录: {result.output_dir}")
            print(f"  设计空间点数: {result.n_design_points}")
            print("  关键点: 保存于 base_gp_key_points.json")
            print("  长度尺度: base_gp_lengthscales.json")
            print("  报告: base_gp_report.md")
            print()
            print("下一步：可在 Phase 2 模型中加载 base_gp_state.pth 作为形状函数先验")
            return

        except Exception as e:
            print(f"[警告] 新 API 运行失败，回退到传统实现: {e}")

    # 传统实现（向后兼容）
    from core.phase1_step3_base_gp import process_step3

    cfg = STEP3_CONFIG
    print("配置参数:")
    print(f"  Phase1数据: {cfg['data_csv_path']}")
    print(f"  设计空间:   {cfg['design_space_csv']}")
    print(f"  被试列:     {cfg['subject_col']}")
    print(f"  响应列:     {cfg['response_col']}")
    print(f"  迭代数:     {cfg['max_iters']}")
    print(f"  学习率:     {cfg['learning_rate']}")
    print(f"  使用CUDA:   {cfg['use_cuda']}")
    print(f"  多样性检查: {cfg.get('ensure_diversity', True)}")
    print(f"  输出目录:   {cfg['output_dir']}")
    print()

    try:
        result = process_step3(
            data_csv_path=cfg["data_csv_path"],
            design_space_csv=cfg["design_space_csv"],
            subject_col=cfg["subject_col"],
            response_col=cfg["response_col"],
            output_dir=cfg["output_dir"],
            max_iters=cfg["max_iters"],
            lr=cfg["learning_rate"],
            use_cuda=cfg["use_cuda"],
            ensure_diversity=cfg.get("ensure_diversity", True),
        )
    except Exception as e:
        print(f"[错误] Step3 失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("[OK] Base GP 训练与扫描完成")
    print(f"  输出目录: {result['output_dir']}")
    print(f"  设计空间点数: {result['n_design_points']}")
    print("  关键点: 保存于 base_gp_key_points.json")
    print("  长度尺度: base_gp_lengthscales.json")
    print("  报告: base_gp_report.md")
    print()
    print("下一步：可在 Phase 2 模型中加载 base_gp_state.pth 作为形状函数先验")


def run_step2_plus_3():
    """整合运行 Step 2 和 Step 3，并生成统一报告"""
    print("=" * 80)
    print("Phase 1 完整分析：Step 2 (交互对) + Step 3 (Base GP)")
    print("=" * 80)
    print()

    # ========== Step 2: 交互对分析 ==========
    print("=" * 80)
    print("第一部分：交互对筛选与 Phase 2 参数估计")
    print("=" * 80)
    print()

    from core.analyze_phase1 import Phase1DataAnalyzer
    import json

    # 运行 Step 2
    try:
        analyzer = Phase1DataAnalyzer(
            data_csv_path=STEP2_CONFIG["data_csv_path"],
            subject_col=STEP2_CONFIG["subject_col"],
            response_col=STEP2_CONFIG["response_col"],
        )

        analysis = analyzer.analyze(
            max_pairs=STEP2_CONFIG["max_pairs"],
            min_pairs=STEP2_CONFIG["min_pairs"],
            selection_method=STEP2_CONFIG["selection_method"],
            verbose=True,
        )

        phase2_config = analyzer.generate_phase2_config(
            n_subjects=STEP2_CONFIG["phase2_n_subjects"],
            trials_per_subject=STEP2_CONFIG["phase2_trials_per_subject"],
            lambda_adjustment=STEP2_CONFIG["lambda_adjustment"],
        )

        exported_files = analyzer.export_report(
            phase2_config=phase2_config,
            output_dir=STEP2_CONFIG["output_dir"],
            prefix=STEP2_CONFIG["prefix"],
            report_format=STEP2_CONFIG["report_format"],
        )

        print(f"\n[OK] Step 2 完成！筛选了 {len(analysis['selected_pairs'])} 个交互对")
        print(f"     报告：{exported_files['report']}")

    except Exception as e:
        print(f"[错误] Step 2 失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ========== Step 3: Base GP 训练 ==========
    print()
    print("=" * 80)
    print("第二部分：Base GP 敏感度分析与关键点选择")
    print("=" * 80)
    print()

    from core.phase1_step3_base_gp import process_step3

    try:
        result_step3 = process_step3(
            data_csv_path=STEP3_CONFIG["data_csv_path"],
            design_space_csv=STEP3_CONFIG["design_space_csv"],
            subject_col=STEP3_CONFIG["subject_col"],
            response_col=STEP3_CONFIG["response_col"],
            output_dir=STEP3_CONFIG["output_dir"],
            max_iters=STEP3_CONFIG["max_iters"],
            lr=STEP3_CONFIG["learning_rate"],
            use_cuda=STEP3_CONFIG["use_cuda"],
            ensure_diversity=STEP3_CONFIG.get("ensure_diversity", True),
        )

        print(f"\n[OK] Step 3 完成！训练了 Base GP 模型")
        print(f"     报告：{Path(result_step3['output_dir']) / 'base_gp_report.md'}")

    except Exception as e:
        print(f"[错误] Step 3 失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # ========== 生成整合报告 ==========
    print()
    print("=" * 80)
    print("第三部分：生成整合分析报告")
    print("=" * 80)
    print()

    _generate_integrated_report(
        step2_dir=Path(exported_files["report"]).parent,
        step3_dir=Path(result_step3["output_dir"]),
        analysis=analysis,
        phase2_config=phase2_config,
        lengthscales=result_step3["lengthscales"],
        key_points=result_step3["key_points"],
    )


def _generate_integrated_report(
    step2_dir: Path,
    step3_dir: Path,
    analysis: dict,
    phase2_config: dict,
    lengthscales: list,
    key_points: dict,
):
    """生成 Step 2 + Step 3 的整合分析报告"""
    import json

    # 创建整合报告目录
    integrated_dir = step2_dir.parent / f"integrated_{step2_dir.name}"
    print(f"[Debug] Step2 dir: {step2_dir}")
    print(f"[Debug] Step2 dir parent: {step2_dir.parent}")
    print(f"[Debug] Integrated dir path: {integrated_dir}")
    integrated_dir.mkdir(exist_ok=True)
    print(f"[Debug] Directory exists: {integrated_dir.exists()}")

    # 读取因子名称
    step3_lengthscales_file = step3_dir / "base_gp_lengthscales.json"
    with open(step3_lengthscales_file) as f:
        ls_data = json.load(f)
    factor_names = ls_data["factor_names"]

    # 生成整合报告
    report_path = integrated_dir / "INTEGRATED_ANALYSIS_REPORT.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 1 完整分析报告\n\n")
        f.write(
            "> **整合了 Step 2 (交互对分析) 和 Step 3 (Base GP 敏感度分析) 的结果**\n\n"
        )
        f.write(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # ===== 核心发现 =====
        f.write("## 核心发现\n\n")

        # 交互对
        f.write("### 1. 关键交互对（来自 Step 2）\n\n")
        f.write(f"筛选出 **{len(analysis['selected_pairs'])}** 个重要交互对：\n\n")
        f.write("| 排名 | 交互对 | 评分 | 系数 |\n")
        f.write("|------|--------|------|------|\n")

        # Get interaction scores and effects
        interaction_scores = analysis.get("diagnostics", {}).get(
            "interaction_scores", {}
        )
        interaction_effects = analysis.get("interaction_effects", {})

        for i, pair in enumerate(analysis["selected_pairs"], 1):
            idx1, idx2 = pair  # pair is a tuple like (2, 5)
            f1 = factor_names[idx1]
            f2 = factor_names[idx2]
            score = interaction_scores.get(pair, 0.0)

            # Get coefficient from interaction_effects if available
            effect_info = interaction_effects.get(pair, {})
            if isinstance(effect_info, dict):
                coef = effect_info.get("coef_interaction", 0.0)
            else:
                coef = 0.0

            f.write(f"| {i} | **{f1}** × **{f2}** | {score:.3f} | {coef:.3f} |\n")

        f.write("\n")

        # 因子敏感度
        f.write("### 2. 因子敏感度排序（来自 Step 3 ARD）\n\n")
        f.write("基于 Base GP 的自动相关性判断（ARD）：\n\n")
        f.write("| 排名 | 因子 | 长度尺度 | 敏感度 | 参与交互数 |\n")
        f.write("|------|------|----------|--------|------------|\n")

        # 统计每个因子参与的交互数
        interaction_count = {i: 0 for i in range(len(factor_names))}
        for pair in analysis["selected_pairs"]:
            idx1, idx2 = pair  # pair is a tuple like (2, 5)
            interaction_count[idx1] += 1
            interaction_count[idx2] += 1

        # 按长度尺度排序（小到大 = 高敏感到低敏感）
        sorted_factors = sorted(enumerate(lengthscales), key=lambda x: x[1])

        for rank, (idx, ls) in enumerate(sorted_factors, 1):
            fname = factor_names[idx]
            if rank <= len(sorted_factors) // 3:
                sensitivity = "*** 高"
            elif rank <= 2 * len(sorted_factors) // 3:
                sensitivity = "** 中"
            else:
                sensitivity = "* 低"

            n_interactions = interaction_count[idx]
            f.write(
                f"| {rank} | {fname} | {ls:.2f} | {sensitivity} | {n_interactions} 个 |\n"
            )

        f.write("\n")

        # ===== 核心洞察 =====
        f.write("## 核心洞察：交互模式分析\n\n")

        # 找出交互最多的因子
        max_interactions = max(interaction_count.values())
        interaction_hubs = [
            factor_names[i]
            for i, count in interaction_count.items()
            if count == max_interactions
        ]

        if max_interactions > 0:
            f.write(f"### 交互核心因子：{', '.join(interaction_hubs)}\n\n")
            f.write(
                f"这些因子参与了 **{max_interactions}/{len(analysis['selected_pairs'])}** 个交互对，表明其效果**高度依赖情境**。\n\n"
            )

        # 对比主效应敏感度和交互参与度
        f.write("### 因子特性对比\n\n")
        f.write("| 因子 | 主效应敏感度 | 交互参与度 | 特性 |\n")
        f.write("|------|--------------|------------|------|\n")

        for idx, fname in enumerate(factor_names):
            ls = lengthscales[idx]
            n_int = interaction_count[idx]

            # 判断敏感度类别
            ls_rank = sorted_factors.index((idx, ls)) + 1
            if ls_rank <= len(sorted_factors) // 3:
                sens = "高"
            elif ls_rank <= 2 * len(sorted_factors) // 3:
                sens = "中"
            else:
                sens = "低"

            # 判断特性
            if sens == "高" and n_int == 0:
                char = "独立主效应"
            elif sens in ["低", "中"] and n_int >= 3:
                char = "**情境依赖型**"
            elif sens == "高" and n_int > 0:
                char = "主效应 + 交互"
            else:
                char = "调节因子"

            f.write(f"| {fname} | {sens} (LS={ls:.2f}) | {n_int} 个 | {char} |\n")

        f.write("\n")

        # ===== 三个关键采样点 =====
        f.write("## 推荐的初始采样点（来自 Step 3）\n\n")
        f.write("这些点可作为 Phase 2 的 warmup 初始化：\n\n")

        # Sample 1: Best
        f.write("### Sample 1: Best Prior（预测最佳）\n\n")
        best_coords = key_points["x_best_prior"]
        f.write(
            f"- **预测得分**: {key_points['best_mean']:.3f} (std={key_points['best_std']:.3f})\n"
        )
        f.write("- **参数配置**:\n")
        for fname in factor_names:
            f.write(f"  - {fname}: {best_coords[fname]}\n")
        f.write("\n")

        # Sample 2: Worst
        f.write("### Sample 2: Worst Prior（预测最差）\n\n")
        worst_coords = key_points["x_worst_prior"]
        f.write(
            f"- **预测得分**: {key_points['worst_mean']:.3f} (std={key_points['worst_std']:.3f})\n"
        )
        f.write("- **参数配置**:\n")
        for fname in factor_names:
            f.write(f"  - {fname}: {worst_coords[fname]}\n")
        f.write("\n")

        # Sample 3: Max Uncertainty
        f.write("### Sample 3: Max Uncertainty（最不确定）\n\n")
        maxstd_coords = key_points["x_max_std"]
        if key_points["used_center_point"]:
            f.write("**注意**: 所有点方差过低，使用设计空间中心点\n\n")
        f.write(
            f"- **不确定性**: std={key_points['max_std']:.3f} (mean={key_points.get('max_std_mean', 0):.3f})\n"
        )
        f.write("- **参数配置**:\n")
        for fname in factor_names:
            f.write(f"  - {fname}: {maxstd_coords[fname]}\n")
        f.write("\n")

        # ===== Phase 2 建议 =====
        f.write("## Phase 2 实验建议\n\n")

        f.write("### 推荐策略\n\n")
        f.write("基于整合分析，建议 Phase 2 采用以下策略：\n\n")
        f.write("1. **EUR-ANOVA 配置**（来自 Step 2）:\n")
        f.write(f"   - 交互对: {len(analysis['selected_pairs'])} 个\n")
        f.write(
            f"   - λ (交互权重): {phase2_config['lambda_init']:.2f} → {phase2_config['lambda_end']:.2f}\n"
        )
        f.write(
            f"   - γ (覆盖权重): {phase2_config['gamma_init']:.2f} → {phase2_config['gamma_end']:.2f}\n"
        )
        f.write(f"   - 总预算: {phase2_config['total_budget']} 次\n\n")

        f.write("2. **初始 Warmup 点**（来自 Step 3）:\n")
        f.write("   - 使用 3 个关键点作为初始采样\n")
        f.write("   - 覆盖设计空间的关键区域（最佳/最差/最不确定）\n\n")

        f.write("3. **探索优先级**:\n")
        high_sens_factors = [
            factor_names[idx] for idx, _ in sorted_factors[: len(sorted_factors) // 3]
        ]
        high_int_factors = [fname for fname in interaction_hubs]

        f.write(f"   - **优先探索主效应**: {', '.join(high_sens_factors)}\n")
        if high_int_factors:
            f.write(
                f"   - **重点探索交互**: 涉及 {', '.join(high_int_factors)} 的组合\n"
            )
        f.write("\n")

        # ===== 输出文件 =====
        f.write("## 📦 输出文件\n\n")
        f.write("### Step 2 输出\n")
        f.write(f"- JSON配置: `{step2_dir / 'phase1_phase2_config.json'}`\n")
        f.write(f"- NumPy调度: `{step2_dir / 'phase1_phase2_schedules.npz'}`\n")
        f.write(f"- 详细报告: `{step2_dir / 'phase1_analysis_report.md'}`\n\n")

        f.write("### Step 3 输出\n")
        f.write(f"- GP模型: `{step3_dir / 'base_gp_state.pth'}`\n")
        f.write(f"- 关键点: `{step3_dir / 'base_gp_key_points.json'}`\n")
        f.write(f"- 长度尺度: `{step3_dir / 'base_gp_lengthscales.json'}`\n")
        f.write(f"- 设计空间扫描: `{step3_dir / 'design_space_scan.csv'}`\n")
        f.write(f"- 详细报告: `{step3_dir / 'base_gp_report.md'}`\n\n")

        f.write("### 整合报告\n")
        f.write(f"- **本报告**: `{report_path}`\n\n")

        f.write("---\n\n")
        f.write("*自动生成于 Phase 1 完整分析流程*\n")

    print(f"[OK] 整合报告已生成：{report_path}")
    print()
    print("=" * 80)
    print("Phase 1 完整分析完成！")
    print("=" * 80)
    print()
    print("查看结果：")
    print(f"  - 整合报告: {report_path}")
    print(f"  - Step 2 详情: {step2_dir}")
    print(f"  - Step 3 详情: {step3_dir}")
    print()


def run_chain12():
    """使用流程管理器运行步骤1->2"""
    print("=" * 80)
    print("链式流程：步骤1 -> 步骤2")
    print("=" * 80)
    print()

    if not API_AVAILABLE:
        print(
            "[错误] 流程管理器需要新的 API 支持，请确保 config_models.py 和 warmup_api.py 可用"
        )
        sys.exit(1)

    try:
        # 创建步骤1配置
        step1_config = _dict_to_step1_config(STEP1_CONFIG)

        # 创建步骤2配置
        step2_config = _dict_to_step2_config(STEP2_CONFIG)

        # 创建并执行链式流程
        chain = Step1Step2Chain(step1_config, step2_config)
        result = chain.execute()

        print("[OK] 链式流程完成！")
        print(f"  步骤1输出: {result.step1_result.output_dir}/")
        print(f"  步骤2输出: {result.step2_result.output_dir}/")
        print(f"  筛选的交互对: {len(result.step2_result.selected_pairs)}个")
        print()
        print("=" * 80)
        print("下一步：可在 Phase 2 模型中使用分析结果")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"[错误] 链式流程失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_chain123():
    """使用流程管理器运行步骤1->2->3"""
    print("=" * 80)
    print("链式流程：步骤1 -> 步骤2 -> 步骤3")
    print("=" * 80)
    print()

    if not API_AVAILABLE:
        print(
            "[错误] 流程管理器需要新的 API 支持，请确保 config_models.py 和 warmup_api.py 可用"
        )
        sys.exit(1)

    try:
        # 创建所有步骤的配置
        step1_config = _dict_to_step1_config(STEP1_CONFIG)
        step2_config = _dict_to_step2_config(STEP2_CONFIG)
        step3_config = _dict_to_step3_config(STEP3_CONFIG)

        # 创建并执行链式流程
        chain = Step1Step2Step3Chain(step1_config, step2_config, step3_config)
        result = chain.execute()

        print("[OK] 完整链式流程完成！")
        print(f"  步骤1输出: {result.step1_result.output_dir}/")
        print(f"  步骤2输出: {result.step2_result.output_dir}/")
        print(f"  步骤3输出: {result.step3_result.output_dir}/")
        print(f"  筛选的交互对: {len(result.step2_result.selected_pairs)}个")
        print(f"  设计空间点数: {result.step3_result.n_design_points}")
        print()
        print("=" * 80)
        print("下一步：可在 Phase 2 模型中使用所有分析结果")
        print("=" * 80)
        print()

    except Exception as e:
        print(f"[错误] 链式流程失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def run_step1_5():
    """执行 Step 1.5: 模拟被试作答"""
    print("=" * 80)
    print("Step 1.5: 模拟被试作答")
    print("=" * 80)
    print()

    # 添加tools目录到路径，使用新的warmup_adapter V3
    tools_path = Path(__file__).parent.parent.parent / "tools"
    sys.path.insert(0, str(tools_path))

    try:
        from subject_simulator_v2.adapters.warmup_adapter import (
            run as simulate_responses,
        )
        import json

        config = STEP1_5_CONFIG.copy()
        input_dir = Path(config.pop("input_dir"))

        # 提取模型显示选项
        print_model = config.pop("print_model", True)
        save_model_summary = config.pop("save_model_summary", True)
        model_summary_format = config.pop("model_summary_format", "txt")

        # 添加design_space_csv参数（V3方法需要）
        # 使用与STEP1相同的设计空间CSV
        if "design_space_csv" not in config:
            config["design_space_csv"] = STEP1_CONFIG["design_csv_path"]

        print(f"输入目录: {input_dir}")
        print(f"随机种子: {config['seed']}")
        print(f"输出类型: {config['output_type']}")
        if config["output_type"] == "likert":
            print(f"  Likert级别: {config['likert_levels']}")
            print(f"  映射模式: {config['likert_mode']}")
        print(f"交互方法: V3 (interaction-as-features，默认)")
        print(f"设计空间CSV: {config.get('design_space_csv', 'N/A')}")
        print()

        # 运行模拟（使用V3方法，默认启用interaction_as_features=True）
        simulate_responses(input_dir=input_dir, **config)

        result_dir = input_dir / "result"

        # ========== 打印和保存模型规格 ==========
        if print_model or save_model_summary:
            # 读取第一个被试的模型规格作为代表
            model_md_files = sorted(list(result_dir.glob("subject_*_model.md")))
            fixed_weights_file = result_dir / "fixed_weights_auto.json"

            if model_md_files:
                print()
                print("=" * 80)
                print("模型规格总览")
                print("=" * 80)
                print()

                # 读取固定权重
                if fixed_weights_file.exists():
                    with open(fixed_weights_file, "r", encoding="utf-8") as f:
                        fixed_weights_data = json.load(f)
                        global_weights = fixed_weights_data.get("global", [])
                else:
                    global_weights = None

                # 构建模型摘要
                model_summary_lines = []
                model_summary_lines.append("=" * 80)
                model_summary_lines.append("Step 1.5 模拟被试模型规格")
                model_summary_lines.append("=" * 80)
                model_summary_lines.append("")
                model_summary_lines.append("## 模型配置")
                model_summary_lines.append(f"- 随机种子: {config['seed']}")
                model_summary_lines.append(
                    f"- 使用潜变量模型: {config.get('use_latent', 'false')}"
                )
                model_summary_lines.append(f"- 输出类型: {config['output_type']}")
                if config["output_type"] == "likert":
                    model_summary_lines.append(
                        f"- Likert级别: {config['likert_levels']}"
                    )
                    model_summary_lines.append(
                        f"- Likert映射: {config.get('likert_mode', 'tanh')}"
                    )
                model_summary_lines.append("")

                model_summary_lines.append("## 数据生成参数")
                model_summary_lines.append(
                    f"- 群体均值: {config.get('population_mean', 0.0)}"
                )
                model_summary_lines.append(
                    f"- 群体标准差: {config.get('population_std', 0.4)}"
                )
                model_summary_lines.append(
                    f"- 个体差异比例: {config.get('individual_std_percent', 1.0)}"
                )
                model_summary_lines.append(
                    f"- 特征间相关: {config.get('individual_corr', 0.0)}"
                )
                model_summary_lines.append("")

                model_summary_lines.append("## 交互效应配置")
                interaction_pairs = config.get("interaction_pairs", [])
                if interaction_pairs:
                    model_summary_lines.append(
                        f"- 指定交互对: {len(interaction_pairs)}个"
                    )
                    for i, (idx1, idx2) in enumerate(interaction_pairs, 1):
                        model_summary_lines.append(f"  {i}. x{idx1} × x{idx2}")
                else:
                    model_summary_lines.append("- 无指定交互对")
                model_summary_lines.append(
                    f"- 随机交互项数: {config.get('num_interactions', 0)}"
                )
                model_summary_lines.append(
                    f"- 交互权重尺度: {config.get('interaction_scale', 1.0)}"
                )
                model_summary_lines.append("")

                # 全局固定权重
                if global_weights:
                    model_summary_lines.append("## 群体固定效应（所有被试共享）")
                    model_summary_lines.append("")
                    for obs_idx, weights in enumerate(global_weights, 1):
                        model_summary_lines.append(f"### 输出变量 {obs_idx}")
                        for feat_idx, w in enumerate(weights, 1):
                            model_summary_lines.append(f"  x{feat_idx-1}: {w:+.5f}")
                        model_summary_lines.append("")

                model_summary_lines.append("## 被试个体差异")
                model_summary_lines.append("每个被试在群体固定效应基础上添加随机偏差，")
                model_summary_lines.append(
                    f"偏差标准差 = {config.get('individual_std_percent', 1.0)} × {config.get('population_std', 0.4)} = {config.get('individual_std_percent', 1.0) * config.get('population_std', 0.4):.4f}"
                )
                model_summary_lines.append("")
                model_summary_lines.append(
                    f"详见各被试模型文件: {result_dir}/subject_*_model.md"
                )
                model_summary_lines.append("")
                model_summary_lines.append("=" * 80)

                # 打印到控制台
                if print_model:
                    for line in model_summary_lines:
                        print(line)
                    print()

                # 保存到文件
                if save_model_summary:
                    if model_summary_format in ["txt", "both"]:
                        summary_txt = result_dir / "MODEL_SUMMARY.txt"
                        with open(summary_txt, "w", encoding="utf-8") as f:
                            f.write("\n".join(model_summary_lines))
                        print(f"[保存] 模型摘要已保存至: {summary_txt}")

                    if model_summary_format in ["md", "both"]:
                        summary_md = result_dir / "MODEL_SUMMARY.md"
                        # Markdown格式稍作调整
                        md_lines = [
                            line.replace("## ", "### ").replace("# ", "## ")
                            for line in model_summary_lines
                        ]
                        md_lines[0] = "# " + md_lines[0].strip("=").strip()  # 标题
                        with open(summary_md, "w", encoding="utf-8") as f:
                            f.write("\n".join(md_lines))
                        print(f"[保存] 模型摘要已保存至: {summary_md}")
                    print()

        print()
        print("=" * 80)
        print("[OK] 模拟应答完成！")
        print(f"输出目录: {input_dir}/result/")
        if save_model_summary:
            print(f"模型摘要: {result_dir}/MODEL_SUMMARY.{model_summary_format}")
        print("=" * 80)
        print()

    except ImportError as e:
        print(f"[错误] 无法导入模拟应答模块: {e}")
        print("请确保 core/simulation_runner.py 和 core/single_output_subject.py 存在")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 模拟应答失败: {e}")
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
    elif MODE == "step1.5":
        run_step1_5()
    elif MODE == "step2":
        run_step2()
    elif MODE == "step3":
        run_step3()
    elif MODE == "step2+3":
        print("[模式] 整合分析：Step 2 + Step 3")
        run_step2_plus_3()
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
    elif MODE == "all":
        print("[模式] 运行 Step1 -> Step1.5(模拟) -> Step2 -> Step3 (all 模式)")

        # 准备统一输出目录结构
        ts = time.strftime("%Y%m%d%H%M")
        base_out = Path(ALL_CONFIG.get("base_output_dir")) / ts
        step1_out = base_out / "step1"
        step1_5_out = base_out / "step1_5"
        step2_out = base_out / "step2"
        step3_out = base_out / "step3"

        # 确保基础目录存在
        base_out.mkdir(parents=True, exist_ok=True)

        # 覆盖各步骤的输出/输入路径
        # 如果 ALL_CONFIG 指定了设计空间文件，优先使用（覆盖 STEP1_CONFIG）
        if ALL_CONFIG.get("design_csv"):
            STEP1_CONFIG["design_csv_path"] = ALL_CONFIG.get("design_csv")

        STEP1_CONFIG["output_dir"] = str(step1_out)

        # Step1.5 的 input_dir 指向 Step1 输出（如果配置允许自动运行模拟）
        if ALL_CONFIG.get("run_step1_5", True):
            STEP1_5_CONFIG["input_dir"] = str(step1_out)
        # Step1.5 的输出会写入 input_dir/result/
        # 如果 ALL_CONFIG 指定了 step2_data_csv，则优先使用
        if ALL_CONFIG.get("step2_data_csv"):
            STEP2_CONFIG["data_csv_path"] = ALL_CONFIG.get("step2_data_csv")
        else:
            # Step2 使用 Step1.5 的 result 目录作为 data source（优先），否则使用 Step1 的 result
            if ALL_CONFIG.get("run_step1_5", True) and ALL_CONFIG.get(
                "step1_5_use_result_dir_for_step2", True
            ):
                STEP2_CONFIG["data_csv_path"] = str(step1_5_out / "result")
            else:
                STEP2_CONFIG["data_csv_path"] = str(step1_out / "result")

        # 覆盖 Step2/Step3 的输出目录
        STEP2_CONFIG["output_dir"] = str(step2_out)
        STEP3_CONFIG["output_dir"] = str(step3_out)

        # Step3 使用 Step2 的数据输出作为输入
        # 如果 ALL_CONFIG 指定了 step3 的 design_space，则覆盖 Step3 的 design_space_csv
        if ALL_CONFIG.get("step3_design_space_csv"):
            STEP3_CONFIG["design_space_csv"] = ALL_CONFIG.get("step3_design_space_csv")

        STEP3_CONFIG["data_csv_path"] = STEP2_CONFIG["data_csv_path"]

        # 运行步骤
        run_step1()
        print()
        print("=" * 80)
        if ALL_CONFIG.get("run_step1_5", True):
            print("自动运行模拟应答 (Step1.5)...")
            print("=" * 80)

            # 将 Step1 的输出复制到单独的 step1_5 输入目录，保证每个步骤有独立目录
            try:
                import shutil

                if Path(STEP1_CONFIG["output_dir"]).exists():
                    step1_5_out.mkdir(parents=True, exist_ok=True)
                    for item in Path(STEP1_CONFIG["output_dir"]).iterdir():
                        dest = step1_5_out / item.name
                        if item.is_dir():
                            if dest.exists():
                                shutil.rmtree(dest)
                            shutil.copytree(item, dest)
                        else:
                            shutil.copy2(item, dest)
                # 将模拟输入指向复制后的目录
                STEP1_5_CONFIG["input_dir"] = str(step1_5_out)
            except Exception as e:
                print(f"[警告] 复制 Step1 输出到 step1_5 目录失败: {e}")
                print("继续使用 Step1 输出目录作为模拟输入")
                STEP1_5_CONFIG["input_dir"] = str(step1_out)

            run_step1_5()
            print()
        print("=" * 80)
        print("继续运行 Step2 分析数据...")
        print("=" * 80)
        print()
        run_step2()
        print()
        print("=" * 80)
        print("继续运行 Step3 (Base GP)...")
        print("=" * 80)
        print()
        run_step3()
    elif MODE == "chain12":
        print("[模式] 使用流程管理器运行步骤1->2")
        run_chain12()
    elif MODE == "chain123":
        print("[模式] 使用流程管理器运行步骤1->2->3")
        run_chain123()
    else:
        print(f"[错误] 未知的模式: {MODE}")
        print(
            "请设置 MODE 为 'step1', 'step1.5', 'step2', 'step3', 'step2+3', 'both', 'all', 'chain12', 或 'chain123'"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
