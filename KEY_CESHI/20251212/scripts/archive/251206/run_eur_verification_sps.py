#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EUR采集函数规范性验证实验
完全规范化版本 - 保持数值变量的原始意义

核心原则：
1. 完全依赖 AEPsych Server 和 Config 机制
2. 保持所有数值变量的原始数值意义
3. 仅映射字符串变量为categorical
4. 从 Server 正确获取所有诊断指标
"""

import sys
import io
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cdist

# 修复编码
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "temp_aepsych"))
sys.path.insert(0, str(PROJECT_ROOT / "extensions" / "dynamic_eur_acquisition"))
sys.path.insert(0, str(PROJECT_ROOT / "extensions" / "custom_generators"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# 导入并注册自定义组件
try:
    from aepsych.config import Config
    from pool_based_generator import PoolBasedGenerator
    from eur_anova_multi import EURAnovaMultiAcqf

    Config.register_object(PoolBasedGenerator)
    Config.register_object(EURAnovaMultiAcqf)
    print("[OK] 自定义组件已注册")
except Exception as e:
    print(f"[ERROR] 无法导入/注册自定义组件: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

from aepsych.server import AEPsychServer
import torch

# 导入被试模拟器
try:
    from subject_simulator import SingleSubject
except ImportError:
    print("[ERROR] 无法导入 subject_simulator")
    sys.exit(1)

# 导入序数模型验证工具
try:
    from verify_ordinal_model import verify_and_save_report
    print("[OK] 序数模型验证工具已导入")
except ImportError:
    print("[WARNING] 无法导入 verify_ordinal_model，序数验证功能将被跳过")
    verify_and_save_report = None

# 解析命令行参数
parser = argparse.ArgumentParser(description="EUR采集函数验证实验")
parser.add_argument(
    "--config",
    type=str,
    default="eur_config_sps.ini",
    help="配置文件名（默认: eur_config_sps.ini）",
)
parser.add_argument(
    "--tag",
    type=str,
    default="",
    help="实验标签（用于区分不同配置的结果）",
)
args = parser.parse_args()

print("\n" + "=" * 80)
print("EUR采集函数规范性验证实验".center(80))
print("=" * 80)
print(f"\n配置文件: {args.config}")
if args.tag:
    print(f"实验标签: {args.tag}")

# 创建结果目录
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.tag:
    result_dir = Path(__file__).parent / "results" / f"{timestamp}_{args.tag}"
else:
    result_dir = Path(__file__).parent / "results" / timestamp
result_dir.mkdir(parents=True, exist_ok=True)
data_dir = result_dir / "data_files"
data_dir.mkdir(exist_ok=True)
figures_dir = result_dir / "figures"
figures_dir.mkdir(exist_ok=True)

print(f"\n结果目录: {result_dir}")

# ========================================
# 步骤1: 加载外部设计空间（保持数值意义）
# ========================================
print("\n" + "=" * 80)
print("步骤1: 加载外部设计空间")
print("=" * 80)

design_space_path = (
    PROJECT_ROOT
    / "data"
    / "only_independences"
    / "data"
    / "only_independences"
    / "6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"
)

try:
    df_design = pd.read_csv(design_space_path)
    print(f"✓ 设计空间加载成功: {df_design.shape}")
    print(f"  文件: {design_space_path.name}")
    print(f"  列名: {list(df_design.columns)}")

    # 检查每列的唯一值
    for col in df_design.columns:
        unique_vals = df_design[col].unique()
        print(f"  {col}: {len(unique_vals)} 个唯一值, 示例: {unique_vals[:5]}")

    # 转换为数值格式（保持数值意义）
    df_numeric = df_design.copy()

    # x0: x1_binary - 保持 0, 1
    df_numeric["x1_binary"] = df_design["x1_binary"].astype(float)

    # x1: x2_5level_discrete - 映射 1,2,3,4,5 → 0,1,2,3,4
    # 公式: value - 1
    df_numeric["x2_5level_discrete"] = (df_design["x2_5level_discrete"] - 1).astype(
        float
    )

    # x2: x3_5level_decimal - 映射 0.0,0.25,0.5,0.75,1.0 → 0,1,2,3,4
    # 公式: value / 0.25
    df_numeric["x3_5level_decimal"] = (df_design["x3_5level_decimal"] / 0.25).astype(
        float
    )

    # x3: x4_4level_categorical - 映射 "low","mid","high","max" → 0,1,2,3
    df_numeric["x4_4level_categorical"] = (
        df_design["x4_4level_categorical"]
        .map({"low": 0, "mid": 1, "high": 2, "max": 3})
        .astype(float)
    )

    # x4: x5_3level_categorical - 映射 "A","B","C" → 0,1,2
    df_numeric["x5_3level_categorical"] = (
        df_design["x5_3level_categorical"].map({"A": 0, "B": 1, "C": 2}).astype(float)
    )

    # x5: x6_binary - 映射 True,False → 1,0
    df_numeric["x6_binary"] = (
        df_design["x6_binary"].map({True: 1, False: 0}).astype(float)
    )

    # 转换为numpy数组
    design_space = df_numeric.values.astype(np.float64)

    # 详细验证
    print(f"\n✓ 设计空间转换完成: {design_space.shape}")
    print(f"  值范围验证:")
    for i, col in enumerate(df_numeric.columns):
        min_val = design_space[:, i].min()
        max_val = design_space[:, i].max()
        unique_count = len(np.unique(design_space[:, i]))
        print(
            f"    x{i} ({col}): [{min_val:.2f}, {max_val:.2f}], {unique_count} 个唯一值"
        )

    # 断言验证
    assert design_space.shape == (1200, 6), f"设计空间形状错误"
    assert np.all((design_space[:, 0] >= 0) & (design_space[:, 0] <= 1)), "x0 超出[0,1]"
    assert np.all((design_space[:, 1] >= 0) & (design_space[:, 1] <= 4)), "x1 超出[0,4]"
    assert np.all((design_space[:, 2] >= 0) & (design_space[:, 2] <= 4)), "x2 超出[0,4]"
    assert np.all((design_space[:, 3] >= 0) & (design_space[:, 3] <= 3)), "x3 超出[0,3]"
    assert np.all((design_space[:, 4] >= 0) & (design_space[:, 4] <= 2)), "x4 超出[0,2]"
    assert np.all((design_space[:, 5] >= 0) & (design_space[:, 5] <= 1)), "x5 超出[0,1]"
    print(f"✓ 所有值范围验证通过")

except Exception as e:
    print(f"⚠ 警告: 无法加载设计空间文件: {e}")
    print("  启用回退逻辑: 生成随机设计空间")
    import traceback

    traceback.print_exc()

    # 回退逻辑：生成符合约束的随机设计空间
    design_space = np.zeros((1200, 6), dtype=np.float64)
    design_space[:, 0] = np.random.randint(0, 2, 1200)  # binary
    design_space[:, 1] = np.random.choice(
        [0.0, 0.25, 0.5, 0.75, 1.0], 1200
    )  # 5-level continuous
    design_space[:, 2] = np.random.choice(
        [0.0, 0.25, 0.5, 0.75, 1.0], 1200
    )  # 5-level continuous
    design_space[:, 3] = np.random.randint(0, 4, 1200)  # 4-level categorical
    design_space[:, 4] = np.random.randint(0, 3, 1200)  # 3-level categorical
    design_space[:, 5] = np.random.randint(0, 2, 1200)  # binary

    print(f"✓ 回退设计空间生成完成: {design_space.shape}")

# ========================================
# 步骤2: 创建并打印Oracle模型
# ========================================
print("\n" + "=" * 80)
print("步骤2: 创建被试模拟器（Oracle）")
print("=" * 80)

# 交互对: [(1, 2), (3, 4), (0, 5)] 对应 [x1, x2], [x3, x4], [x0, x5]
# 模拟真实心理学实验的被试参数
oracle = SingleSubject(
    seed=42,
    likert_levels=5,
    weight_std=0.7,  # 主效应权重标准差（被试对不同因素敏感度差异）
    noise_std=0.35,  # 测量噪声（心理学实验的内在变异性）
    interaction_pairs=[(1, 2), (3, 4), (0, 5)],
    interaction_scale=0.45,  # 交互效应通常比主效应弱
    likert_sensitivity=2.2,  # Likert敏感度（控制输出分布）
)

print(f"✓ Oracle 类型: {type(oracle).__name__}")

# 打印被试模型的详细规格
try:
    model_spec = oracle.get_model_spec()
    print(f"\n{'='*60}")
    print(f"【被试模型规格】")
    print(f"{'='*60}")
    print(f"  模型类型: {model_spec.get('model_type', 'unknown')}")
    print(f"  特征数量: {model_spec.get('num_features', 0)}")
    print(f"  Likert级别: {model_spec.get('likert_levels', 0)}")
    print(f"  噪声标准差: {model_spec.get('noise_std', 0):.4f}")
    print(f"  权重标准差: {model_spec.get('weight_std', 0):.4f}")

    print(f"\n  主效应权重:")
    weights = model_spec.get("weights", [])
    for i, w in enumerate(weights):
        print(f"    x{i}: {w:+.6f}")

    print(f"\n  交互项权重:")
    # 【修复】使用正确的键名
    int_weights = model_spec.get("interaction_terms", {})
    if not int_weights:
        # 备用：尝试旧的键名
        int_weights = model_spec.get("interaction_weights", {})

    if int_weights:
        for term_name, weight in int_weights.items():
            print(f"    {term_name}: {weight:+.6f}")
    else:
        print(f"    (无交互项)")

    print(f"\n  模型公式:")
    print(f"    y = bias + ∑(w_i · x_i) + ∑(w_ij · x_i · x_j) + ε")
    print(f"    其中 ε ~ N(0, {model_spec.get('noise_std', 0):.4f}²)")
    print(f"    bias = {model_spec.get('bias', 0):.6f}")
    print(f"{'='*60}\n")

except Exception as e:
    print(f"⚠ 警告: 无法获取模型规格: {e}")

# ========================================
# 步骤3: 初始化 AEPsych Server
# ========================================
print("\n" + "=" * 80)
print("步骤3: 初始化 AEPsych Server")
print("=" * 80)

config_path = Path(__file__).parent / args.config
db_path = result_dir / "experiment.db"

try:
    # 读取配置文件
    with open(config_path, "r", encoding="utf-8") as f:
        config_str = f.read()

    # 动态替换 pool_points
    pool_points_str = str(design_space.tolist())
    config_str = config_str.replace(
        "pool_points = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]",
        f"pool_points = {pool_points_str}",
    )
    print(f"✓ 已替换 pool_points: {design_space.shape[0]} 个候选点")

    # 创建 Server
    server = AEPsychServer(database_path=str(db_path))
    print(f"✓ Server 已创建")

    # 配置 Server
    setup_message = {
        "type": "setup",
        "message": {"config_str": config_str},
    }

    response = server.handle_request(setup_message)
    print(f"✓ 配置已加载: {config_path.name}")
    print(f"✓ Strategy ID: {response}")

    # 验证组件类型
    if hasattr(server, "strats") and len(server.strats) > 0:
        strat = server.strats[0]
        print(f"✓ Generator: {type(strat.generator).__name__}")
        if hasattr(strat.generator, "acqf"):
            print(f"✓ Acquisition: {type(strat.generator.acqf).__name__}")
        if hasattr(strat, "model"):
            print(f"✓ Model: {type(strat.model).__name__}")

except Exception as e:
    print(f"✗ Server 初始化失败: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ========================================
# 步骤4: 采样循环
# ========================================
print("\n" + "=" * 80)
print("步骤4: 采样循环 (20次)")
print("=" * 80)

BUDGET = 50  # 验证修复在更多迭代中的有效性

# 数据记录
logs = {
    "iteration": [],
    "x_points": [],
    "y_values": [],
    "lambda_t": [],
    "gamma_t": [],
    "r_t": [],
    "n_train": [],
    "model_entropy": [],
    "acqf_mean": [],
    "acqf_std": [],
    "min_distance": [],
}

sampling_history = []
interaction_logs = {}

for t in range(BUDGET):
    try:
        # ========== Ask ==========
        ask_message = {"type": "ask", "message": ""}
        response = server.handle_request(ask_message)

        # 解析采样点
        if "config" in response:
            x_dict = response["config"]
        elif "x" in response:
            x_dict = response["x"]
        else:
            print(f"✗ 迭代 {t}: 响应格式错误")
            break

        # 提取为数组
        def extract_value(val):
            if isinstance(val, list):
                return float(val[0])
            return float(val)

        x_array = np.array(
            [extract_value(x_dict.get(f"x{i}", 0.0)) for i in range(6)],
            dtype=np.float64,
        )

        sampling_history.append(x_array)

        # ========== Evaluate ==========
        # 使用模拟被试的真实输出（Likert 1-5 映射到 0-4）
        y_raw = oracle(x_array)
        y_likert = int(np.clip(y_raw, 1, 5))
        y = y_likert - 1  # 将 1-5 转换为 0-4
        print(f"[迭代 {t}] oracle输出={y_raw:.2f}, Likert={y_likert}, 编码为y={y}")

        # ========== Tell ==========
        x_config = {k: extract_value(v) for k, v in x_dict.items()}
        tell_message = {"type": "tell", "message": {"config": x_config, "outcome": y}}
        server.handle_request(tell_message)

        # ========== 序数模型验证（warmup结束后）==========
        if t == 9:  # 第10次迭代后（min_asks=10）
            print("\n" + "=" * 80)
            print("WARMUP阶段结束 - 运行序数模型验证".center(80))
            print("=" * 80)

            if verify_and_save_report is not None:
                try:
                    verification_report_path = result_dir / "ordinal_verification.json"
                    is_valid = verify_and_save_report(
                        server,
                        output_path=str(verification_report_path),
                        min_training_samples=10
                    )

                    if not is_valid:
                        print("\n" + "!" * 80)
                        print("警告：序数模型配置验证失败！".center(80))
                        print("采集函数可能降级到方差指标（而非熵）".center(80))
                        print("!" * 80)
                        # 不终止实验，但记录警告
                    else:
                        print("\n" + "✓" * 80)
                        print("序数模型配置验证通过！".center(80))
                        print("✓" * 80)

                except Exception as e:
                    print(f"\n序数模型验证失败（异常）: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("\n⚠️  序数模型验证工具未导入，跳过验证")
            print()

        # ========== 获取诊断信息 ==========
        diagnostics = {
            "lambda_t": np.nan,
            "gamma_t": np.nan,
            "r_t": np.nan,
            "n_train": t + 1,
            "model_entropy": np.nan,
            "acqf_mean": np.nan,
            "acqf_std": np.nan,
        }

        if t == 0:
            print(f"\n[DEBUG 迭代 {t}] 开始获取诊断信息")
            print(f"[DEBUG 迭代 {t}] server 有 strats: {hasattr(server, 'strats')}")
            print(f"[DEBUG 迭代 {t}] server 有 _strats: {hasattr(server, '_strats')}")
            if hasattr(server, "_strats"):
                print(f"[DEBUG 迭代 {t}] _strats 长度: {len(server._strats)}")

        try:
            # 【修复】使用 _strats 而不是 strats
            if hasattr(server, "_strats") and len(server._strats) > 0:
                strat = server._strats[0]

                # 1. 从采集函数获取权重
                if hasattr(strat, "generator") and hasattr(strat.generator, "acqf"):
                    # 【修复】使用缓存的采集函数实例，确保状态一致性
                    acqf_instance = None

                    # 优先使用缓存的实例（这是真正被使用的实例）
                    if (
                        hasattr(strat.generator, "_acqf_instance")
                        and strat.generator._acqf_instance is not None
                    ):
                        acqf_instance = strat.generator._acqf_instance
                    else:
                        # 如果缓存不存在，创建新实例
                        acqf_instance = strat.generator._instantiate_acquisition_fn(
                            strat.model
                        )

                    # 【关键修复】确保状态已同步（调用 _ensure_fresh_data）
                    if hasattr(acqf_instance, "_ensure_fresh_data"):
                        acqf_instance._ensure_fresh_data()

                    # 从采集函数直接获取诊断信息
                    if hasattr(acqf_instance, "get_diagnostics"):
                        try:
                            # 【调试】检查 acqf_instance 对象
                            if t == 0:
                                print(
                                    f"\n[DEBUG] acqf_instance 类型: {type(acqf_instance)}"
                                )
                                print(
                                    f"[DEBUG] acqf_instance 有 get_diagnostics: {hasattr(acqf_instance, 'get_diagnostics')}"
                                )
                                print(
                                    f"[DEBUG] get_diagnostics 类型: {type(acqf_instance.get_diagnostics)}"
                                )
                                print(
                                    f"[DEBUG] get_diagnostics 是否可调用: {callable(acqf_instance.get_diagnostics)}"
                                )

                            # 【修复】现在 acqf_instance 是一个真正的实例，可以直接调用
                            acqf_diag = acqf_instance.get_diagnostics()

                            # 【调试】打印完整诊断字典
                            if t == 0:
                                print(f"\n[DEBUG] acqf.get_diagnostics() 返回:")
                                print(f"  类型: {type(acqf_diag)}")
                                print(f"  键: {list(acqf_diag.keys())}")
                                print(f"  lambda_t: {acqf_diag.get('lambda_t')}")
                                print(f"  gamma_t: {acqf_diag.get('gamma_t')}")
                                print(f"  r_t: {acqf_diag.get('r_t')}")

                            # 【修复】直接从诊断字典获取值
                            lambda_t = acqf_diag.get("lambda_t", np.nan)
                            gamma_t = acqf_diag.get("gamma_t", np.nan)
                            r_t = acqf_diag.get("r_t", np.nan)
                            n_train = acqf_diag.get("n_train", t + 1)

                            # 【调试】打印提取的值
                            if t == 0:
                                print(f"\n[DEBUG] 提取的值:")
                                print(
                                    f"  lambda_t: {lambda_t} (isna: {pd.isna(lambda_t)})"
                                )
                                print(
                                    f"  gamma_t: {gamma_t} (isna: {pd.isna(gamma_t)})"
                                )
                                print(f"  r_t: {r_t} (isna: {pd.isna(r_t)})")
                                print(f"  n_train: {n_train}")

                            # 检查获取的值是否有效
                            if not pd.isna(lambda_t):
                                diagnostics["lambda_t"] = lambda_t
                            if not pd.isna(gamma_t):
                                diagnostics["gamma_t"] = gamma_t
                            if not pd.isna(r_t):
                                diagnostics["r_t"] = r_t
                            if not pd.isna(n_train):
                                diagnostics["n_train"] = n_train

                            # 【修复】如果诊断信息仍为空，尝试从weight_engine直接获取
                            if pd.isna(diagnostics["lambda_t"]) and hasattr(
                                acqf_instance, "weight_engine"
                            ):
                                try:
                                    # 直接调用 compute_lambda 和 compute_gamma 获取最新值
                                    lambda_t = (
                                        acqf_instance.weight_engine.compute_lambda()
                                    )
                                    gamma_t = (
                                        acqf_instance.weight_engine.compute_gamma()
                                    )
                                    r_t = (
                                        acqf_instance.weight_engine.compute_relative_main_variance()
                                    )

                                    if t == 0:
                                        print(f"\n[DEBUG] 从 weight_engine 获取的值:")
                                        print(f"  lambda_t: {lambda_t}")
                                        print(f"  gamma_t: {gamma_t}")
                                        print(f"  r_t: {r_t}")

                                    if not pd.isna(lambda_t):
                                        diagnostics["lambda_t"] = lambda_t
                                    if not pd.isna(gamma_t):
                                        diagnostics["gamma_t"] = gamma_t
                                    if not pd.isna(r_t):
                                        diagnostics["r_t"] = r_t

                                except Exception as we_err:
                                    if t >= 10:  # 只在模型应该已经训练后才报警
                                        print(
                                            f"  [迭代 {t}] 警告: 无法从weight_engine获取诊断: {we_err}"
                                        )
                        except Exception as acqf_diag_err:
                            print(
                                f"  [迭代 {t}] 警告: 无法获取acqf诊断: {acqf_diag_err}"
                            )
                            import traceback

                            traceback.print_exc()

                    # 从 weight_engine 获取诊断信息（备用）
                    elif hasattr(acqf_instance, "weight_engine"):
                        try:
                            lambda_t = acqf_instance.weight_engine.compute_lambda()
                            gamma_t = acqf_instance.weight_engine.compute_gamma()
                            r_t = (
                                acqf_instance.weight_engine.compute_relative_main_variance()
                            )

                            diagnostics["lambda_t"] = lambda_t
                            diagnostics["gamma_t"] = gamma_t
                            diagnostics["r_t"] = r_t
                            diagnostics["n_train"] = (
                                acqf_instance.weight_engine._n_train
                            )
                        except Exception as we_err:
                            if t >= 10:  # 只在模型应该已经训练后才报警
                                print(
                                    f"  [迭代 {t}] 警告: 无法获取weight_engine诊断: {we_err}"
                                )

                # 2. 从模型获取 entropy（不确定性）
                if hasattr(strat, "model"):
                    model = strat.model
                    try:
                        # 获取模型在当前已训练数据上的不确定性
                        if (
                            hasattr(model, "train_inputs")
                            and model.train_inputs is not None
                        ):
                            with torch.no_grad():
                                # 使用训练点评估后验方差
                                X_train = model.train_inputs[0]
                                posterior = model.posterior(X_train)
                                variances = posterior.variance.cpu().numpy()
                                # 使用平均方差作为entropy的代理
                                diagnostics["model_entropy"] = float(np.mean(variances))
                        else:
                            # 如果没有训练数据，使用当前采样点评估
                            if len(sampling_history) > 0:
                                with torch.no_grad():
                                    X_hist = torch.tensor(
                                        sampling_history, dtype=torch.float32
                                    )
                                    posterior = model.posterior(X_hist)
                                    variances = posterior.variance.cpu().numpy()
                                    diagnostics["model_entropy"] = float(
                                        np.mean(variances)
                                    )
                    except Exception as model_err:
                        print(f"  [迭代 {t}] 警告: 无法获取model_entropy: {model_err}")

                # 3. 从采集函数获取统计信息
                if hasattr(strat, "generator") and hasattr(strat.generator, "acqf"):
                    try:
                        # 获取池中所有点的采集函数值
                        if hasattr(strat.generator, "pool_points"):
                            pool_points = strat.generator.pool_points

                            # 确保是tensor格式
                            if isinstance(pool_points, np.ndarray):
                                pool_tensor = torch.tensor(
                                    pool_points, dtype=torch.float32
                                )
                            else:
                                pool_tensor = pool_points.float()

                            with torch.no_grad():
                                # 尝试直接调用acqf_instance
                                try:
                                    acqf_values = acqf_instance(
                                        pool_tensor.unsqueeze(-2)
                                    )
                                except Exception as acqf_call_err:
                                    # 如果失败，尝试不加unsqueeze
                                    try:
                                        acqf_values = acqf_instance(pool_tensor)
                                    except Exception as acqf_call_err2:
                                        print(
                                            f"  [迭代 {t}] 警告: acqf调用失败: {acqf_call_err2}"
                                        )
                                        raise

                                if acqf_values.numel() > 0:
                                    diagnostics["acqf_mean"] = float(acqf_values.mean())
                                    diagnostics["acqf_std"] = float(acqf_values.std())
                        else:
                            print(f"  [迭代 {t}] 警告: generator没有pool_points属性")
                    except Exception as acqf_err:
                        print(f"  [迭代 {t}] 警告: 无法获取acqf统计: {acqf_err}")

        except Exception as diag_error:
            print(f"  警告 (迭代 {t}): 诊断信息获取错误: {diag_error}")

        # ========== 计算最小距离 ==========
        min_dist = np.nan
        if len(sampling_history) > 1:
            history_array = np.array(sampling_history[:-1])
            distances = cdist([x_array], history_array, metric="euclidean")[0]
            min_dist = float(distances.min())

        # ========== 记录数据 ==========
        logs["iteration"].append(t + 1)
        logs["x_points"].append(x_array)
        logs["y_values"].append(y)
        logs["lambda_t"].append(diagnostics["lambda_t"])
        logs["gamma_t"].append(diagnostics["gamma_t"])
        logs["r_t"].append(diagnostics["r_t"])
        logs["n_train"].append(diagnostics["n_train"])
        logs["model_entropy"].append(diagnostics["model_entropy"])
        logs["acqf_mean"].append(diagnostics["acqf_mean"])
        logs["acqf_std"].append(diagnostics["acqf_std"])
        logs["min_distance"].append(min_dist)

        # ========== 每100次或最后一次记录交互项 ==========
        if (t + 1) % 100 == 0 or (t + 1) == BUDGET:
            interaction_logs[f"iteration_{t+1}"] = {
                "lambda_t": (
                    float(diagnostics["lambda_t"])
                    if not pd.isna(diagnostics["lambda_t"])
                    else None
                ),
                "gamma_t": (
                    float(diagnostics["gamma_t"])
                    if not pd.isna(diagnostics["gamma_t"])
                    else None
                ),
                "r_t": (
                    float(diagnostics["r_t"])
                    if not pd.isna(diagnostics["r_t"])
                    else None
                ),
                "n_train": int(diagnostics["n_train"]),
            }

        # ========== 进度显示 ==========
        if (t + 1) % 10 == 0 or (t + 1) == BUDGET:
            lambda_str = (
                f"{diagnostics['lambda_t']:.4f}"
                if not pd.isna(diagnostics["lambda_t"])
                else "N/A"
            )
            gamma_str = (
                f"{diagnostics['gamma_t']:.4f}"
                if not pd.isna(diagnostics["gamma_t"])
                else "N/A"
            )
            entropy_str = (
                f"{diagnostics['model_entropy']:.4f}"
                if not pd.isna(diagnostics["model_entropy"])
                else "N/A"
            )
            min_dist_str = f"{min_dist:.4f}" if not pd.isna(min_dist) else "N/A"
            print(
                f"[{t+1:3d}/{BUDGET}] y={y}, "
                f"λ={lambda_str}, γ={gamma_str}, "
                f"H={entropy_str}, "
                f"d_min={min_dist_str}"
            )

    except Exception as e:
        print(f"✗ 迭代 {t} 失败: {e}")
        import traceback

        traceback.print_exc()
        break

print(f"\n✓ 采样完成: {len(logs['y_values'])} 个样本")

# ========================================
# 步骤5: 保存数据文件
# ========================================
print("\n" + "=" * 80)
print("步骤5: 保存数据文件")
print("=" * 80)

# 5.1 保存 test_data.csv
df = pd.DataFrame(
    {
        "iteration": logs["iteration"],
        "y_value": logs["y_values"],
        "lambda_t": logs["lambda_t"],
        "gamma_t": logs["gamma_t"],
        "r_t": logs["r_t"],
        "n_train": logs["n_train"],
        "model_entropy": logs["model_entropy"],
        "acqf_mean": logs["acqf_mean"],
        "acqf_std": logs["acqf_std"],
        "min_distance": logs["min_distance"],
    }
)

for i in range(6):
    df[f"x{i}"] = [x[i] for x in logs["x_points"]]

test_data_file = data_dir / "test_data.csv"
df.to_csv(test_data_file, index=False)
print(f"✓ test_data.csv 已保存: {test_data_file}")

# 5.2 保存 sampling_history.npy
sampling_history_array = np.array(sampling_history, dtype=np.float32)
history_file = data_dir / "sampling_history.npy"
np.save(history_file, sampling_history_array)
print(f"✓ sampling_history.npy 已保存: {history_file}")

# 5.3 保存 interaction_log.json
interaction_log_file = data_dir / "interaction_log.json"
with open(interaction_log_file, "w", encoding="utf-8") as f:
    json.dump(interaction_logs, f, indent=2, ensure_ascii=False)
print(f"✓ interaction_log.json 已保存: {interaction_log_file}")

# 5.4 保存摘要信息
y_array = np.array(logs["y_values"])
summary = {
    "timestamp": timestamp,
    "budget": BUDGET,
    "samples_collected": len(logs["y_values"]),
    "y_statistics": {
        "min": int(y_array.min()),
        "max": int(y_array.max()),
        "mean": float(y_array.mean()),
        "std": float(y_array.std()),
    },
    "lambda_statistics": {
        "min": float(np.nanmin(logs["lambda_t"])),
        "max": float(np.nanmax(logs["lambda_t"])),
        "mean": float(np.nanmean(logs["lambda_t"])),
    },
    "gamma_statistics": {
        "min": float(np.nanmin(logs["gamma_t"])),
        "max": float(np.nanmax(logs["gamma_t"])),
        "mean": float(np.nanmean(logs["gamma_t"])),
    },
    "config_file": str(config_path),
    "result_dir": str(result_dir),
    "design_space_file": str(design_space_path),
    "oracle_seed": 42,
    "interaction_pairs": [[1, 2], [3, 4], [0, 5]],
}

summary_file = data_dir / "summary.json"
with open(summary_file, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"✓ summary.json 已保存: {summary_file}")

print("\n" + "=" * 80)
print("实验完成".center(80))
print(f"结果保存在: {result_dir}".center(80))
print("=" * 80 + "\n")
