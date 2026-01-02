#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
采样循环和诊断信息获取模块
负责执行采样循环并收集诊断数据
"""

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Any


def run_sampling_loop(
    server, oracle, design_space, budget: int, logger=None
) -> Dict[str, List[Any]]:
    """
    执行采样循环

    Args:
        server: AEPsych Server实例
        oracle: Oracle模型实例
        design_space: 设计空间数组
        budget: 采样预算（总迭代次数）
        logger: loguru logger实例（可选）

    Returns:
        logs: 包含所有采样数据的字典
    """
    print("\n" + "=" * 80)
    print("步骤4: 采样循环")
    print("=" * 80)

    BUDGET = budget
    print(f"  Budget: {BUDGET}")

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
            y_likert = oracle(x_array)  # Oracle自动离散化为 {1,2,3,4,5}
            y = float(y_likert - 1)  # 转换为INI空间 {0,1,2,3,4}
            print(f"[迭代 {t}] oracle Likert={int(y_likert)}, INI空间={int(y)}")

            # ========== Tell ==========
            x_config = {k: extract_value(v) for k, v in x_dict.items()}
            tell_message = {
                "type": "tell",
                "message": {"config": x_config, "outcome": y},
            }
            server.handle_request(tell_message)

            # DEBUG: 检查模型训练数据
            if hasattr(server, '_strats') and len(server._strats) > 0:
                strat = server._strats[0]
                if hasattr(strat, 'model') and strat.model is not None:
                    model = strat.model
                    if hasattr(model, 'train_inputs') and model.train_inputs is not None and len(model.train_inputs) > 0:
                        n_inputs = len(model.train_inputs[0])
                        if hasattr(model, 'train_targets') and model.train_targets is not None:
                            n_targets = len(model.train_targets)
                            print(f"  [DEBUG after tell {t}] train_inputs: {n_inputs}, train_targets: {n_targets}")
                            if n_inputs != n_targets:
                                print(f"  ⚠️ WARNING: Shape mismatch detected!")
                        else:
                            print(f"  [DEBUG after tell {t}] train_inputs: {n_inputs}, train_targets: None")

            # ========== 获取诊断信息 ==========
            diagnostics = _get_diagnostics(server, sampling_history, t, logger)

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

            # ========== 每10次或最后一次记录 ==========
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
    return logs


def _get_diagnostics(server, sampling_history, t: int, logger=None) -> Dict[str, float]:
    """
    获取当前迭代的诊断信息

    Args:
        server: AEPsych Server实例
        sampling_history: 采样历史列表
        t: 当前迭代索引
        logger: loguru logger实例（可选）

    Returns:
        diagnostics: 诊断信息字典
    """
    diagnostics = {
        "lambda_t": np.nan,
        "gamma_t": np.nan,
        "r_t": np.nan,
        "n_train": t + 1,
        "model_entropy": np.nan,
        "acqf_mean": np.nan,
        "acqf_std": np.nan,
    }

    try:
        if hasattr(server, "_strats") and len(server._strats) > 0:
            strat = server._strats[0]

            # 1. 从采集函数获取权重
            if hasattr(strat, "generator") and hasattr(strat.generator, "acqf"):
                acqf_instance = None

                if (
                    hasattr(strat.generator, "_acqf_instance")
                    and strat.generator._acqf_instance is not None
                ):
                    acqf_instance = strat.generator._acqf_instance
                else:
                    acqf_instance = strat.generator._instantiate_acquisition_fn(
                        strat.model
                    )

                if hasattr(acqf_instance, "_ensure_fresh_data"):
                    acqf_instance._ensure_fresh_data()

                # 从采集函数获取诊断信息
                if hasattr(acqf_instance, "get_diagnostics"):
                    try:
                        acqf_diag = acqf_instance.get_diagnostics()
                        lambda_t = acqf_diag.get("lambda_t", np.nan)
                        gamma_t = acqf_diag.get("gamma_t", np.nan)
                        r_t = acqf_diag.get("r_t", np.nan)
                        n_train = acqf_diag.get("n_train", t + 1)

                        if not pd.isna(lambda_t):
                            diagnostics["lambda_t"] = lambda_t
                        if not pd.isna(gamma_t):
                            diagnostics["gamma_t"] = gamma_t
                        if not pd.isna(r_t):
                            diagnostics["r_t"] = r_t
                        if not pd.isna(n_train):
                            diagnostics["n_train"] = n_train

                        # 如果诊断信息仍为空，尝试从weight_engine直接获取
                        if pd.isna(diagnostics["lambda_t"]) and hasattr(
                            acqf_instance, "weight_engine"
                        ):
                            try:
                                lambda_t = acqf_instance.weight_engine.compute_lambda()
                                gamma_t = acqf_instance.weight_engine.compute_gamma()
                                r_t = (
                                    acqf_instance.weight_engine.compute_relative_main_variance()
                                )

                                if not pd.isna(lambda_t):
                                    diagnostics["lambda_t"] = lambda_t
                                if not pd.isna(gamma_t):
                                    diagnostics["gamma_t"] = gamma_t
                                if not pd.isna(r_t):
                                    diagnostics["r_t"] = r_t

                            except Exception as we_err:
                                if t >= 10 and logger:
                                    logger.debug(
                                        f"[迭代 {t}] 无法从weight_engine获取诊断: {we_err}"
                                    )

                    except Exception as acqf_diag_err:
                        if logger:
                            logger.debug(
                                f"[迭代 {t}] 无法获取acqf诊断: {acqf_diag_err}"
                            )

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
                        diagnostics["n_train"] = acqf_instance.weight_engine._n_train

                    except Exception as we_err:
                        if t >= 10 and logger:
                            logger.debug(
                                f"[迭代 {t}] 无法获取weight_engine诊断: {we_err}"
                            )

            # 2. 从模型获取 entropy（不确定性）
            if hasattr(strat, "model"):
                model = strat.model
                try:
                    if (
                        hasattr(model, "train_inputs")
                        and model.train_inputs is not None
                    ):
                        with torch.no_grad():
                            X_train = model.train_inputs[0]
                            posterior = model.posterior(X_train)
                            variances = posterior.variance.cpu().numpy()
                            diagnostics["model_entropy"] = float(np.mean(variances))
                    else:
                        if len(sampling_history) > 0:
                            with torch.no_grad():
                                X_hist = torch.tensor(
                                    sampling_history, dtype=torch.float32
                                )
                                posterior = model.posterior(X_hist)
                                variances = posterior.variance.cpu().numpy()
                                diagnostics["model_entropy"] = float(np.mean(variances))

                except Exception as model_err:
                    if logger:
                        logger.debug(f"[迭代 {t}] 无法获取model_entropy: {model_err}")

            # 3. 从采集函数获取统计信息
            if hasattr(strat, "generator") and hasattr(strat.generator, "pool_points"):
                try:
                    pool_points = strat.generator.pool_points

                    if isinstance(pool_points, np.ndarray):
                        pool_tensor = torch.tensor(pool_points, dtype=torch.float32)
                    else:
                        pool_tensor = pool_points.float()

                    with torch.no_grad():
                        try:
                            acqf_values = acqf_instance(pool_tensor.unsqueeze(-2))
                        except Exception:
                            try:
                                acqf_values = acqf_instance(pool_tensor)
                            except Exception as acqf_call_err:
                                if logger:
                                    logger.debug(
                                        f"[迭代 {t}] acqf调用失败: {acqf_call_err}"
                                    )
                                raise

                        if acqf_values.numel() > 0:
                            diagnostics["acqf_mean"] = float(acqf_values.mean())
                            diagnostics["acqf_std"] = float(acqf_values.std())

                except Exception as acqf_err:
                    if logger:
                        logger.debug(f"[迭代 {t}] 无法获取acqf统计: {acqf_err}")

    except Exception as diag_error:
        if logger:
            logger.debug(f"[迭代 {t}] 诊断信息获取错误: {diag_error}")

    return diagnostics
