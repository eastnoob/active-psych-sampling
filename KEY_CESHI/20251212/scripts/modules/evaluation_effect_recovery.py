#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的效应识别和预测质量评估 (v2)

核心改进：
1. 直接用线性回归系数作为effect估计，而不是方差
2. 计算与Oracle权重的Spearman相关性，量化采样质量
3. 支持序数和回归两种模型类型
4. 精确ANOVA分解（在边际期望下）
"""

import torch
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")


def evaluate_effect_recovery_v2(
    server,
    oracle,
    design_space: np.ndarray,
    configured_pairs: Optional[List[Tuple[int, int]]] = None,
    n_grid: int = 500,
    test_fraction: float = 0.3,
) -> Dict:
    """
    通过线性回归系数与Oracle权重对标，评估采样策略的效应识别能力

    核心思路：
    1. 从model的训练数据中拟合线性模型: y ~ x + interactions
    2. 提取回归系数作为ANOVA效应估计
    3. 与Oracle的真实权重计算相关性
    4. 相关性高 = 采样策略好，低 = 采样不足或分布偏

    Args:
        server: AEPsych server (包含训练好的模型)
        oracle: SingleSubject Oracle实例
        design_space: 完整的设计空间(1200×6)
        configured_pairs: 配置的交互对列表
        n_grid: 用于边际期望估计的网格大小
        test_fraction: 测试集比例

    Returns:
        dict: 包含相关性、系数对比、预测质量的完整评估
    """
    try:
        strat = server._strats[0]
        model = strat.model

        # ========== 第一步：获取Oracle真实权重 ==========
        oracle_spec = oracle.get_model_spec()
        oracle_weights = np.array(oracle_spec.get("weights", [0] * 6))

        oracle_interactions = {}
        int_terms = oracle_spec.get("interaction_terms", {})
        for term_name, weight in int_terms.items():
            # 解析交互项名称，支持1-indexed格式
            parts = term_name.replace("interaction_", "").replace("x", "").split("*")
            if len(parts) >= 2:
                try:
                    i, j = int(parts[0]), int(parts[1])
                    # 如果是1-indexed，转换为0-indexed
                    if min(i, j) >= 1:
                        i, j = i - 1, j - 1
                    oracle_interactions[(min(i, j), max(i, j))] = weight
                except (ValueError, IndexError):
                    continue

        # 确定要评估的交互对
        if configured_pairs is not None:
            pairs_to_eval = sorted(configured_pairs)
        else:
            pairs_to_eval = list(oracle_interactions.keys())

        print(f"\n{'='*70}")
        print(f"【Oracle规格 (Ground Truth)】")
        print(f"{'='*70}")
        print(f"  主效应权重: {oracle_weights}")
        print(f"  交互权重: {oracle_interactions}")
        print(f"  评估交互对: {pairs_to_eval}")

        # ========== 第二步：从模型训练数据拟合线性模型 ==========
        # 获取训练数据
        train_X = (
            model.train_inputs[0].cpu().numpy()
            if hasattr(model.train_inputs[0], "cpu")
            else model.train_inputs[0]
        )
        train_y_raw = (
            model.train_targets.cpu().numpy()
            if hasattr(model.train_targets, "cpu")
            else model.train_targets
        )

        n_train = train_X.shape[0]
        print(f"\n  训练样本数: {n_train}")

        # 构造特征矩阵: [1, x0, x1, ..., x5, x0*x5, x1*x2, x3*x4]
        feature_cols = ["intercept"] + [f"x{i}" for i in range(6)]
        feature_cols += [f"x{i}*x{j}" for i, j in pairs_to_eval]

        X_design = np.column_stack(
            [
                np.ones(n_train),  # intercept
                train_X,  # main effects
                *(
                    train_X[:, i] * train_X[:, j] for i, j in pairs_to_eval
                ),  # interactions
            ]
        )

        # 对于序数模型，需要转换target
        if hasattr(model.likelihood, "cutpoints"):
            # 序数模型：将概率转换为期望评分
            print(f"  模型类型: 序数 (Ordinal)")
            with torch.no_grad():
                probs = model.predict_probs(torch.from_numpy(train_X).to(torch.float64))
                probs_np = probs.cpu().numpy()
                cutpoints = model.likelihood.cutpoints.detach().cpu().numpy()
                # 期望评分 = sum(class_idx * prob_class)
                train_y = np.array(
                    [
                        np.sum(np.arange(len(probs_np[i])) * probs_np[i])
                        for i in range(len(probs_np))
                    ]
                )
        else:
            print(f"  模型类型: 回归 (Regression)")
            # Debug: check shape before flatten
            print(f"  [DEBUG] train_y_raw shape: {train_y_raw.shape if hasattr(train_y_raw, 'shape') else len(train_y_raw)}")
            print(f"  [DEBUG] train_X shape: {train_X.shape}")

            # Fix: ensure train_y matches train_X length
            if hasattr(train_y_raw, "flatten"):
                train_y = train_y_raw.flatten()
            else:
                train_y = train_y_raw

            # If shapes don't match, something is wrong
            if len(train_y) != len(train_X):
                print(f"  [WARNING] Shape mismatch: train_y={len(train_y)}, train_X={len(train_X)}")
                print(f"  [WARNING] This might indicate a model data issue, skipping evaluation")
                raise ValueError(f"train_y shape {len(train_y)} doesn't match train_X shape {len(train_X)}")

        # 拟合线性模型
        lr_model = LinearRegression()
        lr_model.fit(X_design, train_y)

        # 提取系数
        lr_intercept = lr_model.intercept_
        lr_coef = lr_model.coef_

        # 系数对应关系
        lr_main = lr_coef[1:7]  # x0-x5的系数
        lr_interaction = lr_coef[7 : 7 + len(pairs_to_eval)]  # 交互项系数

        print(f"\n{'='*70}")
        print(f"【线性回归系数 (模型估计)】")
        print(f"{'='*70}")
        print(f"  拟合R²: {lr_model.score(X_design, train_y):.4f}")
        print(f"  截距: {lr_intercept:.6f}")
        print(f"  主效应系数: {lr_main}")

        print(f"\n  交互系数:")
        for (i, j), coef in zip(pairs_to_eval, lr_interaction):
            oracle_val = oracle_interactions.get((i, j), 0)
            print(f"    x{i}*x{j}: {coef:+.6f} (Oracle: {oracle_val:+.6f})")

        # ========== 第三步：计算相关性 ==========
        # 主效应相关性
        main_corr, main_pval = spearmanr(oracle_weights, lr_main)
        main_corr_pearson, _ = pearsonr(oracle_weights, lr_main)

        # 交互相关性
        oracle_int_vals = np.array(
            [oracle_interactions.get((i, j), 0) for i, j in pairs_to_eval]
        )
        if len(pairs_to_eval) > 1:
            int_corr, int_pval = spearmanr(oracle_int_vals, lr_interaction)
            int_corr_pearson, _ = pearsonr(oracle_int_vals, lr_interaction)
        else:
            int_corr = int_pval = int_corr_pearson = np.nan

        print(f"\n{'='*70}")
        print(f"【效应恢复质量评估】")
        print(f"{'='*70}")
        print(f"  主效应 Spearman相关性: {main_corr:.4f} (p={main_pval:.4f})")
        print(f"  主效应 Pearson相关性:  {main_corr_pearson:.4f}")
        if not np.isnan(int_corr):
            print(f"  交互效应 Spearman相关性: {int_corr:.4f} (p={int_pval:.4f})")
            print(f"  交互效应 Pearson相关性:  {int_corr_pearson:.4f}")

        # ========== 第四步：计算RMSE和MAE ==========
        main_rmse = np.sqrt(np.mean((oracle_weights - lr_main) ** 2))
        main_mae = np.mean(np.abs(oracle_weights - lr_main))

        int_rmse = np.sqrt(np.mean((oracle_int_vals - lr_interaction) ** 2))
        int_mae = np.mean(np.abs(oracle_int_vals - lr_interaction))

        print(f"\n  主效应 RMSE: {main_rmse:.6f}")
        print(f"  主效应 MAE:  {main_mae:.6f}")
        print(f"  交互效应 RMSE: {int_rmse:.6f}")
        print(f"  交互效应 MAE:  {int_mae:.6f}")

        # ========== 第五步：预测质量 ==========
        # 在完整设计空间上评估预测
        X_test = torch.from_numpy(design_space).to(torch.float64)
        y_oracle = np.array([oracle(x) - 1 for x in design_space])  # Likert转为0-3

        with torch.no_grad():
            if hasattr(model.likelihood, "cutpoints"):
                probs = model.predict_probs(X_test)
                y_pred = torch.argmax(probs, dim=1).cpu().numpy()
            else:
                posterior = model.posterior(X_test)
                y_pred = posterior.mean.squeeze(-1).cpu().numpy()
                y_pred = np.clip(y_pred, 0, 4)

        pred_rmse = np.sqrt(np.mean((y_oracle - y_pred) ** 2))
        pred_mae = np.mean(np.abs(y_oracle - y_pred))

        # R²分数
        ss_res = np.sum((y_oracle - y_pred) ** 2)
        ss_tot = np.sum((y_oracle - y_oracle.mean()) ** 2)
        pred_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        print(f"\n{'='*70}")
        print(f"【预测质量 (设计空间 n={len(design_space)})】")
        print(f"{'='*70}")
        print(f"  R²:   {pred_r2:.4f}")
        print(f"  RMSE: {pred_rmse:.4f}")
        print(f"  MAE:  {pred_mae:.4f}")

        # ========== 第六步：采样覆盖分析 ==========
        # 检查采样点分布
        print(f"\n{'='*70}")
        print(f"【采样分布分析】")
        print(f"{'='*70}")
        print(f"  采样点总数: {n_train}")

        # 计算采样密度（按维度）
        for dim in range(6):
            train_min, train_max = train_X[:, dim].min(), train_X[:, dim].max()
            design_min, design_max = (
                design_space[:, dim].min(),
                design_space[:, dim].max(),
            )
            coverage = (train_max - train_min) / (design_max - design_min + 1e-8)
            print(
                f"  维度x{dim}: [{train_min:.2f}, {train_max:.2f}] / [{design_min:.2f}, {design_max:.2f}] (覆盖率: {coverage:.1%})"
            )

        print(f"\n{'='*70}\n")

        return {
            "main_correlation": {
                "spearman": float(main_corr),
                "spearman_pval": float(main_pval),
                "pearson": float(main_corr_pearson),
            },
            "interaction_correlation": {
                "spearman": float(int_corr) if not np.isnan(int_corr) else None,
                "spearman_pval": float(int_pval) if not np.isnan(int_pval) else None,
                "pearson": (
                    float(int_corr_pearson) if not np.isnan(int_corr_pearson) else None
                ),
            },
            "main_effects": {
                "oracle": oracle_weights.tolist(),
                "estimated": lr_main.tolist(),
                "rmse": float(main_rmse),
                "mae": float(main_mae),
            },
            "interaction_effects": {
                "pairs": pairs_to_eval,
                "oracle": oracle_int_vals.tolist(),
                "estimated": lr_interaction.tolist(),
                "rmse": float(int_rmse),
                "mae": float(int_mae),
            },
            "prediction_quality": {
                "R2": float(pred_r2),
                "RMSE": float(pred_rmse),
                "MAE": float(pred_mae),
                "n_test": len(design_space),
            },
            "model_fit": {
                "lr_r2": float(lr_model.score(X_design, train_y)),
                "n_train": n_train,
                "model_type": (
                    "ordinal"
                    if hasattr(model.likelihood, "cutpoints")
                    else "regression"
                ),
            },
        }

    except Exception as e:
        print(f"\n✗ 效应恢复评估失败: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "main_correlation": {},
            "interaction_correlation": {},
            "main_effects": {},
            "interaction_effects": {},
            "prediction_quality": {},
            "model_fit": {},
        }


def evaluate_effect_recovery_from_samples(
    X_train: np.ndarray,
    y_train: np.ndarray,
    oracle,
    design_space: np.ndarray,
    configured_pairs: Optional[List[Tuple[int, int]]] = None,
) -> Dict:
    """
    直接从采样数据（X, y）评估effect恢复能力，不需要model实例

    Args:
        X_train: 采样点 (n, 6)
        y_train: Oracle对应的输出值 (n,)
        oracle: SingleSubject Oracle实例
        design_space: 完整设计空间
        configured_pairs: 要评估的交互对

    Returns:
        dict: 评估结果
    """
    try:
        # ========== 第一步：获取Oracle真实权重 ==========
        oracle_spec = oracle.get_model_spec()
        oracle_weights = np.array(oracle_spec.get("weights", [0] * 6))

        # 重要：从Oracle获取实际的交互对，不要依赖配置的交互对
        # 因为配置可能与Oracle实际定义不一致
        oracle_interactions = {}
        int_terms = oracle_spec.get("interaction_terms", {})
        for term_name, weight in int_terms.items():
            # 支持多种格式：
            # - "x2*x3" → 可能是 0-indexed (2,3) 或 1-indexed (2-1, 3-1)=(1,2)
            # - "x1*x6" 一定是 1-indexed，因为没有 x0*x5 这样的写法
            cleaned = term_name.replace("interaction_", "")
            parts = cleaned.replace("x", "").split("*")
            if len(parts) >= 2:
                try:
                    i, j = int(parts[0]), int(parts[1])
                    # 关键：检测是否为1-indexed
                    # 如果最小索引是1而不是0，说明是1-indexed，需要转换
                    if min(i, j) >= 1:
                        # 1-indexed → 0-indexed
                        i, j = i - 1, j - 1
                    oracle_interactions[(min(i, j), max(i, j))] = weight
                except (ValueError, IndexError):
                    continue

        # 用Oracle的实际交互对，而不是配置的
        # 如果配置也提供了，用配置的（有时候可能有意义），否则用Oracle的
        if configured_pairs is not None:
            # 配置优先，但要检查一致性
            configured_set = set(tuple(sorted(p)) for p in configured_pairs)
            oracle_set = set(oracle_interactions.keys())

            if configured_set != oracle_set:
                print(f"\n[WARNING] 配置交互对与Oracle不一致！")
                print(f"  配置的: {sorted(configured_set)}")
                print(f"  Oracle的: {sorted(oracle_set)}")
                print(f"  使用Oracle的交互对进行评估")

            pairs_to_eval = sorted(oracle_set)
        else:
            pairs_to_eval = sorted(oracle_interactions.keys())

        print(f"\n{'='*70}")
        print(f"【Oracle规格 (Ground Truth)】")
        print(f"{'='*70}")
        print(f"  主效应权重: {oracle_weights}")
        print(f"  交互权重: {oracle_interactions}")
        print(f"  评估交互对: {pairs_to_eval}")

        # ========== 第二步：拟合线性模型 ==========
        feature_cols = ["intercept"] + [f"x{i}" for i in range(6)]
        feature_cols += [f"x{i}*x{j}" for i, j in pairs_to_eval]

        X_design = np.column_stack(
            [
                np.ones(len(X_train)),
                X_train,
                *(X_train[:, i] * X_train[:, j] for i, j in pairs_to_eval),
            ]
        )

        lr_model = LinearRegression()
        lr_model.fit(X_design, y_train)

        lr_intercept = lr_model.intercept_
        lr_coef = lr_model.coef_

        lr_main = lr_coef[1:7]
        lr_interaction = lr_coef[7 : 7 + len(pairs_to_eval)]

        print(f"\n{'='*70}")
        print(f"【线性回归系数 (从采样数据拟合)】")
        print(f"{'='*70}")
        print(f"  拟合R²: {lr_model.score(X_design, y_train):.4f}")
        print(f"  截距: {lr_intercept:.6f}")
        print(f"  主效应系数: {lr_main}")

        print(f"\n  交互系数:")
        for (i, j), coef in zip(pairs_to_eval, lr_interaction):
            oracle_val = oracle_interactions.get((i, j), 0)
            print(f"    x{i}*x{j}: {coef:+.6f} (Oracle: {oracle_val:+.6f})")

        # ========== 第三步：计算相关性 ==========
        main_corr, main_pval = spearmanr(oracle_weights, lr_main)
        main_corr_pearson, _ = pearsonr(oracle_weights, lr_main)

        oracle_int_vals = np.array(
            [oracle_interactions.get((i, j), 0) for i, j in pairs_to_eval]
        )
        if len(pairs_to_eval) > 1:
            int_corr, int_pval = spearmanr(oracle_int_vals, lr_interaction)
            int_corr_pearson, _ = pearsonr(oracle_int_vals, lr_interaction)
        else:
            int_corr = int_pval = int_corr_pearson = np.nan

        print(f"\n{'='*70}")
        print(f"【效应恢复质量评估】")
        print(f"{'='*70}")
        print(f"  主效应 Spearman相关性: {main_corr:.4f} (p={main_pval:.4f})")
        print(f"  主效应 Pearson相关性:  {main_corr_pearson:.4f}")
        if not np.isnan(int_corr):
            print(f"  交互效应 Spearman相关性: {int_corr:.4f} (p={int_pval:.4f})")
            print(f"  交互效应 Pearson相关性:  {int_corr_pearson:.4f}")

        # ========== 第四步：计算RMSE和MAE ==========
        main_rmse = np.sqrt(np.mean((oracle_weights - lr_main) ** 2))
        main_mae = np.mean(np.abs(oracle_weights - lr_main))

        int_rmse = np.sqrt(np.mean((oracle_int_vals - lr_interaction) ** 2))
        int_mae = np.mean(np.abs(oracle_int_vals - lr_interaction))

        print(f"\n  主效应 RMSE: {main_rmse:.6f}")
        print(f"  主效应 MAE:  {main_mae:.6f}")
        print(f"  交互效应 RMSE: {int_rmse:.6f}")
        print(f"  交互效应 MAE:  {int_mae:.6f}")

        # ========== 第五步：采样覆盖分析 ==========
        print(f"\n{'='*70}")
        print(f"【采样分布分析】")
        print(f"{'='*70}")
        print(f"  采样点总数: {len(X_train)}")

        for dim in range(6):
            train_min, train_max = X_train[:, dim].min(), X_train[:, dim].max()
            design_min, design_max = (
                design_space[:, dim].min(),
                design_space[:, dim].max(),
            )
            coverage = (train_max - train_min) / (design_max - design_min + 1e-8)
            print(
                f"  维度x{dim}: [{train_min:.2f}, {train_max:.2f}] / [{design_min:.2f}, {design_max:.2f}] (覆盖率: {coverage:.1%})"
            )

        print(f"\n{'='*70}\n")

        return {
            "main_correlation": {
                "spearman": float(main_corr),
                "spearman_pval": float(main_pval),
                "pearson": float(main_corr_pearson),
            },
            "interaction_correlation": {
                "spearman": float(int_corr) if not np.isnan(int_corr) else None,
                "spearman_pval": float(int_pval) if not np.isnan(int_pval) else None,
                "pearson": (
                    float(int_corr_pearson) if not np.isnan(int_corr_pearson) else None
                ),
            },
            "main_effects": {
                "oracle": oracle_weights.tolist(),
                "estimated": lr_main.tolist(),
                "rmse": float(main_rmse),
                "mae": float(main_mae),
            },
            "interaction_effects": {
                "pairs": pairs_to_eval,
                "oracle": oracle_int_vals.tolist(),
                "estimated": lr_interaction.tolist(),
                "rmse": float(int_rmse),
                "mae": float(int_mae),
            },
            "model_fit": {
                "lr_r2": float(lr_model.score(X_design, y_train)),
                "n_train": len(X_train),
            },
        }

    except Exception as e:
        print(f"\n✗ 效应恢复评估失败: {e}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}

    """
    对比多个采样策略的效应恢复能力

    Args:
        results_list: 多个evaluation_v2返回结果的列表
    """
    print(f"\n{'='*70}")
    print(f"【采样策略对比】")
    print(f"{'='*70}")

    # 提取关键指标
    for i, res in enumerate(results_list):
        if "error" in res:
            print(f"\n策略 {i}: 失败 ({res['error']})")
            continue

        main_corr = res["main_correlation"].get("spearman", np.nan)
        int_corr = res["interaction_correlation"].get("spearman", np.nan)
        r2 = res["prediction_quality"]["R2"]
        lr_r2 = res["model_fit"]["lr_r2"]

        print(f"\n策略 {i}:")
        print(f"  主效应恢复 (Spearman): {main_corr:+.4f}")
        print(f"  交互恢复 (Spearman):   {int_corr:+.4f}")
        print(f"  预测R²:              {r2:.4f}")
        print(f"  线性拟合R²:          {lr_r2:.4f}")
        print(f"  样本数:              {res['model_fit']['n_train']}")
