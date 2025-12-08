"""
Phase 1 数据分析模块
用于从预热阶段数据中提取关键信息，为Phase 2主动学习做准备
"""

import numpy as np
import pandas as pd
import itertools
from typing import List, Tuple, Dict, Any
import warnings


def analyze_phase1_data(
    X_warmup: np.ndarray,
    y_warmup: np.ndarray,
    subject_ids: np.ndarray,
    factor_names: List[str] = None,
    max_pairs: int = 5,
    min_pairs: int = 3,
    selection_method: str = "elbow",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    分析Phase 1预热数据，提取关键信息供Phase 2使用

    这是连接Phase 1和Phase 2的核心桥梁函数

    Args:
        X_warmup: Phase 1数据，形状 (n_samples, n_factors)
        y_warmup: Phase 1响应变量，形状 (n_samples,)
        subject_ids: 被试ID，形状 (n_samples,)
        factor_names: 因子名称列表
        max_pairs: 最多选择的交互对数量
        min_pairs: 最少选择的交互对数量
        selection_method: 交互对选择方法 ('elbow', 'bic_threshold', 'top_k')
        verbose: 是否打印详细信息

    Returns:
        dict: 包含以下关键信息
            - selected_pairs: 筛选出的交互对 (list of tuples)
            - lambda_init: 初始λ权重
            - main_effects: 主效应估计
            - interaction_effects: 交互效应估计
            - diagnostics: 诊断信息
    """
    n_samples, d = X_warmup.shape

    if factor_names is None:
        factor_names = [f"factor_{i}" for i in range(d)]
    elif len(factor_names) != d:
        # 如果提供的factor_names长度不匹配，使用默认名称
        print(
            f"警告: factor_names长度({len(factor_names)})与因子数({d})不匹配，使用默认名称"
        )
        factor_names = [f"factor_{i}" for i in range(d)]

    if verbose:
        print("=" * 80)
        print("Phase 1 数据分析")
        print("=" * 80)
        print(f"样本数: {n_samples}")
        print(f"因子数: {d}")
        print(f"被试数: {len(np.unique(subject_ids))}")
        print()

    # === Step 1: 交互对筛选 ===
    if verbose:
        print("Step 1: 交互对筛选...")

    selected_pairs, interaction_scores = _select_interaction_pairs(
        X_warmup, y_warmup, d, max_pairs, min_pairs, selection_method, verbose
    )

    # === Step 2: λ参数估计 ===
    if verbose:
        print("\nStep 2: λ参数估计...")

    lambda_init, var_decomposition = _estimate_lambda(
        X_warmup, y_warmup, subject_ids, selected_pairs, d, verbose
    )

    # === Step 3: 主效应和交互效应估计 ===
    if verbose:
        print("\nStep 3: 效应估计...")

    main_effects, interaction_effects = _estimate_effects(
        X_warmup, y_warmup, subject_ids, selected_pairs, d, factor_names, verbose
    )

    # === Step 4: 诊断信息 ===
    diagnostics = {
        "n_samples": n_samples,
        "n_subjects": len(np.unique(subject_ids)),
        "n_factors": d,
        "n_selected_pairs": len(selected_pairs),
        "interaction_scores": dict(interaction_scores[:15]),  # top-15
        "var_decomposition": var_decomposition,
        "selection_method": selection_method,
    }

    if verbose:
        print("\n" + "=" * 80)
        print("Phase 1 分析完成!")
        print("=" * 80)
        print(f"筛选出的交互对: {len(selected_pairs)}个")
        for i, pair in enumerate(selected_pairs):
            pair_name = f"({factor_names[pair[0]]}, {factor_names[pair[1]]})"
            score = dict(interaction_scores).get(pair, 0)
            print(f"  {i+1}. {pair_name}: score={score:.3f}")
        print(f"\n初始λ估计: {lambda_init:.3f}")
        print()

    return {
        "selected_pairs": selected_pairs,
        "lambda_init": lambda_init,
        "main_effects": main_effects,
        "interaction_effects": interaction_effects,
        "diagnostics": diagnostics,
    }


def _select_interaction_pairs(
    X: np.ndarray,
    y: np.ndarray,
    d: int,
    max_pairs: int,
    min_pairs: int,
    method: str,
    verbose: bool,
) -> Tuple[List[Tuple[int, int]], List[Tuple[Tuple[int, int], float]]]:
    """
    筛选显著的交互对

    方法：
    1. 拟合加性模型（只有主效应）
    2. 分析残差中的交互模式
    3. 对每个可能的交互对评分
    4. 按选择方法确定最终交互对
    """
    from sklearn.linear_model import LinearRegression

    # 拟合加性模型（只有主效应）
    additive_model = LinearRegression()
    additive_model.fit(X, y)
    residuals = y - additive_model.predict(X)

    # 计算加性模型的BIC
    n = len(y)
    rss_additive = np.sum(residuals**2)
    bic_additive = n * np.log(rss_additive / n) + (d + 1) * np.log(n)

    # 对每个可能的交互对评分
    interaction_scores = []
    all_pairs = list(itertools.combinations(range(d), 2))

    for i, j in all_pairs:
        # 方法1: 残差模式评分（四象限分析）
        pattern_score = _compute_interaction_pattern_score(residuals, X[:, [i, j]])

        # 方法2: BIC增益（拟合包含该交互的模型）
        X_with_interaction = np.column_stack([X, X[:, i] * X[:, j]])
        model_with_interaction = LinearRegression()
        model_with_interaction.fit(X_with_interaction, y)
        residuals_with_int = y - model_with_interaction.predict(X_with_interaction)
        rss_with_int = np.sum(residuals_with_int**2)
        bic_with_int = n * np.log(rss_with_int / n) + (d + 2) * np.log(n)
        bic_gain = bic_additive - bic_with_int  # 正值表示改进

        # 方法3: 方差解释（该交互解释的额外方差占比）
        var_total = np.var(y)
        var_explained_additive = np.var(additive_model.predict(X))
        var_explained_with_int = np.var(
            model_with_interaction.predict(X_with_interaction)
        )
        var_gain = (var_explained_with_int - var_explained_additive) / var_total

        # 综合评分（加权平均）
        score = 0.3 * pattern_score + 0.5 * max(bic_gain, 0) + 0.2 * max(var_gain, 0)

        interaction_scores.append(((i, j), score))

    # 排序
    interaction_scores.sort(key=lambda x: x[1], reverse=True)

    # 根据方法选择交互对
    if method == "top_k":
        # 固定选择top-K
        k = max_pairs
        selected_pairs = [pair for pair, score in interaction_scores[:k]]

    elif method == "bic_threshold":
        # 只选择BIC增益 > 阈值的交互对
        selected_pairs = []
        for pair, score in interaction_scores:
            if score > 0 and len(selected_pairs) < max_pairs:
                selected_pairs.append(pair)
        # 确保至少有min_pairs个
        if len(selected_pairs) < min_pairs:
            selected_pairs = [pair for pair, score in interaction_scores[:min_pairs]]

    elif method == "elbow":
        # 肘部法则：找拐点
        scores = [score for pair, score in interaction_scores[: max_pairs * 2]]
        k = _find_elbow_point(scores, min_pairs, max_pairs)
        selected_pairs = [pair for pair, score in interaction_scores[:k]]

    else:
        raise ValueError(f"Unknown selection method: {method}")

    if verbose:
        print(f"  候选交互对总数: {len(all_pairs)}")
        print(f"  筛选方法: {method}")
        print(f"  选择的交互对数: {len(selected_pairs)}")

    return selected_pairs, interaction_scores


def _compute_interaction_pattern_score(
    residuals: np.ndarray, X_pair: np.ndarray
) -> float:
    """
    计算交互模式得分（基于四象限残差分析）

    如果存在交互效应，四个象限的残差应该呈现特定模式
    """
    # 将两个因子二值化（中位数分割）
    median_0 = np.median(X_pair[:, 0])
    median_1 = np.median(X_pair[:, 1])

    # 四个象限
    quad_00 = residuals[(X_pair[:, 0] <= median_0) & (X_pair[:, 1] <= median_1)]
    quad_01 = residuals[(X_pair[:, 0] <= median_0) & (X_pair[:, 1] > median_1)]
    quad_10 = residuals[(X_pair[:, 0] > median_0) & (X_pair[:, 1] <= median_1)]
    quad_11 = residuals[(X_pair[:, 0] > median_0) & (X_pair[:, 1] > median_1)]

    # 如果某个象限样本太少，返回0
    if any(len(q) < 5 for q in [quad_00, quad_01, quad_10, quad_11]):
        return 0.0

    # 计算各象限的平均残差
    mean_00 = np.mean(quad_00)
    mean_01 = np.mean(quad_01)
    mean_10 = np.mean(quad_10)
    mean_11 = np.mean(quad_11)

    # 交互模式：对角线象限的残差应该同号，反对角线应该异号
    # 计算"交互信号强度"
    diagonal_diff = abs((mean_00 + mean_11) - (mean_01 + mean_10))
    overall_std = np.std(residuals)

    # 标准化得分
    if overall_std > 1e-6:
        pattern_score = diagonal_diff / overall_std
    else:
        pattern_score = 0.0

    return pattern_score


def _find_elbow_point(scores: List[float], min_k: int, max_k: int) -> int:
    """
    使用肘部法则找拐点
    """
    if len(scores) < min_k:
        return len(scores)

    # 计算一阶差分
    diffs = np.diff(scores)

    # 找最大下降的位置
    elbow_idx = np.argmin(diffs) + 1

    # 限制在[min_k, max_k]范围内
    k = max(min_k, min(elbow_idx, max_k))

    return k


def _estimate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    selected_pairs: List[Tuple[int, int]],
    d: int,
    verbose: bool,
) -> Tuple[float, Dict[str, float]]:
    """
    估算λ_max参数（Phase 2中λ的目标上限）

    改进方法（使用 Adjusted R²）：
    1. 分别拟合仅包含主效应的模型和包含交互的完整模型
    2. 使用 Adjusted R² 而非普通 R²，防止参数堆积导致的虚高
    3. 计算 Δ_adj = max(0, R²_adj_full - R²_adj_main)
    4. λ_max = clamp(0.2 + 1.5 × Raw_Ratio, min=0.2, max=0.9)
       其中 Raw_Ratio = Δ_adj / R²_adj_full
    """
    try:
        from sklearn.linear_model import LinearRegression

        n = len(y)

        # 模型1：仅主效应
        model_main = LinearRegression()
        model_main.fit(X, y)
        y_pred_main = model_main.predict(X)
        rss_main = np.sum((y - y_pred_main) ** 2)
        r2_main = model_main.score(X, y)

        # 计算 Adjusted R² for main model
        p_main = d  # 主效应参数数量
        r2_adj_main = (
            1 - (1 - r2_main) * (n - 1) / (n - p_main - 1)
            if n > p_main + 1
            else r2_main
        )

        # 模型2：主效应 + 交互
        X_with_interactions = X.copy()
        for i, j in selected_pairs:
            interaction_col = X[:, i] * X[:, j]
            X_with_interactions = np.column_stack(
                [X_with_interactions, interaction_col]
            )

        model_full = LinearRegression()
        model_full.fit(X_with_interactions, y)
        y_pred_full = model_full.predict(X_with_interactions)
        rss_full = np.sum((y - y_pred_full) ** 2)
        r2_full = model_full.score(X_with_interactions, y)

        # 计算 Adjusted R² for full model
        p_full = d + len(selected_pairs)  # 主效应 + 交互项参数数量
        r2_adj_full = (
            1 - (1 - r2_full) * (n - 1) / (n - p_full - 1)
            if n > p_full + 1
            else r2_full
        )

        # 计算交互的真实贡献（使用 Adjusted R²）
        delta_adj = max(0, r2_adj_full - r2_adj_main)

        # 如果 Δ_adj ≤ 0，说明交互效应不存在或被参数堆积掩盖
        if delta_adj <= 0 or r2_adj_full < 0.01:
            lambda_max = 0.2  # 最小值，Phase 2 只需轻微关注交互
            if verbose:
                print(f"  ⚠️  Adjusted R² 未显示交互效应提升，λ_max 设为最小值 0.2")
        else:
            # 计算 Raw Ratio
            raw_ratio = delta_adj / r2_adj_full if r2_adj_full > 0 else 0

            # 映射到 λ_max（线性截断方式）
            lambda_max = np.clip(0.2 + 1.5 * raw_ratio, 0.2, 0.9)

        # 方差分解：基于 Adjusted R²
        y_var = np.var(y)
        var_explained_main = y_var * r2_adj_main
        var_explained_interaction = y_var * delta_adj
        var_residual = y_var * (1 - r2_adj_full)

        var_decomposition = {
            "main_variance": float(var_explained_main),
            "interaction_variance": float(var_explained_interaction),
            "residual_variance": float(var_residual),
            "r2_main": float(r2_main),
            "r2_full": float(r2_full),
            "r2_adj_main": float(r2_adj_main),
            "r2_adj_full": float(r2_adj_full),
            "delta_adj": float(delta_adj),
            "raw_ratio": float(raw_ratio) if delta_adj > 0 else 0.0,
        }

        if verbose:
            print(f"  主效应 R²: {r2_main:.4f}, Adj R²: {r2_adj_main:.4f}")
            print(f"  完整模型 R²: {r2_full:.4f}, Adj R²: {r2_adj_full:.4f}")
            print(f"  Δ_adj (交互真实贡献): {delta_adj:.4f}")
            if delta_adj > 0:
                print(f"  Raw Ratio: {raw_ratio:.4f}")
            print(f"  λ_max 估计: {lambda_max:.3f}")

    except Exception as e:
        warnings.warn(f"Lambda estimation failed: {e}. Using default lambda_max=0.5")
        lambda_max = 0.5
        var_decomposition = {
            "main_variance": 0.0,
            "interaction_variance": 0.0,
            "residual_variance": 0.0,
            "error": str(e),
        }

    return lambda_max, var_decomposition


def _estimate_effects(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    selected_pairs: List[Tuple[int, int]],
    d: int,
    factor_names: List[str],
    verbose: bool,
) -> Tuple[Dict[str, Any], Dict[Tuple[int, int], Any]]:
    """
    估计主效应和交互效应
    """
    from sklearn.linear_model import LinearRegression

    # 主效应估计（简单线性回归）
    main_effects = {}
    for i in range(d):
        model = LinearRegression()
        model.fit(X[:, [i]], y)
        main_effects[factor_names[i]] = {
            "coef": float(model.coef_[0]),
            "intercept": float(model.intercept_),
        }

    # 交互效应估计
    interaction_effects = {}
    for i, j in selected_pairs:
        # 拟合包含该交互的模型
        X_with_int = np.column_stack([X[:, i], X[:, j], X[:, i] * X[:, j]])
        model = LinearRegression()
        model.fit(X_with_int, y)

        interaction_effects[(i, j)] = {
            "coef_i": float(model.coef_[0]),
            "coef_j": float(model.coef_[1]),
            "coef_interaction": float(model.coef_[2]),
            "pair_name": f"({factor_names[i]}, {factor_names[j]})",
        }

    if verbose:
        print(f"  主效应估计完成: {d}个因子")
        print(f"  交互效应估计完成: {len(selected_pairs)}个交互对")

    return main_effects, interaction_effects
