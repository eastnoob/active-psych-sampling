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
        print(f"警告: factor_names长度({len(factor_names)})与因子数({d})不匹配，使用默认名称")
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
    rss_additive = np.sum(residuals ** 2)
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
        rss_with_int = np.sum(residuals_with_int ** 2)
        bic_with_int = n * np.log(rss_with_int / n) + (d + 2) * np.log(n)
        bic_gain = bic_additive - bic_with_int  # 正值表示改进

        # 方法3: 方差解释（该交互解释的额外方差占比）
        var_total = np.var(y)
        var_explained_additive = np.var(additive_model.predict(X))
        var_explained_with_int = np.var(model_with_interaction.predict(X_with_interaction))
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
        scores = [score for pair, score in interaction_scores[:max_pairs * 2]]
        k = _find_elbow_point(scores, min_pairs, max_pairs)
        selected_pairs = [pair for pair, score in interaction_scores[:k]]

    else:
        raise ValueError(f"Unknown selection method: {method}")

    if verbose:
        print(f"  候选交互对总数: {len(all_pairs)}")
        print(f"  筛选方法: {method}")
        print(f"  选择的交互对数: {len(selected_pairs)}")

    return selected_pairs, interaction_scores


def _compute_interaction_pattern_score(residuals: np.ndarray, X_pair: np.ndarray) -> float:
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
    估算λ参数（主效应 vs 交互效应的相对权重）

    方法：
    1. 拟合包含主效应和选定交互的混合效应模型
    2. 计算主效应和交互效应解释的方差
    3. λ = 交互方差 / (主方差 + 交互方差)
    """
    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        # 准备数据
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
        df["y"] = y
        df["subject"] = subject_ids

        # 添加交互项
        for i, j in selected_pairs:
            df[f"x{i}_x{j}"] = X[:, i] * X[:, j]

        # 构建公式
        main_terms = " + ".join([f"x{i}" for i in range(d)])
        interaction_terms = " + ".join([f"x{i}_x{j}" for i, j in selected_pairs])
        formula = f"y ~ {main_terms}"
        if interaction_terms:
            formula += f" + {interaction_terms}"

        # 拟合混合效应模型
        model = MixedLM.from_formula(formula, data=df, groups=df["subject"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method="lbfgs", maxiter=100)

        # 提取系数
        params = result.params
        main_coefs = [params[f"x{i}"] for i in range(d)]
        interaction_coefs = [params[f"x{i}_x{j}"] for i, j in selected_pairs]

        # 计算方差
        var_main = np.var(main_coefs) if len(main_coefs) > 0 else 0
        var_interaction = np.var(interaction_coefs) if len(interaction_coefs) > 0 else 0

        # 估算λ
        total_var = var_main + var_interaction
        if total_var > 1e-6:
            lambda_init = var_interaction / total_var
        else:
            lambda_init = 0.5  # 默认值

        # 限制在合理范围[0.2, 0.9]
        lambda_init = np.clip(lambda_init, 0.2, 0.9)

        var_decomposition = {
            "var_main": float(var_main),
            "var_interaction": float(var_interaction),
            "total_var": float(total_var),
        }

        if verbose:
            print(f"  主效应方差: {var_main:.4f}")
            print(f"  交互效应方差: {var_interaction:.4f}")
            print(f"  λ估计: {lambda_init:.3f}")

    except Exception as e:
        warnings.warn(f"Mixed model fitting failed: {e}. Using default lambda=0.5")
        lambda_init = 0.5
        var_decomposition = {
            "var_main": 0.0,
            "var_interaction": 0.0,
            "total_var": 0.0,
            "error": str(e),
        }

    return lambda_init, var_decomposition


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
