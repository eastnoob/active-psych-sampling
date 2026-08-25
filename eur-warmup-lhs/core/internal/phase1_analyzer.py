"""
Phase 1 数据分析模块
用于从预热阶段数据中提取关键信息，为Phase 2主动学习做准备
"""

import numpy as np
import pandas as pd
import itertools
from typing import List, Tuple, Dict, Any
import warnings
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from loguru import logger


def analyze_phase1_data(
    X_warmup: np.ndarray,
    y_warmup: np.ndarray,
    subject_ids: np.ndarray,
    factor_names: List[str] = None,
    max_pairs: int = 5,
    min_pairs: int = 3,
    selection_method: str = "elbow",
    suspected_pairs: List[Tuple[int, int]] = None,
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
        suspected_pairs: 值得怀疑的交互对 (Phase 2 强制保留)
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
        logger.warning(
            f"factor_names length ({len(factor_names)}) does not match number of factors ({d}), using default names"
        )
        factor_names = [f"factor_{i}" for i in range(d)]

    if verbose:
        print("=" * 40)
        print("Phase 1 Data Analysis")
        print("=" * 40)
        print(f"Number of samples: {n_samples}")
        print(f"Number of factors: {d}")
        print(f"Number of subjects: {len(np.unique(subject_ids))}")

    # === Step 1: 交互对筛选 ===
    if verbose:
        print("Step 1: Interaction Pair Screening (Lasso Discovery + Prior Fusion)...")

    # 1.1 自动发现交互对 (使用 Lasso 筛选)
    discovered_pairs, interaction_scores = _select_interaction_pairs(
        X_warmup, y_warmup, d, max_pairs, min_pairs, selection_method, verbose
    )

    # 1.2 融合先验怀疑项与熔断平衡
    selected_pairs = _fuse_and_balance_pairs(
        discovered_pairs, 
        suspected_pairs, 
        interaction_scores, 
        X_warmup, 
        y_warmup, 
        d,
        max_pairs, 
        verbose
    )

    if verbose:
        print(f"  Final selected interaction pairs: {selected_pairs}")

    # === Step 2: λ参数估计 ===
    if verbose:
        print("Step 2: λ Parameter Estimation...")

    lambda_init, var_decomposition = _estimate_lambda(
        X_warmup, y_warmup, subject_ids, selected_pairs, d, verbose
    )

    # === Step 3: 主效应和交互效应估计 ===
    if verbose:
        print("Step 3: Effect Estimation...")

    main_effects, interaction_effects = _estimate_effects(
        X_warmup, y_warmup, subject_ids, selected_pairs, d, factor_names, verbose
    )

    # === Step 4: 生成模型规格 (Model Specification) ===
    if verbose:
        print("Step 4: Generating Model Specification...")
    
    model_spec = _generate_model_spec(
        selected_pairs, 
        lambda_init, 
        var_decomposition, 
        main_effects, 
        interaction_effects, 
        factor_names
    )

    # === Step 5: 诊断信息 ===
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
        print("=" * 40)
        print("Phase 1 Analysis Complete!")
        print("=" * 40)
        print(f"Selected interaction pairs: {len(selected_pairs)}")
        for i, pair in enumerate(selected_pairs):
            pair_name = f"({factor_names[pair[0]]}, {factor_names[pair[1]]})"
            score = dict(interaction_scores).get(pair, 0)
            print(f"  {i+1}. {pair_name}: score={score:.3f}")
        print(f"Initial λ estimate: {lambda_init:.3f}")
        print(f"Suggested noise floor: {model_spec['noise_floor']:.4f}")

    return {
        "selected_pairs": selected_pairs,
        "lambda_init": lambda_init,
        "main_effects": main_effects,
        "interaction_effects": interaction_effects,
        "model_spec": model_spec,
        "diagnostics": diagnostics,
    }


def _generate_model_spec(
    selected_pairs: List[Tuple[int, int]],
    lambda_init: float,
    var_decomp: Dict[str, float],
    main_effects: Dict[str, Any],
    interaction_effects: Dict[Tuple[int, int], Any],
    factor_names: List[str]
) -> Dict[str, Any]:
    """
    根据分析结果生成 BaseGP 的配置建议 (Recipe)
    """
    # 1. 长度尺度建议 (Lengthscale Priors)
    # 逻辑：效应越强，长度尺度越短（变化越剧烈）
    # 默认长度尺度设为 0.5 (在 [0, 1] 空间)
    lengthscale_priors = {}
    
    # 主效应长度尺度
    for name, effect in main_effects.items():
        coef_abs = abs(effect["coef"])
        # 映射：coef=0 -> 0.8, coef=2 -> 0.2
        ls = np.clip(0.8 - 0.3 * coef_abs, 0.1, 0.8)
        lengthscale_priors[name] = float(ls)
        
    # 2. 噪声底噪建议
    # 逻辑：基于残差方差占比
    total_var = (
        var_decomp.get("main_variance", 0) 
        + var_decomp.get("interaction_variance", 0) 
        + var_decomp.get("residual_variance", 0)
        + 1e-9 # 增加 epsilon 防止除零
    )
    residual_ratio = var_decomp.get("residual_variance", 0) / total_var
    # 噪声底噪通常在 0.01 到 0.2 之间
    noise_floor = np.clip(residual_ratio * 0.5, 0.01, 0.2)
    
    return {
        "interaction_pairs": selected_pairs,
        "lambda_init": float(lambda_init),
        "lengthscale_priors": lengthscale_priors,
        "noise_floor": float(noise_floor),
        "kernel_type": "custom_anova" if len(selected_pairs) > 0 else "default",
        "interaction_mode": "list" if len(selected_pairs) > 0 else "none"
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
    使用 Lasso 筛选显著的交互对 (Joint Effect Screening)
    """
    # 1. 构建全交互特征矩阵
    # 重要：对所有特征（主效应+交互）统一中心化，确保Lasso系数可比
    # 中心化解决两个问题：
    # (1) 分类变量索引为0导致的乘积全为0（Zero-Product Bias）
    # (2) 使主效应和交互项在相同的尺度上，Lasso惩罚才公平
    X_centered = X - np.mean(X, axis=0)

    # 1.5 效应遗传筛选（可选）
    # 先估计主效应强度，只在主效应显著的因子之间搜索交互
    candidate_pairs = None
    if method in ["stability", "bic"]:  # 对小样本方法启用效应遗传
        main_effects_strength = []
        for i in range(d):
            # 简单线性回归估计主效应
            model = LinearRegression()
            model.fit(X[:, [i]], y)
            strength = abs(model.coef_[0])
            main_effects_strength.append((i, strength))

        # 按强度排序，选择前 k 个显著因子
        main_effects_strength.sort(key=lambda x: x[1], reverse=True)
        # 自适应阈值：至少保留前50%的因子，或者强度 > 中位数的因子
        median_strength = np.median([s for _, s in main_effects_strength])
        significant_factors = [i for i, s in main_effects_strength if s >= median_strength * 0.5]

        # 只在显著因子之间构建交互对（弱遗传原则）
        candidate_pairs = list(itertools.combinations(significant_factors, 2))

        if verbose:
            print(f"  Effect Heredity: {len(significant_factors)}/{d} factors with strong main effects")
            print(f"  Candidate interaction pairs reduced from {d*(d-1)//2} to {len(candidate_pairs)}")

    # 使用候选对或全部对
    all_pairs = candidate_pairs if candidate_pairs else list(itertools.combinations(range(d), 2))
    X_interactions = np.zeros((X.shape[0], len(all_pairs)))

    for idx, (i, j) in enumerate(all_pairs):
        X_interactions[:, idx] = X_centered[:, i] * X_centered[:, j]

    # 使用中心化后的主效应 + 交互项
    X_full = np.column_stack([X_centered, X_interactions])

    # 2. 标准化 (Lasso 对量纲敏感)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)

    # 3. 拟合 LassoCV 自动选择惩罚强度
    try:
        # 使用较小的 cv 值以适应小样本
        lasso = LassoCV(cv=min(5, X.shape[0] // 5), random_state=42, max_iter=2000)
        lasso.fit(X_scaled, y)
        coefs = lasso.coef_
    except Exception as e:
        if verbose:
            logger.warning(f"  LassoCV failed ({e}), falling back to fixed alpha Lasso")
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_scaled, y)
        coefs = lasso.coef_

    # 4. 提取交互项系数并评分
    # 前 d 个是主效应，后面是交互项
    interaction_coefs = coefs[d:]
    interaction_scores = []
    for idx, pair in enumerate(all_pairs):
        score = abs(interaction_coefs[idx])
        interaction_scores.append((pair, score))

    # 排序
    interaction_scores.sort(key=lambda x: x[1], reverse=True)

    # 5. 根据方法选择交互对
    if method == "top_k":
        k = max_pairs
        selected_pairs = [pair for pair, score in interaction_scores[:k] if score > 1e-5]
    elif method == "elbow":
        scores = [score for pair, score in interaction_scores if score > 1e-5]
        if len(scores) > 0:
            k = _find_elbow_point(scores, min_pairs, max_pairs)
            selected_pairs = [pair for pair, score in interaction_scores[:k]]
        else:
            selected_pairs = []
    elif method == "bic":
        # 使用 BIC 准则选择最优交互对数量
        k, bic_values = _select_pairs_by_bic(X, y, d, interaction_scores, min_pairs, max_pairs, verbose)
        selected_pairs = [pair for pair, score in interaction_scores[:k]]
    elif method == "stability":
        # 使用 Stability Selection (推荐用于小样本)
        selected_pairs, stability_scores = _stability_selection(
            X, y, d,
            n_bootstrap=100,
            subsample_ratio=0.5,
            stability_threshold=0.6,
            max_pairs=max_pairs,
            verbose=verbose
        )
        # 将稳定性得分转换为 interaction_scores 格式以便后续使用
        interaction_scores = [(pair, stability_scores.get(pair, 0.0)) for pair in
                             [p for p, _ in interaction_scores]]
        interaction_scores.sort(key=lambda x: x[1], reverse=True)
    else:
        # 默认只选非零项
        selected_pairs = [pair for pair, score in interaction_scores if score > 1e-5]
        selected_pairs = selected_pairs[:max_pairs]

    # 确保至少有 min_pairs (如果可能)
    if len(selected_pairs) < min_pairs:
        additional = [p for p, s in interaction_scores if p not in selected_pairs]
        selected_pairs.extend(additional[:(min_pairs - len(selected_pairs))])

    if verbose:
        print(f"  Lasso discovered non-zero interaction terms: {len([s for p, s in interaction_scores if s > 1e-5])}")
        print(f"  Initial screened interaction pairs: {selected_pairs}")

    return selected_pairs, interaction_scores


def _fuse_and_balance_pairs(
    discovered_pairs: List[Tuple[int, int]],
    suspected_pairs: List[Tuple[int, int]],
    interaction_scores: List[Tuple[Tuple[int, int], float]],
    X: np.ndarray,
    y: np.ndarray,
    d: int,
    max_pairs: int,
    verbose: bool,
) -> List[Tuple[int, int]]:
    """
    融合发现项与怀疑项，并执行柔性熔断逻辑
    """
    # 标准化先验对
    if suspected_pairs:
        suspected_pairs = [tuple(sorted(p)) for p in suspected_pairs]
    else:
        suspected_pairs = []

    # 合并去重
    all_candidates = list(suspected_pairs)
    for p in discovered_pairs:
        if p not in all_candidates:
            all_candidates.append(p)
    
    if len(all_candidates) <= max_pairs:
        return all_candidates

    # 如果超过限制，执行柔性熔断
    if verbose:
        logger.info(f"  [Circuit Breaker Check] Candidate interaction pairs ({len(all_candidates)}) exceed recommended limit ({max_pairs})")

    # 1. 强制保留怀疑项
    final_pairs = list(suspected_pairs)
    
    # 2. 评估剩余的发现项
    remaining_discovered = [p for p in discovered_pairs if p not in final_pairs]
    
    # 计算基础模型的 Adj R2 (主效应 + 怀疑项)
    base_adj_r2 = _calculate_adj_r2(X, y, d, final_pairs)
    
    for p in remaining_discovered:
        # 如果已经达到硬上限 (max_pairs + 3)，除非证据极强否则停止
        if len(final_pairs) >= max_pairs + 3:
            break
            
        # 计算加入该项后的 Adj R2 增益
        new_adj_r2 = _calculate_adj_r2(X, y, d, final_pairs + [p])
        gain = new_adj_r2 - base_adj_r2
        
        # 阈值逻辑：
        # 如果在 max_pairs 以内，只要 gain > 0 就保留
        # 如果超过 max_pairs，需要 gain > 0.02 (显著增益)
        threshold = 0.02 if len(final_pairs) >= max_pairs else 0.001
        
        if gain > threshold:
            final_pairs.append(p)
            base_adj_r2 = new_adj_r2
            if len(final_pairs) > max_pairs and verbose:
                logger.info(f"  [Limit Break] Interaction pair {p} brings significant gain ({gain:.4f}), retained")
        elif len(final_pairs) < max_pairs:
            # 即使增益不显著，但在名额内，也保留（尊重 Lasso 结果）
            final_pairs.append(p)
            base_adj_r2 = new_adj_r2

    if len(final_pairs) > max_pairs and verbose:
        logger.warning(f"  Final number of interaction pairs ({len(final_pairs)}) exceeds recommended value, risk of overfitting in small samples")

    return final_pairs


def _calculate_adj_r2(X: np.ndarray, y: np.ndarray, d: int, pairs: List[Tuple[int, int]]) -> float:
    """计算 Adjusted R2"""
    n = X.shape[0]
    p = d + len(pairs)
    if n <= p + 1:
        return -1.0
        
    X_full = X.copy()
    for i, j in pairs:
        X_full = np.column_stack([X_full, X[:, i] * X[:, j]])
    
    model = LinearRegression().fit(X_full, y)
    r2 = model.score(X_full, y)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adj_r2


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
    改进的肘部检测算法 - 使用二阶导数（曲率）而非一阶差分

    原理：
    - 一阶差分（斜率）在噪声数据中不稳定
    - 二阶差分（曲率）能更好地识别真正的"拐点"
    - 加入阈值过滤，避免将随机波动误判为肘部

    Args:
        scores: 降序排列的得分列表
        min_k: 最小交互对数量
        max_k: 最大交互对数量

    Returns:
        k: 肘部位置（交互对数量）
    """
    if len(scores) < 3:
        # 样本太少，返回中间值
        return max(min_k, min(len(scores), max_k))

    # 归一化得分到 [0, 1]，使阈值具有可比性
    scores_norm = np.array(scores)
    scores_norm = (scores_norm - scores_norm.min()) / (scores_norm.max() - scores_norm.min() + 1e-10)

    # 计算一阶差分（斜率）
    first_diff = -np.diff(scores_norm)  # 负号因为scores是降序

    # 计算二阶差分（曲率）
    # 曲率大的地方是真正的"拐点"
    if len(first_diff) < 2:
        second_diff = first_diff
    else:
        second_diff = np.diff(first_diff)

    # 找到曲率最大的点（二阶差分最大）
    # 这表示斜率变化最剧烈的地方
    if len(second_diff) > 0:
        # 只考虑前 max_k 个位置
        search_range = min(len(second_diff), max_k - 1)
        elbow_idx = int(np.argmax(second_diff[:search_range])) + 1

        # 验证：这个肘部是否显著？
        # 要求：(1) 曲率足够大 (2) 前面的得分下降足够明显
        if search_range > 0:
            max_curvature = second_diff[elbow_idx - 1] if elbow_idx - 1 < len(second_diff) else 0
            avg_curvature = np.mean(np.abs(second_diff[:search_range]))

            # 如果最大曲率不显著（小于平均值的1.5倍），说明没有明显肘部
            if max_curvature < avg_curvature * 1.5:
                # 退化策略：选择得分下降最快的位置
                if len(first_diff) > 0:
                    elbow_idx = int(np.argmax(first_diff[:search_range])) + 1
                else:
                    elbow_idx = min(3, len(scores))  # 默认选3个
    else:
        # 没有足够数据计算二阶差分，使用一阶差分
        elbow_idx = int(np.argmax(first_diff[:min(len(first_diff), max_k - 1)])) + 1

    # 限制在[min_k, max_k]范围内
    k = max(min_k, min(elbow_idx, max_k))

    return k


def _select_pairs_by_bic(
    X: np.ndarray,
    y: np.ndarray,
    d: int,
    interaction_scores: List[Tuple[Tuple[int, int], float]],
    min_pairs: int,
    max_pairs: int,
    verbose: bool,
) -> Tuple[int, List[float]]:
    """
    使用 BIC (Bayesian Information Criterion) 选择最优交互对数量

    BIC = n * log(RSS/n) + k * log(n)
    其中 RSS 是残差平方和，k 是参数数量，n 是样本数

    Returns:
        optimal_k: 最优交互对数量
        bic_values: 每个 k 对应的 BIC 值
    """
    n = X.shape[0]
    bic_values = []

    # 测试 k = 0 到 max_pairs 的所有可能
    for k in range(0, max_pairs + 1):
        # 选择得分最高的前 k 个交互对
        if k == 0:
            selected_pairs = []
        else:
            selected_pairs = [pair for pair, score in interaction_scores[:k]]

        # 构建特征矩阵
        X_model = X.copy()
        for i, j in selected_pairs:
            X_model = np.column_stack([X_model, X[:, i] * X[:, j]])

        # 拟合线性模型
        try:
            model = LinearRegression()
            model.fit(X_model, y)
            y_pred = model.predict(X_model)
            rss = np.sum((y - y_pred) ** 2)

            # 计算 BIC
            # 参数数量 = 主效应 (d) + 交互项 (k) + 截距 (1)
            num_params = d + k + 1

            # 防止 log(0)
            if rss < 1e-10:
                rss = 1e-10

            bic = n * np.log(rss / n) + num_params * np.log(n)
            bic_values.append(bic)

        except Exception as e:
            if verbose:
                logger.warning(f"  BIC calculation failed for k={k}: {e}")
            bic_values.append(np.inf)

    # 找到 BIC 最小的 k
    optimal_k = int(np.argmin(bic_values))

    # 确保在 [min_pairs, max_pairs] 范围内
    # 但如果 BIC 明确指示更少的交互对，尊重 BIC 的判断
    if optimal_k < min_pairs and bic_values[optimal_k] < bic_values[min_pairs] * 0.95:
        # BIC 显著更好（5%阈值），使用 optimal_k
        if verbose:
            logger.info(f"  BIC suggests {optimal_k} pairs (below min_pairs={min_pairs}), but BIC improvement is significant")
    else:
        optimal_k = max(min_pairs, min(optimal_k, max_pairs))

    if verbose:
        print(f"  BIC values for k=0 to {max_pairs}: {[f'{v:.2f}' for v in bic_values]}")
        print(f"  Optimal k selected by BIC: {optimal_k}")

    return optimal_k, bic_values


def _stability_selection(
    X: np.ndarray,
    y: np.ndarray,
    d: int,
    n_bootstrap: int = 100,
    subsample_ratio: float = 0.5,
    stability_threshold: float = 0.6,
    max_pairs: int = 10,
    verbose: bool = False,
) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    使用 Stability Selection 识别稳定的交互对

    通过多次 Bootstrap 重采样，只保留在多数采样中稳定出现的交互对

    Args:
        X: 特征矩阵
        y: 响应变量
        d: 特征数量
        n_bootstrap: Bootstrap 重采样次数
        subsample_ratio: 每次采样的样本比例
        stability_threshold: 稳定性阈值 (0-1)，交互对需要在至少这个比例的采样中出现
        max_pairs: 每次 Lasso 最多选择的交互对数量
        verbose: 是否打印详细信息

    Returns:
        stable_pairs: 稳定的交互对列表
        stability_scores: 每个交互对的稳定性得分 (出现频率)
    """
    n_samples = X.shape[0]
    subsample_size = int(n_samples * subsample_ratio)
    all_pairs = list(itertools.combinations(range(d), 2))

    # 记录每个交互对在多少次采样中被选中
    selection_counts = {pair: 0 for pair in all_pairs}

    if verbose:
        print(f"  Running Stability Selection with {n_bootstrap} bootstrap iterations...")

    for b in range(n_bootstrap):
        # 随机抽取子样本
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        X_sub = X[indices]
        y_sub = y[indices]

        # 在子样本上运行 Lasso
        try:
            # 构建交互特征
            X_centered = X_sub - np.mean(X_sub, axis=0)
            X_interactions = np.zeros((X_sub.shape[0], len(all_pairs)))

            for idx, (i, j) in enumerate(all_pairs):
                X_interactions[:, idx] = X_centered[:, i] * X_centered[:, j]

            X_full = np.column_stack([X_sub, X_interactions])

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_full)

            # 拟合 Lasso
            lasso = LassoCV(cv=min(3, subsample_size // 5), random_state=b, max_iter=2000)
            lasso.fit(X_scaled, y_sub)
            coefs = lasso.coef_

            # 提取非零交互项
            interaction_coefs = coefs[d:]
            selected_in_this_run = []

            for idx, pair in enumerate(all_pairs):
                if abs(interaction_coefs[idx]) > 1e-5:
                    selected_in_this_run.append(pair)

            # 只保留得分最高的 max_pairs 个
            if len(selected_in_this_run) > max_pairs:
                scores = [(pair, abs(interaction_coefs[all_pairs.index(pair)]))
                         for pair in selected_in_this_run]
                scores.sort(key=lambda x: x[1], reverse=True)
                selected_in_this_run = [pair for pair, _ in scores[:max_pairs]]

            # 更新计数
            for pair in selected_in_this_run:
                selection_counts[pair] += 1

        except Exception as e:
            if verbose and b < 5:  # 只打印前几次的错误
                logger.warning(f"  Bootstrap iteration {b} failed: {e}")
            continue

    # 计算稳定性得分 (选择频率)
    stability_scores = {pair: count / n_bootstrap for pair, count in selection_counts.items()}

    # 筛选稳定的交互对
    stable_pairs = [pair for pair, score in stability_scores.items()
                   if score >= stability_threshold]

    # 按稳定性得分排序
    stable_pairs.sort(key=lambda p: stability_scores[p], reverse=True)

    if verbose:
        print(f"  Stability Selection complete:")
        print(f"    Pairs with stability >= {stability_threshold}: {len(stable_pairs)}")
        if stable_pairs:
            print(f"    Top stable pairs:")
            for pair in stable_pairs[:10]:
                print(f"      {pair}: {stability_scores[pair]:.3f}")

    return stable_pairs, stability_scores


def _estimate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    selected_pairs: List[Tuple[int, int]],
    d: int,
    verbose: bool,
) -> Tuple[float, Dict[str, float]]:
    """
    估算lambda_max参数（Phase 2中lambda的目标上限）

    改进方法（使用 Adjusted R2）：
    1. 分别拟合仅包含主效应的模型和包含交互的完整模型
    2. 使用 Adjusted R2 而非普通 R2，防止参数堆积导致的虚高
    3. 计算 Delta_adj = max(0, R2_adj_full - R2_adj_main)
    4. lambda_max = clamp(0.2 + 1.5 × Raw_Ratio, min=0.2, max=0.9)
       其中 Raw_Ratio = Delta_adj / R2_adj_full
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

        # 计算交互的真实贡献（使用 Adjusted R2）
        delta_adj = max(0, r2_adj_full - r2_adj_main)

        # 如果 Delta_adj <= 0，说明交互效应不存在或被参数堆积掩盖
        if delta_adj <= 0 or r2_adj_full < 0.01:
            lambda_max = 0.2  # 最小值，Phase 2 只需轻微关注交互
            if verbose:
                logger.warning(f"  Adjusted R2 showed no improvement from interaction effects, lambda_max set to minimum 0.2")
        else:
            # 计算 Raw Ratio
            raw_ratio = delta_adj / r2_adj_full if r2_adj_full > 0 else 0

            # 映射到 lambda_max（线性截断方式）
            lambda_max = np.clip(0.2 + 1.5 * raw_ratio, 0.2, 0.9)

        # 方差分解：基于 Adjusted R2
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
            print(f"  Main effect R2: {r2_main:.4f}, Adj R2: {r2_adj_main:.4f}")
            print(f"  Full model R2: {r2_full:.4f}, Adj R2: {r2_adj_full:.4f}")
            print(f"  Delta_adj (True interaction contribution): {delta_adj:.4f}")
            if delta_adj > 0:
                print(f"  Raw Ratio: {raw_ratio:.4f}")
            print(f"  lambda_max estimate: {lambda_max:.3f}")

    except Exception as e:
        logger.error(f"Lambda estimation failed: {e}. Using default lambda_max=0.5")
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
        print(f"  Main effect estimation complete: {d} factors")
        print(f"  Interaction effect estimation complete: {len(selected_pairs)} pairs")

    return main_effects, interaction_effects
