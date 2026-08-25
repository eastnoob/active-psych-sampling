#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互敏感度分析模块

在已训练的主效应模型上计算两两维度的交互敏感度，
无需修改kernel或重新训练模型。

策略：基于设计空间扫描的预测结果，使用多种方法检测显著的交互对。
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger


class InteractionSensitivityAnalyzer:
    """基于预测的交互对检测"""

    def __init__(
        self,
        design_df: pd.DataFrame,
        predictions: np.ndarray,  # shape (N,) or (N, 2) for mean and std
        factor_names: List[str],
        residuals: Optional[np.ndarray] = None,  # 可选：模型残差
    ):
        """
        Args:
            design_df: 设计空间数据框（已编码的物理值）
            predictions: 模型预测，shape (N,) 或 (N, 2)
            factor_names: 因子名称列表
            residuals: 可选的残差 shape (N,)
        """
        self.design_df = design_df
        self.factor_names = factor_names
        self.d = len(factor_names)
        
        if predictions.ndim == 1:
            self.pred_mean = predictions
            self.pred_std = None
        else:
            self.pred_mean = predictions[:, 0]
            self.pred_std = predictions[:, 1] if predictions.shape[1] > 1 else None
        
        self.residuals = residuals
        self.X = design_df.values.astype(float)
        
        logger.info(f"[InteractionAnalyzer] Initialized with {self.d} factors, {len(design_df)} points")

    def compute_all_pairwise_scores(self) -> List[Tuple[Tuple[int, int], float]]:
        """计算所有可能的交互对的得分"""
        all_pairs = list(itertools.combinations(range(self.d), 2))
        scores = []
        
        for i, j in all_pairs:
            # 综合多种方法的得分
            score = 0.0
            
            # 方法1: 四象限差异（基于预测均值）
            if self.pred_mean is not None:
                q_score = self._quadrant_variance_score(i, j)
                score += 0.4 * q_score
            
            # 方法2: 预测不确定性交互（基于std）
            if self.pred_std is not None:
                unc_score = self._uncertainty_interaction_score(i, j)
                score += 0.3 * unc_score
            
            # 方法3: 残差模式检测
            if self.residuals is not None:
                res_score = self._residual_pattern_score(i, j)
                score += 0.3 * res_score
            
            scores.append(((i, j), score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _quadrant_variance_score(self, i: int, j: int) -> float:
        """
        基于四象限的方差差异检测交互
        
        交互模式：两个因子的组合会产生不同方向的效果
        """
        # 在维度i和j的中位数处分割
        median_i = np.median(self.X[:, i])
        median_j = np.median(self.X[:, j])
        
        quad_00 = self.pred_mean[(self.X[:, i] <= median_i) & (self.X[:, j] <= median_j)]
        quad_01 = self.pred_mean[(self.X[:, i] <= median_i) & (self.X[:, j] > median_j)]
        quad_10 = self.pred_mean[(self.X[:, i] > median_i) & (self.X[:, j] <= median_j)]
        quad_11 = self.pred_mean[(self.X[:, i] > median_i) & (self.X[:, j] > median_j)]
        
        # 如果任何象限样本太少，返回0
        if any(len(q) < 3 for q in [quad_00, quad_01, quad_10, quad_11]):
            return 0.0
        
        mean_00 = np.mean(quad_00)
        mean_01 = np.mean(quad_01)
        mean_10 = np.mean(quad_10)
        mean_11 = np.mean(quad_11)
        
        # 对角线和反对角线的差异
        # 交互存在 → 对角线效果强度不同
        diagonal_diff = abs((mean_00 + mean_11) - (mean_01 + mean_10))
        
        # 标准化
        all_means = np.concatenate([quad_00, quad_01, quad_10, quad_11])
        overall_std = np.std(all_means)
        
        if overall_std < 1e-6:
            return 0.0
        
        score = diagonal_diff / overall_std
        return max(0.0, min(1.0, score))  # 归一化到[0, 1]

    def _uncertainty_interaction_score(self, i: int, j: int) -> float:
        """
        基于预测不确定性的交互检测
        
        如果(i,j)存在交互，当两者同时变化时不确定性应该较高
        """
        # 将两个因子离散化为high/low
        median_i = np.median(self.X[:, i])
        median_j = np.median(self.X[:, j])
        
        std_00 = self.pred_std[(self.X[:, i] <= median_i) & (self.X[:, j] <= median_j)]
        std_01 = self.pred_std[(self.X[:, i] <= median_i) & (self.X[:, j] > median_j)]
        std_10 = self.pred_std[(self.X[:, i] > median_i) & (self.X[:, j] <= median_j)]
        std_11 = self.pred_std[(self.X[:, i] > median_i) & (self.X[:, j] > median_j)]
        
        if any(len(s) < 3 for s in [std_00, std_01, std_10, std_11]):
            return 0.0
        
        # 交互区域（同时高）的不确定性是否特别高？
        mean_std_interaction = np.mean(std_11)
        mean_std_main = np.mean(np.concatenate([
            self.pred_std[(self.X[:, i] > median_i)],
            self.pred_std[(self.X[:, j] > median_j)]
        ]))
        
        if mean_std_main < 1e-6:
            return 0.0
        
        score = mean_std_interaction / (mean_std_main + 1e-6)
        return max(0.0, min(1.0, score))

    def _residual_pattern_score(self, i: int, j: int) -> float:
        """
        基于残差模式的交互检测
        
        需要原始数据的残差。交互存在时四象限的残差应有特定模式
        仅在有完整残差数据时使用（避免伪残差导致的虚假信号）
        """
        # 只有当残差与设计空间大小一致时才运行
        if len(self.residuals) != len(self.X):
            return 0.0
        
        median_i = np.median(self.X[:, i])
        median_j = np.median(self.X[:, j])
        
        res_00 = self.residuals[(self.X[:, i] <= median_i) & (self.X[:, j] <= median_j)]
        res_01 = self.residuals[(self.X[:, i] <= median_i) & (self.X[:, j] > median_j)]
        res_10 = self.residuals[(self.X[:, i] > median_i) & (self.X[:, j] <= median_j)]
        res_11 = self.residuals[(self.X[:, i] > median_i) & (self.X[:, j] > median_j)]
        
        if any(len(r) < 3 for r in [res_00, res_01, res_10, res_11]):
            return 0.0
        
        mean_res_00 = np.mean(res_00)
        mean_res_01 = np.mean(res_01)
        mean_res_10 = np.mean(res_10)
        mean_res_11 = np.mean(res_11)
        
        # 交互模式：对角线同号，反对角线异号
        diagonal_same = (mean_res_00 * mean_res_11 > 0)
        anti_diag_diff = (mean_res_01 * mean_res_10 < 0)
        
        if diagonal_same and anti_diag_diff:
            pattern_strength = abs((mean_res_00 + mean_res_11) - (mean_res_01 + mean_res_10))
            overall_std = np.std(self.residuals)
            if overall_std < 1e-6:
                return 0.0
            score = pattern_strength / overall_std
            return max(0.0, min(1.0, score))
        else:
            return 0.0

    def select_top_interactions(
        self, k: int = 6, threshold: float = 0.0
    ) -> List[Tuple[Tuple[int, int], float]]:
        """选择top-k的交互对"""
        all_scores = self.compute_all_pairwise_scores()
        
        # 按阈值筛选
        filtered = [(pair, score) for pair, score in all_scores if score >= threshold]
        
        # 取top-k
        selected = filtered[:k]
        
        logger.info(f"[InteractionAnalyzer] Found {len(selected)} interactions above threshold {threshold}")
        for (i, j), score in selected:
            logger.info(f"  ({self.factor_names[i]}, {self.factor_names[j]}): {score:.4f}")
        
        return selected


def analyze_interactions_for_step3(
    design_df: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
    residuals: Optional[np.ndarray],
    factor_names: List[str],
    k: int = 6,
) -> Dict:
    """
    集成函数：用于Step3在非ANOVA模式下调用
    
    Returns:
        {
            "interactions": [(pair_tuple, score), ...],
            "interaction_dict": {"(i,j)": score, ...}
        }
    """
    # 将均值和标准差组合成二维数组
    predictions = np.column_stack([means, stds])
    
    analyzer = InteractionSensitivityAnalyzer(
        design_df=design_df,
        predictions=predictions,  # 包含均值和标准差
        factor_names=factor_names,
        residuals=residuals,
    )
    
    interactions = analyzer.select_top_interactions(k=k, threshold=0.0)
    
    # 转换为字典格式
    interaction_dict = {}
    for (i, j), score in interactions:
        key = f"({i}, {j})"
        interaction_dict[key] = {
            "factors": [factor_names[i], factor_names[j]],
            "score": float(score),
        }
    
    return {
        "interactions": interactions,
        "interaction_dict": interaction_dict,
    }


if __name__ == "__main__":
    # 简单测试
    print("Interaction Sensitivity Analyzer loaded successfully")
