"""
Phase 1 预热采样策略实现
基于设计文档 phase1_warmup_strategy_new.md

目标：
- 175次预算 (7人×25次)
- Core-1: 56次 (8个固定点×7人)
- Core-2: 70次 (主效应45次 + 交互25次)
- 个体点: 49次 (边界20次 + LHS 29次)
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.metrics import pairwise_distances
from typing import List, Tuple, Dict, Any, Optional
import logging
import json
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phase1WarmupSampler:
    """
    Phase 1 预热采样器

    作用：筛选器 + 初始化器
    输入：完整设计空间CSV（1200个配置）
    输出：3个候选交互对 + 初始GP + 不确定性地图
    预算：7人 × 25次 = 175次（23%总预算）
    """

    def __init__(
        self,
        design_df: pd.DataFrame,
        n_subjects: int = 7,
        trials_per_subject: int = 25,
        priority_pairs: Optional[List[Tuple[int, int]]] = None,
        interaction_selection: str = "auto",
        seed: Optional[int] = None,
    ):
        """
        初始化Phase 1预热采样器

        Args:
            design_df: 设计空间DataFrame，包含f1...fd列
            n_subjects: 受试者数量（默认7人）
            trials_per_subject: 每人试验次数（默认25次）
            priority_pairs: 优先交互对列表 [(f1_idx,f2_idx), ...]，最多5对
            interaction_selection: 交互对选择方法 ('auto', 'variance', 'correlation')
            seed: 随机种子
        """
        self.design_df = design_df.copy()
        self.n_subjects = n_subjects
        self.trials_per_subject = trials_per_subject
        self.total_budget = n_subjects * trials_per_subject
        self.seed = seed or 42

        np.random.seed(self.seed)

        # 检测因子
        self.factor_names = [col for col in design_df.columns if col.startswith("f")]
        self.d = len(self.factor_names)

        # 优先交互对（最多5对）
        self.priority_pairs = priority_pairs[:5] if priority_pairs else None
        self.interaction_selection = interaction_selection

        # 预算分配
        self.budget = {
            "core1_unique": 8,  # Core-1固定点数
            "core1_total": 8 * n_subjects,  # Core-1总次数
            "core2_main": 45,
            "core2_inter": 25,
            "boundary": 20,
            "lhs": 29,
        }

        # 采样结果
        self.core1_points = None
        self.selected_samples = {
            "core1": [],
            "core2_main": [],
            "core2_inter": [],
            "boundary": [],
            "lhs": [],
        }

        logger.info(
            f"初始化Phase1WarmupSampler: d={self.d}, n_subjects={n_subjects}, "
            f"total_budget={self.total_budget}"
        )

    def select_core1_points(self) -> pd.DataFrame:
        """
        选择Core-1固定重复点（8个）
        策略：角点 + 中心点

        Returns:
            包含8个Core-1点的DataFrame
        """
        logger.info("=== 开始选择Core-1固定重复点 ===")

        candidates = []

        # 计算每个因子的范围
        factor_stats = {}
        for factor in self.factor_names:
            values = self.design_df[factor].values
            factor_stats[factor] = {
                "min": np.min(values),
                "max": np.max(values),
                "median": np.median(values),
            }

        # 1. 全取最小值角点
        all_min = {f: factor_stats[f]["min"] for f in self.factor_names}
        candidates.append(("all_min", all_min))

        # 2. 全取最大值角点
        all_max = {f: factor_stats[f]["max"] for f in self.factor_names}
        candidates.append(("all_max", all_max))

        # 3-6. 交替模式角点（4个）
        if self.d >= 2:
            # 奇数维高，偶数维低
            alternating_1 = {}
            for i, f in enumerate(self.factor_names):
                alternating_1[f] = (
                    factor_stats[f]["max"] if i % 2 == 0 else factor_stats[f]["min"]
                )
            candidates.append(("alternating_1", alternating_1))

            # 奇数维低，偶数维高
            alternating_2 = {}
            for i, f in enumerate(self.factor_names):
                alternating_2[f] = (
                    factor_stats[f]["min"] if i % 2 == 0 else factor_stats[f]["max"]
                )
            candidates.append(("alternating_2", alternating_2))

            # 前半高，后半低
            alternating_3 = {}
            half = self.d // 2
            for i, f in enumerate(self.factor_names):
                alternating_3[f] = (
                    factor_stats[f]["max"] if i < half else factor_stats[f]["min"]
                )
            candidates.append(("alternating_3", alternating_3))

            # 前半低，后半高
            alternating_4 = {}
            for i, f in enumerate(self.factor_names):
                alternating_4[f] = (
                    factor_stats[f]["min"] if i < half else factor_stats[f]["max"]
                )
            candidates.append(("alternating_4", alternating_4))

        # 7. 所有维度取中位数
        median_point = {f: factor_stats[f]["median"] for f in self.factor_names}
        candidates.append(("median", median_point))

        # 8. 中位数 + 小扰动
        perturbed_center = {}
        for f in self.factor_names:
            range_f = factor_stats[f]["max"] - factor_stats[f]["min"]
            perturbation = 0.1 * range_f * (np.random.random() - 0.5)
            perturbed_center[f] = factor_stats[f]["median"] + perturbation
        candidates.append(("perturbed", perturbed_center))

        # 从design_space中找最近邻匹配
        selected_indices = []
        for name, target in candidates[:8]:  # 确保只选8个
            idx = self._find_nearest_match(target)
            selected_indices.append(idx)
            logger.debug(f"Core-1点 {name}: 找到匹配索引 {idx}")

        self.core1_points = self.design_df.iloc[selected_indices].copy()
        self.core1_points["core1_type"] = [c[0] for c in candidates[:8]]

        logger.info(f"成功选择 {len(self.core1_points)} 个Core-1点")
        return self.core1_points

    def select_core2_main_effects(self) -> List[int]:
        """
        选择Core-2a主效应覆盖点（45次）
        策略：D-optimal设计，确保每个因子水平≥7次

        Returns:
            选中的design_df索引列表
        """
        logger.info("=== 开始选择Core-2a主效应覆盖点 ===")

        n_select = self.budget["core2_main"]

        # 排除Core-1已选的点
        core1_indices = (
            self.core1_points.index.tolist() if self.core1_points is not None else []
        )
        candidate_pool = self.design_df.drop(core1_indices, errors="ignore")

        # D-optimal选点
        selected = []
        X_selected = None

        for i in range(n_select):
            if len(candidate_pool) == 0:
                logger.warning("候选池已空，提前结束")
                break

            best_idx = None
            best_score = -np.inf

            # 对每个候选点评估D-optimal分数
            for idx in candidate_pool.index[
                : min(100, len(candidate_pool))
            ]:  # 采样加速
                # 构建设计矩阵
                if len(selected) == 0:
                    X_temp = candidate_pool.loc[[idx], self.factor_names].values
                else:
                    X_temp = np.vstack(
                        [
                            self.design_df.loc[selected, self.factor_names].values,
                            candidate_pool.loc[[idx], self.factor_names].values,
                        ]
                    )

                # 添加截距列
                X_temp = np.column_stack([np.ones(len(X_temp)), X_temp])

                # 计算信息矩阵
                try:
                    info_matrix = X_temp.T @ X_temp
                    score = np.linalg.slogdet(info_matrix)[1]  # log(det)

                    if score > best_score:
                        best_score = score
                        best_idx = idx
                except:
                    continue

            if best_idx is not None:
                selected.append(best_idx)
                candidate_pool = candidate_pool.drop(best_idx)
            else:
                # 随机选择
                random_idx = np.random.choice(candidate_pool.index)
                selected.append(random_idx)
                candidate_pool = candidate_pool.drop(random_idx)

        self.selected_samples["core2_main"] = selected
        logger.info(f"成功选择 {len(selected)} 个主效应覆盖点")
        return selected

    def select_core2_interactions(self) -> List[Tuple[int, str]]:
        """
        选择Core-2b交互初筛点（25次）
        策略：象限均衡采样，每对5次

        Returns:
            列表，每个元素为(design_idx, pair_id)
        """
        logger.info("=== 开始选择Core-2b交互初筛点 ===")

        # 确定要测试的5个交互对
        if self.priority_pairs and len(self.priority_pairs) >= 5:
            pairs = self.priority_pairs[:5]
            logger.info(f"使用用户指定的交互对")
        else:
            # 自动选择交互对
            pairs = self._select_interaction_pairs()

        logger.info(f"测试交互对: {pairs}")

        # 已选点（避免重复）
        all_selected = []
        if self.core1_points is not None:
            all_selected.extend(self.core1_points.index.tolist())
        all_selected.extend(self.selected_samples.get("core2_main", []))

        selected_with_pair = []

        for pair_idx, (fi, fj) in enumerate(pairs):
            f1_name = self.factor_names[fi]
            f2_name = self.factor_names[fj]

            # 计算中位数
            f1_median = self.design_df[f1_name].median()
            f2_median = self.design_df[f2_name].median()

            # 定义4个象限
            quadrants = {
                "low_low": (self.design_df[f1_name] <= f1_median)
                & (self.design_df[f2_name] <= f2_median),
                "low_high": (self.design_df[f1_name] <= f1_median)
                & (self.design_df[f2_name] > f2_median),
                "high_low": (self.design_df[f1_name] > f1_median)
                & (self.design_df[f2_name] <= f2_median),
                "high_high": (self.design_df[f1_name] > f1_median)
                & (self.design_df[f2_name] > f2_median),
            }

            # 从每个象限采样1次
            for q_name, q_mask in quadrants.items():
                candidates_q = self.design_df[q_mask]
                # 排除已选点
                candidates_q = candidates_q[~candidates_q.index.isin(all_selected)]

                if len(candidates_q) > 0:
                    # 选择最接近象限中心的点
                    q_center = {
                        f1_name: candidates_q[f1_name].median(),
                        f2_name: candidates_q[f2_name].median(),
                    }
                    best_idx = self._find_nearest_match(q_center, candidates_q)
                    selected_with_pair.append((best_idx, f"pair_{pair_idx}_{q_name}"))
                    all_selected.append(best_idx)

            # 第5次：从任意剩余点中随机选择
            remaining = self.design_df[~self.design_df.index.isin(all_selected)]
            if len(remaining) > 0:
                random_idx = np.random.choice(remaining.index)
                selected_with_pair.append((random_idx, f"pair_{pair_idx}_random"))
                all_selected.append(random_idx)

        self.selected_samples["core2_inter"] = selected_with_pair
        logger.info(f"成功选择 {len(selected_with_pair)} 个交互初筛点")
        return selected_with_pair

    def select_boundary_points(self) -> List[int]:
        """
        选择个体点a边界极端点（20次）
        策略：分层极端点

        Returns:
            选中的design_df索引列表
        """
        logger.info("=== 开始选择边界极端点 ===")

        boundary_candidates = []

        # 计算因子统计
        factor_stats = {}
        for f in self.factor_names:
            factor_stats[f] = {
                "min": self.design_df[f].min(),
                "max": self.design_df[f].max(),
                "median": self.design_df[f].median(),
            }

        # 1. 单维极端（d个维度，每维2个点）
        for f in self.factor_names:
            # 只有该因子取极值，其他取中位数
            config_min = {fi: factor_stats[fi]["median"] for fi in self.factor_names}
            config_min[f] = factor_stats[f]["min"]
            boundary_candidates.append(("uni_min_" + f, config_min))

            config_max = {fi: factor_stats[fi]["median"] for fi in self.factor_names}
            config_max[f] = factor_stats[f]["max"]
            boundary_candidates.append(("uni_max_" + f, config_max))

        # 2. 全局极端（2个）
        all_min = {f: factor_stats[f]["min"] for f in self.factor_names}
        boundary_candidates.append(("global_min", all_min))

        all_max = {f: factor_stats[f]["max"] for f in self.factor_names}
        boundary_candidates.append(("global_max", all_max))

        # 从design_space中找最近邻
        boundary_indices = []
        for name, target in boundary_candidates:
            idx = self._find_nearest_match(target)
            if idx not in boundary_indices:  # 去重
                boundary_indices.append(idx)

        # 如果不足20个，用maximin补充
        all_selected = set()
        if self.core1_points is not None:
            all_selected.update(self.core1_points.index.tolist())
        all_selected.update(self.selected_samples.get("core2_main", []))
        all_selected.update(
            [x[0] for x in self.selected_samples.get("core2_inter", [])]
        )

        while len(boundary_indices) < self.budget["boundary"]:
            remaining = [
                i
                for i in self.design_df.index
                if i not in all_selected and i not in boundary_indices
            ]
            if len(remaining) == 0:
                break

            # 选择距离已选点最远的
            remaining_vals = self.design_df.loc[remaining, self.factor_names].values
            selected_vals = self.design_df.loc[
                boundary_indices, self.factor_names
            ].values

            dists = cdist(remaining_vals, selected_vals, metric="euclidean").min(axis=1)
            best_idx = remaining[np.argmax(dists)]
            boundary_indices.append(best_idx)

        self.selected_samples["boundary"] = boundary_indices[: self.budget["boundary"]]
        logger.info(f"成功选择 {len(self.selected_samples['boundary'])} 个边界极端点")
        return self.selected_samples["boundary"]

    def select_lhs_points(self) -> List[int]:
        """
        选择个体点b分层LHS点（29次）
        策略：约束LHS + Gower距离

        Returns:
            选中的design_df索引列表
        """
        logger.info("=== 开始选择分层LHS点 ===")

        n_lhs = self.budget["lhs"]

        # 生成LHS样本（连续空间）
        sampler = qmc.LatinHypercube(d=self.d, seed=self.seed)
        lhs_samples = sampler.random(n=n_lhs)

        # 映射到实际因子范围
        lhs_scaled = np.zeros_like(lhs_samples)
        for i, f in enumerate(self.factor_names):
            f_min = self.design_df[f].min()
            f_max = self.design_df[f].max()
            lhs_scaled[:, i] = f_min + lhs_samples[:, i] * (f_max - f_min)

        # 找最近邻匹配
        all_selected = set()
        if self.core1_points is not None:
            all_selected.update(self.core1_points.index.tolist())
        all_selected.update(self.selected_samples.get("core2_main", []))
        all_selected.update(
            [x[0] for x in self.selected_samples.get("core2_inter", [])]
        )
        all_selected.update(self.selected_samples.get("boundary", []))

        lhs_indices = []
        design_vals = self.design_df[self.factor_names].values

        for lhs_point in lhs_scaled:
            # 计算到所有候选点的距离
            dists = np.sqrt(((design_vals - lhs_point) ** 2).sum(axis=1))

            # 找最近的未选择点
            sorted_indices = np.argsort(dists)
            for idx in sorted_indices:
                if idx not in all_selected and idx not in lhs_indices:
                    lhs_indices.append(idx)
                    all_selected.add(idx)
                    break

        self.selected_samples["lhs"] = lhs_indices
        logger.info(f"成功选择 {len(lhs_indices)} 个LHS点")
        return lhs_indices

    def _select_interaction_pairs(self) -> List[Tuple[int, int]]:
        """
        自动选择交互对（支持多种方法）

        Returns:
            选中的交互对列表
        """
        method = self.interaction_selection

        if method == "variance":
            # 基于方差：选择高方差因子
            factor_vars = self.design_df[self.factor_names].var()
            sorted_factors = factor_vars.argsort()[::-1]
            pairs = []
            for i in range(min(5, len(sorted_factors))):
                for j in range(i + 1, min(5, len(sorted_factors))):
                    if len(pairs) < 5:
                        pairs.append((int(sorted_factors[i]), int(sorted_factors[j])))
            return pairs

        elif method == "correlation":
            # 基于相关性：选择低相关性+高方差因子对
            corr_matrix = self.design_df[self.factor_names].corr().abs()
            factor_vars = self.design_df[self.factor_names].var()

            # 候选所有因子对
            all_pairs = []
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    # 评分 = 高方差 + 低相关性
                    var_score = (factor_vars.iloc[i] + factor_vars.iloc[j]) / 2
                    corr_penalty = corr_matrix.iloc[i, j]
                    score = var_score * (1 - corr_penalty)  # 方差高、相关低 → 分数高
                    all_pairs.append(((i, j), score))

            # 选择top-5
            all_pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = [p[0] for p in all_pairs[:5]]
            logger.info(
                f"基于相关性选择，avg_corr={np.mean([corr_matrix.iloc[i,j] for i,j in pairs]):.3f}"
            )
            return pairs

        else:  # 'auto' - 结合方差和相关性的平衡策略
            corr_matrix = self.design_df[self.factor_names].corr().abs()
            factor_vars = self.design_df[self.factor_names].var()

            # 平衡策略：方差权重0.6，相关性权重0.4
            all_pairs = []
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    var_score = (factor_vars.iloc[i] + factor_vars.iloc[j]) / 2
                    var_score_norm = var_score / factor_vars.max()  # 归一化到[0,1]
                    corr_score = 1 - corr_matrix.iloc[i, j]  # 低相关性=高分

                    # 加权得分
                    score = 0.6 * var_score_norm + 0.4 * corr_score
                    all_pairs.append(((i, j), score))

            all_pairs.sort(key=lambda x: x[1], reverse=True)
            pairs = [p[0] for p in all_pairs[:5]]
            logger.info(
                f"自动选择，avg_var={np.mean([factor_vars.iloc[i]+factor_vars.iloc[j] for i,j in pairs]):.3f}, "
                f"avg_corr={np.mean([corr_matrix.iloc[i,j] for i,j in pairs]):.3f}"
            )
            return pairs

    def evaluate_sampling_quality(self, trials_df: pd.DataFrame) -> Dict[str, Any]:
        """
        评估采样质量

        Args:
            trials_df: 试验清单DataFrame

        Returns:
            质量指标字典
        """
        quality = {}

        # 1. 最小距离检查（避免过于密集）
        trial_vals = trials_df[self.factor_names].values
        if len(trial_vals) > 1:
            dists = pdist(trial_vals, metric="euclidean")
            quality["min_dist"] = float(np.min(dists))
            quality["median_dist"] = float(np.median(dists))

            # 警告阈值：设计空间对角线的1%
            design_vals = self.design_df[self.factor_names].values
            design_range = design_vals.max(axis=0) - design_vals.min(axis=0)
            diagonal = np.sqrt(np.sum(design_range**2))
            threshold = 0.01 * diagonal

            if quality["min_dist"] < threshold:
                warning = f"采样点过于密集: min_dist={quality['min_dist']:.4f} < threshold={threshold:.4f}"
                logger.warning(warning)
                quality["warnings"] = quality.get("warnings", []) + [warning]

        # 2. 覆盖率统计
        unique_configs = trials_df[self.factor_names].drop_duplicates()
        quality["n_unique_configs"] = len(unique_configs)
        quality["coverage_rate"] = len(unique_configs) / len(self.design_df)

        if quality["coverage_rate"] < 0.05:
            warning = f"覆盖率过低: {quality['coverage_rate']:.2%}"
            logger.warning(warning)
            quality["warnings"] = quality.get("warnings", []) + [warning]

        # 3. 因子水平分布检查
        level_balance = {}
        for f in self.factor_names:
            value_counts = trials_df[f].value_counts()
            level_balance[f] = {
                "n_levels": len(value_counts),
                "min_count": int(value_counts.min()),
                "max_count": int(value_counts.max()),
                "gini": self._compute_gini_for_factor(value_counts),
            }

            # 检查分布不均
            if level_balance[f]["gini"] > 0.5:
                warning = f"因子{f}分布不均: Gini={level_balance[f]['gini']:.3f}"
                logger.warning(warning)
                quality["warnings"] = quality.get("warnings", []) + [warning]

        quality["level_balance"] = level_balance

        logger.info(
            f"质量评估: 覆盖率={quality['coverage_rate']:.2%}, "
            f"最小距离={quality.get('min_dist', 0):.4f}, "
            f"唯一配置={quality['n_unique_configs']}"
        )

        return quality

    def _compute_gini_for_factor(self, value_counts: pd.Series) -> float:
        """计算单个因子的Gini系数"""
        counts = np.sort(value_counts.values)
        n = len(counts)
        if n == 0 or counts.sum() == 0:
            return 0.0
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * counts)) / (n * counts.sum()) - (n + 1) / n)

    def run_sampling(self) -> Dict[str, Any]:
        """
        执行完整的Phase 1采样流程

        Returns:
            包含所有采样结果的字典
        """
        logger.info("=" * 60)
        logger.info("开始执行Phase 1预热采样")
        logger.info("=" * 60)

        # 1. Core-1固定重复点
        self.select_core1_points()

        # 2. Core-2a主效应覆盖
        self.select_core2_main_effects()

        # 3. Core-2b交互初筛
        self.select_core2_interactions()

        # 4. 边界极端点
        self.select_boundary_points()

        # 5. 分层LHS点
        self.select_lhs_points()

        # 生成试验清单
        trials = self._generate_trial_list()

        # 质量检查
        quality = self.evaluate_sampling_quality(trials)

        logger.info("=" * 60)
        logger.info("Phase 1采样完成")
        logger.info(f"总试验数: {len(trials)}")
        logger.info("=" * 60)

        return {
            "trials": trials,
            "core1_points": self.core1_points,
            "selected_samples": self.selected_samples,
            "budget": self.budget,
            "quality": quality,
        }

    def _find_nearest_match(
        self, target: Dict[str, float], candidates: Optional[pd.DataFrame] = None
    ) -> int:
        """
        在候选集中找到最接近目标配置的点

        Args:
            target: 目标配置字典 {factor_name: value}
            candidates: 候选DataFrame，默认为self.design_df

        Returns:
            最接近的点的索引
        """
        if candidates is None:
            candidates = self.design_df

        # 计算欧氏距离
        distances = np.zeros(len(candidates))
        for f in self.factor_names:
            if f in target:
                distances += (candidates[f].values - target[f]) ** 2

        distances = np.sqrt(distances)
        return candidates.index[np.argmin(distances)]

    def _generate_trial_list(self) -> pd.DataFrame:
        """
        生成完整的试验清单

        Returns:
            试验清单DataFrame
        """
        trials = []
        trial_id = 0

        # Core-1: 每个受试者都做全部8个点
        for subject_id in range(self.n_subjects):
            for idx, row in self.core1_points.iterrows():
                trial = {
                    "trial_id": trial_id,
                    "subject_id": subject_id,
                    "block_type": "core1",
                    "design_idx": idx,
                }
                for f in self.factor_names:
                    trial[f] = row[f]
                trials.append(trial)
                trial_id += 1

        # Core-2 main: 分配给不同受试者
        for i, idx in enumerate(self.selected_samples["core2_main"]):
            subject_id = i % self.n_subjects
            trial = {
                "trial_id": trial_id,
                "subject_id": subject_id,
                "block_type": "core2_main",
                "design_idx": idx,
            }
            for f in self.factor_names:
                trial[f] = self.design_df.loc[idx, f]
            trials.append(trial)
            trial_id += 1

        # Core-2 interaction: 分配给不同受试者
        for i, (idx, pair_id) in enumerate(self.selected_samples["core2_inter"]):
            subject_id = i % self.n_subjects
            trial = {
                "trial_id": trial_id,
                "subject_id": subject_id,
                "block_type": "core2_inter",
                "design_idx": idx,
                "pair_id": pair_id,
            }
            for f in self.factor_names:
                trial[f] = self.design_df.loc[idx, f]
            trials.append(trial)
            trial_id += 1

        # Boundary: 分配给不同受试者
        for i, idx in enumerate(self.selected_samples["boundary"]):
            subject_id = i % self.n_subjects
            trial = {
                "trial_id": trial_id,
                "subject_id": subject_id,
                "block_type": "boundary",
                "design_idx": idx,
            }
            for f in self.factor_names:
                trial[f] = self.design_df.loc[idx, f]
            trials.append(trial)
            trial_id += 1

        # LHS: 分配给不同受试者
        for i, idx in enumerate(self.selected_samples["lhs"]):
            subject_id = i % self.n_subjects
            trial = {
                "trial_id": trial_id,
                "subject_id": subject_id,
                "block_type": "lhs",
                "design_idx": idx,
            }
            for f in self.factor_names:
                trial[f] = self.design_df.loc[idx, f]
            trials.append(trial)
            trial_id += 1

        return pd.DataFrame(trials)

    def export_results(self, output_dir: str = ".") -> None:
        """
        导出采样结果

        Args:
            output_dir: 输出目录
        """
        import os

        os.makedirs(output_dir, exist_ok=True)

        # 导出试验清单
        if "trials" in self.__dict__:
            trials_df = self.trials
            trials_df.to_csv(f"{output_dir}/phase1_trials.csv", index=False)
            logger.info(f"试验清单已导出到 {output_dir}/phase1_trials.csv")

        # 导出Core-1点
        if self.core1_points is not None:
            self.core1_points.to_csv(
                f"{output_dir}/phase1_core1_points.csv", index=False
            )
            logger.info(f"Core-1点已导出到 {output_dir}/phase1_core1_points.csv")

        # 导出采样摘要
        summary = {
            "n_subjects": self.n_subjects,
            "trials_per_subject": self.trials_per_subject,
            "total_budget": self.total_budget,
            "d": self.d,
            "factor_names": self.factor_names,
            "budget": self.budget,
            "selected_counts": {
                k: len(v) if isinstance(v, list) else 0
                for k, v in self.selected_samples.items()
            },
        }

        with open(f"{output_dir}/phase1_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"采样摘要已导出到 {output_dir}/phase1_summary.json")


def main():
    """示例：使用Phase1WarmupSampler"""
    # 生成模拟设计空间
    np.random.seed(42)
    n_configs = 1200
    d = 5

    design_data = {}
    for i in range(d):
        design_data[f"f{i+1}"] = np.random.rand(n_configs)

    design_df = pd.DataFrame(design_data)
    print(f"设计空间: {len(design_df)} 个配置, {d} 个因子")

    # 创建采样器
    sampler = Phase1WarmupSampler(
        design_df=design_df, n_subjects=7, trials_per_subject=25, seed=42
    )

    # 执行采样
    results = sampler.run_sampling()

    # 导出结果
    sampler.export_results(output_dir="./phase1_results")

    print("\n采样完成!")
    print(f"总试验数: {len(results['trials'])}")
    print(f"Core-1点数: {len(results['core1_points'])}")


if __name__ == "__main__":
    main()
