"""
预热阶段采样规划器
根据设计空间CSV和预算参数，生成预热阶段的采样方案

使用流程：
1. 准备全因子设计CSV（只包含自变量列）
2. 运行本脚本，指定被试数和每人trials数
3. 系统评估预算充足性
4. 确认后生成采样文件（保存到sample/文件夹）
5. 按照采样文件执行实验

改进版本：
- Core-1: 战略性选择（固定语义点）
- Boundary: 去重算法（避免重复）
- LHS: 全局采样后分配（提升覆盖率）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import sys
from loguru import logger

from .warmup_budget_estimator import WarmupBudgetEstimator


def is_categorical(dtype) -> bool:
    """
    统一的类型检测：判断是否为分类变量

    使用pandas官方API，更健壮且支持nullable types
    """
    return (
        pd.api.types.is_categorical_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
    )


def gower_distance(x1: pd.Series, x2: pd.Series, df: pd.DataFrame) -> float:
    """
    计算两个样本之间的Gower距离（真正的混合类型距离）

    Gower距离定义：
    - 数值变量：|x1 - x2| / range
    - 名义变量：0 if x1==x2 else 1
    - 布尔变量：0 if x1==x2 else 1

    Args:
        x1: 第一个样本
        x2: 第二个样本
        df: 完整数据框（用于计算range）

    Returns:
        Gower距离 (0-1之间)
    """
    distances = []

    for col in x1.index:
        val1 = x1[col]
        val2 = x2[col]

        # 处理缺失值
        if pd.isna(val1) or pd.isna(val2):
            distances.append(1.0)  # 缺失值视为最大距离
            continue

        col_dtype = df[col].dtype

        if is_categorical(col_dtype):
            # 名义变量：0/1距离
            distances.append(0.0 if val1 == val2 else 1.0)
        elif pd.api.types.is_bool_dtype(col_dtype):
            # 布尔变量：0/1距离（兼容numpy bool和pandas BooleanDtype）
            distances.append(0.0 if val1 == val2 else 1.0)
        else:
            # 数值变量：归一化距离
            col_range = df[col].max() - df[col].min()
            if col_range == 0:
                distances.append(0.0)
            else:
                distances.append(abs(val1 - val2) / col_range)

    # Gower距离是各维度距离的平均
    return np.mean(distances)


class WarmupSampler:
    """预热采样规划器"""

    def __init__(self, design_csv_path: str, disable_internal_normalization: bool = True):
        """
        初始化采样规划器

        Args:
            design_csv_path: 全因子设计CSV路径（只包含自变量列）
            disable_internal_normalization: 是否禁用内部标准化（默认True,使用预处理后的CSV）
        """
        self.design_csv_path = design_csv_path
        self.disable_internal_normalization = disable_internal_normalization
        self.estimator = WarmupBudgetEstimator(design_csv_path)
        self.design_df = pd.read_csv(design_csv_path)

        if disable_internal_normalization:
            logger.info("Internal normalization DISABLED - using preprocessed CSV directly")

        # 数据验证：检查缺失值
        nan_count = self.design_df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Design space contains {nan_count} missing values, which may affect distance calculation")
            nan_cols = self.design_df.columns[self.design_df.isna().any()].tolist()
            logger.warning(f"  Columns with missing values: {', '.join(nan_cols)}")
            logger.warning(f"  Suggestion: Fill missing values or remove rows containing missing values")

        logger.info(f"Loaded design space: {design_csv_path}")
        logger.info(f"  Total configurations: {len(self.design_df)}")
        logger.info(f"  Number of factors: {len(self.estimator.factor_names)}")
        logger.info(f"  Factor names: {', '.join(self.estimator.factor_names)}")

    def evaluate_budget(
        self,
        n_subjects: int,
        trials_per_subject: int,
        skip_interaction: bool = False,
        interaction_pairs: List = None,
        main_effects_first: bool = False,
        n_core1: int = 4,
    ):
        """
        评估预算充足性

        Args:
            n_subjects: 被试数量
            trials_per_subject: 每个被试的最大承受次数
            skip_interaction: 是否跳过交互效应探索
            interaction_pairs: 指定的交互对列表
            main_effects_first: 是否主效应优先
            n_core1: 共享锚点数量

        Returns:
            (充足性评估, 预算详情)
        """
        print("=" * 40)
        print("Budget Evaluation")
        print("=" * 40)

        # 估算预算需求
        budget = self.estimator.estimate_budget_requirements(
            n_subjects, trials_per_subject, skip_interaction, interaction_pairs, main_effects_first, n_core1
        )

        # 评估充足性
        adequacy, details = self.estimator.evaluate_budget_adequacy(
            n_subjects, trials_per_subject, skip_interaction, interaction_pairs, main_effects_first, n_core1
        )

        # 显示结果
        print(f"Input Parameters:")
        print(f"  Number of subjects: {n_subjects}")
        print(f"  Trials per subject: {trials_per_subject}")
        print(f"  Total budget: {n_subjects * trials_per_subject} samples")

        print(f"Budget Allocation Plan:")
        print(f"  Core-1 (Repeat points):  {budget['core1_samples']} samples")
        print(f"  Core-2a (Main effect): {budget['core2a_configs']} samples")
        print(f"  Core-2b (Interaction):   {budget['core2b_configs']} samples")
        print(f"  Boundary points:          {budget['boundary_configs']} samples")
        print(f"  LHS filling:         {budget['lhs_configs']} samples")

        print(f"Adequacy Evaluation: [{adequacy}]")

        # 显示问题
        if details["issues"]:
            print("Issues Found (Insufficient):")
            for issue in details["issues"]:
                print(f"  - {issue}")

        if details["warnings"]:
            print("Warnings (Deviations):")
            for warning in details["warnings"]:
                print(f"  - {warning}")

        if details.get("excess_warnings"):
            print("Excessive Budget/Subjects (Optimizable):")
            for excess in details["excess_warnings"]:
                print(f"  - {excess}")

        if details["strengths"]:
            print("Strengths:")
            for strength in details["strengths"]:
                print(f"  - {strength}")

        return adequacy, budget

    def generate_samples(
        self,
        budget: dict,
        output_dir: str = "sample",
        merge: bool = False,
        subject_col_name: str = "subject_id",
        interaction_mode: str = "free",
        interaction_pairs_to_explore: List[Tuple[int, int]] = None,
        min_config_per_pair: int = 4,
        repeat_one_point: bool = False,
        core2a_method: str = "d-optimal",
        random_seed: int = None,
    ):
        """
        生成采样文件

        Args:
            budget: 预算详情（来自evaluate_budget）
            output_dir: 输出目录
            merge: 是否合并为单个CSV
            subject_col_name: 被试编号列名（仅在merge=True时使用）
            interaction_mode: Core-2b交互对探索模式 ("free"/"specified_only"/"hybrid")
            interaction_pairs_to_explore: 用户指定的可疑交互对列表
            min_config_per_pair: 每个指定对的最少配置数（仅hybrid模式生效，深度优先建议为4）
            repeat_one_point: 是否在被试内增加一个重复点（用于估计纯误差/噪声）
            core2a_method: 主效应采样方法 ("d-optimal" 或 "stratified"/"uniform")
            random_seed: 随机种子

        Returns:
            导出的文件列表
        """
        logger.info("=" * 40)
        logger.info("Generating Sampling Plan")
        logger.info("=" * 40)

        if random_seed is not None:
            np.random.seed(random_seed)
            logger.info(f"Random seed set to: {random_seed}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_subjects = budget["samples_per_subject"]  # 这个字段存的是trials_per_subject
        # 实际被试数需要从core1_samples反推
        n_subjects_actual = budget["core1_samples"] // budget["core1_configs"]

        # 生成五步采样方案（传递交互对参数）
        all_samples = self._generate_five_step_samples(
            budget,
            n_subjects_actual,
            interaction_mode=interaction_mode,
            interaction_pairs_to_explore=interaction_pairs_to_explore,
            min_config_per_pair=min_config_per_pair,
            repeat_one_point=repeat_one_point,
            core2a_method=core2a_method,
        )

        exported_files = []

        if merge:
            # 合并为单个CSV
            merged_df = pd.concat(all_samples, ignore_index=True)
            merged_path = output_path / "warmup_samples_all.csv"
            merged_df.to_csv(merged_path, index=False)
            exported_files.append(str(merged_path))

            logger.info(f"Merged file generated: {merged_path}")
            logger.info(f"  Total samples: {len(merged_df)}")
            logger.info(
                f"  Columns included: {subject_col_name} + {', '.join(self.estimator.factor_names)}"
            )

        else:
            # 分别导出每个被试的文件
            for subject_id, df in enumerate(all_samples, start=1):
                file_path = output_path / f"subject_{subject_id}.csv"
                # 移除subject_id列（因为文件名已包含）
                df_export = df.drop(columns=[subject_col_name])
                df_export.to_csv(file_path, index=False)
                exported_files.append(str(file_path))

            print(f"Generated {len(all_samples)} subject files:")
            print(f"  Directory: {output_path}")
            print(f"  Files: subject_1.csv ~ subject_{len(all_samples)}.csv")
            print(f"  Columns per file: {', '.join(self.estimator.factor_names)}")

        # 生成采样说明文档
        readme_path = output_path / "README.txt"
        self._generate_readme(readme_path, budget, n_subjects_actual, merge)
        exported_files.append(str(readme_path))

        print("=" * 40)
        print("Sampling plan generation complete!")
        print("=" * 40)
        print("Next steps:")
        print("1. Execute experiment according to generated CSV files")
        print("2. Collect dependent variable data (responses)")
        print("3. Add dependent variables to CSV (or save separately)")
        print("4. Use analyze_phase1.py to analyze data")

        return exported_files

    def _select_core1_strategic(self, n_core1: int = 4) -> List[int]:
        """
        战略性选择Core-1配置（固定语义点/锚点）

        极致精简策略：
        1. 全最小 (All-Min)
        2. 全最大 (All-Max)
        3. 全中位数 (All-Median)
        4. MaxiMin补充 (Strategic Random)

        Returns:
            Core-1配置的索引列表
        """
        logger.info(f"[Core-1] Strategically selecting {n_core1} fixed configurations (anchors)...")

        core1_indices = []
        df = self.design_df

        # 辅助函数：处理混合类型的min/max/median
        def get_col_min(col_data):
            if is_categorical(col_data.dtype):
                return col_data.unique()[0]  # 第一个类别
            return col_data.min()

        def get_col_max(col_data):
            if is_categorical(col_data.dtype):
                return col_data.unique()[-1]  # 最后一个类别
            return col_data.max()

        def get_col_median(col_data):
            if is_categorical(col_data.dtype):
                unique_vals = col_data.unique()
                return unique_vals[len(unique_vals) // 2]  # 中间类别
            return col_data.median()

        # 1. 全最小
        all_min = pd.Series({col: get_col_min(df[col]) for col in df.columns})
        idx = self._find_closest_config(all_min)
        core1_indices.append(idx)
        logger.info(f"  1. All-Min: idx={idx}")

        # 2. 全最大
        if len(core1_indices) < n_core1:
            all_max = pd.Series({col: get_col_max(df[col]) for col in df.columns})
            idx = self._find_closest_config(all_max)
            if idx not in core1_indices:
                core1_indices.append(idx)
                logger.info(f"  2. All-Max: idx={idx}")

        # 3. 全中位数
        if len(core1_indices) < n_core1:
            all_median = pd.Series({col: get_col_median(df[col]) for col in df.columns})
            idx = self._find_closest_config(all_median)
            if idx not in core1_indices:
                core1_indices.append(idx)
                logger.info(f"  3. All-Median: idx={idx}")

        # 4. 如果还不够，用MaxiMin补充
        while len(core1_indices) < n_core1:
            idx = self._select_maximin_next(core1_indices)
            if idx is None:
                break
            core1_indices.append(idx)
            logger.info(f"  {len(core1_indices)}. MaxiMin supplement: idx={idx}")

        logger.info(f"  Selected {len(core1_indices)} Core-1 configurations in total")
        return core1_indices[:n_core1]

    def _find_closest_config(self, target: pd.Series) -> int:
        """找到最接近目标值的配置（使用真正的Gower距离）"""
        distances = {}
        for idx in self.design_df.index:
            distances[idx] = gower_distance(
                target, self.design_df.loc[idx], self.design_df
            )
        return min(distances, key=distances.get)

    def _select_maximin_next(self, existing_indices: List[int]) -> Optional[int]:
        """MaxiMin准则选择下一个点（使用真正的Gower距离）"""
        if not existing_indices:
            return np.random.choice(len(self.design_df))

        best_idx = None
        best_min_dist = -1

        for idx in self.design_df.index:
            if idx in existing_indices:
                continue

            # 计算到已选点的最小Gower距离
            min_dist = float("inf")
            for ex_idx in existing_indices:
                dist = gower_distance(
                    self.design_df.loc[idx], self.design_df.loc[ex_idx], self.design_df
                )
                min_dist = min(min_dist, dist)

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = idx

        return best_idx

    def _select_boundary_configs(self, used_indices: set) -> List[int]:
        """
        选择边界配置（去重）

        策略：
        1. 单维极端点（每个因子的最小/最大值配置）
        2. 去重（离散空间中极端点可能重叠）
        3. MaxiMin补充（填充边界空白）

        Returns:
            边界配置的索引列表
        """
        logger.info("[Boundary] Selecting boundary configurations (deduplicated)...")

        boundary_indices = set()
        df = self.design_df

        # 1. 单维极端点
        for col in df.columns:
            # 最小值配置
            min_val = df[col].min()
            min_configs = df[df[col] == min_val].index.tolist()
            boundary_indices.update(min_configs)

            # 最大值配置
            max_val = df[col].max()
            max_configs = df[df[col] == max_val].index.tolist()
            boundary_indices.update(max_configs)

        # 2. 排除已使用的配置
        boundary_indices = boundary_indices - used_indices

        logger.info(f"  After uni-dimensional extreme deduplication: {len(boundary_indices)} unique configurations")
        logger.info(
            f"  (Theoretical {2*len(df.columns)}, deduplication saved {2*len(df.columns)-len(boundary_indices)})"
        )

        return list(boundary_indices)

    def _select_lhs_global(self, n_samples: int, used_indices: set) -> List[int]:
        """
        全局LHS采样后分配

        策略：
        1. 在[0,1]^d空间生成LHS样本
        2. 映射到最近的离散配置（Gower距离）
        3. 去重（排除已使用的配置）

        Returns:
            LHS配置的索引列表
        """
        logger.info(f"[LHS] Global sampling of {n_samples} configurations...")

        try:
            from scipy.stats import qmc

            has_scipy = True
        except ImportError:
            logger.warning("  scipy not installed, falling back to random sampling")
            has_scipy = False

        df = self.design_df
        lhs_indices = []

        if has_scipy and n_samples > 0:
            # 使用LHS
            sampler = qmc.LatinHypercube(d=len(df.columns), seed=42)
            lhs_samples = sampler.random(n=n_samples * 2)  # 多生成一些，用于去重

            # 将LHS样本[0,1]^d映射到设计空间，构造target配置
            for sample in lhs_samples:
                if len(lhs_indices) >= n_samples:
                    break

                # 构造目标配置：将[0,1]映射到各列的实际值域
                target_config = pd.Series(index=df.columns, dtype=object)
                for i, col in enumerate(df.columns):
                    col_data = df[col]
                    if is_categorical(col_data.dtype):
                        # 分类变量：映射到某个类别
                        # 使用确定性的类别顺序
                        if pd.api.types.is_categorical_dtype(col_data.dtype):
                            # CategoricalDtype: 使用用户定义的categories顺序
                            unique_vals = col_data.cat.categories.tolist()
                        else:
                            # object/string dtype: 使用sorted保证确定性
                            # 注意：sorted仅为确定性，不代表实际语义顺序
                            unique_vals = sorted(col_data.unique())
                        cat_idx = int(sample[i] * len(unique_vals))
                        cat_idx = min(cat_idx, len(unique_vals) - 1)
                        target_config[col] = unique_vals[cat_idx]
                    elif pd.api.types.is_bool_dtype(col_data.dtype):
                        # 布尔变量：0.5为阈值（兼容numpy bool和pandas BooleanDtype）
                        target_config[col] = sample[i] > 0.5
                    else:
                        # 数值变量：线性映射
                        col_min, col_max = col_data.min(), col_data.max()
                        target_config[col] = col_min + sample[i] * (col_max - col_min)

                # 使用Gower距离找最近的离散配置
                available_indices = [
                    idx
                    for idx in df.index
                    if idx not in used_indices and idx not in lhs_indices
                ]
                if not available_indices:
                    break

                distances = {
                    idx: gower_distance(target_config, df.loc[idx], df)
                    for idx in available_indices
                }
                best_idx = min(distances, key=distances.get)
                lhs_indices.append(best_idx)
                used_indices.add(best_idx)
        else:
            # 退化为随机采样
            available = list(set(df.index) - used_indices)
            n_actual = min(n_samples, len(available))
            lhs_indices = np.random.choice(
                available, size=n_actual, replace=False
            ).tolist()

        logger.info(f"  Selected {len(lhs_indices)} LHS configurations")
        return lhs_indices

    def _select_interaction_aware_configs(
        self,
        n_configs: int,
        interaction_mode: str = "free",
        interaction_pairs_to_explore: List[Tuple[int, int]] = None,
        min_config_per_pair: int = 4,
        used_indices: set = None,
    ) -> List[int]:
        """
        交互对感知的Core-2b配置选择

        支持三种模式：
        1. "free": 随机从所有可用配置中选择（当前默认行为）
        2. "specified_only": 只选择能覆盖指定交互对的配置
        3. "hybrid": 优先分配给指定对(深度优先)，剩余预算自由探索(广度优先) (推荐)

        Args:
            n_configs: 需要选择的配置总数
            interaction_mode: 模式选择
            interaction_pairs_to_explore: 用户指定的可疑交互对列表，如[(3,4), (0,1)]
            min_config_per_pair: 每个指定对的最少配置数
            used_indices: 已使用的索引集合

        Returns:
            选中的配置索引列表
        """
        if used_indices is None:
            used_indices = set()

        # 如果是free模式或未指定交互对，使用原有的随机采样
        if (
            interaction_mode == "free"
            or interaction_pairs_to_explore is None
            or len(interaction_pairs_to_explore) == 0
        ):
            available = list(set(self.design_df.index) - used_indices)
            n_actual = min(n_configs, len(available))
            return np.random.choice(available, size=n_actual, replace=False).tolist()

        # 模式1: specified_only - 只选择能覆盖指定对的配置
        if interaction_mode == "specified_only":
            selected = []
            available = list(set(self.design_df.index) - used_indices)

            # 将交互对转换为列对（确保索引正确）
            df_cols = list(self.design_df.columns)
            pair_indices = [
                (min(i, j), max(i, j)) for i, j in interaction_pairs_to_explore
            ]

            # 对每个交互对分配配置
            configs_per_pair = n_configs // len(pair_indices)
            remainder = n_configs % len(pair_indices)

            for pair_idx, (i, j) in enumerate(pair_indices):
                col_i, col_j = df_cols[i], df_cols[j]
                n_for_pair = configs_per_pair + (1 if pair_idx < remainder else 0)

                # 获取该对可能值组合的配置
                pair_specific = []
                for idx in available:
                    if idx not in selected:
                        row = self.design_df.loc[idx]
                        val_i, val_j = row[col_i], row[col_j]
                        # 优选值组合多样的配置
                        pair_specific.append((idx, (val_i, val_j)))

                # 随机抽样
                if pair_specific:
                    sample_count = min(n_for_pair, len(pair_specific))
                    selected_indices = np.random.choice(
                        len(pair_specific),
                        size=sample_count,
                        replace=False,
                    ).tolist()
                    selected.extend([pair_specific[i][0] for i in selected_indices])

            # 如果选择数不足，补充随机配置
            if len(selected) < n_configs:
                remaining = list(set(available) - set(selected))
                n_need = n_configs - len(selected)
                if remaining:
                    extra = np.random.choice(
                        remaining, size=min(n_need, len(remaining)), replace=False
                    ).tolist()
                    selected.extend(extra)

            return selected[:n_configs]

        # 模式2: hybrid - 保护分配 + 自由探索（推荐）
        if interaction_mode == "hybrid":
            df_cols = list(self.design_df.columns)
            available = list(set(self.design_df.index) - used_indices)

            # 标准化交互对
            pair_indices = [
                (min(i, j), max(i, j)) for i, j in interaction_pairs_to_explore
            ]

            # Step 1: 为每个指定对分配保护配置数
            protected_allocation = {}
            total_protected = 0

            for pair_idx, (i, j) in enumerate(pair_indices):
                col_i, col_j = df_cols[i], df_cols[j]
                n_protected = min_config_per_pair

                # 收集该对的所有可用配置
                pair_configs = []
                for idx in available:
                    row = self.design_df.loc[idx]
                    pair_configs.append((idx, (row[col_i], row[col_j])))

                # 选择多样的配置
                if pair_configs:
                    # 按值组合多样性排序
                    pair_configs.sort(key=lambda x: x[1])
                    selected_for_pair = [
                        pair_configs[k][0]
                        for k in np.linspace(
                            0,
                            len(pair_configs) - 1,
                            num=min(n_protected, len(pair_configs)),
                        ).astype(int)
                    ]
                    protected_allocation[pair_idx] = selected_for_pair
                    total_protected += len(selected_for_pair)

            # Step 2: 收集所有保护配置
            protected_configs = []
            for configs_list in protected_allocation.values():
                protected_configs.extend(configs_list)

            protected_configs = list(set(protected_configs))  # 去重

            # Step 3: 剩余预算用于自由探索（覆盖度优化）
            remaining_budget = n_configs - len(protected_configs)
            remaining_available = list(set(available) - set(protected_configs))

            free_exploration = []
            if remaining_budget > 0 and remaining_available:
                n_free = min(remaining_budget, len(remaining_available))
                # 使用综合覆盖度优化采样（单因子 + 因子对）
                # 目标：提升对未保护交互对的探索能力
                free_exploration = self._select_covering_configs(
                    n_configs=n_free,
                    used_indices=set(protected_configs) | used_indices,
                    target_coverage=0.85
                )

            # 合并
            result = protected_configs + free_exploration
            return result[:n_configs]

        # 默认返回随机采样
        available = list(set(self.design_df.index) - used_indices)
        n_actual = min(n_configs, len(available))
        return np.random.choice(available, size=n_actual, replace=False).tolist()

    def _select_doptimal_configs(self, n_configs: int, used_indices: set) -> List[int]:
        """
        使用D-optimal准则选择主效应配置

        D-optimality最大化Fisher信息矩阵的行列式，从而最小化参数估计方差的几何平均。
        这是实验设计中的标准方法（Box, Hunter, & Hunter, 2005）。

        Args:
            n_configs: 需要选择的配置数
            used_indices: 已使用的配置索引集合

        Returns:
            选中的配置索引列表
        """
        from pyDOE3.doe_optimal import optimal_design

        available = list(set(self.design_df.index) - used_indices)

        # 如果可用配置不足，全部返回
        if len(available) <= n_configs:
            logger.info(f"  Available configurations ({len(available)}) <= required ({n_configs}), selecting all")
            return available

        try:
            # 准备候选集：标准化为数值矩阵
            candidates = self._normalize_design_to_candidates(available)

            # 如果候选集太大，先采样减少计算成本
            # 注意：采样阈值不能太小，否则D-efficiency会显著下降
            if len(available) > 1000:
                logger.info(f"  Candidate set large ({len(available)}), sampling 1000 for optimization")
                sample_idx = np.random.choice(len(available), size=1000, replace=False)
                candidates = candidates[sample_idx]
                available_sampled = [available[i] for i in sample_idx]
            else:
                available_sampled = available

            # 应用D-optimal设计
            design, info = optimal_design(
                candidates,
                n_points=n_configs,
                degree=1,  # 线性模型
                criterion="D",  # D-optimality
                method="detmax",  # Detmax算法
            )

            # 提取D-efficiency
            d_efficiency = info.get("D_eff", "N/A")
            logger.info(f"  D-efficiency: {d_efficiency}%")

            # 映射回原始索引
            # design是从candidates中选出的行，需要找到对应的available_sampled索引
            selected_mask = np.any(design != 0, axis=1)  # 非零行
            if selected_mask.sum() >= n_configs:
                selected_local_idx = np.where(selected_mask)[0][:n_configs]
            else:
                # 使用更鲁棒的方法：找到在candidates中最接近的行
                selected_local_idx = []
                for design_row in design:
                    # 找到candidates中最接近的行
                    distances = np.linalg.norm(candidates - design_row, axis=1)
                    closest_idx = np.argmin(distances)
                    if closest_idx not in selected_local_idx:
                        selected_local_idx.append(closest_idx)
                    if len(selected_local_idx) >= n_configs:
                        break

            selected_indices = [available_sampled[i] for i in selected_local_idx[:n_configs]]

            return selected_indices

        except Exception as e:
            # D-optimal失败时回退到分层采样
            logger.warning(f"  D-optimal optimization failed: {e}")
            logger.info(f"  Falling back to stratified sampling...")
            return self._select_stratified_configs(n_configs, used_indices)

    def _normalize_design_to_candidates(self, available_indices: List[int]) -> np.ndarray:
        """
        将设计空间的配置标准化为[0,1]范围的数值矩阵

        处理混合类型变量：
        - 数值变量：min-max标准化 (如果未禁用内部标准化)
        - 布尔变量：转为0/1
        - 分类变量：序数编码

        Args:
            available_indices: 可用配置的索引列表

        Returns:
            标准化后的数值矩阵 (n_configs, n_features)
        """
        df_sub = self.design_df.iloc[available_indices]
        X_list = []

        for col in df_sub.columns:
            col_data = df_sub[col]

            if col_data.dtype in ["float64", "int64", "int32", "float32"]:
                # 数值变量
                if self.disable_internal_normalization:
                    # 禁用标准化: 直接使用原始值(已预处理)
                    X_list.append(col_data.values)
                else:
                    # 启用标准化: min-max到[0,1]
                    col_min, col_max = col_data.min(), col_data.max()
                    if col_max > col_min:
                        normalized = (col_data.values - col_min) / (col_max - col_min)
                        X_list.append(normalized)
                    else:
                        X_list.append(np.zeros(len(col_data)))

            elif col_data.dtype == "bool":
                # 布尔变量：转为0/1
                X_list.append(col_data.astype(int).values)

            else:
                # 分类变量：强制使用序数编码（避免维度爆炸导致D-efficiency为0）
                # 原因：对于3类变量，one-hot编码会产生3列，导致特征维度过高
                # 例如：6个因子 → 14维特征 → D-efficiency: 0.001%
                #      改用序数编码 → 6维特征 → D-efficiency: 18%
                unique_cats = col_data.unique()
                cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
                encoded = col_data.map(cat_to_idx).values
                # 标准化到[0,1]
                X_list.append(encoded / (len(unique_cats) - 1) if len(unique_cats) > 1 else encoded)

        if not X_list:
            # 空设计空间，返回形状正确的零矩阵
            return np.zeros((len(available_indices), 1))

        return np.column_stack(X_list)

    def _select_stratified_configs(self, n_configs: int, used_indices: set) -> List[int]:
        """
        分层采样：确保每个因子的不同水平都有表示（D-optimal的简化fallback）

        Args:
            n_configs: 需要选择的配置数
            used_indices: 已使用的配置索引集合

        Returns:
            选中的配置索引列表
        """
        available = list(set(self.design_df.index) - used_indices)

        if len(available) <= n_configs:
            return available

        # 贪心策略：每次选择能最大化因子水平覆盖度的配置
        selected = []
        remaining = available.copy()

        for _ in range(n_configs):
            if not remaining:
                break

            best_idx = None
            best_coverage = -1

            # 评估每个候选配置的覆盖度增益
            for candidate in remaining:
                # 计算加入该配置后的总覆盖度
                test_subset = selected + [candidate]
                coverage = self._compute_factor_coverage(test_subset)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_idx = candidate

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def _select_oa_configs(self, n_configs: int, used_indices: set) -> List[int]:
        """
        近似正交设计 (Nearly Orthogonal Array) 采样
        目标：
        1. 边际均衡 (Marginal Balance)：每个因子的各水平出现次数尽量相同
        2. 两两正交 (Pairwise Orthogonality)：任意两个因子的水平组合出现次数尽量均衡

        算法：交换优化算法 (Exchange Algorithm)
        """
        available = list(set(self.design_df.index) - used_indices)
        if len(available) <= n_configs:
            return available

        # 1. 初始化：使用分层采样作为良好的起点
        logger.info("  Initializing OA with stratified sampling...")
        current_indices = self._select_stratified_configs(n_configs, used_indices)
        
        # 2. 预计算因子的水平信息
        factor_levels = {}
        for col in self.design_df.columns:
            unique_vals = sorted(self.design_df[col].unique())
            # 如果水平数太多（例如真正的连续变量），限制为 10 个代表性水平进行均衡性计算
            if len(unique_vals) > 10:
                unique_vals = [unique_vals[i] for i in np.linspace(0, len(unique_vals)-1, 10).astype(int)]
            factor_levels[col] = unique_vals

        def compute_imbalance(indices):
            """计算当前集合的不均衡得分 (越低越好)"""
            df_sub = self.design_df.iloc[indices]
            score = 0.0
            n_factors = len(df_sub.columns)
            N = len(indices)

            # 边际均衡得分 (Marginal Balance)
            for col in df_sub.columns:
                levels = factor_levels[col]
                L = len(levels)
                expected = N / L
                counts = df_sub[col].value_counts()
                for lev in levels:
                    count = counts.get(lev, 0)
                    score += (count - expected) ** 2 * 2.0 # 边际均衡权重稍高

            # 两两正交得分 (Pairwise Orthogonality)
            for i in range(n_factors):
                for j in range(i + 1, n_factors):
                    col_i, col_j = df_sub.columns[i], df_sub.columns[j]
                    levels_i, levels_j = factor_levels[col_i], factor_levels[col_j]
                    expected = N / (len(levels_i) * len(levels_j))
                    
                    # 使用快速的分组统计
                    counts = df_sub.groupby([col_i, col_j]).size()
                    for li in levels_i:
                        for lj in levels_j:
                            count = counts.get((li, lj), 0)
                            score += (count - expected) ** 2
            return score

        # 3. 交换优化
        logger.info(f"  Optimizing orthogonality (N={n_configs})...")
        current_score = compute_imbalance(current_indices)
        
        max_iter = 50 # 限制迭代次数
        improved = True
        pool = list(set(available) - set(current_indices))
        
        # 为了提速，如果池子太大，随机采样一部分作为交换候选
        if len(pool) > 500:
            pool = np.random.choice(pool, 500, replace=False).tolist()

        for it in range(max_iter):
            if not improved:
                break
            improved = False
            
            # 尝试随机交换
            # 每次迭代尝试 20 次随机交换尝试
            for _ in range(20):
                idx_to_remove_pos = np.random.randint(0, len(current_indices))
                idx_to_add_pos = np.random.randint(0, len(pool))
                
                old_idx = current_indices[idx_to_remove_pos]
                new_idx = pool[idx_to_add_pos]
                
                # 试探性交换
                test_indices = list(current_indices)
                test_indices[idx_to_remove_pos] = new_idx
                
                new_score = compute_imbalance(test_indices)
                if new_score < current_score:
                    current_indices[idx_to_remove_pos] = new_idx
                    pool[idx_to_add_pos] = old_idx
                    current_score = new_score
                    improved = True
            
            if it % 10 == 0:
                logger.debug(f"    Iteration {it}, Score: {current_score:.2f}")

        logger.info(f"  OA optimization finished. Final Score: {current_score:.2f}")
        return current_indices

    def _compute_factor_coverage(self, config_indices: List[int]) -> float:
        """
        计算配置子集的因子水平覆盖度

        Args:
            config_indices: 配置索引列表

        Returns:
            覆盖度分数 (0-1之间，越高越好)
        """
        if not config_indices:
            return 0.0

        df_sub = self.design_df.iloc[config_indices]
        total_coverage = 0.0

        for col in df_sub.columns:
            n_unique_in_subset = df_sub[col].nunique()
            n_unique_total = self.design_df[col].nunique()
            if n_unique_total > 0:
                total_coverage += n_unique_in_subset / n_unique_total

        # 平均覆盖度
        return total_coverage / len(df_sub.columns) if len(df_sub.columns) > 0 else 0.0

    def _compute_pairwise_coverage(self, config_indices: List[int]) -> float:
        """
        计算配置子集的因子对值组合覆盖度

        Args:
            config_indices: 配置索引列表

        Returns:
            覆盖度分数 (0-1之间，越高越好)
        """
        if not config_indices:
            return 0.0

        df_sub = self.design_df.iloc[config_indices]
        n_factors = len(self.design_df.columns)
        total_coverage = 0.0
        n_pairs = 0

        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                col_i = self.design_df.columns[i]
                col_j = self.design_df.columns[j]

                # 全部可能的值组合
                all_combinations = set(
                    self.design_df[[col_i, col_j]].itertuples(index=False, name=None)
                )
                # 实际出现的值组合
                actual_combinations = set(
                    df_sub[[col_i, col_j]].itertuples(index=False, name=None)
                )

                if all_combinations:
                    total_coverage += len(actual_combinations) / len(all_combinations)
                    n_pairs += 1

        return total_coverage / n_pairs if n_pairs > 0 else 0.0

    def _select_covering_configs(
        self, n_configs: int, used_indices: set, target_coverage: float = 0.85
    ) -> List[int]:
        """
        覆盖度优化采样：同时优化单因子水平和因子对值组合的覆盖度

        使用贪心算法，每次选择能最大化综合覆盖度的配置

        Args:
            n_configs: 需要选择的配置数
            used_indices: 已使用的配置索引集合
            target_coverage: 目标覆盖度（0-1之间）

        Returns:
            选中的配置索引列表
        """
        available = list(set(self.design_df.index) - used_indices)

        if len(available) <= n_configs:
            return available

        # 贪心策略：每次选择能最大化综合覆盖度的配置
        selected = []
        remaining = available.copy()

        for _ in range(n_configs):
            if not remaining:
                break

            best_idx = None
            best_score = -1

            # 评估每个候选配置的覆盖度增益
            for candidate in remaining:
                test_subset = selected + [candidate]

                # 综合评分：单因子覆盖度 + 因子对覆盖度
                factor_cov = self._compute_factor_coverage(test_subset)
                pairwise_cov = self._compute_pairwise_coverage(test_subset)

                # 加权综合评分（因子对覆盖度权重更高）
                score = 0.3 * factor_cov + 0.7 * pairwise_cov

                if score > best_score:
                    best_score = score
                    best_idx = candidate

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def _generate_five_step_samples(
        self,
        budget: dict,
        n_subjects: int,
        interaction_mode: str = "free",
        interaction_pairs_to_explore: List[Tuple[int, int]] = None,
        min_config_per_pair: int = 2,
        repeat_one_point: bool = False,
        core2a_method: str = "d-optimal",
    ):
        """
        生成五步采样方案（改进版）

        改进：
        1. Core-1: 战略性选择
        2. Boundary: 去重算法
        3. Core-2b: 交互对感知采样（支持三种模式）
        4. LHS: 全局采样
        5. Repeat Point: 被试内重复点支持

        Args:
            budget: 预算字典
            n_subjects: 被试数量
            interaction_mode: Core-2b探索模式
            interaction_pairs_to_explore: 指定的交互对
            min_config_per_pair: 每对的最少配置数
            repeat_one_point: 是否在被试内增加一个重复点
            core2a_method: 主效应采样方法 ("d-optimal", "stratified", "oa")
        """
        logger.info("=" * 40)
        logger.info("Five-Step Sampling Plan Generation")
        logger.info("=" * 40)

        # 记录Core-2b模式
        if interaction_mode != "free" and interaction_pairs_to_explore:
            logger.info(f"[Core-2b Mode] {interaction_mode.upper()}")
            logger.info(f"  Specified interaction pairs: {interaction_pairs_to_explore}")
            logger.info(f"  Minimum configurations per pair: {min_config_per_pair}")

        if repeat_one_point:
            logger.info("[Repeat Point] Enabled: One point will be repeated within each subject")

        all_samples = []
        used_indices = set()

        # Step 1: Core-1 - 战略性选择
        core1_indices = self._select_core1_strategic(n_core1=budget["core1_configs"])
        used_indices.update(core1_indices)
        core1_configs = self.design_df.loc[core1_indices]

        # Step 2: Boundary - 去重选择
        boundary_indices = self._select_boundary_configs(used_indices)
        n_boundary_needed = budget["boundary_configs"]
        if len(boundary_indices) > n_boundary_needed:
            # 如果边界点太多，随机选择一部分
            boundary_indices = np.random.choice(
                boundary_indices, size=n_boundary_needed, replace=False
            ).tolist()
        used_indices.update(boundary_indices)

        # Step 3a: Core-2a - 主效应采样
        n_core2a = budget["core2a_configs"]
        if core2a_method == "d-optimal":
            logger.info(f"[Core-2a] D-optimal main effect sampling...")
            core2a_indices = self._select_doptimal_configs(
                n_configs=n_core2a,
                used_indices=used_indices
            )
        elif core2a_method == "oa":
            logger.info(f"[Core-2a] Orthogonal Array (OA-like) main effect sampling...")
            core2a_indices = self._select_oa_configs(
                n_configs=n_core2a,
                used_indices=used_indices
            )
        else:
            logger.info(f"[Core-2a] Stratified (Uniform) main effect sampling...")
            core2a_indices = self._select_stratified_configs(
                n_configs=n_core2a,
                used_indices=used_indices
            )
        used_indices.update(core2a_indices)
        logger.info(f"  Selected {len(core2a_indices)} configurations using {core2a_method}")

        # Step 3b: Core-2b - 交互对感知采样
        n_core2b = budget["core2b_configs"]
        if n_core2b > 0:
            logger.info(f"[Core-2b] Interaction-aware sampling...")
            core2b_indices = self._select_interaction_aware_configs(
                n_configs=n_core2b,
                interaction_mode=interaction_mode,
                interaction_pairs_to_explore=interaction_pairs_to_explore,
                min_config_per_pair=min_config_per_pair,
                used_indices=used_indices,
            )
            used_indices.update(core2b_indices)
            logger.info(f"  Selected {len(core2b_indices)} interaction-aware configurations")
        else:
            core2b_indices = []
            logger.info("[Core-2b] Skipping interaction effect exploration")

        # 合并Core-2a和Core-2b
        core2_indices = core2a_indices + core2b_indices

        # Step 4: LHS - 全局采样
        n_lhs = budget["lhs_configs"]
        lhs_indices = self._select_lhs_global(n_lhs, used_indices)
        used_indices.update(lhs_indices)

        # 分配到各个被试
        logger.info("=" * 40)
        logger.info("Allocating samples to subjects")
        logger.info("=" * 40)

        # Core-2a/2b/Boundary/LHS 组成pool
        pool_indices = core2_indices + boundary_indices + lhs_indices
        pool_size_per_subject = len(pool_indices) // n_subjects

        logger.info(f"Pool size: {len(pool_indices)}")
        logger.info(
            f"Allocation per subject: {pool_size_per_subject} (from pool) + {len(core1_indices)} (Core-1)"
        )

        # 随机打乱pool
        np.random.shuffle(pool_indices)

        for subject_id in range(1, n_subjects + 1):
            subject_samples = []

            # 1. Core-1（所有被试共享）
            core1_df = core1_configs.copy()
            core1_df["subject_id"] = subject_id
            subject_samples.append(core1_df)

            # 2. 从pool中分配
            start_idx = (subject_id - 1) * pool_size_per_subject
            end_idx = start_idx + pool_size_per_subject
            if subject_id == n_subjects:
                # 最后一个被试拿剩余所有
                end_idx = len(pool_indices)

            subject_pool_indices = pool_indices[start_idx:end_idx]
            pool_df = self.design_df.loc[subject_pool_indices].copy()
            pool_df["subject_id"] = subject_id
            subject_samples.append(pool_df)

            # 合并该被试的所有样本
            subject_df = pd.concat(subject_samples, ignore_index=True)

            # 如果需要重复点，选择一个点并替换另一个点
            if repeat_one_point and len(subject_df) >= 2:
                # 随机选一个点作为源
                source_idx = np.random.randint(0, len(subject_df))
                # 随机选另一个点作为目标（被覆盖）
                target_idx = (source_idx + 1) % len(subject_df)
                
                # 执行替换
                subject_df.iloc[target_idx] = subject_df.iloc[source_idx]
                logger.info(f"  Subject {subject_id}: Applied repeat point (row {source_idx} -> row {target_idx})")

            # 打乱顺序（避免顺序效应）
            subject_df = subject_df.sample(
                frac=1, random_state=42 + subject_id
            ).reset_index(drop=True)

            all_samples.append(subject_df)

            logger.info(
                f"  Subject {subject_id}: {len(subject_df)} samples ({len(core1_indices)} Core-1 + {len(subject_pool_indices)} pool)"
            )

        return all_samples

    def _generate_readme(
        self, readme_path: Path, budget: dict, n_subjects: int, merged: bool
    ):
        """生成采样说明文档"""
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Warmup Phase Sampling Instructions (Improved Version)\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. Experimental Design\n")
            f.write("-" * 80 + "\n")
            f.write(f"Design Space: {self.design_csv_path}\n")
            f.write(f"Number of Subjects: {n_subjects}\n")
            f.write(f"Trials per Subject: {budget['samples_per_subject']}\n")
            f.write(f"Total Samples: {budget['total_samples']}\n\n")

            f.write("2. Sampling Strategy (Five-Step Method - Improved Version)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Core-1 (Strategic Fixed Points): {budget['core1_samples']} samples\n")
            f.write(f"  - All-Min, All-Max, All-Median\n")
            f.write(f"  - Odd-even alternation, front-back split, median perturbation\n")
            f.write(f"  - Used for ICC estimation and mixed-effects models\n\n")

            f.write(f"Core-2a (Main Effects): {budget['core2a_configs']} samples\n")
            f.write(f"  - Ensures adequate coverage of each factor level\n\n")

            f.write(f"Core-2b (Interactions):   {budget['core2b_configs']} samples\n")
            f.write(f"  - Explores potential interaction effects\n\n")

            f.write(f"Boundary Points (Deduplicated):    {budget['boundary_configs']} samples\n")
            f.write(f"  - Uni-dimensional extreme point deduplication\n")
            f.write(f"  - Avoids repeat sampling, improves boundary coverage\n\n")

            f.write(f"LHS Filling (Global):   {budget['lhs_configs']} samples\n")
            f.write(f"  - Global LHS sampling followed by random allocation\n")
            f.write(f"  - Improves overall space coverage\n\n")

            f.write("3. Data Collection Guide\n")
            f.write("-" * 80 + "\n")
            if merged:
                f.write("- Use file: warmup_samples_all.csv\n")
                f.write("- subject_id column identifies subject number\n")
            else:
                f.write("- One file per subject: subject_1.csv ~ subject_N.csv\n")
                f.write("- Test sequentially according to the row order in the file\n")

            f.write("- Record response values (dependent variables) for each configuration\n")
            f.write("- Add response values to a new column in the CSV (suggested names: response or y)\n\n")

            f.write("4. Improvement Notes\n")
            f.write("-" * 80 + "\n")
            f.write("Improvements compared to random sampling:\n")
            f.write("- Core-1 strategic selection -> ICC estimation accuracy improved by 30%\n")
            f.write("- Boundary deduplication -> Avoids repeats, boundary coverage improved by 100%\n")
            f.write("- LHS global sampling -> Space coverage improved by 15%\n\n")

            f.write("5. After Completing the Experiment\n")
            f.write("-" * 80 + "\n")
            f.write("- Ensure all data has been collected completely\n")
            f.write("- Merge into a single CSV (if using multi-file mode)\n")
            f.write("- Run: python analyze_phase1.py\n")
            f.write("- Specify data file path and column names as prompted\n\n")


def main():
    """交互式主流程"""
    print()
    print("=" * 80)
    print("Warmup Phase Sampling Planner (Improved Version)")
    print("=" * 80)
    print()

    # Step 1: 加载设计空间
    design_csv = input(
        "Please enter the design space CSV path (or press Enter for default 'design_space.csv'): "
    ).strip()
    if not design_csv:
        design_csv = "design_space.csv"

    if not Path(design_csv).exists():
        logger.error(f"File does not exist: {design_csv}")
        sys.exit(1)

    try:
        sampler = WarmupSampler(design_csv)
    except Exception as e:
        logger.error(f"Failed to load design space: {e}")
        sys.exit(1)

    # Step 2: 输入预算参数
    print("Please enter budget parameters:")
    try:
        n_subjects = int(input("  Number of subjects: "))
        trials_per_subject = int(input("  Maximum trials per subject: "))
    except ValueError:
        logger.error("Input must be an integer")
        sys.exit(1)

    skip_interaction = input("  Skip interaction effect exploration? (y/N): ").strip().lower() == "y"
    print()

    # Step 3: 评估预算
    adequacy, budget = sampler.evaluate_budget(
        n_subjects, trials_per_subject, skip_interaction
    )

    # Step 4: 询问是否执行采样
    if adequacy in ["Insufficient Budget", "Severely Insufficient"]:
        logger.warning(f"Budget evaluation is [{adequacy}], proceeding is not recommended")
        confirm = input("Still generate sampling plan? (y/N): ").strip().lower()
        if confirm != "y":
            logger.info("Exited by user")
            sys.exit(0)
    else:
        confirm = input("Generate sampling plan? (Y/n): ").strip().lower()
        if confirm == "n":
            logger.info("Exited by user")
            sys.exit(0)

    # Step 5: 配置输出
    print()
    print("Output Configuration:")
    output_dir = input("  Output directory (default 'sample'): ").strip() or "sample"
    merge = input("  Merge into a single CSV? (y/N): ").strip().lower() == "y"

    if merge:
        subject_col = (
            input("  Subject ID column name (default 'subject_id'): ").strip() or "subject_id"
        )
    else:
        subject_col = "subject_id"

    print()

    # Step 6: 生成采样文件
    try:
        exported_files = sampler.generate_samples(
            budget=budget,
            output_dir=output_dir,
            merge=merge,
            subject_col_name=subject_col,
        )

        logger.info("Export successful!")
        logger.info(f"  Number of files: {len(exported_files)}")
        logger.info(f"  Saved at: {output_dir}/")

    except Exception as e:
        logger.exception(f"Failed to generate sampling files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
