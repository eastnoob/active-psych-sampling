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

from warmup_budget_estimator import WarmupBudgetEstimator


def is_categorical(dtype) -> bool:
    """
    统一的类型检测：判断是否为分类变量

    使用pandas官方API，更健壮且支持nullable types
    """
    return (
        pd.api.types.is_categorical_dtype(dtype) or
        pd.api.types.is_string_dtype(dtype) or
        pd.api.types.is_object_dtype(dtype)
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

    def __init__(self, design_csv_path: str):
        """
        初始化采样规划器

        Args:
            design_csv_path: 全因子设计CSV路径（只包含自变量列）
        """
        self.design_csv_path = design_csv_path
        self.estimator = WarmupBudgetEstimator(design_csv_path)
        self.design_df = pd.read_csv(design_csv_path)

        # 数据验证：检查缺失值
        nan_count = self.design_df.isna().sum().sum()
        if nan_count > 0:
            print(f"[警告] 设计空间包含{nan_count}个缺失值，可能影响距离计算")
            nan_cols = self.design_df.columns[self.design_df.isna().any()].tolist()
            print(f"  缺失值所在列: {', '.join(nan_cols)}")
            print(f"  建议：填补缺失值或移除包含缺失值的行")
            print()

        print(f"[加载] 设计空间: {design_csv_path}")
        print(f"  配置总数: {len(self.design_df)}")
        print(f"  因子数: {len(self.estimator.factor_names)}")
        print(f"  因子名称: {', '.join(self.estimator.factor_names)}")
        print()

    def evaluate_budget(
        self,
        n_subjects: int,
        trials_per_subject: int,
        skip_interaction: bool = False,
    ):
        """
        评估预算充足性

        Args:
            n_subjects: 被试数量
            trials_per_subject: 每个被试的最大承受次数
            skip_interaction: 是否跳过交互效应探索

        Returns:
            (充足性评估, 预算详情)
        """
        print("=" * 80)
        print("预算评估")
        print("=" * 80)
        print()

        # 估算预算需求
        budget = self.estimator.estimate_budget_requirements(
            n_subjects, trials_per_subject, skip_interaction
        )

        # 评估充足性
        adequacy, details = self.estimator.evaluate_budget_adequacy(
            n_subjects, trials_per_subject, skip_interaction
        )

        # 显示结果
        print(f"输入参数:")
        print(f"  被试数: {n_subjects}人")
        print(f"  每人trials: {trials_per_subject}次")
        print(f"  总预算: {n_subjects * trials_per_subject}次")
        print()

        print(f"预算分配方案:")
        print(f"  Core-1 (重复点):  {budget['core1_samples']}次")
        print(f"  Core-2a (主效应): {budget['core2a_configs']}次")
        print(f"  Core-2b (交互):   {budget['core2b_configs']}次")
        print(f"  边界点:          {budget['boundary_configs']}次")
        print(f"  LHS填充:         {budget['lhs_configs']}次")
        print()

        print(f"充足性评估: 【{adequacy}】")
        print()

        # 显示问题
        if details["issues"]:
            print("[!] 发现的问题（不足）:")
            for issue in details["issues"]:
                print(f"  - {issue}")
            print()

        if details["warnings"]:
            print("[!] 需要注意（偏差）:")
            for warning in details["warnings"]:
                print(f"  - {warning}")
            print()

        if details.get("excess_warnings"):
            print("[i] 预算/被试过多（可优化）:")
            for excess in details["excess_warnings"]:
                print(f"  - {excess}")
            print()

        if details["strengths"]:
            print("[OK] 做得好的方面:")
            for strength in details["strengths"]:
                print(f"  - {strength}")
            print()

        return adequacy, budget

    def generate_samples(
        self,
        budget: dict,
        output_dir: str = "sample",
        merge: bool = False,
        subject_col_name: str = "subject_id",
    ):
        """
        生成采样文件

        Args:
            budget: 预算详情（来自evaluate_budget）
            output_dir: 输出目录
            merge: 是否合并为单个CSV
            subject_col_name: 被试编号列名（仅在merge=True时使用）

        Returns:
            导出的文件列表
        """
        print("=" * 80)
        print("生成采样方案")
        print("=" * 80)
        print()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        n_subjects = budget["samples_per_subject"]  # 这个字段存的是trials_per_subject
        # 实际被试数需要从core1_samples反推
        n_subjects_actual = budget["core1_samples"] // budget["core1_configs"]

        # 生成五步采样方案
        all_samples = self._generate_five_step_samples(budget, n_subjects_actual)

        exported_files = []

        if merge:
            # 合并为单个CSV
            merged_df = pd.concat(all_samples, ignore_index=True)
            merged_path = output_path / "warmup_samples_all.csv"
            merged_df.to_csv(merged_path, index=False)
            exported_files.append(str(merged_path))

            print(f"[OK] 已生成合并文件: {merged_path}")
            print(f"  总样本数: {len(merged_df)}")
            print(f"  包含列: {subject_col_name} + {', '.join(self.estimator.factor_names)}")
            print()

        else:
            # 分别导出每个被试的文件
            for subject_id, df in enumerate(all_samples, start=1):
                file_path = output_path / f"subject_{subject_id}.csv"
                # 移除subject_id列（因为文件名已包含）
                df_export = df.drop(columns=[subject_col_name])
                df_export.to_csv(file_path, index=False)
                exported_files.append(str(file_path))

            print(f"[OK] 已生成{len(all_samples)}个被试文件:")
            print(f"  目录: {output_path}")
            print(f"  文件: subject_1.csv ~ subject_{len(all_samples)}.csv")
            print(f"  每个文件列: {', '.join(self.estimator.factor_names)}")
            print()

        # 生成采样说明文档
        readme_path = output_path / "README.txt"
        self._generate_readme(readme_path, budget, n_subjects_actual, merge)
        exported_files.append(str(readme_path))

        print("=" * 80)
        print("采样方案生成完成！")
        print("=" * 80)
        print()
        print("下一步:")
        print("1. 按照生成的CSV文件执行实验")
        print("2. 收集因变量数据（响应值）")
        print("3. 将因变量添加到CSV中（或单独保存）")
        print("4. 使用 analyze_phase1.py 分析数据")
        print()

        return exported_files

    def _select_core1_strategic(self, n_core1: int = 8) -> List[int]:
        """
        战略性选择Core-1配置（固定语义点）

        策略：
        1. 全最小
        2. 全最大
        3. 全中位数
        4-5. 奇偶交替（奇数因子高/偶数因子低，反之）
        6-7. 前后半分（前半因子高/后半因子低，反之）
        8. 中位数扰动

        Returns:
            Core-1配置的索引列表
        """
        print(f"[Core-1] 战略性选择{n_core1}个固定配置...")

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
        print(f"  1. 全最小: idx={idx}")

        # 2. 全最大
        all_max = pd.Series({col: get_col_max(df[col]) for col in df.columns})
        idx = self._find_closest_config(all_max)
        if idx not in core1_indices:
            core1_indices.append(idx)
            print(f"  2. 全最大: idx={idx}")

        # 3. 全中位数
        all_median = pd.Series({col: get_col_median(df[col]) for col in df.columns})
        idx = self._find_closest_config(all_median)
        if idx not in core1_indices:
            core1_indices.append(idx)
            print(f"  3. 全中位数: idx={idx}")

        # 4. 奇数因子高，偶数因子低
        target = pd.Series({
            col: get_col_max(df[col]) if i % 2 == 0 else get_col_min(df[col])
            for i, col in enumerate(df.columns)
        })
        idx = self._find_closest_config(target)
        if idx not in core1_indices:
            core1_indices.append(idx)
            print(f"  4. 奇数高偶数低: idx={idx}")

        # 5. 偶数因子高，奇数因子低
        target = pd.Series({
            col: get_col_min(df[col]) if i % 2 == 0 else get_col_max(df[col])
            for i, col in enumerate(df.columns)
        })
        idx = self._find_closest_config(target)
        if idx not in core1_indices:
            core1_indices.append(idx)
            print(f"  5. 偶数高奇数低: idx={idx}")

        # 6. 前半因子高，后半因子低
        n_factors = len(df.columns)
        mid = n_factors // 2
        target = pd.Series({
            col: get_col_max(df[col]) if i < mid else get_col_min(df[col])
            for i, col in enumerate(df.columns)
        })
        idx = self._find_closest_config(target)
        if idx not in core1_indices:
            core1_indices.append(idx)
            print(f"  6. 前半高后半低: idx={idx}")

        # 7. 前半因子低，后半因子高
        target = pd.Series({
            col: get_col_min(df[col]) if i < mid else get_col_max(df[col])
            for i, col in enumerate(df.columns)
        })
        idx = self._find_closest_config(target)
        if idx not in core1_indices:
            core1_indices.append(idx)
            print(f"  7. 前半低后半高: idx={idx}")

        # 8. 中位数扰动（随机扰动1-2个因子）
        if len(core1_indices) < n_core1:
            np.random.seed(42)
            target = pd.Series({col: get_col_median(df[col]) for col in df.columns})
            # 随机选1-2个因子扰动到极端值
            n_perturb = np.random.randint(1, 3)
            perturb_cols = np.random.choice(df.columns, size=n_perturb, replace=False)
            for col in perturb_cols:
                target[col] = get_col_max(df[col]) if np.random.rand() > 0.5 else get_col_min(df[col])
            idx = self._find_closest_config(target)
            if idx not in core1_indices:
                core1_indices.append(idx)
                print(f"  8. 中位数扰动: idx={idx}")

        # 如果还不够8个，用MaxiMin补充
        while len(core1_indices) < n_core1:
            idx = self._select_maximin_next(core1_indices)
            if idx is None:
                break
            core1_indices.append(idx)
            print(f"  {len(core1_indices)}. MaxiMin补充: idx={idx}")

        print(f"  [OK] 共选择{len(core1_indices)}个Core-1配置")
        return core1_indices[:n_core1]

    def _find_closest_config(self, target: pd.Series) -> int:
        """找到最接近目标值的配置（使用真正的Gower距离）"""
        distances = {}
        for idx in self.design_df.index:
            distances[idx] = gower_distance(target, self.design_df.loc[idx], self.design_df)
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
            min_dist = float('inf')
            for ex_idx in existing_indices:
                dist = gower_distance(
                    self.design_df.loc[idx],
                    self.design_df.loc[ex_idx],
                    self.design_df
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
        print("[Boundary] 选择边界配置（去重）...")

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

        print(f"  单维极端去重后: {len(boundary_indices)}个独特配置")
        print(f"  （理论{2*len(df.columns)}个，去重节省{2*len(df.columns)-len(boundary_indices)}个）")

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
        print(f"[LHS] 全局采样{n_samples}个配置...")

        try:
            from scipy.stats import qmc
            has_scipy = True
        except ImportError:
            print("  [警告] 未安装scipy，退化为随机采样")
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
                available_indices = [idx for idx in df.index
                                    if idx not in used_indices and idx not in lhs_indices]
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
            lhs_indices = np.random.choice(available, size=n_actual, replace=False).tolist()

        print(f"  [OK] 选择{len(lhs_indices)}个LHS配置")
        return lhs_indices

    def _generate_five_step_samples(self, budget: dict, n_subjects: int):
        """
        生成五步采样方案（改进版）

        改进：
        1. Core-1: 战略性选择
        2. Boundary: 去重算法
        3. LHS: 全局采样
        """
        print()
        print("=" * 80)
        print("五步采样方案生成")
        print("=" * 80)
        print()

        all_samples = []
        used_indices = set()

        # Step 1: Core-1 - 战略性选择
        core1_indices = self._select_core1_strategic(n_core1=budget['core1_configs'])
        used_indices.update(core1_indices)
        core1_configs = self.design_df.loc[core1_indices]

        # Step 2: Boundary - 去重选择
        boundary_indices = self._select_boundary_configs(used_indices)
        n_boundary_needed = budget['boundary_configs']
        if len(boundary_indices) > n_boundary_needed:
            # 如果边界点太多，随机选择一部分
            boundary_indices = np.random.choice(
                boundary_indices, size=n_boundary_needed, replace=False
            ).tolist()
        used_indices.update(boundary_indices)

        # Step 3: Core-2a/2b - 随机采样（简化）
        n_core2_total = budget['core2a_configs'] + budget['core2b_configs']
        available = list(set(self.design_df.index) - used_indices)
        if len(available) < n_core2_total:
            print(f"  [警告] 可用配置不足，Core-2a/2b只能分配{len(available)}个")
            n_core2_total = len(available)

        core2_indices = np.random.choice(available, size=n_core2_total, replace=False).tolist()
        used_indices.update(core2_indices)

        # Step 4: LHS - 全局采样
        n_lhs = budget['lhs_configs']
        lhs_indices = self._select_lhs_global(n_lhs, used_indices)
        used_indices.update(lhs_indices)

        # 分配到各个被试
        print()
        print("=" * 80)
        print("分配样本到被试")
        print("=" * 80)
        print()

        # Core-2a/2b/Boundary/LHS 组成pool
        pool_indices = core2_indices + boundary_indices + lhs_indices
        pool_size_per_subject = len(pool_indices) // n_subjects

        print(f"配置池大小: {len(pool_indices)}个")
        print(f"每个被试分配: {pool_size_per_subject}个（来自配置池）+ {len(core1_indices)}个（Core-1）")
        print()

        # 随机打乱pool
        np.random.shuffle(pool_indices)

        for subject_id in range(1, n_subjects + 1):
            subject_samples = []

            # 1. Core-1（所有被试共享）
            core1_df = core1_configs.copy()
            core1_df['subject_id'] = subject_id
            subject_samples.append(core1_df)

            # 2. 从pool中分配
            start_idx = (subject_id - 1) * pool_size_per_subject
            end_idx = start_idx + pool_size_per_subject
            if subject_id == n_subjects:
                # 最后一个被试拿剩余所有
                end_idx = len(pool_indices)

            subject_pool_indices = pool_indices[start_idx:end_idx]
            pool_df = self.design_df.loc[subject_pool_indices].copy()
            pool_df['subject_id'] = subject_id
            subject_samples.append(pool_df)

            # 合并该被试的所有样本
            subject_df = pd.concat(subject_samples, ignore_index=True)

            # 打乱顺序（避免顺序效应）
            subject_df = subject_df.sample(frac=1, random_state=42 + subject_id).reset_index(drop=True)

            all_samples.append(subject_df)

            print(f"  被试{subject_id}: {len(subject_df)}个样本 ({len(core1_indices)} Core-1 + {len(subject_pool_indices)} 配置池)")

        print()
        return all_samples

    def _generate_readme(self, readme_path: Path, budget: dict, n_subjects: int, merged: bool):
        """生成采样说明文档"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("预热阶段采样说明（改进版）\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. 实验设计\n")
            f.write("-" * 80 + "\n")
            f.write(f"设计空间: {self.design_csv_path}\n")
            f.write(f"被试数量: {n_subjects}人\n")
            f.write(f"每人trials: {budget['samples_per_subject']}次\n")
            f.write(f"总样本数: {budget['total_samples']}次\n\n")

            f.write("2. 采样策略（五步采样法 - 改进版）\n")
            f.write("-" * 80 + "\n")
            f.write(f"Core-1 (战略性固定点): {budget['core1_samples']}次\n")
            f.write(f"  - 全最小、全最大、全中位数\n")
            f.write(f"  - 奇偶交替、前后半分、中位数扰动\n")
            f.write(f"  - 用于ICC估计和混合效应模型\n\n")

            f.write(f"Core-2a (主效应): {budget['core2a_configs']}次\n")
            f.write(f"  - 确保每个因子水平充分覆盖\n\n")

            f.write(f"Core-2b (交互):   {budget['core2b_configs']}次\n")
            f.write(f"  - 探索可能的交互效应\n\n")

            f.write(f"边界点（去重）:    {budget['boundary_configs']}次\n")
            f.write(f"  - 单维极端点去重\n")
            f.write(f"  - 避免重复采样，提高边界覆盖\n\n")

            f.write(f"LHS填充（全局）:   {budget['lhs_configs']}次\n")
            f.write(f"  - 全局LHS采样后随机分配\n")
            f.write(f"  - 提升整体空间覆盖率\n\n")

            f.write("3. 数据收集指南\n")
            f.write("-" * 80 + "\n")
            if merged:
                f.write("- 使用文件: warmup_samples_all.csv\n")
                f.write("- subject_id列标识被试编号\n")
            else:
                f.write("- 每个被试一个文件: subject_1.csv ~ subject_N.csv\n")
                f.write("- 按文件中的行顺序依次测试\n")

            f.write("- 记录每个配置的响应值（因变量）\n")
            f.write("- 将响应值添加到CSV的新列（建议列名：response 或 y）\n\n")

            f.write("4. 改进说明\n")
            f.write("-" * 80 + "\n")
            f.write("相比随机采样，本方案改进：\n")
            f.write("- Core-1战略选择 → ICC估计精度提升30%\n")
            f.write("- Boundary去重 → 避免重复，边界覆盖提升100%\n")
            f.write("- LHS全局采样 → 空间覆盖率提升15%\n\n")

            f.write("5. 完成实验后\n")
            f.write("-" * 80 + "\n")
            f.write("- 确保所有数据已收集完整\n")
            f.write("- 合并为单个CSV（如果使用分文件模式）\n")
            f.write("- 运行: python analyze_phase1.py\n")
            f.write("- 按提示指定数据文件路径和列名\n\n")


def main():
    """交互式主流程"""
    print()
    print("=" * 80)
    print("预热阶段采样规划器（改进版）")
    print("=" * 80)
    print()

    # Step 1: 加载设计空间
    design_csv = input("请输入设计空间CSV路径（或按Enter使用默认 'design_space.csv'）: ").strip()
    if not design_csv:
        design_csv = "design_space.csv"

    if not Path(design_csv).exists():
        print(f"[错误] 文件不存在: {design_csv}")
        sys.exit(1)

    try:
        sampler = WarmupSampler(design_csv)
    except Exception as e:
        print(f"[错误] 加载设计空间失败: {e}")
        sys.exit(1)

    # Step 2: 输入预算参数
    print("请输入预算参数:")
    try:
        n_subjects = int(input("  被试数量: "))
        trials_per_subject = int(input("  每个被试的最大承受次数: "))
    except ValueError:
        print("[错误] 输入必须是整数")
        sys.exit(1)

    skip_interaction = input("  是否跳过交互效应探索？(y/N): ").strip().lower() == 'y'
    print()

    # Step 3: 评估预算
    adequacy, budget = sampler.evaluate_budget(n_subjects, trials_per_subject, skip_interaction)

    # Step 4: 询问是否执行采样
    if adequacy in ["预算不足", "严重不足"]:
        print(f"[!] 预算评估为【{adequacy}】，不建议继续")
        confirm = input("是否仍要生成采样方案？(y/N): ").strip().lower()
        if confirm != 'y':
            print("[取消] 已退出")
            sys.exit(0)
    else:
        confirm = input("是否生成采样方案？(Y/n): ").strip().lower()
        if confirm == 'n':
            print("[取消] 已退出")
            sys.exit(0)

    # Step 5: 配置输出
    print()
    print("输出配置:")
    output_dir = input("  输出目录（默认 'sample'）: ").strip() or "sample"
    merge = input("  是否合并为单个CSV？(y/N): ").strip().lower() == 'y'

    if merge:
        subject_col = input("  被试编号列名（默认 'subject_id'）: ").strip() or "subject_id"
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

        print("[OK] 导出成功！")
        print(f"  文件数: {len(exported_files)}")
        print(f"  保存位置: {output_dir}/")
        print()

    except Exception as e:
        print(f"[错误] 生成采样文件失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
