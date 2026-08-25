"""
预热采样预算估算工具 - 基于五步采样法
估算所需的总采样次数

功能：
1. 读取设计空间CSV
2. 分析变量分布
3. 根据五步采样策略估算所需预算
4. 判断输入的被试数量是否满足要求（不足/刚好/充足）
"""

import pandas as pd
import numpy as np
import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from loguru import logger


class WarmupBudgetEstimator:
    """预热采样预算估算器"""

    def __init__(self, design_csv_path: str):
        """
        初始化估算器

        Args:
            design_csv_path: 设计空间CSV文件路径
        """
        self.design_df = pd.read_csv(design_csv_path)

        # 检测因子列（排除常见的响应变量列名）
        response_col_patterns = [
            r"^y$",  # 只有单独的y列
            r"^response$",
            r"^outcome$",
            r"^result$",
            r"^target$",
            r"^label$",
            r"^class$",
            r"^category$",
            r".*_id$",  # 以_id结尾的列
            r"^condition.*",  # 以condition开头的列
        ]

        self.factor_names = [
            col
            for col in self.design_df.columns
            if not any(
                re.search(pattern, col.lower()) for pattern in response_col_patterns
            )
        ]

        if not self.factor_names:
            raise ValueError(
                f"No factor columns found in CSV. CSV columns: {list(self.design_df.columns)}"
            )

        self.d = len(self.factor_names)
        self.n_configs = len(self.design_df)

    def analyze_design_space(self):
        """分析设计空间的统计特征"""
        print("=" * 70)
        print("Design Space Analysis")
        print("=" * 70)
        print(f"Total configurations: {self.n_configs}")
        print(f"Number of factors: {self.d}")
        print(f"Factor names: {', '.join(self.factor_names)}")

        print("Factor distribution statistics:")
        print("-" * 70)
        for factor in self.factor_names:
            values = self.design_df[factor]
            unique_vals = values.unique()
            msg = f"  {factor}:\n"

            # 检测变量类型
            if values.dtype in ["object", "bool"]:
                # Categorical or boolean
                msg += f"    - Type: Categorical\n"
                msg += f"    - Unique values: {len(unique_vals)}\n"
                msg += f"    - Samples: {list(unique_vals)[:10]}"
            else:
                # Numeric
                msg += f"    - Type: Numeric\n"
                msg += f"    - Range: [{values.min():.3f}, {values.max():.3f}]\n"
                msg += f"    - Unique values: {len(unique_vals)}\n"
                msg += f"    - Mean: {values.mean():.3f}\n"
                msg += f"    - Std: {values.std():.3f}"
            print(msg)

    def estimate_budget_requirements(
        self, n_subjects: int, budget_per_subject: int, skip_interaction: bool = False,
        interaction_pairs: List = None,
        main_effects_first: bool = False,
        n_core1: int = 4
    ):
        """
        估算预算需求 - 基于五步采样法

        采样策略:
        - Core-1: 4个固定配置(锚点)，每个被试都测 → 4×N次采样
        - Core-2a: D-optimal配置池，分配给各被试
        - Core-2b: 交互对配置池，分配给各被试（可选）
        - Boundary: 边界极端配置池，分配给各被试
        - LHS: 均匀填充配置池，分配给各被试

        Args:
            n_subjects: 被试数量
            budget_per_subject: 每个被试的测试次数
            skip_interaction: 是否跳过交互效应探索（Core-2b）
            interaction_pairs: 指定的交互对列表
            main_effects_first: 是否主效应优先 (增加主效应预算，减少交互和探索)
            n_core1: 共享锚点数量

        Returns:
            dict: 预算细节
        """
        if interaction_pairs is None:
            interaction_pairs = []

        # 总预算
        total_budget = n_subjects * budget_per_subject

        # Step 1: Core-1固定重复点 (锚点)
        # 极致精简：从8个降至4个，足以锚定ICC和BaseGP量程
        n_core1_configs = n_core1
        n_core1_samples = n_core1_configs * n_subjects  # 每人都测这4个

        # Step 2: 剩余预算（用于分配其他配置）
        remaining_budget = total_budget - n_core1_samples

        # Step 3-5: 按比例分配剩余预算
        # 策略：剩余预算按固定比例分配，确保总和=100%
        # 目标比例（相对于剩余预算）:
        # - Core-2a: 40% (换算成总预算约37%，接近目标27%)
        # - Core-2b: 28% (换算成总预算约26%，接近目标17%)
        # - 边界+LHS: 32% (换算成总预算约30%，符合目标25-35%)
        # 注：这些比例加起来=100%，避免剩余预算失控

        # 计算最小需求
        max_levels = max([len(self.design_df[f].unique()) for f in self.factor_names])

        # ======================== 核心思路转变 ========================
        # 从"基于百分比分配"改为"基于最小配置数的绝对值约束"
        # 原理：
        # - Core-2a、Core-2b、探索都有最小配置数需求（不是百分比）
        # - 剩余预算足以满足最小值后，再按比例分配超出部分
        # ============================================================

        # 1. 基于因子数计算各模块的最小配置数（绝对值）
        # Core-2a最小值：主效应覆盖 = 因子数 × 因子内不同水平的测试次数
        # 保守估计：每个因子至少3-5个配置来探索效应
        n_core2a_min_abs = max(
            15,  # 绝对最小值
            int(
                self.d * 3 * (0.3 + n_subjects / 20.0)
            ),  # 随被试增长：3人=0.45*d, 8人=0.7*d
        )

        # 如果主效应优先，增加最小保证
        if main_effects_first:
            n_core2a_min_abs = int(n_core2a_min_abs * 1.5)

        # Core-2b最小值：交互项覆盖
        # 如果有指定交互对，执行深度优先策略：每对至少4个配置
        if not skip_interaction:
            if interaction_pairs:
                # 深度优先：指定对每对4个点
                n_core2b_min_abs = len(interaction_pairs) * 4
            else:
                # 广度优先：约15对×0.7个配置 ≈ 10-12
                n_core2b_min_abs = max(15, int(self.d * (self.d - 1) * 0.7))
            
            # 如果主效应优先，减少交互项最小保证
            if main_effects_first:
                n_core2b_min_abs = int(n_core2b_min_abs * 0.5)
        else:
            n_core2b_min_abs = 0

        # 探索预算最小值：至少覆盖边界和充分的随机点
        # 边界 ≈ 2^d的顶点 = 64（6因子），实际通常只取部分
        # 最小值：d*2 = 12个边界 + d*2 = 12个LHS = 24个最少
        n_explore_min_abs = max(20, int(self.d * 3.5))  # 绝对最小值  # 6因子≈21

        # 2. 检查剩余预算是否足以满足最小值
        min_sum = n_core2a_min_abs + n_core2b_min_abs + n_explore_min_abs

        if remaining_budget < min_sum:
            # 预算紧张：按比例缩减，但优先保护探索和Core-2a
            scale_factor = remaining_budget / min_sum
            n_core2a = max(8, int(n_core2a_min_abs * scale_factor))
            n_core2b = (
                max(6, int(n_core2b_min_abs * scale_factor))
                if not skip_interaction
                else 0
            )
            n_explore = remaining_budget - n_core2a - n_core2b
        else:
            # 预算充足：按最小值 + 按比例分配超出部分
            surplus = remaining_budget - min_sum

            if not skip_interaction:
                if main_effects_first:
                    # 主效应优先比例：Core-2a 85% / Core-2b 5% / Explore 10% (极度倾斜)
                    n_core2a = n_core2a_min_abs + int(surplus * 0.85)
                    n_core2b = n_core2b_min_abs + int(surplus * 0.05)
                    n_explore = remaining_budget - n_core2a - n_core2b
                else:
                    # 比例：Core-2a 40% / Core-2b 30% / Explore 30%
                    n_core2a = n_core2a_min_abs + int(surplus * 0.40)
                    n_core2b = n_core2b_min_abs + int(surplus * 0.30)
                    n_explore = remaining_budget - n_core2a - n_core2b
            else:
                if main_effects_first:
                    # 纯主效应模式：100% 分配给 Core-2a，彻底关闭交互与探索
                    n_core2a = remaining_budget
                    n_core2b = 0
                    n_explore = 0
                    n_boundary = 0
                    n_lhs = 0
                else:
                    # 没有Core-2b：比例 55% / 45%
                    n_core2a = n_core2a_min_abs + int(surplus * 0.55)
                    n_core2b = 0
                    n_explore = remaining_budget - n_core2a
        # 3. 分配探索预算：边界40%，LHS60%
        # 只有在非纯主效应模式下才计算这些
        if not (skip_interaction and main_effects_first):
            n_boundary_min = max(2 * self.d, int(3 * min(1.0, n_subjects / 3.0)))
            n_boundary = max(n_boundary_min, int(n_explore * 0.40))
            n_lhs = max(0, n_explore - n_boundary)
        else:
            n_boundary = 0
            n_lhs = 0

        # 汇总结果
        budget_details = {
            # Core-1
            "core1_configs": n_core1_configs,
            "core1_samples": n_core1_samples,
            # Core-2a
            "core2a_configs": n_core2a,
            # Core-2b
            "core2b_configs": n_core2b,
            "skip_interaction": skip_interaction,
            "main_effects_first": main_effects_first,
            # Boundary
            "boundary_configs": n_boundary,
            # LHS
            "lhs_configs": n_lhs,
            # 配置池总数（Core-2a/2b/Boundary/LHS）
            "pool_configs": n_core2a + n_core2b + n_boundary + n_lhs,
            # 总采样次数
            "total_samples": n_core1_samples + n_core2a + n_core2b + n_boundary + n_lhs,
            # 独立配置总数
            "unique_configs": n_core1_configs
            + n_core2a
            + n_core2b
            + n_boundary
            + n_lhs,
            # 每人平均测试次数
            "samples_per_subject": budget_per_subject,
        }

        return budget_details

    def evaluate_budget_adequacy(
        self, n_subjects: int, trials_per_subject: int, skip_interaction: bool = False,
        interaction_pairs: List = None,
        main_effects_first: bool = False,
        n_core1: int = 4
    ):
        """
        评估预算充足性 - 基于覆盖性标准

        评估维度：
        1. 空间覆盖度：每个水平的采样次数、独特配置占比
        2. 信息密度：重复点数量、ICC估计能力
        3. 结构平衡性：模块预算比例

        Args:
            n_subjects: 被试数量
            trials_per_subject: 每个被试的测试次数
            skip_interaction: 是否跳过交互效应探索
            interaction_pairs: 指定的交互对列表
            main_effects_first: 是否主效应优先

        Returns:
            tuple: (评估结果, 详细信息dict)
        """
        budget = self.estimate_budget_requirements(
            n_subjects, trials_per_subject, skip_interaction, interaction_pairs, main_effects_first, n_core1
        )


        total_samples = budget["total_samples"]
        total_available = n_subjects * trials_per_subject

        # 计算关键指标
        unique_ratio = budget["unique_configs"] / total_samples  # 独特配置占比

        # 估算每个水平的理论采样次数
        # 注意：这里应该用独特配置数而不是总采样次数，因为重复采样不增加水平覆盖
        max_levels = max([len(self.design_df[f].unique()) for f in self.factor_names])

        # 每个水平的预期采样次数 = 独特配置数 / 总水平数
        # （假设配置均匀分布在各水平上）
        avg_unique_samples_per_level = budget["unique_configs"] / (self.d * max_levels)

        # 考虑Core-1重复后的实际采样次数
        # Core-1的8个配置会被重复n_subjects次
        # 假设Core-1均匀覆盖水平，每个水平从Core-1获得的重复 ≈ 8/总水平数 * n_subjects
        core1_repeats_per_level = (
            budget["core1_configs"] / (self.d * max_levels)
        ) * n_subjects
        avg_samples_per_level = avg_unique_samples_per_level + core1_repeats_per_level

        # 模块预算比例
        core1_ratio = budget["core1_samples"] / total_samples
        core2a_ratio = budget["core2a_configs"] / total_samples
        core2b_ratio = budget["core2b_configs"] / total_samples
        explore_ratio = (
            budget["boundary_configs"] + budget["lhs_configs"]
        ) / total_samples

        # ICC估计能力（Core-1重复人数）
        icc_subjects = n_subjects

        # 评分系统
        issues = []  # 严重不足的问题
        warnings = []  # 偏差（偏低/偏高）
        excess_warnings = []  # 过多/过度的问题
        strengths = []

        # === 维度1：空间覆盖度 ===
        # 1.1 边缘覆盖：每个水平的独特配置数（更重要的指标）
        if avg_unique_samples_per_level < 2:
            issues.append(
                f"边缘覆盖严重不足：平均每个水平仅有 {avg_unique_samples_per_level:.1f} 个独特配置 (需 ≥2)"
            )
        elif avg_unique_samples_per_level < 3:
            warnings.append(
                f"边缘覆盖偏低：平均每个水平有 {avg_unique_samples_per_level:.1f} 个独特配置 (建议 3-5)"
            )
        elif avg_unique_samples_per_level <= 5:
            strengths.append(
                f"边缘覆盖良好：平均每个水平有 {avg_unique_samples_per_level:.1f} 个独特配置"
            )
        elif avg_unique_samples_per_level <= 8:
            strengths.append(
                f"边缘覆盖充足：平均每个水平有 {avg_unique_samples_per_level:.1f} 个独特配置"
            )
        else:
            excess_warnings.append(
                f"边缘覆盖过剩：平均每个水平有 {avg_unique_samples_per_level:.1f} 个独特配置 (>8 边际收益递减，建议减少被试或每人 trials)"
            )

        # 1.2 独特配置占比
        if unique_ratio < 0.60:
            issues.append(f"独特配置占比过低: {unique_ratio*100:.1f}% (需 ≥60%)")
        elif unique_ratio < 0.70:
            warnings.append(f"独特配置占比偏低: {unique_ratio*100:.1f}% (建议 70-80%)")
        elif unique_ratio <= 0.85:
            strengths.append(f"独特配置占比合理: {unique_ratio*100:.1f}%")
        else:
            excess_warnings.append(
                f"独特配置占比过高: {unique_ratio*100:.1f}% (重复点可能不足，建议增加 Core-1 配置)"
            )

        # === 维度2：信息密度 ===
        # 2.1 ICC估计能力
        if icc_subjects < 3:
            issues.append(f"ICC 估计能力严重不足: 仅 {icc_subjects} 名被试 (需 ≥3)")
        elif icc_subjects < 5:
            warnings.append(f"ICC 估计能力偏弱: {icc_subjects} 名被试 (建议 5-7)")
        elif icc_subjects <= 7:
            strengths.append(f"ICC 估计能力良好: {icc_subjects} 名被试")
        elif icc_subjects <= 10:
            strengths.append(f"ICC 估计能力充足: {icc_subjects} 名被试")
        else:
            excess_warnings.append(
                f"被试数量过多: {icc_subjects} 名 (>10 对 ICC 提升有限，建议减少被试)"
            )

        # 2.2 重复配置数量
        n_repeated_configs = budget["core1_configs"]
        if n_repeated_configs < 5:
            issues.append(f"重复配置数量不足: 仅 {n_repeated_configs} 个 (需 ≥5)")
        elif n_repeated_configs < 6:
            warnings.append(f"重复配置数量偏低: {n_repeated_configs} 个 (建议 6-10)")
        elif n_repeated_configs <= 10:
            strengths.append(f"重复配置数量合理: {n_repeated_configs}")
        else:
            excess_warnings.append(
                f"重复配置数量过多: {n_repeated_configs} (>10 占用过多预算，建议减少)"
            )

        # === 维度3：结构平衡性 ===
        # 改为基于配置数的绝对值评估，并用动态阈值适应被试数变化
        # 原理：更多被试 → 更多预算 → 更多配置数（是正常现象，不是过度）

        # 计算动态阈值：随被试数增长而增长
        # 基础值 + 被试增量
        base_core2a = 25  # 3人时的目标
        base_core2b = 20
        base_explore = 25
        per_subject_add_2a = 4.0  # 每增加1人，增加4.0个配置
        per_subject_add_2b = 4.0  # 每增加1人，增加4.0个配置
        per_subject_add_exp = 4.0

        # 动态阈值（基础 + 被试增量）
        target_core2a = base_core2a + (n_subjects - 3) * per_subject_add_2a
        target_core2b = base_core2b + (n_subjects - 3) * per_subject_add_2b
        target_explore = base_explore + (n_subjects - 3) * per_subject_add_exp

        # 3.1 Core-2a评估：基于动态阈值
        if budget["core2a_configs"] < 15:
            issues.append(
                f"Core-2a 严重不足：仅 {budget['core2a_configs']} 个配置 (需 ≥15)"
            )
        elif budget["core2a_configs"] < 20:
            warnings.append(
                f"Core-2a 偏低：{budget['core2a_configs']} 个配置 (建议 ≥20)"
            )
        elif budget["core2a_configs"] < target_core2a * 1.2:
            strengths.append(f"Core-2a 数量合理：{budget['core2a_configs']} 个配置")
        elif budget["core2a_configs"] < target_core2a * 1.5:
            warnings.append(f"Core-2a 偏高：{budget['core2a_configs']} 个配置 (可适当减少)")
        else:
            excess_warnings.append(
                f"Core-2a 过多：{budget['core2a_configs']} 个配置 (预算浪费)"
            )

        # 3.2 Core-2b评估：基于动态阈值
        if not skip_interaction:
            if budget["core2b_configs"] < 10:
                issues.append(
                    f"Core-2b 严重不足：仅 {budget['core2b_configs']} 个配置 (需 ≥10)"
                )
            elif budget["core2b_configs"] < 15:
                warnings.append(
                    f"Core-2b 偏低：{budget['core2b_configs']} 个配置 (建议 ≥15)"
                )
            elif budget["core2b_configs"] < target_core2b * 1.2:
                strengths.append(f"Core-2b 数量合理：{budget['core2b_configs']} 个配置")
            elif budget["core2b_configs"] < target_core2b * 1.5:
                warnings.append(
                    f"Core-2b 偏高：{budget['core2b_configs']} 个配置 (可适当减少)"
                )
            else:
                excess_warnings.append(
                    f"Core-2b 过多：{budget['core2b_configs']} 个配置 (预算浪费)"
                )

        # 3.3 探索预算评估：基于动态阈值
        explore_configs = budget["boundary_configs"] + budget["lhs_configs"]
        if explore_configs < 15:
            issues.append(f"探索预算严重不足：仅 {explore_configs} 个配置 (需 ≥15)")
        elif explore_configs < 20:
            warnings.append(f"探索预算偏低：{explore_configs} 个配置 (建议 ≥20)")
        elif explore_configs < target_explore * 1.2:
            strengths.append(f"探索预算合理：{explore_configs} 个配置")
        elif explore_configs < target_explore * 1.5:
            warnings.append(f"探索预算偏高：{explore_configs} 个配置 (可适当减少)")
        else:
            excess_warnings.append(f"探索预算过多：{explore_configs} 个配置 (预算浪费)")

        # 3.4 Core-1评估：基于配置数和被试数
        # Core-1固定8个，关键是被试数足够支持ICC估计
        if budget["core1_configs"] < 6:
            issues.append(f"Core-1 配置数不足：仅 {budget['core1_configs']} (需 ≥6)")
        elif budget["core1_configs"] <= 10:
            strengths.append(f"Core-1 配置数合理：{budget['core1_configs']}")
        else:
            excess_warnings.append(
                f"Core-1 配置数过多：{budget['core1_configs']} (占比 {core1_ratio*100:.1f}%)"
            )

        # === 综合评估 ===
        # 统计各类问题数量
        n_issues = len(issues)
        n_warnings = len(warnings)
        n_excess = len(excess_warnings)
        n_total_problems = n_issues + n_warnings + n_excess

        # 评价逻辑优先级（从高到低）：
        # 1. 预算短缺（总预算不够）→ 最严重
        # 2. 严重不足（多个关键指标不足）→ 无法使用
        # 3. 不足（有关键指标不足）→ 勉强可用但有风险
        # 4. 过度充足（资源浪费但无不足问题）→ 需要优化
        # 5. 勉强（偏差较多但资源量合理）→ 可用但不理想
        # 6. 基本满足/充分/刚好（合理范围）→ 推荐使用

        if total_samples > total_available * 1.05:
            # 总预算不够
            overall = "Insufficient Budget"
            issues.insert(
                0, f"Total budget shortage: need {total_samples}, only have {total_available}"
            )
        elif n_issues >= 3:
            # 多个关键指标严重不足
            overall = "Severely Insufficient"
        elif n_issues >= 1:
            # 【优先级提前】：只要有关键指标不足，就是"不足"
            # 即使同时有过度问题，也优先解决不足
            overall = "Insufficient"
        elif n_excess >= 2:
            # 资源过度（无不足问题的前提下）
            # 降低阈值到2个，只要有2个过度警告就说明资源明显过剩
            overall = "Overly Sufficient (Optimizable)"
        elif n_warnings >= 4:
            # 有较多偏差类警告，但没有过度问题（否则前面已判断）
            overall = "Marginal"
        elif n_warnings >= 2:
            # 有少量偏差
            overall = "Basic Satisfaction"
        elif n_excess == 1:
            # 只有1个过度问题，整体还算合理
            overall = "Basic Satisfaction"
        elif n_total_problems == 0 and len(strengths) >= 5:
            # 无任何问题，且有5个以上优点
            overall = "Sufficient"
        elif n_total_problems == 0:
            # 无问题，但优点不够多
            overall = "Adequate"
        else:
            # 其他情况
            overall = "Basic Satisfaction"

        return overall, {
            "total_samples": total_samples,
            "total_available": total_available,
            "issues": issues,
            "warnings": warnings,
            "excess_warnings": excess_warnings,
            "strengths": strengths,
            "metrics": {
                "unique_ratio": unique_ratio,
                "avg_unique_samples_per_level": avg_unique_samples_per_level,
                "avg_samples_per_level_with_repeats": avg_samples_per_level,
                "icc_subjects": icc_subjects,
                "n_repeated_configs": n_repeated_configs,
                "core1_ratio": core1_ratio,
                "core2a_ratio": core2a_ratio,
                "core2b_ratio": core2b_ratio,
                "explore_ratio": explore_ratio,
            },
        }

    def print_budget_report(
        self,
        n_subjects: int,
        trials_per_subject: int,
        skip_interaction: bool = False,
        show_comparison: bool = False,
        interaction_pairs: List = None,
        main_effects_first: bool = False,
        n_core1: int = 4,
    ):
        """
        打印完整的预算报告 - 基于五步采样法

        Args:
            n_subjects: 被试数量
            trials_per_subject: 每个被试的测试次数
            skip_interaction: 是否跳过交互效应探索（Core-2b）
            show_comparison: 是否显示有/无交互效应的对比
            interaction_pairs: 指定的交互对列表
            main_effects_first: 是否主效应优先
        """
        logger.info("=" * 70)
        logger.info("Warmup Sampling Budget Estimation Report (Five-Step Method)")
        logger.info("=" * 70)

        # 输入参数
        logger.info("Input Parameters:")
        logger.info("-" * 70)
        logger.info(f"  Number of subjects: {n_subjects}")
        logger.info(f"  Trials per subject: {trials_per_subject}")
        logger.info(f"  Total available budget: {n_subjects * trials_per_subject}")
        logger.info(f"  Main effects first: {main_effects_first}")

        # 采样策略详情
        budget = self.estimate_budget_requirements(
            n_subjects, trials_per_subject, skip_interaction, interaction_pairs, main_effects_first, n_core1
        )

        if show_comparison and not skip_interaction:
            budget_no_inter = self.estimate_budget_requirements(
                n_subjects, trials_per_subject, skip_interaction=True, interaction_pairs=None, main_effects_first=main_effects_first, n_core1=n_core1
            )

        logger.info("Sampling Strategy and Budget Allocation:")
        logger.info("-" * 70)

        mode_desc = (
            "(Interaction exploration skipped)" if skip_interaction else "(Interaction exploration included)"
        )
        logger.info(f"Current mode: {mode_desc}")
        if interaction_pairs and not skip_interaction:
            logger.info(f"  - Specified interaction pairs: {len(interaction_pairs)}")

        logger.info(f"  1. Core-1 Fixed Repeat Points:")
        logger.info(f"     - Configurations: {budget['core1_configs']}")
        logger.info(
            f"     - Samples: {budget['core1_samples']} ({budget['core1_configs']} × {n_subjects})"
        )
        logger.info(f"     - Description: {n_subjects} subjects share the same {budget['core1_configs']} configurations")

        logger.info(f"  2. Core-2a Main Effect Coverage (Pool):")
        logger.info(f"     - Configurations: {budget['core2a_configs']}")
        logger.info(f"     - Description: D-optimal design, distributed among subjects")

        if not skip_interaction:
            logger.info(f"  3. Core-2b Interaction Screening (Pool):")
            logger.info(f"     - Configurations: {budget['core2b_configs']}")
            logger.info(f"     - Description: Interaction-aware sampling (supports free/specified_only/hybrid modes)")
        else:
            logger.info(f"  3. Core-2b Interaction Screening: Skipped")

        logger.info(f"  4. Boundary Extreme Points (Pool):")
        logger.info(f"     - Configurations: {budget['boundary_configs']}")
        logger.info(f"     - Description: Uni-dimensional/Bi-dimensional/Global extremes, distributed among subjects")

        logger.info(f"  5. Stratified LHS Points (Pool):")
        logger.info(f"     - Configurations: {budget['lhs_configs']}")
        logger.info(f"     - Description: Constrained LHS + Gower distance matching, distributed among subjects")

        # 总预算需求
        total_samples = budget["total_samples"]
        total_available = n_subjects * trials_per_subject

        logger.info("Budget Requirement Summary:")
        logger.info("-" * 70)
        logger.info(f"  Total sample requirement: {total_samples}")
        logger.info(f"    - Core-1 (Shared): {budget['core1_samples']} samples")
        logger.info(f"    - Pool (Distributed): {budget['pool_configs']} samples")
        logger.info(f"  Total available samples: {total_available}")
        logger.info(f"  Difference: {total_available - total_samples:+d} samples")
        logger.info(f"  Utilization: {(total_samples/total_available)*100:.1f}%")

        # 充足性评估
        adequacy, eval_details = self.evaluate_budget_adequacy(
            n_subjects, trials_per_subject, skip_interaction, interaction_pairs, main_effects_first, n_core1
        )

        logger.info("=" * 70)
        logger.info("Adequacy Evaluation Results (Based on Coverage Standards)")
        logger.info("=" * 70)

        # 显示综合评估
        status_icons = {
            "Insufficient Budget": "[X]",
            "Severely Insufficient": "[XX]",
            "Insufficient": "[!]",
            "Marginal": "[~]",
            "Basic Satisfaction": "[OK]",
            "Adequate": "[OK+]",
            "Sufficient": "[++]",
            "Overly Sufficient (Optimizable)": "[++!]",
        }
        icon = status_icons.get(adequacy, "[?]")
        logger.info(f"{icon} Budget Adequacy: [{adequacy}]")

        # 显示关键指标
        metrics = eval_details["metrics"]
        logger.info("Key Metrics:")
        logger.info("-" * 70)
        logger.info(
            f"  Total samples: {eval_details['total_samples']} / {eval_details['total_available']}"
        )
        logger.info(f"  Unique configuration ratio: {metrics['unique_ratio']*100:.1f}% (suggest 70-80%)")
        logger.info(
            f"  Avg unique configurations per level: {metrics['avg_unique_samples_per_level']:.1f} (suggest 3-5)"
        )
        logger.info(
            f"  Avg samples per level: {metrics['avg_samples_per_level_with_repeats']:.1f} (including repeats)"
        )
        logger.info(f"  Subjects for ICC estimation: {metrics['icc_subjects']} (suggest 5-7)")
        logger.info(f"  Repeat configurations: {metrics['n_repeated_configs']} (suggest 6-10)")

        logger.info("Module Budget Ratios:")
        logger.info("-" * 70)
        logger.info(f"  Core-1:  {metrics['core1_ratio']*100:5.1f}% (varies with subjects)")
        logger.info(f"  Core-2a: {metrics['core2a_ratio']*100:5.1f}% (suggest 32-40%)")
        if not skip_interaction:
            logger.info(f"  Core-2b: {metrics['core2b_ratio']*100:5.1f}% (suggest 22-28%)")
        logger.info(f"  Exploration: {metrics['explore_ratio']*100:5.1f}% (suggest 25-35%)")

        # 显示问题
        if eval_details["issues"]:
            logger.warning("Issues Found (Insufficient):")
            for issue in eval_details["issues"]:
                logger.warning(f"  [X] {issue}")

        # 显示警告
        if eval_details["warnings"]:
            logger.warning("Warnings (Deviations):")
            for warning in eval_details["warnings"]:
                logger.warning(f"  [!] {warning}")

        # 显示过度警告
        if eval_details["excess_warnings"]:
            logger.info("Excessive Budget/Subjects (Optimizable):")
            for excess in eval_details["excess_warnings"]:
                logger.info(f"  [++] {excess}")

        # 显示优点
        if eval_details["strengths"]:
            logger.info("Strengths:")
            for strength in eval_details["strengths"]:
                logger.info(f"  [+] {strength}")

        # 对比分析
        if show_comparison and not skip_interaction:
            logger.info("=" * 70)
            logger.info("Comparative Analysis: With/Without Interaction Exploration")
            logger.info("=" * 70)
            adequacy_no_inter, eval_no_inter = self.evaluate_budget_adequacy(
                n_subjects, trials_per_subject, skip_interaction=True, n_core1=n_core1
            )
            logger.info(f"  With Interaction (Core-2b):")
            logger.info(f"    - Adequacy: {adequacy}")
            logger.info(f"    - Total samples: {total_samples}")
            logger.info(f"    - Unique configurations: {budget['unique_configs']}")
            logger.info(f"  Without Interaction (Skip Core-2b):")
            logger.info(f"    - Adequacy: {adequacy_no_inter}")
            logger.info(f"    - Total samples: {budget_no_inter['total_samples']}")
            logger.info(f"    - Unique configurations: {budget_no_inter['unique_configs']}")
            logger.info(f"    - Savings: {budget['core2b_configs']} samples")

        # 覆盖率分析
        unique_configs = budget["unique_configs"]
        coverage_rate = unique_configs / self.n_configs
        logger.info("Design Space Coverage Analysis:")
        logger.info("-" * 70)
        logger.info(f"  Total unique configurations: {unique_configs}")
        logger.info(f"    - Core-1 (Shared): {budget['core1_configs']}")
        logger.info(f"    - Pool: {budget['pool_configs']}")
        logger.info(f"  Total design space size: {self.n_configs}")
        logger.info(f"  Coverage rate: {coverage_rate*100:.2f}%")

        # 推荐被试数量范围
        logger.info("Recommended Configuration:")
        logger.info("-" * 70)
        min_trials = budget["core1_configs"] + int(
            np.ceil(budget["pool_configs"] / n_subjects)
        )
        logger.info(f"  For {n_subjects} subjects:")
        logger.info(f"    - Minimum trials per subject: {min_trials}")
        logger.info(
            f"    - Recommended trials per subject: {max(trials_per_subject, min_trials + 5)} (with margin)"
        )
        logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Warmup Sampling Budget Estimation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python warmup_budget_estimator.py space.csv --subjects 7 --trials 127
  python warmup_budget_estimator.py design.csv -n 10 -t 150
        """,
    )

    parser.add_argument("csv_path", type=str, help="Path to design space CSV file")
    parser.add_argument("-n", "--subjects", type=int, required=True, help="Number of subjects")
    parser.add_argument(
        "-t", "--trials", type=int, required=True, help="Number of trials per subject"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.csv_path).exists():
        logger.error(f"Error: File '{args.csv_path}' does not exist")
        sys.exit(1)

    try:
        # 创建估算器
        estimator = WarmupBudgetEstimator(args.csv_path)

        # 分析设计空间
        estimator.analyze_design_space()

        # 打印预算报告
        estimator.print_budget_report(args.subjects, args.trials)

    except Exception as e:
        logger.exception(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
