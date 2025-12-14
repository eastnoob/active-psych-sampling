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
                f"未在CSV中找到因子列。CSV列名: {list(self.design_df.columns)}"
            )

        self.d = len(self.factor_names)
        self.n_configs = len(self.design_df)

    def analyze_design_space(self):
        """分析设计空间的统计特征"""
        print("=" * 70)
        print("设计空间分析")
        print("=" * 70)
        print(f"总配置数量: {self.n_configs}")
        print(f"因子数量: {self.d}")
        print(f"因子名称: {', '.join(self.factor_names)}")
        print()

        print("各因子分布统计:")
        print("-" * 70)
        for factor in self.factor_names:
            values = self.design_df[factor]
            unique_vals = values.unique()
            print(f"  {factor}:")

            # 检测变量类型
            if values.dtype in ["object", "bool"]:
                # 分类变量或布尔变量
                print(f"    - 类型: 分类变量")
                print(f"    - 唯一值数量: {len(unique_vals)}")
                print(f"    - 取值: {list(unique_vals)}")
            else:
                # 数值变量
                print(f"    - 类型: 数值变量")
                print(f"    - 范围: [{values.min():.3f}, {values.max():.3f}]")
                print(f"    - 唯一值数量: {len(unique_vals)}")
                print(f"    - 均值: {values.mean():.3f}")
                print(f"    - 标准差: {values.std():.3f}")
        print()

    def estimate_budget_requirements(
        self, n_subjects: int, budget_per_subject: int, skip_interaction: bool = False
    ):
        """
        估算预算需求 - 基于五步采样法

        采样策略:
        - Core-1: 8个固定配置，每个被试都测 → 8×N次采样
        - Core-2a: D-optimal配置池，分配给各被试
        - Core-2b: 交互对配置池，分配给各被试（可选）
        - Boundary: 边界极端配置池，分配给各被试
        - LHS: 均匀填充配置池，分配给各被试

        Args:
            n_subjects: 被试数量
            budget_per_subject: 每个被试的测试次数
            skip_interaction: 是否跳过交互效应探索（Core-2b）

        Returns:
            dict: 预算细节
        """
        # 总预算
        total_budget = n_subjects * budget_per_subject

        # Step 1: Core-1固定重复点
        n_core1_configs = 8
        n_core1_samples = n_core1_configs * n_subjects  # 每人都测这8个

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

        # Core-2b最小值：交互项覆盖
        # 交互对数 ≈ C(d,2) = d*(d-1)/2 = 15对（6因子情况）
        # 每对至少2-3个配置 → 15*2 = 30个最少
        n_core2b_min_abs = (
            max(15, int(self.d * (self.d - 1) * 0.7))  # 约15对×0.7个配置 ≈ 10-12
            if not skip_interaction
            else 0
        )

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
                # 比例：Core-2a 40% / Core-2b 30% / Explore 30%
                n_core2a = n_core2a_min_abs + int(surplus * 0.40)
                n_core2b = n_core2b_min_abs + int(surplus * 0.30)
                n_explore = remaining_budget - n_core2a - n_core2b
            else:
                # 没有Core-2b：比例 55% / 45%
                n_core2a = n_core2a_min_abs + int(surplus * 0.55)
                n_core2b = 0
                n_explore = remaining_budget - n_core2a

        # 3. 分配探索预算：边界40%，LHS60%
        n_boundary_min = max(2 * self.d, int(3 * min(1.0, n_subjects / 3.0)))
        n_boundary = max(n_boundary_min, int(n_explore * 0.40))
        n_lhs = max(0, n_explore - n_boundary)

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
        self, n_subjects: int, trials_per_subject: int, skip_interaction: bool = False
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

        Returns:
            tuple: (评估结果, 详细信息dict)
        """
        budget = self.estimate_budget_requirements(
            n_subjects, trials_per_subject, skip_interaction
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
                f"边缘覆盖严重不足：平均每水平仅{avg_unique_samples_per_level:.1f}个独特配置（需要≥2个）"
            )
        elif avg_unique_samples_per_level < 3:
            warnings.append(
                f"边缘覆盖不足：平均每水平{avg_unique_samples_per_level:.1f}个独特配置（建议3-5个）"
            )
        elif avg_unique_samples_per_level <= 5:
            strengths.append(
                f"边缘覆盖刚好：平均每水平{avg_unique_samples_per_level:.1f}个独特配置"
            )
        elif avg_unique_samples_per_level <= 8:
            strengths.append(
                f"边缘覆盖充分：平均每水平{avg_unique_samples_per_level:.1f}个独特配置"
            )
        else:
            excess_warnings.append(
                f"边缘覆盖过度：平均每水平{avg_unique_samples_per_level:.1f}个独特配置（>8个边际收益递减，可减少被试或降低每人trials）"
            )

        # 1.2 独特配置占比
        if unique_ratio < 0.60:
            issues.append(f"独特配置占比过低：{unique_ratio*100:.1f}%（需要≥60%）")
        elif unique_ratio < 0.70:
            warnings.append(f"独特配置占比偏低：{unique_ratio*100:.1f}%（建议70-80%）")
        elif unique_ratio <= 0.85:
            strengths.append(f"独特配置占比合理：{unique_ratio*100:.1f}%")
        else:
            excess_warnings.append(
                f"独特配置占比过高：{unique_ratio*100:.1f}%（重复点可能不足，建议增加Core-1配置数）"
            )

        # === 维度2：信息密度 ===
        # 2.1 ICC估计能力
        if icc_subjects < 3:
            issues.append(f"ICC估计不足：仅{icc_subjects}人（需要≥3人）")
        elif icc_subjects < 5:
            warnings.append(f"ICC估计勉强：{icc_subjects}人（建议5-7人）")
        elif icc_subjects <= 7:
            strengths.append(f"ICC估计能力刚好：{icc_subjects}人")
        elif icc_subjects <= 10:
            strengths.append(f"ICC估计能力充分：{icc_subjects}人")
        else:
            excess_warnings.append(
                f"被试数量过多：{icc_subjects}人（>10人对ICC提升有限，可减少被试数）"
            )

        # 2.2 重复配置数量
        n_repeated_configs = budget["core1_configs"]
        if n_repeated_configs < 5:
            issues.append(f"重复配置不足：仅{n_repeated_configs}个（需要≥5个）")
        elif n_repeated_configs < 6:
            warnings.append(f"重复配置偏少：{n_repeated_configs}个（建议6-10个）")
        elif n_repeated_configs <= 10:
            strengths.append(f"重复配置数量合理：{n_repeated_configs}个")
        else:
            excess_warnings.append(
                f"重复配置过多：{n_repeated_configs}个（>10个占用过多预算，建议降低）"
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
                f"Core-2a不足：仅{budget['core2a_configs']}个配置（需要≥15个）"
            )
        elif budget["core2a_configs"] < 20:
            warnings.append(
                f"Core-2a偏低：{budget['core2a_configs']}个配置（建议≥20个）"
            )
        elif budget["core2a_configs"] < target_core2a * 1.2:
            strengths.append(f"Core-2a合理：{budget['core2a_configs']}个配置")
        elif budget["core2a_configs"] < target_core2a * 1.5:
            warnings.append(f"Core-2a偏高：{budget['core2a_configs']}个配置（可减少）")
        else:
            excess_warnings.append(
                f"Core-2a过度：{budget['core2a_configs']}个配置（预算浪费）"
            )

        # 3.2 Core-2b评估：基于动态阈值
        if not skip_interaction:
            if budget["core2b_configs"] < 10:
                issues.append(
                    f"Core-2b不足：仅{budget['core2b_configs']}个配置（需要≥10个）"
                )
            elif budget["core2b_configs"] < 15:
                warnings.append(
                    f"Core-2b偏低：{budget['core2b_configs']}个配置（建议≥15个）"
                )
            elif budget["core2b_configs"] < target_core2b * 1.2:
                strengths.append(f"Core-2b合理：{budget['core2b_configs']}个配置")
            elif budget["core2b_configs"] < target_core2b * 1.5:
                warnings.append(
                    f"Core-2b偏高：{budget['core2b_configs']}个配置（可减少）"
                )
            else:
                excess_warnings.append(
                    f"Core-2b过度：{budget['core2b_configs']}个配置（预算浪费）"
                )

        # 3.3 探索预算评估：基于动态阈值
        explore_configs = budget["boundary_configs"] + budget["lhs_configs"]
        if explore_configs < 15:
            issues.append(f"探索不足：仅{explore_configs}个配置（需要≥15个）")
        elif explore_configs < 20:
            warnings.append(f"探索偏低：{explore_configs}个配置（建议≥20个）")
        elif explore_configs < target_explore * 1.2:
            strengths.append(f"探索合理：{explore_configs}个配置")
        elif explore_configs < target_explore * 1.5:
            warnings.append(f"探索偏高：{explore_configs}个配置（可减少）")
        else:
            excess_warnings.append(f"探索过度：{explore_configs}个配置（预算浪费）")

        # 3.4 Core-1评估：基于配置数和被试数
        # Core-1固定8个，关键是被试数足够支持ICC估计
        if budget["core1_configs"] < 6:
            issues.append(f"Core-1配置不足：仅{budget['core1_configs']}个（需要≥6个）")
        elif budget["core1_configs"] <= 10:
            strengths.append(f"Core-1配置数合理：{budget['core1_configs']}个")
        else:
            excess_warnings.append(
                f"Core-1配置数过多：{budget['core1_configs']}个（占用{core1_ratio*100:.1f}%）"
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
            overall = "预算不足"
            issues.insert(
                0, f"总预算短缺：需要{total_samples}次，仅有{total_available}次"
            )
        elif n_issues >= 3:
            # 多个关键指标严重不足
            overall = "严重不足"
        elif n_issues >= 1:
            # 【优先级提前】：只要有关键指标不足，就是"不足"
            # 即使同时有过度问题，也优先解决不足
            overall = "不足"
        elif n_excess >= 2:
            # 资源过度（无不足问题的前提下）
            # 降低阈值到2个，只要有2个过度警告就说明资源明显过剩
            overall = "过度充足（可优化）"
        elif n_warnings >= 4:
            # 有较多偏差类警告，但没有过度问题（否则前面已判断）
            overall = "勉强"
        elif n_warnings >= 2:
            # 有少量偏差
            overall = "基本满足"
        elif n_excess == 1:
            # 只有1个过度问题，整体还算合理
            overall = "基本满足"
        elif n_total_problems == 0 and len(strengths) >= 5:
            # 无任何问题，且有5个以上优点
            overall = "充分"
        elif n_total_problems == 0:
            # 无问题，但优点不够多
            overall = "刚好"
        else:
            # 其他情况
            overall = "基本满足"

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
    ):
        """
        打印完整的预算报告 - 基于五步采样法

        Args:
            n_subjects: 被试数量
            trials_per_subject: 每个被试的测试次数
            skip_interaction: 是否跳过交互效应探索（Core-2b）
            show_comparison: 是否显示有/无交互效应的对比
        """
        print("=" * 70)
        print("预热采样预算估算报告（五步采样法）")
        print("=" * 70)
        print()

        # 输入参数
        print("输入参数:")
        print("-" * 70)
        print(f"  被试数量: {n_subjects}")
        print(f"  每人测试次数: {trials_per_subject}")
        print(f"  总可用预算: {n_subjects * trials_per_subject}")
        print()

        # 采样策略详情
        budget = self.estimate_budget_requirements(
            n_subjects, trials_per_subject, skip_interaction
        )

        if show_comparison and not skip_interaction:
            budget_no_inter = self.estimate_budget_requirements(
                n_subjects, trials_per_subject, skip_interaction=True
            )

        print("采样策略及预算分配:")
        print("-" * 70)

        mode_desc = (
            "（已跳过交互效应探索）" if skip_interaction else "（包含交互效应探索）"
        )
        print(f"当前模式: {mode_desc}")
        print()

        print(f"  1. Core-1固定重复点:")
        print(f"     - 配置数: {budget['core1_configs']} 个")
        print(
            f"     - 采样次数: {budget['core1_samples']} 次 ({budget['core1_configs']} × {n_subjects})"
        )
        print(f"     - 说明: {n_subjects}人共享相同的{budget['core1_configs']}个配置")
        print()

        print(f"  2. Core-2a主效应覆盖（配置池）:")
        print(f"     - 配置数: {budget['core2a_configs']} 个")
        print(f"     - 说明: D-optimal设计，分配给各被试")
        print()

        if not skip_interaction:
            print(f"  3. Core-2b交互初筛（配置池）:")
            print(f"     - 配置数: {budget['core2b_configs']} 个")
            print(f"     - 说明: 交互对感知采样（支持free/specified_only/hybrid模式）")
            print()
        else:
            print(f"  3. Core-2b交互初筛: 已跳过")
            print()

        print(f"  4. 边界极端点（配置池）:")
        print(f"     - 配置数: {budget['boundary_configs']} 个")
        print(f"     - 说明: 单维/二维/全局极端，分配给各被试")
        print()

        print(f"  5. 分层LHS点（配置池）:")
        print(f"     - 配置数: {budget['lhs_configs']} 个")
        print(f"     - 说明: 约束LHS + Gower距离匹配，分配给各被试")
        print()

        # 总预算需求
        total_samples = budget["total_samples"]
        total_available = n_subjects * trials_per_subject

        print("预算需求汇总:")
        print("-" * 70)
        print(f"  总采样次数需求: {total_samples}")
        print(f"    - Core-1（共享）: {budget['core1_samples']} 次")
        print(f"    - 配置池（分配）: {budget['pool_configs']} 次")
        print()
        print(f"  总可用采样次数: {total_available}")
        print(f"  差额: {total_available - total_samples:+d} 次")
        print(f"  利用率: {(total_samples/total_available)*100:.1f}%")
        print()

        # 充足性评估
        adequacy, eval_details = self.evaluate_budget_adequacy(
            n_subjects, trials_per_subject, skip_interaction
        )

        print("=" * 70)
        print("充足性评估结果（基于覆盖性标准）")
        print("=" * 70)

        # 显示综合评估
        status_icons = {
            "预算不足": "[X]",
            "严重不足": "[XX]",
            "不足": "[!]",
            "勉强": "[~]",
            "基本满足": "[OK]",
            "刚好": "[OK+]",
            "充分": "[++]",
            "过度充足（可优化）": "[++!]",
        }
        icon = status_icons.get(adequacy, "[?]")
        print(f"{icon} 预算充足性：【{adequacy}】")
        print()

        # 显示关键指标
        metrics = eval_details["metrics"]
        print("关键指标:")
        print("-" * 70)
        print(
            f"  总采样次数: {eval_details['total_samples']} / {eval_details['total_available']}"
        )
        print(f"  独特配置占比: {metrics['unique_ratio']*100:.1f}% （建议70-80%）")
        print(
            f"  平均每水平独特配置数: {metrics['avg_unique_samples_per_level']:.1f}个 （建议3-5个）"
        )
        print(
            f"  平均每水平采样次数: {metrics['avg_samples_per_level_with_repeats']:.1f}次 （含重复）"
        )
        print(f"  ICC估计被试数: {metrics['icc_subjects']}人 （建议5-7人）")
        print(f"  重复配置数: {metrics['n_repeated_configs']}个 （建议6-10个）")
        print()

        print("模块预算比例:")
        print("-" * 70)
        print(f"  Core-1:  {metrics['core1_ratio']*100:5.1f}% （随被试数变化）")
        print(f"  Core-2a: {metrics['core2a_ratio']*100:5.1f}% （建议32-40%）")
        if not skip_interaction:
            print(f"  Core-2b: {metrics['core2b_ratio']*100:5.1f}% （建议22-28%）")
        print(f"  探索:    {metrics['explore_ratio']*100:5.1f}% （建议25-35%）")
        print()

        # 显示问题
        if eval_details["issues"]:
            print("发现的问题（不足）:")
            print("-" * 70)
            for issue in eval_details["issues"]:
                print(f"  [X] {issue}")
            print()

        # 显示警告
        if eval_details["warnings"]:
            print("需要注意（偏差）:")
            print("-" * 70)
            for warning in eval_details["warnings"]:
                print(f"  [!] {warning}")
            print()

        # 显示过度警告
        if eval_details["excess_warnings"]:
            print("预算/被试过多（可优化）:")
            print("-" * 70)
            for excess in eval_details["excess_warnings"]:
                print(f"  [++] {excess}")
            print()

        # 显示优点
        if eval_details["strengths"]:
            print("做得好的方面:")
            print("-" * 70)
            for strength in eval_details["strengths"]:
                print(f"  [+] {strength}")
            print()

        # 对比分析
        if show_comparison and not skip_interaction:
            print("=" * 70)
            print("对比分析：有/无交互效应探索")
            print("=" * 70)
            adequacy_no_inter, eval_no_inter = self.evaluate_budget_adequacy(
                n_subjects, trials_per_subject, skip_interaction=True
            )
            print(f"  包含交互效应（Core-2b）:")
            print(f"    - 充足性: {adequacy}")
            print(f"    - 总采样: {total_samples}次")
            print(f"    - 独立配置: {budget['unique_configs']}个")
            print()
            print(f"  不含交互效应（跳过Core-2b）:")
            print(f"    - 充足性: {adequacy_no_inter}")
            print(f"    - 总采样: {budget_no_inter['total_samples']}次")
            print(f"    - 独立配置: {budget_no_inter['unique_configs']}个")
            print(f"    - 节省: {budget['core2b_configs']}次采样")
            print()

        print("=" * 70)
        print()

        # 覆盖率分析
        unique_configs = budget["unique_configs"]
        coverage_rate = unique_configs / self.n_configs
        print("设计空间覆盖率分析:")
        print("-" * 70)
        print(f"  独立配置总数: {unique_configs}")
        print(f"    - Core-1（共享）: {budget['core1_configs']} 个")
        print(f"    - 配置池: {budget['pool_configs']} 个")
        print(f"  设计空间总数: {self.n_configs}")
        print(f"  覆盖率: {coverage_rate*100:.2f}%")
        print()

        # 推荐被试数量范围
        print("推荐配置:")
        print("-" * 70)
        min_trials = budget["core1_configs"] + int(
            np.ceil(budget["pool_configs"] / n_subjects)
        )
        print(f"  对于 {n_subjects} 名被试:")
        print(f"    - 最少每人测试: {min_trials} 次")
        print(
            f"    - 推荐每人测试: {max(trials_per_subject, min_trials + 5)} 次（留有余地）"
        )
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="预热采样预算估算工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python warmup_budget_estimator.py space.csv --subjects 7 --trials 127
  python warmup_budget_estimator.py design.csv -n 10 -t 150
        """,
    )

    parser.add_argument("csv_path", type=str, help="设计空间CSV文件路径")
    parser.add_argument("-n", "--subjects", type=int, required=True, help="被试数量")
    parser.add_argument(
        "-t", "--trials", type=int, required=True, help="单个被试的测试次数"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not Path(args.csv_path).exists():
        print(f"错误: 文件 '{args.csv_path}' 不存在")
        sys.exit(1)

    try:
        # 创建估算器
        estimator = WarmupBudgetEstimator(args.csv_path)

        # 分析设计空间
        estimator.analyze_design_space()

        # 打印预算报告
        estimator.print_budget_report(args.subjects, args.trials)

    except Exception as e:
        print(f"错误: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
