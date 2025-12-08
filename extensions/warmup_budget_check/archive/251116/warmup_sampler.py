"""
预热阶段采样规划器
根据设计空间CSV和预算参数，生成预热阶段的采样方案

使用流程：
1. 准备全因子设计CSV（只包含自变量列）
2. 运行本脚本，指定被试数和每人trials数
3. 系统评估预算充足性
4. 确认后生成采样文件（保存到sample/文件夹）
5. 按照采样文件执行实验
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import sys

from warmup_budget_estimator import WarmupBudgetEstimator


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

    def _generate_five_step_samples(self, budget: dict, n_subjects: int):
        """
        生成五步采样方案（简化版）

        注意：这里使用随机采样作为示例
        实际应用中应该使用D-optimal、LHS等专业采样方法
        """
        n_configs = len(self.design_df)
        all_samples = []

        # Core-1: 8个固定配置，每个被试都测
        core1_indices = np.random.choice(n_configs, size=budget['core1_configs'], replace=False)
        core1_configs = self.design_df.iloc[core1_indices]

        # 为每个被试生成样本
        for subject_id in range(1, n_subjects + 1):
            subject_samples = []

            # 1. Core-1（所有被试共享）
            core1_df = core1_configs.copy()
            core1_df['subject_id'] = subject_id
            subject_samples.append(core1_df)

            # 2. 配置池（Core-2a + Core-2b + Boundary + LHS）
            pool_size = budget['pool_configs'] // n_subjects  # 平均分配

            # 从剩余配置中随机采样
            remaining_indices = np.setdiff1d(np.arange(n_configs), core1_indices)
            pool_indices = np.random.choice(
                remaining_indices,
                size=min(pool_size, len(remaining_indices)),
                replace=False
            )
            pool_df = self.design_df.iloc[pool_indices].copy()
            pool_df['subject_id'] = subject_id
            subject_samples.append(pool_df)

            # 合并该被试的所有样本
            subject_df = pd.concat(subject_samples, ignore_index=True)

            # 打乱顺序（避免顺序效应）
            subject_df = subject_df.sample(frac=1, random_state=42 + subject_id).reset_index(drop=True)

            all_samples.append(subject_df)

        return all_samples

    def _generate_readme(self, readme_path: Path, budget: dict, n_subjects: int, merged: bool):
        """生成采样说明文档"""
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("预热阶段采样说明\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. 实验设计\n")
            f.write("-" * 80 + "\n")
            f.write(f"设计空间: {self.design_csv_path}\n")
            f.write(f"被试数量: {n_subjects}人\n")
            f.write(f"每人trials: {budget['samples_per_subject']}次\n")
            f.write(f"总样本数: {budget['total_samples']}次\n\n")

            f.write("2. 采样策略（五步采样法）\n")
            f.write("-" * 80 + "\n")
            f.write(f"Core-1 (重复点):  {budget['core1_samples']}次\n")
            f.write(f"Core-2a (主效应): {budget['core2a_configs']}次\n")
            f.write(f"Core-2b (交互):   {budget['core2b_configs']}次\n")
            f.write(f"边界点:          {budget['boundary_configs']}次\n")
            f.write(f"LHS填充:         {budget['lhs_configs']}次\n\n")

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

            f.write("4. 完成实验后\n")
            f.write("-" * 80 + "\n")
            f.write("- 确保所有数据已收集完整\n")
            f.write("- 合并为单个CSV（如果使用分文件模式）\n")
            f.write("- 运行: python analyze_phase1.py\n")
            f.write("- 按提示指定数据文件路径和列名\n\n")


def main():
    """交互式主流程"""
    print()
    print("=" * 80)
    print("预热阶段采样规划器")
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
