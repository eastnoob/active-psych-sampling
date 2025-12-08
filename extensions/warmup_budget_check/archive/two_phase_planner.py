"""
两阶段实验规划器
整合 Phase 1（预热） + Phase 2（主动学习）的完整规划
"""

import numpy as np
import json
import pickle
from typing import Dict, Any, List, Tuple
from pathlib import Path

from warmup_budget_estimator import WarmupBudgetEstimator
from phase1_analyzer import analyze_phase1_data


class TwoPhaseExperimentPlanner:
    """
    整合Phase 1（预热）+ Phase 2（主动学习）的完整规划器

    使用流程：
    1. plan_phase1() - 规划Phase 1预算
    2. [执行Phase 1数据收集]
    3. analyze_phase1_data() - 分析Phase 1数据
    4. plan_phase2() - 规划Phase 2参数（自动从Phase 1继承）
    5. export_phase1_output() - 导出Phase 1输出供Phase 2使用
    """

    def __init__(self, design_csv_path: str):
        """
        初始化规划器

        Args:
            design_csv_path: 设计空间CSV文件路径
        """
        self.estimator = WarmupBudgetEstimator(design_csv_path)
        self.design_csv_path = design_csv_path

        # Phase 1数据和分析结果
        self.phase1_data = None
        self.phase1_analysis = None
        self.phase1_plan = None

    def plan_phase1(
        self,
        n_subjects: int,
        trials_per_subject: int,
        skip_interaction: bool = False,
        core2b_mode: str = "broad_coverage",
    ) -> Dict[str, Any]:
        """
        规划Phase 1（预热阶段）

        Args:
            n_subjects: Phase 1被试数量
            trials_per_subject: 每个被试的测试次数
            skip_interaction: 是否跳过交互效应探索
            core2b_mode: Core-2b模式
                - 'broad_coverage': 测试所有交互对，每对少量采样（推荐）
                - 'preset': 预设交互对数量（当前实现）

        Returns:
            Phase 1规划详情
        """
        budget = self.estimator.estimate_budget_requirements(
            n_subjects, trials_per_subject, skip_interaction
        )

        self.phase1_plan = {
            "phase": 1,
            "n_subjects": n_subjects,
            "trials_per_subject": trials_per_subject,
            "total_budget": budget["total_samples"],
            "sampling_design": budget,
            "core2b_mode": core2b_mode,
            "note": "执行五步采样法，收集数据后运行analyze_phase1_data()",
        }

        print("=" * 80)
        print("Phase 1 规划完成")
        print("=" * 80)
        print(f"被试数: {n_subjects}")
        print(f"每人trials: {trials_per_subject}")
        print(f"总预算: {budget['total_samples']}")
        print()
        print("预算分配:")
        print(f"  Core-1 (重复点):  {budget['core1_samples']}")
        print(f"  Core-2a (主效应): {budget['core2a_configs']}")
        print(f"  Core-2b (交互):   {budget['core2b_configs']}")
        print(f"  边界:            {budget['boundary_configs']}")
        print(f"  LHS:             {budget['lhs_configs']}")
        print()
        print("[!] 注意: Phase 1完成后，请运行 analyze_phase1_data() 来分析数据")
        print("=" * 80)
        print()

        return self.phase1_plan

    def analyze_phase1_data(
        self,
        X_warmup: np.ndarray,
        y_warmup: np.ndarray,
        subject_ids: np.ndarray,
        max_pairs: int = 5,
        min_pairs: int = 3,
        selection_method: str = "elbow",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        分析Phase 1数据（关键桥梁函数）

        这是连接Phase 1和Phase 2的核心步骤

        Args:
            X_warmup: Phase 1数据，形状 (n_samples, n_factors)
            y_warmup: Phase 1响应变量，形状 (n_samples,)
            subject_ids: 被试ID，形状 (n_samples,)
            max_pairs: 最多选择的交互对数量
            min_pairs: 最少选择的交互对数量
            selection_method: 交互对选择方法
            verbose: 是否打印详细信息

        Returns:
            Phase 1分析结果
        """
        if self.phase1_plan is None:
            print("警告: 尚未运行plan_phase1()，将使用默认设置")

        # 保存原始数据
        self.phase1_data = {
            "X": X_warmup,
            "y": y_warmup,
            "subject_ids": subject_ids,
        }

        # 运行分析
        self.phase1_analysis = analyze_phase1_data(
            X_warmup,
            y_warmup,
            subject_ids,
            factor_names=self.estimator.factor_names,
            max_pairs=max_pairs,
            min_pairs=min_pairs,
            selection_method=selection_method,
            verbose=verbose,
        )

        # 添加额外信息
        self.phase1_analysis["design_csv_path"] = self.design_csv_path
        self.phase1_analysis["phase1_plan"] = self.phase1_plan

        return self.phase1_analysis

    def plan_phase2(
        self,
        n_subjects: int,
        trials_per_subject: int,
        use_phase1_estimates: bool = True,
        lambda_adjustment: float = 1.2,
    ) -> Dict[str, Any]:
        """
        规划Phase 2（主动学习阶段）

        参数自动从Phase 1分析结果中获取

        Args:
            n_subjects: Phase 2被试数量
            trials_per_subject: 每人测试次数
            use_phase1_estimates: 是否使用Phase 1的估计
            lambda_adjustment: λ调整系数（Phase 2前期应更关注交互）

        Returns:
            Phase 2规划详情
        """
        if self.phase1_analysis is None:
            raise ValueError(
                "必须先运行analyze_phase1_data()才能规划Phase 2！\n"
                "请先收集Phase 1数据并运行分析。"
            )

        total_budget = n_subjects * trials_per_subject

        # 从Phase 1获取初始参数
        if use_phase1_estimates:
            lambda_init = self.phase1_analysis["lambda_init"]
            # 调整：Phase 2前期应该更关注交互（乘以adjustment）
            lambda_init = min(lambda_init * lambda_adjustment, 0.95)
        else:
            lambda_init = 0.5  # 备用默认值

        # γ初始值：根据Phase 1的覆盖情况
        # 如果Phase 1覆盖率低，Phase 2应该更关注探索
        phase1_coverage = self._estimate_phase1_coverage()
        if phase1_coverage < 0.20:
            gamma_init = 0.4  # 覆盖率很低，增加探索
        elif phase1_coverage < 0.30:
            gamma_init = 0.3  # 覆盖率偏低
        else:
            gamma_init = 0.2  # 覆盖率可接受

        phase2_plan = {
            "phase": 2,
            "n_subjects": n_subjects,
            "trials_per_subject": trials_per_subject,
            "total_budget": total_budget,
            # 从Phase 1继承的关键信息
            "interaction_pairs": self.phase1_analysis["selected_pairs"],
            "n_interaction_pairs": len(self.phase1_analysis["selected_pairs"]),
            # EUR-ANOVA参数
            "lambda_init": lambda_init,
            "gamma_init": gamma_init,
            "lambda_schedule": self._compute_lambda_schedule(total_budget, lambda_init),
            "gamma_schedule": self._compute_gamma_schedule(total_budget, gamma_init),
            # 中期诊断配置
            "mid_diagnostic_trial": int(total_budget * 0.67),  # 2/3处进行诊断
            # Phase 1信息
            "phase1_lambda": self.phase1_analysis["lambda_init"],
            "phase1_coverage": phase1_coverage,
        }

        print("=" * 80)
        print("Phase 2 规划完成")
        print("=" * 80)
        print(f"被试数: {n_subjects}")
        print(f"每人trials: {trials_per_subject}")
        print(f"总预算: {total_budget}")
        print()
        print("从Phase 1继承的参数:")
        print(f"  筛选出的交互对: {len(phase2_plan['interaction_pairs'])}个")
        for i, pair in enumerate(phase2_plan["interaction_pairs"]):
            # 安全地获取因子名称
            if hasattr(self.estimator, 'factor_names') and len(self.estimator.factor_names) > max(pair):
                pair_names = (
                    self.estimator.factor_names[pair[0]],
                    self.estimator.factor_names[pair[1]],
                )
            else:
                pair_names = (f"factor_{pair[0]}", f"factor_{pair[1]}")
            print(f"    {i+1}. {pair_names}")
        print(f"  Phase 1 λ估计: {self.phase1_analysis['lambda_init']:.3f}")
        print(f"  Phase 2 λ初始: {lambda_init:.3f} (调整系数={lambda_adjustment})")
        print(f"  Phase 2 γ初始: {gamma_init:.3f}")
        print()
        print("动态调整:")
        print(f"  λ衰减: {lambda_init:.3f} → 0.20 (逐渐降低交互权重)")
        print(f"  γ衰减: {gamma_init:.3f} → {gamma_init*0.2:.3f} (逐渐降低覆盖权重)")
        print()
        print(f"中期诊断点: 第 {phase2_plan['mid_diagnostic_trial']} 次")
        print("=" * 80)
        print()

        return phase2_plan

    def _estimate_phase1_coverage(self) -> float:
        """估算Phase 1的空间覆盖率"""
        if self.phase1_data is None or self.phase1_plan is None:
            return 0.3  # 默认值

        n_unique_configs = self.phase1_plan["sampling_design"]["unique_configs"]
        total_configs = len(self.estimator.design_df)

        coverage = n_unique_configs / total_configs
        return coverage

    def _compute_lambda_schedule(
        self, total_trials: int, lambda_init: float
    ) -> List[Tuple[int, float]]:
        """
        计算λ的动态调整schedule

        策略：从lambda_init开始，逐渐递减
        - 前40%: lambda_init → lambda_init * 0.7
        - 中40%: lambda_init * 0.7 → lambda_init * 0.4
        - 后20%: lambda_init * 0.4 → 0.2
        """
        schedule = []
        n1 = int(total_trials * 0.4)
        n2 = int(total_trials * 0.8)

        for t in range(1, total_trials + 1):
            if t <= n1:
                # 前40%
                progress = t / n1
                lambda_t = lambda_init * (1.0 - 0.3 * progress)
            elif t <= n2:
                # 中40%
                progress = (t - n1) / (n2 - n1)
                lambda_t = lambda_init * (0.7 - 0.3 * progress)
            else:
                # 后20%
                progress = (t - n2) / (total_trials - n2)
                lambda_t = max(lambda_init * 0.4 * (1.0 - 0.5 * progress), 0.2)

            schedule.append((t, lambda_t))

        return schedule

    def _compute_gamma_schedule(
        self, total_trials: int, gamma_init: float
    ) -> List[Tuple[int, float]]:
        """
        计算γ的动态调整schedule

        策略：从gamma_init开始，逐渐递减（降低覆盖权重）
        最终降到 gamma_init * 0.2
        """
        schedule = []
        for t in range(1, total_trials + 1):
            progress = t / total_trials
            gamma_t = gamma_init * (1.0 - 0.8 * progress)  # 最终降到20%
            schedule.append((t, gamma_t))

        return schedule

    def export_phase1_output(
        self, output_dir: str = ".", prefix: str = "phase1_output"
    ) -> Dict[str, str]:
        """
        导出Phase 1的所有输出（供Phase 2使用）

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀

        Returns:
            导出的文件路径字典
        """
        if self.phase1_analysis is None:
            raise ValueError("尚未分析Phase 1数据！请先运行 analyze_phase1_data()")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 1. 导出JSON格式的分析结果（不包含大数组）
        json_path = output_dir / f"{prefix}.json"
        json_output = {
            "selected_pairs": self.phase1_analysis["selected_pairs"],
            "lambda_init": float(self.phase1_analysis["lambda_init"]),
            "main_effects": {
                k: {kk: float(vv) for kk, vv in v.items()}
                for k, v in self.phase1_analysis["main_effects"].items()
            },
            "interaction_effects": {
                str(k): {kk: float(vv) if isinstance(vv, (int, float)) else vv
                         for kk, vv in v.items()}
                for k, v in self.phase1_analysis["interaction_effects"].items()
            },
            "diagnostics": {
                k: v
                for k, v in self.phase1_analysis["diagnostics"].items()
                if k not in ["interaction_scores"]  # 太大，单独保存
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_output, f, indent=2, ensure_ascii=False)

        exported_files["json"] = str(json_path)
        print(f"[OK] JSON输出已保存: {json_path}")

        # 2. 导出完整的Phase 1分析结果（pickle格式）
        pkl_path = output_dir / f"{prefix}_full.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(self.phase1_analysis, f)

        exported_files["pickle"] = str(pkl_path)
        print(f"[OK] 完整分析结果已保存: {pkl_path}")

        # 3. 导出Phase 1数据
        data_path = output_dir / f"{prefix}_data.npz"
        np.savez(
            data_path,
            X=self.phase1_data["X"],
            y=self.phase1_data["y"],
            subject_ids=self.phase1_data["subject_ids"],
        )

        exported_files["data"] = str(data_path)
        print(f"[OK] Phase 1数据已保存: {data_path}")

        # 4. 生成人类可读的报告
        report_path = output_dir / f"{prefix}_report.txt"
        self._generate_text_report(report_path)

        exported_files["report"] = str(report_path)
        print(f"[OK] 文本报告已保存: {report_path}")

        print()
        print("=" * 80)
        print("Phase 1 输出导出完成！")
        print("=" * 80)
        print("导出的文件:")
        for key, path in exported_files.items():
            print(f"  {key}: {path}")
        print()

        return exported_files

    def _generate_text_report(self, report_path: Path):
        """生成文本格式的分析报告"""
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Phase 1 数据分析报告\n")
            f.write("=" * 80 + "\n\n")

            # 基本信息
            f.write("基本信息:\n")
            f.write("-" * 80 + "\n")
            diag = self.phase1_analysis["diagnostics"]
            f.write(f"样本数: {diag['n_samples']}\n")
            f.write(f"被试数: {diag['n_subjects']}\n")
            f.write(f"因子数: {diag['n_factors']}\n")
            f.write(f"筛选方法: {diag['selection_method']}\n\n")

            # 筛选出的交互对
            f.write("筛选出的交互对:\n")
            f.write("-" * 80 + "\n")
            for i, pair in enumerate(self.phase1_analysis["selected_pairs"]):
                # 安全地获取因子名称
                if hasattr(self.estimator, 'factor_names') and len(self.estimator.factor_names) > max(pair):
                    pair_names = (
                        self.estimator.factor_names[pair[0]],
                        self.estimator.factor_names[pair[1]],
                    )
                else:
                    pair_names = (f"factor_{pair[0]}", f"factor_{pair[1]}")
                score = diag["interaction_scores"].get(pair, 0.0)
                f.write(f"{i+1}. {pair_names}: score={score:.3f}\n")
            f.write("\n")

            # λ估计
            f.write("λ参数估计:\n")
            f.write("-" * 80 + "\n")
            f.write(f"λ初始值: {self.phase1_analysis['lambda_init']:.3f}\n")
            var_decomp = diag["var_decomposition"]
            f.write(f"主效应方差: {var_decomp.get('var_main', 0):.4f}\n")
            f.write(f"交互效应方差: {var_decomp.get('var_interaction', 0):.4f}\n")
            f.write("\n")

            # 主效应
            f.write("主效应估计:\n")
            f.write("-" * 80 + "\n")
            for factor, effect in self.phase1_analysis["main_effects"].items():
                f.write(f"{factor}:\n")
                f.write(f"  系数: {effect['coef']:.4f}\n")
                f.write(f"  截距: {effect['intercept']:.4f}\n")
            f.write("\n")

            # 交互效应
            f.write("交互效应估计:\n")
            f.write("-" * 80 + "\n")
            for pair, effect in self.phase1_analysis["interaction_effects"].items():
                f.write(f"{effect['pair_name']}:\n")
                f.write(f"  交互系数: {effect['coef_interaction']:.4f}\n")
                f.write(f"  因子i系数: {effect['coef_i']:.4f}\n")
                f.write(f"  因子j系数: {effect['coef_j']:.4f}\n")
            f.write("\n")

    @staticmethod
    def load_phase1_output(output_path: str) -> Dict[str, Any]:
        """
        加载Phase 1输出

        Args:
            output_path: phase1_output_full.pkl的路径

        Returns:
            Phase 1分析结果
        """
        with open(output_path, "rb") as f:
            phase1_analysis = pickle.load(f)

        print("[OK] Phase 1输出已加载")
        print(f"  筛选出的交互对: {len(phase1_analysis['selected_pairs'])}个")
        print(f"  λ估计: {phase1_analysis['lambda_init']:.3f}")

        return phase1_analysis
