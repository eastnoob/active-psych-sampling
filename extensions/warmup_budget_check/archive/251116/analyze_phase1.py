"""
Phase 1数据分析脚本（独立使用）
在预热实验完成后，分析收集的数据并生成Phase 2参数

使用流程：
1. 完成预热阶段实验，收集响应数据
2. 将响应值添加到采样CSV中（或准备包含因变量的CSV）
3. 运行本脚本，指定数据文件路径
4. 告知subject_id和响应列名称
5. 系统分析数据，输出Phase 2参数报告

输出内容：
- 筛选出的交互对（用于EUR-ANOVA）
- λ初始值（交互权重参数）
- γ初始值（覆盖权重参数）
- 主效应和交互效应估计
- 详细的文本报告
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Optional, Dict, Any
import json

from phase1_analyzer import analyze_phase1_data


class Phase1DataAnalyzer:
    """Phase 1数据分析器（独立工具）"""

    def __init__(
        self,
        data_csv_path: str,
        subject_col: str = "subject_id",
        response_col: str = "response",
    ):
        """
        初始化分析器

        Args:
            data_csv_path: 实验数据CSV路径（包含自变量和因变量）
            subject_col: 被试编号列名
            response_col: 响应变量列名
        """
        self.data_csv_path = data_csv_path
        self.subject_col = subject_col
        self.response_col = response_col

        # 加载数据
        print(f"[加载] 实验数据: {data_csv_path}")
        self.df = pd.read_csv(data_csv_path)

        # 验证列存在
        if subject_col not in self.df.columns:
            raise ValueError(f"未找到被试列: '{subject_col}'")
        if response_col not in self.df.columns:
            raise ValueError(f"未找到响应列: '{response_col}'")

        # 提取数据
        self.subject_ids = self.df[subject_col].values
        self.y_warmup = self.df[response_col].values

        # 提取因子列（排除subject_id和response）
        self.factor_cols = [
            col for col in self.df.columns
            if col not in [subject_col, response_col]
        ]
        self.X_warmup = self.df[self.factor_cols].values

        print(f"  样本数: {len(self.df)}")
        print(f"  被试数: {len(np.unique(self.subject_ids))}")
        print(f"  因子数: {len(self.factor_cols)}")
        print(f"  因子名称: {', '.join(self.factor_cols)}")
        print()

    def analyze(
        self,
        max_pairs: int = 5,
        min_pairs: int = 3,
        selection_method: str = "elbow",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        分析Phase 1数据

        Args:
            max_pairs: 最多选择的交互对数量
            min_pairs: 最少选择的交互对数量
            selection_method: 选择方法 ('elbow', 'bic_threshold', 'top_k')
            verbose: 是否显示详细输出

        Returns:
            分析结果字典
        """
        print("=" * 80)
        print("Phase 1数据分析")
        print("=" * 80)
        print()

        # 调用核心分析函数
        analysis = analyze_phase1_data(
            X_warmup=self.X_warmup,
            y_warmup=self.y_warmup,
            subject_ids=self.subject_ids,
            factor_names=self.factor_cols,
            max_pairs=max_pairs,
            min_pairs=min_pairs,
            selection_method=selection_method,
            verbose=verbose,
        )

        # 存储结果
        self.analysis = analysis
        return analysis

    def generate_phase2_config(
        self,
        n_subjects: int,
        trials_per_subject: int,
        lambda_adjustment: float = 1.2,
    ) -> Dict[str, Any]:
        """
        生成Phase 2配置参数

        Args:
            n_subjects: Phase 2被试数
            trials_per_subject: 每个被试的测试次数
            lambda_adjustment: λ调整系数（相对于Phase 1估计）

        Returns:
            Phase 2配置字典
        """
        if not hasattr(self, 'analysis'):
            raise RuntimeError("请先运行analyze()方法")

        total_budget = n_subjects * trials_per_subject

        # 计算λ初始值
        lambda_phase1 = self.analysis['lambda_init']
        lambda_init = min(lambda_phase1 * lambda_adjustment, 1.0)

        # 计算γ初始值（基于预算）
        # 前期高γ（探索），后期低γ（精化）
        gamma_init = 0.3  # 默认初始γ

        # 计算λ衰减终点
        lambda_end = 0.2  # Phase 2后期降到0.2

        # 计算γ衰减终点
        gamma_end = 0.06  # Phase 2后期降到0.06

        # 中期诊断位置（2/3处）
        mid_diagnostic_trial = int(total_budget * 0.67)

        config = {
            "n_subjects": n_subjects,
            "trials_per_subject": trials_per_subject,
            "total_budget": total_budget,

            # 交互对（用于EUR-ANOVA）
            "interaction_pairs": self.analysis['selected_pairs'],
            "n_interaction_pairs": len(self.analysis['selected_pairs']),

            # λ参数（交互权重）
            "lambda_phase1": lambda_phase1,
            "lambda_adjustment": lambda_adjustment,
            "lambda_init": lambda_init,
            "lambda_end": lambda_end,

            # γ参数（覆盖权重）
            "gamma_init": gamma_init,
            "gamma_end": gamma_end,

            # 动态调度
            "lambda_schedule": self._compute_lambda_schedule(
                total_budget, lambda_init, lambda_end
            ),
            "gamma_schedule": self._compute_gamma_schedule(
                total_budget, gamma_init, gamma_end
            ),

            # 诊断位置
            "mid_diagnostic_trial": mid_diagnostic_trial,

            # Phase 1传递的信息
            "from_phase1": {
                "main_effects": self.analysis.get('main_effects', {}),
                "interaction_effects": self.analysis.get('interaction_effects', {}),
                "variance_decomposition": self.analysis.get('variance_decomposition', {}),
            }
        }

        return config

    def _compute_lambda_schedule(
        self, total_budget: int, lambda_init: float, lambda_end: float
    ):
        """计算λ动态调度（指数衰减）"""
        trials = np.arange(1, total_budget + 1)
        decay_rate = np.log(lambda_end / lambda_init) / total_budget
        lambda_values = lambda_init * np.exp(decay_rate * (trials - 1))
        return [(int(t), float(lam)) for t, lam in zip(trials, lambda_values)]

    def _compute_gamma_schedule(
        self, total_budget: int, gamma_init: float, gamma_end: float
    ):
        """计算γ动态调度（指数衰减）"""
        trials = np.arange(1, total_budget + 1)
        decay_rate = np.log(gamma_end / gamma_init) / total_budget
        gamma_values = gamma_init * np.exp(decay_rate * (trials - 1))
        return [(int(t), float(gam)) for t, gam in zip(trials, gamma_values)]

    def export_report(
        self,
        phase2_config: Dict[str, Any],
        output_dir: str = "phase1_analysis_output",
        prefix: str = "phase1",
    ) -> Dict[str, str]:
        """
        导出分析报告和配置文件

        Args:
            phase2_config: Phase 2配置
            output_dir: 输出目录
            prefix: 文件名前缀

        Returns:
            导出的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 1. JSON配置文件（供程序读取）
        json_path = output_path / f"{prefix}_phase2_config.json"
        json_data = {
            "interaction_pairs": phase2_config["interaction_pairs"],
            "lambda_init": phase2_config["lambda_init"],
            "lambda_end": phase2_config["lambda_end"],
            "gamma_init": phase2_config["gamma_init"],
            "gamma_end": phase2_config["gamma_end"],
            "total_budget": phase2_config["total_budget"],
            "mid_diagnostic_trial": phase2_config["mid_diagnostic_trial"],
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        exported_files['json_config'] = str(json_path)

        # 2. NumPy配置文件（供程序读取）
        npz_path = output_path / f"{prefix}_phase2_schedules.npz"
        np.savez(
            npz_path,
            lambda_schedule=np.array(phase2_config["lambda_schedule"]),
            gamma_schedule=np.array(phase2_config["gamma_schedule"]),
            interaction_pairs=np.array(phase2_config["interaction_pairs"]),
        )
        exported_files['npz_schedules'] = str(npz_path)

        # 3. 人类可读报告
        report_path = output_path / f"{prefix}_analysis_report.txt"
        self._write_text_report(report_path, phase2_config)
        exported_files['txt_report'] = str(report_path)

        # 4. Phase 2使用指南
        guide_path = output_path / f"PHASE2_USAGE_GUIDE.txt"
        self._write_usage_guide(guide_path, phase2_config)
        exported_files['usage_guide'] = str(guide_path)

        print()
        print("=" * 80)
        print("导出完成")
        print("=" * 80)
        print()
        print("生成的文件:")
        for key, path in exported_files.items():
            print(f"  {key:15s}: {path}")
        print()

        return exported_files

    def _write_text_report(self, path: Path, phase2_config: Dict[str, Any]):
        """生成人类可读的文本报告"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Phase 1 数据分析报告\n")
            f.write("=" * 80 + "\n\n")

            # 数据概览
            f.write("1. 数据概览\n")
            f.write("-" * 80 + "\n")
            f.write(f"数据文件: {self.data_csv_path}\n")
            f.write(f"样本总数: {len(self.df)}\n")
            f.write(f"被试数量: {len(np.unique(self.subject_ids))}\n")
            f.write(f"因子数量: {len(self.factor_cols)}\n")
            f.write(f"因子名称: {', '.join(self.factor_cols)}\n\n")

            # 交互对筛选结果
            f.write("2. 筛选出的交互对（用于Phase 2）\n")
            f.write("-" * 80 + "\n")
            f.write(f"数量: {len(self.analysis['selected_pairs'])}个\n\n")

            # 获取interaction_scores
            interaction_scores = self.analysis.get('diagnostics', {}).get('interaction_scores', {})

            for rank, pair in enumerate(self.analysis['selected_pairs'], 1):
                score = interaction_scores.get(pair, 0.0)
                factor1 = self.factor_cols[pair[0]]
                factor2 = self.factor_cols[pair[1]]
                f.write(f"  {rank}. ({factor1}, {factor2})\n")
                f.write(f"     索引: ({pair[0]}, {pair[1]})\n")
                f.write(f"     综合评分: {score:.3f}\n\n")

            # λ估计
            f.write("3. 交互权重参数（λ）\n")
            f.write("-" * 80 + "\n")
            f.write(f"Phase 1估计: {self.analysis['lambda_init']:.3f}\n")
            f.write(f"调整系数: {phase2_config['lambda_adjustment']:.2f}\n")
            f.write(f"Phase 2初始: {phase2_config['lambda_init']:.3f}\n")
            f.write(f"Phase 2终点: {phase2_config['lambda_end']:.3f}\n\n")

            # 方差分解
            diagnostics = self.analysis.get('diagnostics', {})
            var_decomp = diagnostics.get('var_decomposition', {})
            if var_decomp:
                f.write("方差分解:\n")
                f.write(f"  主效应方差: {var_decomp.get('main_variance', 0):.4f}\n")
                f.write(f"  交互方差: {var_decomp.get('interaction_variance', 0):.4f}\n")
                f.write(f"  残差方差: {var_decomp.get('residual_variance', 0):.4f}\n\n")

            # γ参数
            f.write("4. 覆盖权重参数（γ）\n")
            f.write("-" * 80 + "\n")
            f.write(f"Phase 2初始: {phase2_config['gamma_init']:.3f}\n")
            f.write(f"Phase 2终点: {phase2_config['gamma_end']:.3f}\n\n")

            # Phase 2配置
            f.write("5. Phase 2配置\n")
            f.write("-" * 80 + "\n")
            f.write(f"被试数: {phase2_config['n_subjects']}人\n")
            f.write(f"每人trials: {phase2_config['trials_per_subject']}次\n")
            f.write(f"总预算: {phase2_config['total_budget']}次\n")
            f.write(f"中期诊断位置: 第{phase2_config['mid_diagnostic_trial']}次trial\n\n")

            # 主效应估计
            f.write("6. 主效应估计\n")
            f.write("-" * 80 + "\n")
            main_effects = self.analysis.get('main_effects', {})
            if main_effects:
                for factor, effect_info in main_effects.items():
                    if isinstance(effect_info, dict):
                        coef = effect_info.get('coef', 0)
                        f.write(f"  {factor}: {coef:.4f}\n")
                    else:
                        f.write(f"  {factor}: {effect_info:.4f}\n")
            else:
                f.write("  （未估计）\n")
            f.write("\n")

            # 交互效应估计
            f.write("7. 交互效应估计\n")
            f.write("-" * 80 + "\n")
            interaction_effects = self.analysis.get('interaction_effects', {})
            if interaction_effects:
                for pair, effect_info in interaction_effects.items():
                    if isinstance(effect_info, dict):
                        pair_name = effect_info.get('pair_name', str(pair))
                        coef_int = effect_info.get('coef_interaction', 0)
                        f.write(f"  {pair_name}: {coef_int:.4f}\n")
                    else:
                        f.write(f"  {pair}: {effect_info:.4f}\n")
            else:
                f.write("  （未估计）\n")
            f.write("\n")

            # 使用说明
            f.write("8. 下一步\n")
            f.write("-" * 80 + "\n")
            f.write("1. 查看 PHASE2_USAGE_GUIDE.txt 了解如何使用这些参数\n")
            f.write("2. 在EUR-ANOVA中使用筛选出的交互对\n")
            f.write("3. 使用λ和γ动态调度表\n")
            f.write(f"4. 在第{phase2_config['mid_diagnostic_trial']}次trial进行中期诊断\n\n")

    def _write_usage_guide(self, path: Path, phase2_config: Dict[str, Any]):
        """生成Phase 2使用指南"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Phase 2 参数使用指南\n")
            f.write("=" * 80 + "\n\n")

            f.write("本指南说明如何在EUR-ANOVA主动学习中使用Phase 1分析结果\n\n")

            # 加载配置
            f.write("1. 加载Phase 2配置\n")
            f.write("-" * 80 + "\n")
            f.write("```python\n")
            f.write("import numpy as np\n")
            f.write("import json\n\n")
            f.write("# 加载JSON配置\n")
            f.write("with open('phase1_phase2_config.json') as f:\n")
            f.write("    config = json.load(f)\n\n")
            f.write("# 加载动态调度\n")
            f.write("data = np.load('phase1_phase2_schedules.npz')\n")
            f.write("lambda_schedule = data['lambda_schedule']  # (n_trials, 2)\n")
            f.write("gamma_schedule = data['gamma_schedule']    # (n_trials, 2)\n")
            f.write("interaction_pairs = data['interaction_pairs'].tolist()\n")
            f.write("```\n\n")

            # 初始化EUR-ANOVA
            f.write("2. 初始化EUR-ANOVA采集函数\n")
            f.write("-" * 80 + "\n")
            f.write("```python\n")
            f.write("from eur_anova_pair import EURAnovaPairAcqf\n\n")
            f.write("# 交互对列表（从Phase 1筛选）\n")
            f.write(f"interaction_pairs = {phase2_config['interaction_pairs']}\n\n")
            f.write("# 初始化采集函数\n")
            f.write("acqf = EURAnovaPairAcqf(\n")
            f.write("    model=your_gp_model,\n")
            f.write(f"    gamma={phase2_config['gamma_init']:.3f},  # γ初始值\n")
            f.write("    lambda_min=0.1,\n")
            f.write("    lambda_max=1.0,\n")
            f.write("    interaction_pairs=interaction_pairs,  # 从Phase 1\n")
            f.write("    tau1=0.7,\n")
            f.write("    tau2=0.3,\n")
            f.write(f"    tau_n_max={phase2_config['total_budget']},\n")
            f.write(f"    gamma_min={phase2_config['gamma_end']:.3f}\n")
            f.write(")\n")
            f.write("```\n\n")

            # 主循环
            f.write("3. Phase 2主动学习循环\n")
            f.write("-" * 80 + "\n")
            f.write("```python\n")
            f.write(f"for trial in range(1, {phase2_config['total_budget']} + 1):\n")
            f.write("    # 获取当前λ和γ\n")
            f.write("    lambda_t = lambda_schedule[trial - 1, 1]\n")
            f.write("    gamma_t = gamma_schedule[trial - 1, 1]\n\n")
            f.write("    # 更新采集函数参数（如果需要）\n")
            f.write("    # acqf.gamma = gamma_t\n\n")
            f.write("    # EUR-ANOVA采集\n")
            f.write("    scores = acqf(X_candidates)\n")
            f.write("    next_idx = scores.argmax()\n")
            f.write("    X_next = X_candidates[next_idx]\n\n")
            f.write("    # 执行实验\n")
            f.write("    y_next = conduct_experiment(X_next)\n\n")
            f.write("    # 更新GP模型\n")
            f.write("    your_gp_model.update(X_next, y_next)\n\n")
            f.write("    # 中期诊断\n")
            f.write(f"    if trial == {phase2_config['mid_diagnostic_trial']}:\n")
            f.write("        run_mid_phase_diagnostic()\n")
            f.write("```\n\n")

            # 参数说明
            f.write("4. 参数说明\n")
            f.write("-" * 80 + "\n")
            f.write(f"interaction_pairs: {len(phase2_config['interaction_pairs'])}个交互对\n")
            f.write(f"  {phase2_config['interaction_pairs']}\n\n")
            f.write(f"lambda_init: {phase2_config['lambda_init']:.3f}\n")
            f.write("  控制交互效应探索的权重\n")
            f.write("  λ越大，越关注交互；λ越小，越关注主效应\n\n")
            f.write(f"gamma_init: {phase2_config['gamma_init']:.3f}\n")
            f.write("  控制覆盖性（exploration）的权重\n")
            f.write("  γ越大，越关注未探索区域；γ越小，越关注精化已知区域\n\n")
            f.write(f"衰减策略: 指数衰减\n")
            f.write(f"  λ: {phase2_config['lambda_init']:.3f} → {phase2_config['lambda_end']:.3f}\n")
            f.write(f"  γ: {phase2_config['gamma_init']:.3f} → {phase2_config['gamma_end']:.3f}\n\n")

            # 注意事项
            f.write("5. 注意事项\n")
            f.write("-" * 80 + "\n")
            f.write("- 交互对索引基于因子顺序，请确保与设计空间一致\n")
            f.write("- λ和γ调度已预先计算，可直接使用或根据需要调整\n")
            f.write("- 中期诊断用于检查模型拟合质量和参数合理性\n")
            f.write("- 如发现Phase 2前期交互探索不足，可手动增大λ_adjustment\n\n")


def main():
    """交互式主流程"""
    print()
    print("=" * 80)
    print("Phase 1 数据分析工具")
    print("=" * 80)
    print()

    # Step 1: 输入数据文件路径
    data_csv = input("请输入实验数据CSV路径（或按Enter使用默认 'warmup_data.csv'）: ").strip()
    if not data_csv:
        data_csv = "warmup_data.csv"

    if not Path(data_csv).exists():
        print(f"[错误] 文件不存在: {data_csv}")
        print()
        print("提示: 请确保CSV包含以下列:")
        print("  - 被试编号列（默认: subject_id）")
        print("  - 响应变量列（默认: response）")
        print("  - 所有因子列")
        sys.exit(1)

    # Step 2: 输入列名
    print()
    print("请指定列名:")
    subject_col = input("  被试编号列名（默认 'subject_id'）: ").strip() or "subject_id"
    response_col = input("  响应变量列名（默认 'response'）: ").strip() or "response"
    print()

    # Step 3: 加载数据
    try:
        analyzer = Phase1DataAnalyzer(
            data_csv_path=data_csv,
            subject_col=subject_col,
            response_col=response_col,
        )
    except Exception as e:
        print(f"[错误] 加载数据失败: {e}")
        sys.exit(1)

    # Step 4: 配置分析参数
    print("请配置分析参数:")
    try:
        max_pairs = int(input("  最多选择交互对数量（默认 5）: ").strip() or "5")
        min_pairs = int(input("  最少选择交互对数量（默认 3）: ").strip() or "3")
    except ValueError:
        print("[错误] 输入必须是整数")
        sys.exit(1)

    selection_method = input("  选择方法 (elbow/bic_threshold/top_k，默认 elbow): ").strip() or "elbow"
    print()

    # Step 5: 执行分析
    try:
        analysis = analyzer.analyze(
            max_pairs=max_pairs,
            min_pairs=min_pairs,
            selection_method=selection_method,
            verbose=True,
        )
    except Exception as e:
        print(f"[错误] 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 6: 配置Phase 2
    print()
    print("=" * 80)
    print("Phase 2配置")
    print("=" * 80)
    print()
    print("请输入Phase 2参数:")
    try:
        n_subjects = int(input("  被试数量: "))
        trials_per_subject = int(input("  每个被试的测试次数: "))
        lambda_adjustment_str = input("  λ调整系数（默认 1.2）: ").strip()
        lambda_adjustment = float(lambda_adjustment_str) if lambda_adjustment_str else 1.2
    except ValueError:
        print("[错误] 输入格式错误")
        sys.exit(1)

    print()

    # Step 7: 生成Phase 2配置
    try:
        phase2_config = analyzer.generate_phase2_config(
            n_subjects=n_subjects,
            trials_per_subject=trials_per_subject,
            lambda_adjustment=lambda_adjustment,
        )

        print("=" * 80)
        print("Phase 2配置生成完成")
        print("=" * 80)
        print()
        print(f"总预算: {phase2_config['total_budget']}次")
        print(f"筛选的交互对: {len(phase2_config['interaction_pairs'])}个")
        print(f"λ初始: {phase2_config['lambda_init']:.3f} (Phase 1: {phase2_config['lambda_phase1']:.3f})")
        print(f"λ终点: {phase2_config['lambda_end']:.3f}")
        print(f"γ初始: {phase2_config['gamma_init']:.3f}")
        print(f"γ终点: {phase2_config['gamma_end']:.3f}")
        print(f"中期诊断: 第{phase2_config['mid_diagnostic_trial']}次trial")
        print()

    except Exception as e:
        print(f"[错误] 生成Phase 2配置失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 8: 导出报告
    output_dir = input("输出目录（默认 'phase1_analysis_output'）: ").strip() or "phase1_analysis_output"
    prefix = input("文件名前缀（默认 'phase1'）: ").strip() or "phase1"
    print()

    try:
        exported_files = analyzer.export_report(
            phase2_config=phase2_config,
            output_dir=output_dir,
            prefix=prefix,
        )

        print("=" * 80)
        print("分析完成！")
        print("=" * 80)
        print()
        print("下一步:")
        print("1. 查看分析报告: " + exported_files['txt_report'])
        print("2. 阅读使用指南: " + exported_files['usage_guide'])
        print("3. 在Phase 2中加载配置文件:")
        print(f"   - JSON: {exported_files['json_config']}")
        print(f"   - NumPy: {exported_files['npz_schedules']}")
        print()

    except Exception as e:
        print(f"[错误] 导出报告失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
