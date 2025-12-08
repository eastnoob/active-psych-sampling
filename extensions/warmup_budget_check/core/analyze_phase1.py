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
from typing import Optional, Dict, Any, List
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
            data_csv_path: 实验数据CSV路径或目录路径
                         - 如果是文件: 直接读取（需包含subject_col和response_col）
                         - 如果是目录: 读取所有subject_*.csv，每个文件代表一个被试
            subject_col: 被试编号列名
            response_col: 响应变量列名
        """
        self.data_csv_path = data_csv_path
        self.subject_col = subject_col
        self.response_col = response_col

        # 检查是文件还是目录
        data_path = Path(data_csv_path)

        if data_path.is_dir():
            # 目录模式：读取所有 subject_*.csv
            print(f"[加载] 从目录读取被试数据: {data_csv_path}")
            subject_csvs = sorted(data_path.glob("subject_*.csv"))

            if not subject_csvs:
                raise FileNotFoundError(f"目录中未找到 subject_*.csv 文件: {data_csv_path}")

            print(f"  找到 {len(subject_csvs)} 个被试文件")

            # 读取每个被试文件并添加subject列
            all_dfs = []
            for i, csv_path in enumerate(subject_csvs, start=1):
                df_subject = pd.read_csv(csv_path)

                # 验证响应列存在
                if response_col not in df_subject.columns:
                    raise ValueError(f"文件 {csv_path.name} 中未找到响应列: '{response_col}'")

                # 添加被试列（如果不存在）
                if subject_col not in df_subject.columns:
                    # 从文件名提取被试编号 (subject_1.csv -> subject_1)
                    subject_id = csv_path.stem  # "subject_1"
                    df_subject.insert(0, subject_col, subject_id)

                all_dfs.append(df_subject)
                print(f"    - {csv_path.name}: {len(df_subject)} 行")

            # 合并所有数据
            self.df = pd.concat(all_dfs, ignore_index=True)
            print(f"  合并后总计: {len(self.df)} 行")

        else:
            # 文件模式：直接读取
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
            col for col in self.df.columns if col not in [subject_col, response_col]
        ]

        # 编码分类变量和布尔变量
        df_encoded = self.df[self.factor_cols].copy()
        for col in df_encoded.columns:
            # 检查是否为数值类型
            if df_encoded[col].dtype == "object":
                # 分类变量：使用Label Encoding
                unique_vals = df_encoded[col].unique()
                encode_dict = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                df_encoded[col] = df_encoded[col].map(encode_dict)
            elif df_encoded[col].dtype == "bool":
                # 布尔变量：转换为0/1
                df_encoded[col] = df_encoded[col].astype(int)

        self.X_warmup = df_encoded.values

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
        if not hasattr(self, "analysis"):
            raise RuntimeError("请先运行analyze()方法")

        total_budget = n_subjects * trials_per_subject

        # 计算λ_max（Phase 2的目标上限）
        lambda_max_phase1 = self.analysis["lambda_init"]  # 实际上是lambda_max
        lambda_max = min(lambda_max_phase1 * lambda_adjustment, 0.9)

        # 计算γ初始值（基于预算）
        # 前期高γ（探索），后期低γ（精化）
        gamma_init = 0.3  # 默认初始γ

        # Phase 2的λ起点和终点
        lambda_start = 0.1  # Phase 2初期从0.1开始（稳固主效应）
        lambda_end = lambda_max  # Phase 2后期达到lambda_max（探索交互）

        # 计算γ衰减终点
        gamma_end = 0.06  # Phase 2后期降到0.06

        # 中期诊断位置（2/3处）
        mid_diagnostic_trial = int(total_budget * 0.67)

        config = {
            "n_subjects": n_subjects,
            "trials_per_subject": trials_per_subject,
            "total_budget": total_budget,
            # 交互对（用于EUR-ANOVA）
            "interaction_pairs": self.analysis["selected_pairs"],
            "n_interaction_pairs": len(self.analysis["selected_pairs"]),
            # λ参数（交互权重）- 新语义：从低到高探索
            "lambda_max_phase1": lambda_max_phase1,  # Phase 1估计的上限
            "lambda_adjustment": lambda_adjustment,
            "lambda_start": lambda_start,  # Phase 2起点（稳固主效应）
            "lambda_max": lambda_max,  # Phase 2终点（探索交互）
            "lambda_init": lambda_start,  # 兼容旧代码
            "lambda_end": lambda_end,  # 兼容旧代码
            # γ参数（覆盖权重）
            "gamma_init": gamma_init,
            "gamma_end": gamma_end,
            # 动态调度（从低到高增长）
            "lambda_schedule": self._compute_lambda_schedule(
                total_budget, lambda_start, lambda_end
            ),
            "gamma_schedule": self._compute_gamma_schedule(
                total_budget, gamma_init, gamma_end
            ),
            # 诊断位置
            "mid_diagnostic_trial": mid_diagnostic_trial,
            # Phase 1传递的信息
            "from_phase1": {
                "main_effects": self.analysis.get("main_effects", {}),
                "interaction_effects": self.analysis.get("interaction_effects", {}),
                "variance_decomposition": self.analysis.get(
                    "variance_decomposition", {}
                ),
            },
        }

        return config

    def _compute_lambda_schedule(
        self, total_budget: int, lambda_start: float, lambda_end: float
    ):
        """计算λ动态调度（指数增长：从低到高）"""
        trials = np.arange(1, total_budget + 1)
        # 从 lambda_start 增长到 lambda_end
        growth_rate = np.log(lambda_end / lambda_start) / total_budget
        lambda_values = lambda_start * np.exp(growth_rate * (trials - 1))
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
        report_format: str = "md",
    ) -> Dict[str, str]:
        """
        导出分析报告和配置文件

        Args:
            phase2_config: Phase 2配置
            output_dir: 输出目录
            prefix: 文件名前缀
            report_format: 报告格式，'md'或'txt'（默认'md'）

        Returns:
            导出的文件路径字典
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # 1. JSON配置文件（供程序读取）
        json_path = output_path / f"{prefix}_phase2_config.json"

        # 获取方差分解信息
        var_decomp = phase2_config.get("from_phase1", {}).get(
            "variance_decomposition", {}
        )

        json_data = {
            "interaction_pairs": phase2_config["interaction_pairs"],
            "lambda_start": phase2_config["lambda_start"],
            "lambda_max": phase2_config["lambda_max"],
            "lambda_init": phase2_config["lambda_init"],  # 兼容性
            "lambda_end": phase2_config["lambda_end"],  # 兼容性
            "gamma_init": phase2_config["gamma_init"],
            "gamma_end": phase2_config["gamma_end"],
            "total_budget": phase2_config["total_budget"],
            "mid_diagnostic_trial": phase2_config["mid_diagnostic_trial"],
            # 新增：Phase 1 诊断信息
            "phase1_diagnostics": {
                "r2_adj_main": var_decomp.get("r2_adj_main", 0),
                "r2_adj_full": var_decomp.get("r2_adj_full", 0),
                "delta_adj": var_decomp.get("delta_adj", 0),
                "raw_ratio": var_decomp.get("raw_ratio", 0),
                "noise_level_estimate": var_decomp.get("residual_variance", 0),
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        exported_files["json_config"] = str(json_path)

        # 2. NumPy配置文件（供程序读取）
        npz_path = output_path / f"{prefix}_phase2_schedules.npz"
        np.savez(
            npz_path,
            lambda_schedule=np.array(phase2_config["lambda_schedule"]),
            gamma_schedule=np.array(phase2_config["gamma_schedule"]),
            interaction_pairs=np.array(phase2_config["interaction_pairs"]),
        )
        exported_files["npz_schedules"] = str(npz_path)

        # 3. 人类可读报告（支持MD和TXT）
        if report_format.lower() == "md":
            report_path = output_path / f"{prefix}_analysis_report.md"
            self._write_markdown_report(report_path, phase2_config)
        else:
            report_path = output_path / f"{prefix}_analysis_report.txt"
            self._write_text_report(report_path, phase2_config)
        exported_files["report"] = str(report_path)

        # 4. Phase 2使用指南
        if report_format.lower() == "md":
            guide_path = output_path / f"PHASE2_USAGE_GUIDE.md"
            self._write_usage_guide_markdown(guide_path, phase2_config)
        else:
            guide_path = output_path / f"PHASE2_USAGE_GUIDE.txt"
            self._write_usage_guide(guide_path, phase2_config)
        exported_files["usage_guide"] = str(guide_path)

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
        quality_metrics = self._calculate_quality_metrics()
        with open(path, "w", encoding="utf-8") as f:
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

            f.write("数据质量指标（预热阶段）\n")
            f.write("-" * 80 + "\n")

            def write_metric(label: str, metric_key: str, formatter) -> None:
                metric = quality_metrics.get(metric_key, {})
                value = metric.get("value")
                if value is None:
                    msg = metric.get("error", "计算失败")
                    f.write(f"{label}: 暂无结果（{msg}）\n")
                else:
                    f.write(
                        f"{label}: {formatter(value)} | {metric.get('purpose', '')}\n"
                    )

            write_metric("ICC (可靠性基线)", "icc", lambda v: f"{v:.3f}")
            write_metric(
                "主效应SE (粗估计精度)", "main_effect_se", lambda v: f"{v:.4f}"
            )
            write_metric("批次效应 (系统偏差)", "batch_effect", lambda v: f"{v:.4f}")
            write_metric("重测信度 (时间稳定性)", "test_retest", lambda v: f"{v:.3f}")
            write_metric("GP CV-RMSE (初始模型)", "gp_cv_rmse", lambda v: f"{v:.4f}")
            write_metric("覆盖率 (空间探索)", "coverage", lambda v: f"{v*100:.1f}%")
            write_metric("基尼系数 (分布均匀)", "gini", lambda v: f"{v:.4f}")
            write_metric(
                "筛选对数 (下阶段focus)",
                "n_interaction_pairs",
                lambda v: f"{int(v)} 个",
            )
            f.write("\n")

            insights = self._build_quality_insights(quality_metrics)
            if insights:
                f.write("质量指标解读:\n")
                for line in insights:
                    f.write(f"  {line.lstrip('- ')}\n")
                f.write("\n")

            # 交互对筛选结果
            f.write("2. 筛选出的交互对（用于Phase 2）\n")
            f.write("-" * 80 + "\n")
            f.write(f"数量: {len(self.analysis['selected_pairs'])}个\n\n")

            # 获取interaction_scores
            interaction_scores = self.analysis.get("diagnostics", {}).get(
                "interaction_scores", {}
            )

            for rank, pair in enumerate(self.analysis["selected_pairs"], 1):
                score = interaction_scores.get(pair, 0.0)
                factor1 = self.factor_cols[pair[0]]
                factor2 = self.factor_cols[pair[1]]
                f.write(f"  {rank}. ({factor1}, {factor2})\n")
                f.write(f"     索引: ({pair[0]}, {pair[1]})\n")
                f.write(f"     综合评分: {score:.3f}\n\n")

            # λ估计
            f.write("3. 交互权重参数（λ）- 新策略：从低到高\n")
            f.write("-" * 80 + "\n")
            f.write(f"Phase 1估计λ_max: {self.analysis['lambda_init']:.3f}\n")
            f.write(f"调整系数: {phase2_config['lambda_adjustment']:.2f}\n")
            f.write(f"Phase 2起点: {phase2_config['lambda_start']:.3f} (稳固主效应)\n")
            f.write(f"Phase 2终点: {phase2_config['lambda_max']:.3f} (探索交互)\n\n")
            f.write("策略说明:\n")
            f.write("  - Phase 2前期：λ低 → 先稳固主效应估计\n")
            f.write("  - Phase 2后期：λ高 → 再探索交互效应\n")
            f.write("  - 避免过早探索交互导致主效应估计不准\n\n")

            # 方差分解
            diagnostics = self.analysis.get("diagnostics", {})
            var_decomp = diagnostics.get("var_decomposition", {})
            if var_decomp:
                f.write("方差分解:\n")
                f.write(f"  主效应方差: {var_decomp.get('main_variance', 0):.4f}\n")
                f.write(
                    f"  交互方差: {var_decomp.get('interaction_variance', 0):.4f}\n"
                )
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
            f.write(
                f"中期诊断位置: 第{phase2_config['mid_diagnostic_trial']}次trial\n\n"
            )

            # 主效应估计
            f.write("6. 主效应估计\n")
            f.write("-" * 80 + "\n")
            main_effects = self.analysis.get("main_effects", {})
            if main_effects:
                for factor, effect_info in main_effects.items():
                    if isinstance(effect_info, dict):
                        coef = effect_info.get("coef", 0)
                        f.write(f"  {factor}: {coef:.4f}\n")
                    else:
                        f.write(f"  {factor}: {effect_info:.4f}\n")
            else:
                f.write("  （未估计）\n")
            f.write("\n")

            # 交互效应估计
            f.write("7. 交互效应估计\n")
            f.write("-" * 80 + "\n")
            interaction_effects = self.analysis.get("interaction_effects", {})
            if interaction_effects:
                for pair, effect_info in interaction_effects.items():
                    if isinstance(effect_info, dict):
                        pair_name = effect_info.get("pair_name", str(pair))
                        coef_int = effect_info.get("coef_interaction", 0)
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
            f.write(
                f"4. 在第{phase2_config['mid_diagnostic_trial']}次trial进行中期诊断\n\n"
            )

    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """计算数据质量指标（预热阶段）

        注：预热阶段是探索性实验，标准比Phase 2更宽松
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        metrics = {}

        try:
            # 1. ICC（被试内相关系数）- 可靠性基线
            # 衡量被试间的一致性，预热阶段≥0.30即可接受
            subject_means = {}
            for subj in np.unique(self.subject_ids):
                subject_means[subj] = np.mean(self.y_warmup[self.subject_ids == subj])

            grand_mean = np.mean(self.y_warmup)

            # 计算组内方差（within-subject variance）
            within_var = 0.0
            for subj in subject_means:
                subj_mask = self.subject_ids == subj
                within_var += np.sum(
                    (self.y_warmup[subj_mask] - subject_means[subj]) ** 2
                )
            within_var /= len(self.y_warmup) - len(subject_means)

            # 计算组间方差（between-subject variance）
            between_var = 0.0
            for subj in subject_means:
                n_subj = np.sum(self.subject_ids == subj)
                between_var += n_subj * (subject_means[subj] - grand_mean) ** 2
            between_var /= (len(subject_means) - 1) if len(subject_means) > 1 else 1

            icc = (
                between_var / (between_var + within_var)
                if (between_var + within_var) > 0
                else 0
            )
            metrics["icc"] = {
                "value": icc,
                "threshold_good": 0.45,
                "threshold_min": 0.30,  # 预热阶段的最低标准
                "purpose": "可靠性基线",
            }
        except Exception as e:
            metrics["icc"] = {"value": None, "error": f"计算失败: {str(e)}"}

        try:
            # 2. 主效应SE（标准误）- 粗估计精度
            # 衡量主效应系数估计的精确度，预热阶段<0.15即可
            model = LinearRegression().fit(self.X_warmup, self.y_warmup)
            residuals = self.y_warmup - model.predict(self.X_warmup)
            mse = np.sum(residuals**2) / (len(self.y_warmup) - self.X_warmup.shape[1])
            X_with_const = np.column_stack([np.ones(len(self.y_warmup)), self.X_warmup])

            try:
                var_covar = mse * np.linalg.inv(X_with_const.T @ X_with_const)
                se_values = np.sqrt(np.diag(var_covar))[1:]  # 忽略常数项
                se_mean = np.mean(se_values)
                metrics["main_effect_se"] = {
                    "value": se_mean,
                    "threshold_good": 0.12,  # Phase 2标准
                    "threshold_min": 0.15,  # 预热阶段标准
                    "purpose": "粗估计精度",
                }
            except np.linalg.LinAlgError:
                metrics["main_effect_se"] = {"value": None, "error": "矩阵奇异"}
        except:
            metrics["main_effect_se"] = {"value": None, "error": "计算失败"}

        try:
            # 3. 批次效应 - 系统偏差控制
            # 衡量早期vs晚期的稳定性，<0.30表示稳定
            n_quarter = len(self.y_warmup) // 4
            early_mean = np.mean(self.y_warmup[:n_quarter])
            late_mean = np.mean(self.y_warmup[-n_quarter:])
            batch_effect = abs(late_mean - early_mean) / (np.std(self.y_warmup) + 1e-6)
            metrics["batch_effect"] = {
                "value": batch_effect,
                "threshold_good": 0.20,  # Phase 2标准
                "threshold_min": 0.30,  # 预热阶段标准
                "purpose": "系统偏差控制",
            }
        except:
            metrics["batch_effect"] = {"value": None, "error": "计算失败"}

        try:
            # 4. 重测信度 - 时间稳定性
            # 对每位被试分别计算前半程与后半程的平均响应，再用Spearman相关衡量整体一致性
            from scipy.stats import spearmanr

            subject_pairs = []
            for subj in np.unique(self.subject_ids):
                subj_mask = self.subject_ids == subj
                subj_series = self.y_warmup[subj_mask]
                if len(subj_series) < 4:
                    continue
                mid_local = len(subj_series) // 2
                if mid_local == 0 or len(subj_series) - mid_local == 0:
                    continue
                first_mean = float(np.mean(subj_series[:mid_local]))
                second_mean = float(np.mean(subj_series[mid_local:]))
                subject_pairs.append((first_mean, second_mean))

            if len(subject_pairs) >= 2:
                first_vals, second_vals = zip(*subject_pairs)
                test_retest_corr, _ = spearmanr(first_vals, second_vals)
                if np.isnan(test_retest_corr):
                    raise ValueError("Spearman correlation is NaN")
            else:
                raise ValueError("Not enough subjects with split-half data")

            metrics["test_retest"] = {
                "value": float(test_retest_corr),
                "threshold_good": 0.80,  # Phase 2标准
                "threshold_min": 0.70,  # 预热阶段标准
                "purpose": "时间稳定性",
            }
        except Exception as e:
            metrics["test_retest"] = {"value": None, "error": f"计算失败: {e}"}

        try:
            # 5. GP CV-RMSE - 初始模型泛化能力
            model = LinearRegression()
            cv_scores = cross_val_score(
                model,
                self.X_warmup,
                self.y_warmup,
                cv=min(5, len(np.unique(self.subject_ids))),
                scoring="neg_mean_squared_error",
            )
            cv_rmse = np.sqrt(-cv_scores.mean())
            metrics["gp_cv_rmse"] = {
                "value": cv_rmse,
                "threshold_good": 0.85,  # Phase 2标准
                "threshold_min": 1.00,  # 预热阶段标准
                "purpose": "初始模型",
            }
        except:
            metrics["gp_cv_rmse"] = {"value": None, "error": "计算失败"}

        try:
            # 6. 空间探索 - 覆盖率
            # 按照唯一设计点占理论组合数量的比例来衡量覆盖度
            factor_df = self.df[self.factor_cols]
            unique_design_points = len(factor_df.drop_duplicates())
            level_counts = [
                max(factor_df[col].nunique(), 1) for col in self.factor_cols
            ]
            theoretical_combos = 1
            for count in level_counts:
                theoretical_combos *= count

            if theoretical_combos == 0:
                raise ValueError("Theoretical combination count is zero")

            # 如果理论组合数过大（连续变量场景），退化为 unique / 总样本
            if theoretical_combos > 1_000_000:
                coverage_ratio = unique_design_points / len(factor_df)
            else:
                coverage_ratio = unique_design_points / theoretical_combos

            metrics["coverage"] = {
                "value": float(coverage_ratio),
                "threshold_good": 0.10,  # Phase 2标准
                "threshold_min": 0.08,  # 预热阶段标准
                "purpose": "空间探索",
            }
        except Exception as e:
            metrics["coverage"] = {"value": None, "error": f"计算失败: {e}"}

        try:
            # 7. 分布均匀性 - 基尼系数
            # 衡量响应值分布是否均衡，<0.40表示分布较均匀
            y_sorted = np.sort(self.y_warmup)
            n = len(y_sorted)
            gini = (2 * np.sum((np.arange(1, n + 1)) * y_sorted)) / (
                n * np.sum(y_sorted)
            ) - (n + 1) / n
            metrics["gini"] = {
                "value": gini,
                "threshold_good": 0.40,  # Phase 2标准
                "threshold_min": 0.50,  # 预热阶段标准
                "purpose": "分布均匀性",
            }
        except:
            metrics["gini"] = {"value": None, "error": "计算失败"}

        try:
            # 8. 下阶段focus - 筛选出的交互对数
            # 诊断信息：有多少个交互对将在Phase 2重点探索
            n_pairs = len(self.analysis.get("selected_pairs", []))
            metrics["n_interaction_pairs"] = {
                "value": n_pairs,
                "threshold_good": 3,  # 通常3-5个为宜
                "threshold_min": 2,  # 至少2个
                "purpose": "下阶段focus",
            }
        except:
            metrics["n_interaction_pairs"] = {"value": None, "error": "计算失败"}

        return metrics

    def _build_quality_insights(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """根据质量指标生成可读性更高的解释文本"""

        insights: List[str] = []

        icc_val = quality_metrics.get("icc", {}).get("value")
        test_retest_val = quality_metrics.get("test_retest", {}).get("value")
        if icc_val is not None or test_retest_val is not None:
            if (icc_val is None or icc_val >= 0.45) and (
                test_retest_val is None or test_retest_val >= 0.80
            ):
                insights.append(
                    "- **被试稳定性**：ICC"
                    + (f"={icc_val:.2f}" if icc_val is not None else "≈N/A")
                    + "、重测信度"
                    + (
                        f"={test_retest_val:.2f}"
                        if test_retest_val is not None
                        else "≈N/A"
                    )
                    + " 均在绿色区间，说明被试群体反应模式高度一致，Phase 2 可以直接沿用该整体先验，把预算集中在交互探索上。"
                )
            else:
                insights.append(
                    "- **被试稳定性**：ICC"
                    + (f"={icc_val:.2f}" if icc_val is not None else "缺失")
                    + "、重测信度"
                    + (
                        f"={test_retest_val:.2f}"
                        if test_retest_val is not None
                        else "缺失"
                    )
                    + " 暗示个体差异仍然存在，Phase 2 需要在早期多收集不同被试的数据，以防止模型过拟合单个被试。"
                )

        batch_val = quality_metrics.get("batch_effect", {}).get("value")
        se_val = quality_metrics.get("main_effect_se", {}).get("value")
        if batch_val is not None or se_val is not None:
            if (batch_val is None or batch_val < 0.30) and (
                se_val is None or se_val < 0.15
            ):
                insights.append(
                    "- **系统稳定性**：批次效应"
                    + (f"={batch_val:.2f}" if batch_val is not None else "≈N/A")
                    + "、主效应SE"
                    + (f"={se_val:.3f}" if se_val is not None else "≈N/A")
                    + " 表明Phase 1已获得较干净的主效应估计，Phase 2可以减少重复基线测试，直接投入到交互验证。"
                )
            else:
                insights.append(
                    "- **系统稳定性**：批次效应"
                    + (f"={batch_val:.2f}" if batch_val is not None else "缺失")
                    + " 或主效应SE"
                    + (f"={se_val:.3f}" if se_val is not None else "缺失")
                    + " 偏高，提示实验流程仍有偏移，建议在Phase 2 设计额外的基线/校准试次。"
                )

        coverage_val = quality_metrics.get("coverage", {}).get("value")
        gini_val = quality_metrics.get("gini", {}).get("value")
        if coverage_val is not None or gini_val is not None:
            coverage_pct = coverage_val * 100 if coverage_val is not None else None
            if (coverage_val is not None and coverage_val >= 0.10) and (
                gini_val is None or gini_val < 0.40
            ):
                insights.append(
                    "- **空间探索**：覆盖率"
                    + (f"={coverage_pct:.1f}%" if coverage_pct is not None else "≈N/A")
                    + "、基尼系数"
                    + (f"={gini_val:.2f}" if gini_val is not None else "≈N/A")
                    + " 说明Phase 1已经把样本铺在较广的空间，Phase 2 可以更快进入 exploitation 阶段。"
                )
            else:
                insights.append(
                    "- **空间探索**：覆盖率"
                    + (f"={coverage_pct:.1f}%" if coverage_pct is not None else "缺失")
                    + " 或基尼系数"
                    + (f"={gini_val:.2f}" if gini_val is not None else "缺失")
                    + " 暗示仍有未触及的设计区域，Phase 2 前期应保持较高的 γ 来补足探索空白。"
                )

        cv_rmse_val = quality_metrics.get("gp_cv_rmse", {}).get("value")
        if cv_rmse_val is not None:
            if cv_rmse_val < 0.85:
                insights.append(
                    f"- **模型准备度**：GP CV-RMSE={cv_rmse_val:.2f} 已低于基准，当前先验能够较好预测响应，Phase 2 采样可以少量保留 exploitation 以加速收敛。"
                )
            else:
                insights.append(
                    f"- **模型准备度**：GP CV-RMSE={cv_rmse_val:.2f} 仍偏高，Phase 2 需要通过更高的探索权重来提升模型泛化能力。"
                )

        return insights

    def _write_markdown_report(self, path: Path, phase2_config: Dict[str, Any]):
        """生成Markdown格式的报告（增强版，包含更多解释）"""

        # 计算质量指标
        quality_metrics = self._calculate_quality_metrics()

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Phase 1 数据分析报告\n\n")
            f.write(
                "> **本报告汇总Phase 1预热实验的分析结果，为Phase 2主动学习提供参数指导**\n\n"
            )

            # 执行摘要
            f.write("## 📋 执行摘要\n\n")
            f.write(f"Phase 1实验已完成，共收集 **{len(self.df)} 条样本** 数据。")
            f.write(
                f"系统从数据中筛选出 **{len(self.analysis['selected_pairs'])} 个关键交互对**，"
            )
            f.write(f"并估计出交互权重参数λ = **{self.analysis['lambda_init']:.3f}**，")
            f.write(
                f"用于指导 Phase 2 的 **{phase2_config['total_budget']} 次自适应采样**。\n\n"
            )

            # 数据概览
            f.write("## 1️⃣ 数据概览\n\n")
            f.write("**数据质量基本信息**\n\n")
            f.write(f"| 项目 | 值 |\n")
            f.write(f"|------|-------|\n")
            f.write(f"| 数据文件 | {self.data_csv_path} |\n")
            f.write(f"| 样本总数 | {len(self.df)} |\n")
            f.write(f"| 被试数量 | {len(np.unique(self.subject_ids))} |\n")
            f.write(f"| 因子数量 | {len(self.factor_cols)} |\n")
            f.write(f"| 因子名称 | {', '.join(self.factor_cols)} |\n\n")

            f.write("**数据质量指标** *(衡量实验可靠性，预热阶段标准)*\n\n")
            f.write("| 指标 | 值 | 评价 | 说明 |\n")
            f.write("|------|--------|------|----------|\n")

            # ICC - 可靠性基线
            icc_val = quality_metrics["icc"].get("value")
            if icc_val is not None:
                icc_status = (
                    "✅ 优"
                    if icc_val >= 0.45
                    else ("⚠️  中" if icc_val >= 0.30 else "❌ 差")
                )
                f.write(
                    f"| **ICC** (可靠性基线) | {icc_val:.3f} | {icc_status} | 被试间一致性，≥0.30可接受 |\n"
                )

            # 主效应SE - 粗估计精度
            se_val = quality_metrics["main_effect_se"].get("value")
            if se_val is not None:
                se_status = (
                    "✅ 优"
                    if se_val < 0.12
                    else ("⚠️  中" if se_val < 0.15 else "❌ 差")
                )
                f.write(
                    f"| **主效应SE** (粗估计精度) | {se_val:.4f} | {se_status} | 主要因子系数精度，<0.15可接受 |\n"
                )

            # 批次效应 - 系统偏差控制
            batch_val = quality_metrics["batch_effect"].get("value")
            if batch_val is not None:
                batch_status = (
                    "✅ 优"
                    if batch_val < 0.20
                    else ("⚠️  中" if batch_val < 0.30 else "❌ 差")
                )
                f.write(
                    f"| **批次效应** (系统偏差) | {batch_val:.4f} | {batch_status} | 早期vs晚期稳定性，<0.30可接受 |\n"
                )

            # 重测信度 - 时间稳定性
            test_retest_val = quality_metrics["test_retest"].get("value")
            if test_retest_val is not None:
                test_retest_status = (
                    "✅ 优"
                    if test_retest_val >= 0.80
                    else ("⚠️  中" if test_retest_val >= 0.70 else "❌ 差")
                )
                f.write(
                    f"| **重测信度** (时间稳定性) | {test_retest_val:.3f} | {test_retest_status} | 前后两半数据一致性，≥0.70可接受 |\n"
                )

            # GP CV-RMSE - 初始模型
            cv_rmse_val = quality_metrics["gp_cv_rmse"].get("value")
            if cv_rmse_val is not None:
                cv_status = (
                    "✅ 优"
                    if cv_rmse_val < 0.85
                    else ("⚠️  中" if cv_rmse_val < 1.00 else "❌ 差")
                )
                f.write(
                    f"| **GP CV-RMSE** (初始模型) | {cv_rmse_val:.4f} | {cv_status} | 交叉验证误差，<1.00可接受 |\n"
                )

            # 覆盖率 - 空间探索
            coverage_val = quality_metrics["coverage"].get("value")
            if coverage_val is not None:
                coverage_pct = coverage_val * 100
                coverage_status = (
                    "✅ 优"
                    if coverage_val >= 0.10
                    else ("⚠️  中" if coverage_val >= 0.08 else "❌ 差")
                )
                f.write(
                    f"| **覆盖率** (空间探索) | {coverage_pct:.1f}% | {coverage_status} | 设计空间覆盖度，>8%可接受 |\n"
                )

            # Gini系数 - 分布均匀性
            gini_val = quality_metrics["gini"].get("value")
            if gini_val is not None:
                gini_status = (
                    "✅ 优"
                    if gini_val < 0.40
                    else ("⚠️  中" if gini_val < 0.50 else "❌ 差")
                )
                f.write(
                    f"| **基尼系数** (分布均匀) | {gini_val:.4f} | {gini_status} | 响应值分布均衡度，<0.50可接受 |\n"
                )

            # 交互对数 - 下阶段focus
            n_pairs_val = quality_metrics["n_interaction_pairs"].get("value")
            if n_pairs_val is not None:
                n_pairs_status = (
                    "✅ 优"
                    if n_pairs_val >= 3
                    else ("⚠️  中" if n_pairs_val >= 2 else "❌ 差")
                )
                f.write(
                    f"| **筛选对数** (下阶段focus) | {n_pairs_val} 个 | {n_pairs_status} | Phase 2重点探索交互对数，2-5个为宜 |\n"
                )

            f.write("\n")

            insights = self._build_quality_insights(quality_metrics)
            if insights:
                f.write("**如何解读这些指标？**\n\n")
                for line in insights:
                    f.write(line + "\n")
                f.write("\n")

            # 交互对筛选结果
            f.write("## 2️⃣ 交互对筛选结果\n\n")
            f.write("**找到了哪些重要的因子互动？**\n\n")
            f.write(
                f"系统从所有可能的因子对中筛选出 **{len(self.analysis['selected_pairs'])} 个最重要的交互对**，这些交互对的综合评分如下：\n\n"
            )
            f.write("| 排序 | 因子1 | 因子2 | 评分 | 说明 |\n")
            f.write("|------|-------|-------|-------|----------|\n")

            interaction_scores = self.analysis.get("diagnostics", {}).get(
                "interaction_scores", {}
            )
            for rank, pair in enumerate(self.analysis["selected_pairs"], 1):
                score = interaction_scores.get(pair, 0.0)
                factor1 = self.factor_cols[pair[0]]
                factor2 = self.factor_cols[pair[1]]
                f.write(f"| {rank} | {factor1} | {factor2} | {score:.3f} | ")
                f.write(
                    f"{'🔥 强交互' if score > 0.15 else '💡 中交互' if score > 0.10 else '⚡ 弱交互'} |\n"
                )
            f.write("\n")
            f.write("**为什么筛选这些交互对？**\n\n")
            f.write("Phase 1数据中，这些因子对对响应变量有显著的**协同效应**：\n")
            f.write("- 高评分的交互对说明两个因子不是独立作用，而是相互影响\n")
            f.write("- Phase 2会**重点探索**这些交互，以精确估计它们的大小和方向\n")
            f.write("- 帮助EUR-ANOVA采样器避免浪费预算在无关的因子组合\n\n")

            # λ参数详解
            f.write("## 3️⃣ 交互权重参数（λ）- 新策略：从低到高渐进探索\n\n")
            f.write("**参数概览**\n\n")
            f.write(f"| 参数 | 值 | 含义 |\n")
            f.write(f"|------|--------|----------|\n")
            f.write(
                f"| Phase 1估计λ_max | {self.analysis['lambda_init']:.3f} | 从Phase 1数据中估计的交互强度上限 |\n"
            )
            f.write(
                f"| 调整系数 | {phase2_config['lambda_adjustment']:.2f}× | 对Phase 1结果的信心调整 |\n"
            )
            f.write(
                f"| Phase 2起点 | {phase2_config['lambda_start']:.3f} | Phase 2开始时的λ值（低，稳固主效应） |\n"
            )
            f.write(
                f"| Phase 2终点 | {phase2_config['lambda_max']:.3f} | Phase 2后期达到的λ_max（高，探索交互） |\n\n"
            )

            f.write("**λ是什么？**\n\n")
            f.write("- λ控制EUR-ANOVA采样器**探索交互的热情程度**\n")
            f.write("- λ = 0.0 意味着完全忽略交互，只关注主效应\n")
            f.write("- λ = 1.0 意味着交互和主效应同等重要\n")
            f.write(
                f"- Phase 2采用**渐进策略**：从 {phase2_config['lambda_start']:.3f} 增长到 {phase2_config['lambda_max']:.3f}\n\n"
            )

            f.write("**为什么λ要从低到高增长？（新策略）**\n\n")
            f.write("- Phase 2前期：λ值较低 → **稳固阶段**，先获得准确的主效应估计\n")
            f.write("- Phase 2后期：λ值增长 → **探索阶段**，在稳固基础上探索交互效应\n")
            f.write("- 这避免了过早探索交互导致主效应估计不准的问题\n")
            f.write('- 符合"先简单后复杂"的科学探索原则\n\n')

            # 方差分解
            f.write("**方差分解** *(哪些效应最重要？)*\n\n")
            diagnostics = self.analysis.get("diagnostics", {})
            var_decomp = diagnostics.get("var_decomposition", {})
            if var_decomp:
                main_var = var_decomp.get("main_variance", 0)
                inter_var = var_decomp.get("interaction_variance", 0)
                resid_var = var_decomp.get("residual_variance", 0)
                total_var = main_var + inter_var + resid_var + 1e-6

                f.write(f"| 方差来源 | 大小 | 占比 | 含义 |\n")
                f.write(f"|---------|-------|------|----------|\n")
                f.write(
                    f"| 主效应方差 | {main_var:.4f} | {100*main_var/total_var:.1f}% | 单个因子的直接影响 |\n"
                )
                f.write(
                    f"| 交互方差 | {inter_var:.4f} | {100*inter_var/total_var:.1f}% | 因子间的协同效应 |\n"
                )
                f.write(
                    f"| 残差方差 | {resid_var:.4f} | {100*resid_var/total_var:.1f}% | 模型无法解释的部分 |\n\n"
                )

                if inter_var / total_var > 0.20:
                    f.write(
                        "💡 **观察**：交互效应很强（>20%），Phase 2重点探索这些交互是明智的选择\n\n"
                    )
                elif inter_var / total_var > 0.10:
                    f.write(
                        "💡 **观察**：交互效应中等强度（10-20%），既要探索也要精化主效应\n\n"
                    )
                else:
                    f.write("💡 **观察**：交互效应较弱（<10%），主效应是主要贡献者\n\n")

            # γ参数
            f.write("## 4️⃣ 覆盖权重参数（γ）- 如何平衡探索vs精化\n\n")
            f.write("**参数概览**\n\n")
            f.write(f"| 参数 | 值 |\n")
            f.write(f"|------|-------|\n")
            f.write(f"| Phase 2初始 | {phase2_config['gamma_init']:.3f} |\n")
            f.write(f"| Phase 2终点 | {phase2_config['gamma_end']:.3f} |\n\n")

            f.write("**γ是什么？**\n\n")
            f.write("- γ控制EUR-ANOVA采样器**寻找新区域的热情程度**\n")
            f.write("- 高γ值（0.3）：采样器会广泛探索设计空间\n")
            f.write("- 低γ值（0.06）：采样器会集中在已知的高价值区域\n")
            f.write("- **动态衰减策略**：从探索逐步转向精化\n\n")

            # Phase 2配置
            f.write("## 5️⃣ Phase 2采样配置\n\n")
            f.write(f"| 配置项 | 值 | 说明 |\n")
            f.write(f"|--------|--------|----------|\n")
            f.write(
                f"| 被试数 | {phase2_config['n_subjects']} 人 | 将邀请此数量的被试进行Phase 2 |\n"
            )
            f.write(
                f"| 每人trials | {phase2_config['trials_per_subject']} 次 | 每个被试完成的试验次数 |\n"
            )
            f.write(
                f"| 总预算 | {phase2_config['total_budget']} 次 | {phase2_config['n_subjects']} × {phase2_config['trials_per_subject']} |\n"
            )
            f.write(
                f"| 中期诊断 | 第{phase2_config['mid_diagnostic_trial']} 次 | 建议在此时检查是否需要调整策略 |\n\n"
            )

            # 主效应
            f.write("## 6️⃣ 主效应估计 *(单个因子的影响)*\n\n")
            f.write("这些是从Phase 1数据中估计的各因子对响应的直接影响：\n\n")
            main_effects = self.analysis.get("main_effects", {})
            if main_effects:
                f.write("| 因子 | 估计系数 | 方向 |\n")
                f.write("|------|---------|------|\n")
                for factor, effect_info in main_effects.items():
                    if isinstance(effect_info, dict):
                        coef = effect_info.get("coef", 0)
                    else:
                        coef = effect_info
                    direction = (
                        "↑ 正影响"
                        if coef > 0
                        else "↓ 负影响" if coef < 0 else "→ 无影响"
                    )
                    f.write(f"| {factor} | {coef:.4f} | {direction} |\n")
                f.write("\n")
                f.write("**如何理解？**\n")
                f.write("- 正系数(+)：增加该因子的水平会**增加**响应值\n")
                f.write("- 负系数(-)：增加该因子的水平会**减少**响应值\n")
                f.write("- 绝对值越大，影响越强\n\n")
            else:
                f.write("（未估计）\n\n")

            # 交互效应
            f.write("## 7️⃣ 交互效应估计 *(因子间的协同作用)*\n\n")
            f.write("这些是筛选出的交互对的实际效应大小：\n\n")
            interaction_effects = self.analysis.get("interaction_effects", {})
            if interaction_effects:
                f.write("| 交互对 | 估计系数 | 类型 |\n")
                f.write("|--------|---------|------|\n")
                for pair, effect_info in interaction_effects.items():
                    if isinstance(effect_info, dict):
                        pair_name = effect_info.get("pair_name", str(pair))
                        coef_int = effect_info.get("coef_interaction", 0)
                    else:
                        pair_name = str(pair)
                        coef_int = effect_info

                    if coef_int > 0:
                        inter_type = "✅ 协同增强"
                    elif coef_int < 0:
                        inter_type = "⚠️  协同削弱"
                    else:
                        inter_type = "→ 无互动"

                    f.write(f"| {pair_name} | {coef_int:.4f} | {inter_type} |\n")
                f.write("\n")
                f.write("**如何理解交互效应？**\n")
                f.write(
                    "- **协同增强**（+）：两个因子同时增加时，效果大于各自单独的效果\n"
                )
                f.write("- **协同削弱**（-）：两个因子组合的效果反而降低\n")
                f.write("- Phase 2会重点探索这些交互发生的具体条件\n\n")
            else:
                f.write("（未估计）\n\n")

            # 使用说明
            f.write("## 8️⃣ 后续步骤\n\n")
            f.write(
                "1. **查看使用指南**：打开 `PHASE2_USAGE_GUIDE.md` 了解如何在EUR-ANOVA中使用这些参数\n\n"
            )
            f.write("2. **准备Phase 2实验**：\n")
            f.write(f"   - 邀请 {phase2_config['n_subjects']} 个被试\n")
            f.write(f"   - 准备进行 {phase2_config['total_budget']} 次自适应采样\n\n")
            f.write("3. **中期检查**：\n")
            f.write(
                f"   - 在第 {phase2_config['mid_diagnostic_trial']} 次trial时，检查：\n"
            )
            f.write("     - 是否发现了新的重要交互？\n")
            f.write("     - λ和γ的衰减是否合适？\n")
            f.write("     - 是否需要调整Phase 2的后续策略？\n\n")
            f.write("4. **完成Phase 2**：收集所有 500 个样本点，为最终建模准备数据\n\n")

            f.write("---\n\n")
            f.write("*本报告由Phase 1数据分析系统自动生成*\n")

    def _write_usage_guide_markdown(self, path: Path, phase2_config: Dict[str, Any]):
        """生成Markdown格式的Phase 2使用指南（增强版，更易理解）"""
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Phase 2 EUR-ANOVA 使用指南\n\n")
            f.write("> 本指南说明如何在自适应采样中正确使用Phase 1的分析结果\n\n")

            # 快速开始
            f.write("## 🚀 快速开始\n\n")
            f.write("Phase 2使用EUR-ANOVA进行自适应采样。主要思想是：\n")
            f.write("- **根据Phase 1的发现**，智能地选择下一个采样点\n")
            f.write("- **平衡探索和精化**，最大化信息获取\n")
            f.write("- **动态调整参数**，适应实验进展\n\n")

            # 文件说明
            f.write("## 📁 生成的文件说明\n\n")
            f.write("| 文件 | 格式 | 用途 |\n")
            f.write("|------|------|------|\n")
            f.write(
                "| `phase1_phase2_config.json` | JSON | 被程序读取的配置（λ、γ初始值等） |\n"
            )
            f.write(
                "| `phase1_phase2_schedules.npz` | NumPy | λ和γ的动态衰减表（每个trial一行） |\n"
            )
            f.write(
                "| `phase1_analysis_report.md` | Markdown | 分析结果总结（给人看的） |\n"
            )
            f.write("| `PHASE2_USAGE_GUIDE.md` | Markdown | 本指南 |\n\n")

            # 实现步骤
            f.write("## 1️⃣ 第1步：加载配置\n\n")
            f.write("**为什么要这样做？**\n")
            f.write("- Phase 1分析生成的参数需要被EUR-ANOVA采样器读取\n")
            f.write("- JSON文件保存了交互对列表、λ和γ的初始值\n")
            f.write("- NPZ文件保存了整个Phase 2期间的参数衰减表\n\n")
            f.write("**代码实现：**\n\n")
            f.write("```python\n")
            f.write("import numpy as np\n")
            f.write("import json\n\n")
            f.write("# 读取Phase 1的分析结果\n")
            f.write("with open('phase1_phase2_config.json') as f:\n")
            f.write("    config = json.load(f)\n\n")
            f.write("# 交互对列表\n")
            f.write(f"interaction_pairs = {phase2_config['interaction_pairs']}\n")
            f.write('print(f"要探索的交互对: {interaction_pairs}")\n\n')
            f.write("# 加载λ和γ的动态衰减表\n")
            f.write("schedules = np.load('phase1_phase2_schedules.npz')\n")
            f.write(
                "lambda_schedule = schedules['lambda_schedule']  # 500行2列：(trial_idx, lambda_value)\n"
            )
            f.write(
                "gamma_schedule = schedules['gamma_schedule']    # 500行2列：(trial_idx, gamma_value)\n"
            )
            f.write("```\n\n")

            # 初始化采集函数
            f.write("## 2️⃣ 第2步：初始化EUR-ANOVA采集函数\n\n")
            f.write("**为什么要这样做？**\n")
            f.write(
                "- EUR-ANOVA是一种主动学习算法，能根据数据自动选择最有价值的采样点\n"
            )
            f.write("- 通过交互对信息（从Phase 1），它能优先探索有交互效应的因子组合\n")
            f.write('- λ参数告诉它"多重视这些交互"，γ参数告诉它"探索多大范围"\n\n')
            f.write("**代码实现：**\n\n")
            f.write("```python\n")
            f.write("from eur_anova_pair import EURAnovaPairAcqf\n\n")
            f.write("# 初始化采集函数\n")
            f.write("# 注意：这假设你已经有一个GP模型\n")
            f.write("acqf = EURAnovaPairAcqf(\n")
            f.write("    model=your_gp_model,          # 你训练的高斯过程\n")
            f.write(
                f"    lambda_init={phase2_config['lambda_init']:.3f},  # 初始λ（交互权重）\n"
            )
            f.write(
                f"    gamma_init={phase2_config['gamma_init']:.3f},    # 初始γ（探索程度）\n"
            )
            f.write(
                f"    interaction_pairs={phase2_config['interaction_pairs']},  # 要探索的交互\n"
            )
            f.write(
                f"    n_trials={phase2_config['total_budget']},  # 总共500个trial\n"
            )
            f.write(")\n")
            f.write("```\n\n")

            # 主采样循环
            f.write("## 3️⃣ 第3步：主采样循环\n\n")
            f.write("**为什么要这样做？**\n")
            f.write("- λ和γ不是固定不变的，而是根据进度逐步衰减的\n")
            f.write("- 前期：λ高 → 积极探索交互；γ高 → 广泛探索设计空间\n")
            f.write("- 后期：λ低 → 专注主效应；γ低 → 集中在高价值区域\n")
            f.write("- 这样能充分利用500个试验的预算\n\n")
            f.write("**代码实现：**\n\n")
            f.write("```python\n")
            f.write(f"total_budget = {phase2_config['total_budget']}\n\n")
            f.write("for trial in range(total_budget):\n")
            f.write("    # 【关键】从衰减表查询当前trial的λ和γ\n")
            f.write(
                "    current_lambda = lambda_schedule[trial, 1]  # 第trial行，第1列（值）\n"
            )
            f.write(
                "    current_gamma = gamma_schedule[trial, 1]    # 第trial行，第1列（值）\n"
            )
            f.write("    \n")
            f.write("    # 【重要】更新采集函数的参数\n")
            f.write("    # 这样EUR-ANOVA才知道当前应该有多重视交互\n")
            f.write("    acqf.set_lambda(current_lambda)\n")
            f.write("    acqf.set_gamma(current_gamma)\n")
            f.write("    \n")
            f.write("    # 【核心】用EUR-ANOVA选择下一个最有价值的采样点\n")
            f.write("    x_candidates = # ... 从设计空间生成候选点\n")
            f.write("    scores = acqf(x_candidates)  # 评分每个候选点\n")
            f.write("    x_next = x_candidates[np.argmax(scores)]  # 选分数最高的\n")
            f.write("    \n")
            f.write("    # 执行实验\n")
            f.write("    y_next = conduct_experiment(x_next)\n")
            f.write("    \n")
            f.write("    # 更新GP模型\n")
            f.write("    your_gp_model.update(x_next, y_next)\n")
            f.write("    \n")
            f.write("    # 可选：在中期进行诊断\n")
            f.write(f"    if trial == {phase2_config['mid_diagnostic_trial']}:\n")
            f.write('        print("🔍 中期诊断时刻！检查是否需要调整策略...")\n')
            f.write("```\n\n")

            # 中期诊断
            f.write("## 4️⃣ 第4步：中期诊断（可选但推荐）\n\n")
            f.write("**在第 ")
            f.write(
                f"{phase2_config['mid_diagnostic_trial']} 次trial进行诊断，检查：**\n\n"
            )
            f.write("✅ **主效应**\n")
            f.write("- 主效应的估计是否与Phase 1一致？\n")
            f.write("- 是否有因子的效应变化很大（可能有非线性）？\n\n")
            f.write("✅ **交互效应**\n")
            f.write("- 筛选出的交互对是否确实有预期的效应？\n")
            f.write("- 有没有其他意外的强交互出现？\n\n")
            f.write("✅ **参数调整**\n")
            f.write("- λ和γ的衰减速度是否合适？\n")
            f.write("- 需不需要手动调整后续的参数？\n\n")
            f.write("**如何调整（如果需要）：**\n")
            f.write("```python\n")
            f.write("# 如果发现要加强交互探索\n")
            f.write("acqf.set_lambda(0.5)  # 手动提高λ\n\n")
            f.write("# 如果发现应该更聚焦探索\n")
            f.write("acqf.set_gamma(0.1)   # 手动降低γ\n")
            f.write("```\n\n")

            # 参数释义
            f.write("## 📚 关键参数详解\n\n")
            f.write("### λ（Lambda）：交互权重\n\n")
            f.write("| 含义 | λ值 | 采样行为 |\n")
            f.write("|------|--------|----------|\n")
            f.write("| 只关注主效应 | 0.0 | 均匀探索所有点，忽略交互信息 |\n")
            f.write(
                f"| 你的Phase 2初始值 | {phase2_config['lambda_init']:.3f} | **平衡模式**：既探索交互也精化主效应 |\n"
            )
            f.write("| 平衡权重 | 0.5 | 交互和主效应同等重要 |\n")
            f.write("| 完全关注交互 | 1.0 | 集中探索交互对，忽视主效应 |\n\n")
            f.write("**实例：**\n")
            f.write("- 如果λ=0.36，EUR-ANOVA会36%的力气探索选定的交互对，64%探索其他\n")
            f.write("- Phase 2后期λ衰减到0.2，意味着逐步转向主效应精化\n\n")

            f.write("### γ（Gamma）：覆盖权重\n\n")
            f.write("| 含义 | γ值 | 采样行为 |\n")
            f.write("|------|--------|----------|\n")
            f.write("| 完全精化 | 0.0 | 聚焦在已知最优点附近，不探索新区域 |\n")
            f.write(
                f"| 你的Phase 2终点值 | {phase2_config['gamma_end']:.3f} | 精化阶段：主要精化已发现的好点 |\n"
            )
            f.write(
                f"| 你的Phase 2初始值 | {phase2_config['gamma_init']:.3f} | 探索阶段：广泛探索设计空间 |\n"
            )
            f.write("| 完全探索 | 1.0 | 随机探索所有点，不利用已有信息 |\n\n")
            f.write("**实例：**\n")
            f.write('- 如果γ=0.3，采样器会在"已知好的点"和"新颖点"之间平衡\n')
            f.write("- Phase 2后期γ衰减到0.06，意味着逐步聚焦到最有希望的区域\n\n")

            # 高级用法
            f.write("## 🔧 高级用法（可选）\n\n")
            f.write("**动态调整λ**（如果发现某些交互特别重要）\n")
            f.write("```python\n")
            f.write("# 手动提高特定交互对的权重\n")
            f.write("acqf.increase_interaction_weight((3, 4), factor=2.0)\n")
            f.write("```\n\n")
            f.write("**查看采样历史**（理解EUR-ANOVA的决策）\n")
            f.write("```python\n")
            f.write("# 查看每个trial选择的点\n")
            f.write("import pandas as pd\n")
            f.write("sampling_history = pd.DataFrame({\n")
            f.write("    'trial': range(1, total_budget+1),\n")
            f.write("    'lambda': lambda_schedule[:, 1],\n")
            f.write("    'gamma': gamma_schedule[:, 1],\n")
            f.write("    'x_selected': x_history,  # 你保存的采样点\n")
            f.write("    'y_observed': y_history   # 对应的响应\n")
            f.write("})\n")
            f.write("sampling_history.to_csv('phase2_sampling_log.csv', index=False)\n")
            f.write("```\n\n")

            # 常见问题
            f.write("## ❓ 常见问题\n\n")
            f.write("**Q: 为什么要衰减λ和γ？**\n")
            f.write(
                "A: 早期需要探索新区域，后期需要精化已发现的好点。固定参数会浪费预算。\n\n"
            )
            f.write("**Q: 可以不用动态衰减表吗？**\n")
            f.write(
                "A: 可以，但效率会降低。衰减表是Phase 1分析优化的结果，能最大化信息利用。\n\n"
            )
            f.write("**Q: 中期诊断发现了新问题怎么办？**\n")
            f.write("A: 可以手动调整λ、γ或交互对列表，但要记录变更以便后续分析。\n\n")
            f.write("**Q: EUR-ANOVA不收敛怎么办？**\n")
            f.write("A: 检查GP模型是否训练充分，或尝试调整λ和γ的衰减速度。\n\n")

            f.write("---\n\n")
            f.write("*本指南由Phase 1数据分析系统自动生成，最后更新于Phase 2开始前*\n")

    def _write_usage_guide(self, path: Path, phase2_config: Dict[str, Any]):
        """生成Phase 2使用指南（文本格式，增强版）"""
        with open(path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Phase 2 EUR-ANOVA 使用指南\n")
            f.write("=" * 80 + "\n\n")

            f.write("本指南说明如何在自适应采样中正确使用Phase 1的分析结果\n")
            f.write(
                "内容匹配Markdown版本（PHASE2_USAGE_GUIDE.md），但采用纯文本格式\n\n"
            )

            # 快速开始
            f.write("快速开始\n")
            f.write("=" * 80 + "\n\n")
            f.write("Phase 2使用EUR-ANOVA进行自适应采样。主要思想是：\n")
            f.write("  1. 根据Phase 1的发现，智能地选择下一个采样点\n")
            f.write("  2. 平衡探索和精化，最大化信息获取\n")
            f.write("  3. 动态调整参数，适应实验进展\n\n")

            # 文件说明
            f.write("生成的文件说明\n")
            f.write("=" * 80 + "\n\n")
            f.write("下列文件已由Phase 1分析生成，供Phase 2使用：\n\n")
            f.write("  phase1_phase2_config.json\n")
            f.write("    - 格式: JSON\n")
            f.write("    - 用途: 被程序读取的配置（λ、γ初始值等）\n")
            f.write("    - 内容: 交互对列表、λγ初始值、衰减参数等\n\n")
            f.write("  phase1_phase2_schedules.npz\n")
            f.write("    - 格式: NumPy二进制\n")
            f.write("    - 用途: λ和γ的动态衰减表（每个trial一行）\n")
            f.write(
                "    - 用法: 在第t个trial，查询schedules['lambda_schedule'][t, 1]\n\n"
            )
            f.write("  phase1_analysis_report.md\n")
            f.write("    - 格式: Markdown\n")
            f.write("    - 用途: 分析结果总结（给人看的，包含质量评估）\n\n")
            f.write("  PHASE2_USAGE_GUIDE.md\n")
            f.write("    - 格式: Markdown\n")
            f.write("    - 用途: 本指南的Markdown版本（更清晰的格式）\n\n")

            # 实现步骤
            f.write("第1步：加载配置\n")
            f.write("=" * 80 + "\n\n")
            f.write("为什么要这样做？\n")
            f.write("  - Phase 1分析生成的参数需要被EUR-ANOVA采样器读取\n")
            f.write("  - JSON文件保存了交互对列表、λ和γ的初始值\n")
            f.write("  - NPZ文件保存了整个Phase 2期间的参数衰减表\n\n")
            f.write("代码实现：\n\n")
            f.write("import numpy as np\n")
            f.write("import json\n\n")
            f.write("# 读取Phase 1的分析结果\n")
            f.write("with open('phase1_phase2_config.json') as f:\n")
            f.write("    config = json.load(f)\n\n")
            f.write("# 交互对列表\n")
            f.write(f"interaction_pairs = {phase2_config['interaction_pairs']}\n")
            f.write('print(f"要探索的交互对: {{interaction_pairs}}")\n\n')
            f.write("# 加载λ和γ的动态衰减表\n")
            f.write("schedules = np.load('phase1_phase2_schedules.npz')\n")
            f.write("lambda_schedule = schedules['lambda_schedule']  # 500行2列\n")
            f.write("gamma_schedule = schedules['gamma_schedule']    # 500行2列\n\n")

            # 初始化采集函数
            f.write("第2步：初始化EUR-ANOVA采集函数\n")
            f.write("=" * 80 + "\n\n")
            f.write("为什么要这样做？\n")
            f.write(
                "  - EUR-ANOVA是一种主动学习算法，能根据数据自动选择最有价值的采样点\n"
            )
            f.write(
                "  - 通过交互对信息（从Phase 1），它能优先探索有交互效应的因子组合\n"
            )
            f.write('  - λ参数告诉它"多重视这些交互"，γ参数告诉它"探索多大范围"\n\n')
            f.write("代码实现：\n\n")
            f.write("from eur_anova_pair import EURAnovaPairAcqf\n\n")
            f.write("# 初始化采集函数\n")
            f.write("acqf = EURAnovaPairAcqf(\n")
            f.write("    model=your_gp_model,          # 你训练的高斯过程\n")
            f.write(f"    lambda_init={phase2_config['lambda_init']:.3f},  # 初始λ\n")
            f.write(f"    gamma_init={phase2_config['gamma_init']:.3f},    # 初始γ\n")
            f.write(f"    interaction_pairs={phase2_config['interaction_pairs']},\n")
            f.write(f"    n_trials={phase2_config['total_budget']},\n")
            f.write(")\n\n")

            # 主采样循环
            f.write("第3步：主采样循环\n")
            f.write("=" * 80 + "\n\n")
            f.write("为什么要这样做？\n")
            f.write("  - λ和γ不是固定不变的，而是根据进度逐步衰减的\n")
            f.write("  - 前期：λ高 → 积极探索交互；γ高 → 广泛探索设计空间\n")
            f.write("  - 后期：λ低 → 专注主效应；γ低 → 集中在高价值区域\n")
            f.write("  - 这样能充分利用500个试验的预算\n\n")
            f.write("代码实现：\n\n")
            f.write(f"total_budget = {phase2_config['total_budget']}\n\n")
            f.write("for trial in range(total_budget):\n")
            f.write("    # 从衰减表查询当前trial的λ和γ\n")
            f.write("    current_lambda = lambda_schedule[trial, 1]\n")
            f.write("    current_gamma = gamma_schedule[trial, 1]\n\n")
            f.write("    # 更新采集函数的参数\n")
            f.write("    acqf.set_lambda(current_lambda)\n")
            f.write("    acqf.set_gamma(current_gamma)\n\n")
            f.write("    # 用EUR-ANOVA选择下一个最有价值的采样点\n")
            f.write("    x_candidates = # ... 从设计空间生成候选点\n")
            f.write("    scores = acqf(x_candidates)  # 评分每个候选点\n")
            f.write("    x_next = x_candidates[np.argmax(scores)]  # 选分数最高的\n\n")
            f.write("    # 执行实验\n")
            f.write("    y_next = conduct_experiment(x_next)\n\n")
            f.write("    # 更新GP模型\n")
            f.write("    your_gp_model.update(x_next, y_next)\n\n")
            f.write("    # 可选：在中期进行诊断\n")
            f.write(f"    if trial == {phase2_config['mid_diagnostic_trial']}:\n")
            f.write('        print("中期诊断时刻！检查是否需要调整策略...")\n\n')

            # 中期诊断
            f.write("第4步：中期诊断（可选但推荐）\n")
            f.write("=" * 80 + "\n\n")
            f.write(
                f"在第 {phase2_config['mid_diagnostic_trial']} 次trial进行诊断，检查：\n\n"
            )
            f.write("主效应\n")
            f.write("  - 主效应的估计是否与Phase 1一致？\n")
            f.write("  - 是否有因子的效应变化很大（可能有非线性）？\n\n")
            f.write("交互效应\n")
            f.write("  - 筛选出的交互对是否确实有预期的效应？\n")
            f.write("  - 有没有其他意外的强交互出现？\n\n")
            f.write("参数调整\n")
            f.write("  - λ和γ的衰减速度是否合适？\n")
            f.write("  - 需不需要手动调整后续的参数？\n\n")
            f.write("如何调整（如果需要）：\n\n")
            f.write("if need_more_interaction_exploration:\n")
            f.write("    acqf.set_lambda(0.5)  # 手动提高λ\n\n")
            f.write("if need_more_focused_exploration:\n")
            f.write("    acqf.set_gamma(0.1)   # 手动降低γ\n\n")

            # 参数释义
            f.write("关键参数详解\n")
            f.write("=" * 80 + "\n\n")
            f.write("λ（Lambda）：交互权重\n\n")
            f.write("  含义               λ值    采样行为\n")
            f.write("  " + "-" * 76 + "\n")
            f.write("  只关注主效应       0.0    均匀探索所有点，忽略交互信息\n")
            f.write(
                f"  Phase 2初始值      {phase2_config['lambda_init']:.3f}  平衡模式：既探索交互也精化主效应\n"
            )
            f.write("  平衡权重          0.5    交互和主效应同等重要\n")
            f.write("  完全关注交互       1.0    集中探索交互对，忽视主效应\n\n")
            f.write("实例：\n")
            f.write(
                f"  - 如果λ={phase2_config['lambda_init']:.2f}，EUR-ANOVA会用该比例的力气探索选定的交互对\n"
            )
            f.write(
                f"  - Phase 2后期λ衰减到{phase2_config['lambda_end']:.2f}，意味着逐步转向主效应精化\n\n"
            )

            f.write("γ（Gamma）：覆盖权重\n\n")
            f.write("  含义               γ值    采样行为\n")
            f.write("  " + "-" * 76 + "\n")
            f.write("  完全精化          0.0    聚焦在已知最优点附近，不探索新区域\n")
            f.write(
                f"  Phase 2终点值     {phase2_config['gamma_end']:.3f}  精化阶段：主要精化已发现的好点\n"
            )
            f.write(
                f"  Phase 2初始值     {phase2_config['gamma_init']:.3f}  探索阶段：广泛探索设计空间\n"
            )
            f.write("  完全探索          1.0    随机探索所有点，不利用已有信息\n\n")
            f.write("实例：\n")
            f.write(
                f"  - 如果γ={phase2_config['gamma_init']:.2f}，采样器在\"已知好的点\"和\"新颖点\"间平衡\n"
            )
            f.write(
                f"  - Phase 2后期γ衰减到{phase2_config['gamma_end']:.2f}，逐步聚焦到最有希望的区域\n\n"
            )

            # 常见问题
            f.write("常见问题\n")
            f.write("=" * 80 + "\n\n")
            f.write("Q: 为什么要衰减λ和γ？\n")
            f.write(
                "A: 早期需要探索新区域，后期需要精化已发现的好点。固定参数会浪费预算。\n\n"
            )
            f.write("Q: 可以不用动态衰减表吗？\n")
            f.write(
                "A: 可以，但效率会降低。衰减表是Phase 1分析优化的结果，能最大化信息利用。\n\n"
            )
            f.write("Q: 中期诊断发现了新问题怎么办？\n")
            f.write("A: 可以手动调整λ、γ或交互对列表，但要记录变更以便后续分析。\n\n")
            f.write("Q: EUR-ANOVA不收敛怎么办？\n")
            f.write("A: 检查GP模型是否训练充分，或尝试调整λ和γ的衰减速度。\n\n")

            f.write("=" * 80 + "\n")
            f.write("本指南由Phase 1数据分析系统自动生成\n")


def main():
    """交互式主流程"""
    print()
    print("=" * 80)
    print("Phase 1 数据分析工具")
    print("=" * 80)
    print()

    # Step 1: 输入数据文件路径
    data_csv = input(
        "请输入实验数据CSV路径（或按Enter使用默认 'warmup_data.csv'）: "
    ).strip()
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

    selection_method = (
        input("  选择方法 (elbow/bic_threshold/top_k，默认 elbow): ").strip() or "elbow"
    )
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
        lambda_adjustment = (
            float(lambda_adjustment_str) if lambda_adjustment_str else 1.2
        )
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
        print(
            f"λ初始: {phase2_config['lambda_init']:.3f} (Phase 1: {phase2_config['lambda_phase1']:.3f})"
        )
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
    output_dir = (
        input("输出目录（默认 'phase1_analysis_output'）: ").strip()
        or "phase1_analysis_output"
    )
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
        print("1. 查看分析报告: " + exported_files["txt_report"])
        print("2. 阅读使用指南: " + exported_files["usage_guide"])
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
