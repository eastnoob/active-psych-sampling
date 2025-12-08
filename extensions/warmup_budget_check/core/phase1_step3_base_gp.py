"""\n+Phase 1 Step3: Base GP (Matern 2.5 + ARD) 构建与设计空间扫描\n+\n+功能概述:\n+1. 读取 Phase1 数据集 (含因子与响应) \n+2. 对每个被试进行被试内 Z-score 标准化 (y -> y_norm)\n+3. 使用 Matern ν=2.5 Kernel + ARD 训练 Base GP (botorch + gpytorch)\n+4. 扫描用户给定的设计空间 CSV, 计算预测均值/标准差\n+5. 选出: 全局最高点 x_best_prior, 全局最低点 x_worst_prior, 最不确定点 x_max_std (若方差过低则退化为“中心点”)\n+6. 导出模型 state_dict, 长度尺度, 关键点, 设计空间扫描结果, 报告\n+\n+使用方式(交互):\n+  python phase1_step3_base_gp.py\n+  -> 输入 Phase1 数据 CSV 路径 / 设计空间 CSV 路径 等\n+\n+使用方式(配置一次性调用, 推荐结合 quick_start.py):\n+  在 quick_start.py 中设置 MODE='step3' 并填写 STEP3_CONFIG\n+\n+文件输出(默认 output_dir=base_gp_output):\n+  base_gp_state.pth               模型与likelihood state_dict
  base_gp_lengthscales.json       长度尺度与敏感度排序
  base_gp_subject_stats.json      被试标准化统计 (均值/标准差)
  base_gp_encodings.json          分类变量编码映射
  base_gp_key_points.json         三个关键点及预测值
  design_space_scan.csv           设计空间逐点预测 (mean,std)
  base_gp_report.md               报告摘要
\n+依赖: 需要已安装 torch, gpytorch, botorch (在当前 pixi 环境中 aepsych 已依赖 botorch)。\n+"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.optim.fit import fit_gpytorch_mll_torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
except Exception as e:  # pragma: no cover - 环境导入失败时的提示
    print("[错误] 需要安装 torch/gpytorch/botorch: ", e)
    sys.exit(1)


def _infer_encoding_from_sampling(
    data_dir: Path, factor_cols: List[str]
) -> Dict[str, Dict[Any, int]]:
    """从采样方案和模拟结果推断编码映射。

    比较 subject_1.csv (categorical) 和 result/subject_1.csv (numeric)
    来推断哪些列被编码了，以及编码映射是什么。
    """
    encodings: Dict[str, Dict[Any, int]] = {}

    # 查找采样方案文件和结果文件
    sampling_file = data_dir.parent / "subject_1.csv"
    result_file = data_dir / "subject_1.csv"

    if not sampling_file.exists() or not result_file.exists():
        print(f"[Warning] 无法推断编码：找不到采样文件或结果文件")
        return encodings

    df_sampling = pd.read_csv(sampling_file)
    df_result = pd.read_csv(result_file)

    # 对于每个因子列，检查是否需要编码
    for col in factor_cols:
        if col not in df_sampling.columns or col not in df_result.columns:
            continue

        # 如果采样是categorical，结果是numeric，则推断编码
        if df_sampling[col].dtype == "object" and df_result[col].dtype != "object":
            # 收集所有 (categorical_value, numeric_value) 对
            mapping_pairs = []
            for i in range(min(len(df_sampling), len(df_result))):
                cat_val = df_sampling[col].iloc[i]
                num_val = df_result[col].iloc[i]
                if pd.notna(cat_val) and pd.notna(num_val):
                    mapping_pairs.append((cat_val, int(num_val)))

            # 构建映射字典
            mapping = {}
            for cat_val, num_val in mapping_pairs:
                if cat_val not in mapping:
                    mapping[cat_val] = num_val
                elif mapping[cat_val] != num_val:
                    print(f"[Warning] 列 {col} 的编码不一致: {cat_val} -> {mapping[cat_val]} vs {num_val}")

            if mapping:
                encodings[col] = mapping
                print(f"[推断编码] {col}: {mapping}")

    return encodings


def _encode_factor_df(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]]]:
    """对因子列进行编码 (分类变量 label encode, bool->int)。
    返回编码后 DataFrame 与 编码字典。"""
    encoded = df.copy()
    encodings: Dict[str, Dict[Any, int]] = {}
    for col in encoded.columns:
        if encoded[col].dtype == "object":
            unique_vals = sorted(encoded[col].dropna().unique())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            encodings[col] = mapping
            encoded[col] = encoded[col].map(mapping)
        elif encoded[col].dtype == "bool":
            mapping = {False: 0, True: 1}
            encodings[col] = mapping
            encoded[col] = encoded[col].astype(int)
    return encoded, encodings


def _apply_encodings(
    df: pd.DataFrame, encodings: Dict[str, Dict[Any, int]]
) -> pd.DataFrame:
    """将已存在的编码映射应用到新的 DataFrame (设计空间). 新出现的类别报错。"""
    df_new = df.copy()

    # 处理所有列，确保没有遗漏的分类变量
    for col in df_new.columns:
        if col in encodings:
            # 列在编码字典中 - 应用编码
            mapping = encodings[col]
            if df_new[col].dtype == "object":
                unknown = set(df_new[col].dropna().unique()) - set(mapping.keys())
                if unknown:
                    raise ValueError(f"设计空间列 {col} 出现未知类别: {unknown}")
                df_new[col] = df_new[col].map(mapping)
            elif df_new[col].dtype == "bool":
                df_new[col] = df_new[col].astype(int)
        else:
            # 列不在编码字典中 - 检查是否为分类变量（这是错误）
            if df_new[col].dtype == "object":
                raise ValueError(
                    f"设计空间列 '{col}' 是分类变量，但训练数据中该列为数值型。"
                    f"请确保训练数据和设计空间的列类型一致。"
                    f"当前值示例: {df_new[col].head().tolist()}"
                )
            elif df_new[col].dtype == "bool":
                # 布尔型也需要转换
                df_new[col] = df_new[col].astype(int)

    return df_new


def _standardize_subject_wise(
    df: pd.DataFrame, subject_col: str, response_col: str
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """被试内 Z-score 标准化。
    仅返回标准化后的 y_norm 以及 subject_stats；不对 X 做任何处理（由外部编码）。"""
    subject_stats: Dict[str, Dict[str, float]] = {}
    y = df[response_col].values.astype(float)
    subjects = df[subject_col].astype(str).values
    y_norm = np.zeros_like(y)
    global_std = float(np.std(y)) + 1e-6
    for subj in np.unique(subjects):
        mask = subjects == subj
        y_subj = y[mask]
        mean_subj = float(np.mean(y_subj))
        std_subj = float(np.std(y_subj))
        adj_std = std_subj if std_subj > 1e-8 else global_std
        y_norm[mask] = (y_subj - mean_subj) / (adj_std + 1e-12)
        subject_stats[subj] = {
            "mean": mean_subj,
            "std": std_subj,
            "adjusted_std_used": adj_std,
            "n": int(mask.sum()),
        }
    return y_norm.astype(float), subject_stats


class _MaternARDGP(gpytorch.models.ExactGP):
    """自定义 Matern 2.5 + ARD 精确 GP."""

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=train_x.shape[-1],
            )
        )

    def forward(self, x: Tensor):  # type: ignore
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_base_gp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    max_iters: int = 300,
    lr: float = 0.05,
    use_cuda: bool = True,
) -> Tuple[_MaternARDGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    """训练自定义 Matern2.5+ARD GP。返回模型, likelihood, 训练日志。"""
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(train_x).float().to(device)
    y = torch.from_numpy(train_y).float().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = _MaternARDGP(X, y, likelihood).to(device)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=lr)
    mll = ExactMarginalLogLikelihood(likelihood, model)
    log_history: List[Dict[str, float]] = []

    for it in range(1, max_iters + 1):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        if it % 25 == 0 or it == 1:
            lengthscales = (
                model.covar_module.base_kernel.lengthscale.detach()
                .cpu()
                .numpy()
                .ravel()
                .tolist()
            )
            log_history.append(
                {
                    "iter": it,
                    "loss": float(loss.item()),
                    "noise": float(model.likelihood.noise.item()),
                    "lengthscale_mean": float(np.mean(lengthscales)),
                }
            )
    model.eval()
    likelihood.eval()
    return model, likelihood, {"device": str(device), "history": log_history}


def scan_design_space(
    model: _MaternARDGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    design_x: np.ndarray,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """批量预测设计空间 (均值, 标准差)。"""
    device = next(model.parameters()).device
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, design_x.shape[0], batch_size):
            end = start + batch_size
            Xbatch = torch.from_numpy(design_x[start:end]).float().to(device)
            pred = likelihood(model(Xbatch))
            means.append(pred.mean.cpu().numpy())
            stds.append(pred.stddev.cpu().numpy())
    mean_all = np.concatenate(means, axis=0)
    std_all = np.concatenate(stds, axis=0)
    return mean_all, std_all


def select_key_points(
    design_df_encoded: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
    ensure_diversity: bool = True,
) -> Dict[str, Any]:
    """选择三个关键点。若最大 std < 1e-6 则用"中心点"替代不确定点。

    Args:
        ensure_diversity: 若True，若Sample 3与Sample 1/2重复，则选Std第二高的点。
    """
    idx_best = int(np.argmax(means))
    idx_worst = int(np.argmin(means))
    max_std = float(np.max(stds))
    center_point = design_df_encoded.median(numeric_only=True).to_dict()

    # 初始选择：最大std的点
    if max_std < 1e-6:
        idx_std = -1  # 标记使用中心点
        max_std_mean = None
    else:
        idx_std = int(np.argmax(stds))
        max_std_mean = float(means[idx_std])

        # 若启用多样性检查：确保Sample 3与Sample 1/2不重复
        if ensure_diversity and idx_std in (idx_best, idx_worst):
            # 找第二高、第三高等的std点（且不是best/worst）
            sorted_indices = np.argsort(-stds)  # 降序排列
            for candidate_idx in sorted_indices:
                if candidate_idx not in (idx_best, idx_worst):
                    idx_std = int(candidate_idx)
                    max_std = float(stds[idx_std])
                    max_std_mean = float(means[idx_std])
                    break

    return {
        "x_best_prior_index": idx_best,
        "x_best_prior": design_df_encoded.iloc[idx_best].to_dict(),
        "best_mean": float(means[idx_best]),
        "best_std": float(stds[idx_best]),
        "x_worst_prior_index": idx_worst,
        "x_worst_prior": design_df_encoded.iloc[idx_worst].to_dict(),
        "worst_mean": float(means[idx_worst]),
        "worst_std": float(stds[idx_worst]),
        "x_max_std_index": idx_std,
        "x_max_std": (
            design_df_encoded.iloc[idx_std].to_dict() if idx_std >= 0 else center_point
        ),
        "max_std": max_std if idx_std >= 0 else max_std,
        "max_std_mean": max_std_mean,
        "used_center_point": idx_std == -1,
        "center_point": center_point,
        "ensure_diversity": ensure_diversity,
    }


def write_report(
    path: Path,
    factor_names: List[str],
    lengthscales: List[float],
    subject_stats: Dict[str, Dict[str, float]],
    key_points: Dict[str, Any],
    train_meta: Dict[str, Any],
):
    """生成 Markdown 报告。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Base GP (Matern 2.5 + ARD) 报告\n\n")
        f.write("## 📐 模型结构\n")
        f.write("- Kernel: Matern(ν=2.5) + ARD + Scale\n")
        f.write("- 输入维度: {}\n".format(len(factor_names)))
        f.write("- 设备: {}\n".format(train_meta.get("device")))
        f.write("\n## 🔧 训练摘要\n")
        hist = train_meta.get("history", [])
        if hist:
            f.write(
                "| Iter | Loss | Noise | Mean Lengthscale |\n|------|------|-------|------------------|\n"
            )
            for row in hist:
                f.write(
                    f"| {row['iter']} | {row['loss']:.3f} | {row['noise']:.3e} | {row['lengthscale_mean']:.3f} |\n"
                )
        f.write("\n## 🎛️ 长度尺度 (Sensitivity)\n")
        ranked = sorted(zip(factor_names, lengthscales), key=lambda x: x[1])
        f.write(
            "| Rank | Factor | Lengthscale | Interpretation |\n|------|--------|------------:|---------------|\n"
        )
        for rank, (name, ls) in enumerate(ranked, 1):
            interp = (
                "高敏感 (变化小即影响大)"
                if rank <= max(1, len(ranked) // 3)
                else ("中等" if rank <= 2 * len(ranked) // 3 else "低敏感")
            )
            f.write(f"| {rank} | {name} | {ls:.4f} | {interp} |\n")
        f.write("\n## 👥 被试标准化统计\n")
        f.write(
            "| Subject | Mean | Std | Adjusted_Std_Used | N |\n|---------|------|-----|-------------------|---|\n"
        )
        for subj, stats in subject_stats.items():
            f.write(
                f"| {subj} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['adjusted_std_used']:.3f} | {stats['n']} |\n"
            )
        f.write("\n## 📍 关键点 (设计空间) - 三个采样点\n")
        f.write("*供 Phase 2 直接使用的三个关键参数配方*\n\n")

        # Sample 1: Best Prior
        best_coords = key_points["x_best_prior"]
        best_coord_list = [best_coords[f] for f in factor_names]
        f.write("### 1️⃣ Sample 1 (Best Prior)\n")
        f.write(
            "- **Score**: Mean = {:.3f} (Std = {:.3f})\n".format(
                key_points["best_mean"], key_points["best_std"]
            )
        )
        f.write("- **Coordinates**: {}\n".format(best_coord_list))
        f.write("- **Detailed**: ")
        f.write(", ".join([f"{name}={best_coords[name]}" for name in factor_names]))
        f.write("\n\n")

        # Sample 2: Worst Prior
        worst_coords = key_points["x_worst_prior"]
        worst_coord_list = [worst_coords[f] for f in factor_names]
        f.write("### 2️⃣ Sample 2 (Worst Prior)\n")
        f.write(
            "- **Score**: Mean = {:.3f} (Std = {:.3f})\n".format(
                key_points["worst_mean"], key_points["worst_std"]
            )
        )
        f.write("- **Coordinates**: {}\n".format(worst_coord_list))
        f.write("- **Detailed**: ")
        f.write(", ".join([f"{name}={worst_coords[name]}" for name in factor_names]))
        f.write("\n\n")

        # Sample 3: Max Uncertainty
        max_std_coords = key_points["x_max_std"]
        max_std_coord_list = [max_std_coords[f] for f in factor_names]
        f.write("### 3️⃣ Sample 3 (Max Uncertainty / Center)\n")
        if key_points["used_center_point"]:
            f.write(
                "⚠️  **Note**: All points have very low variance (<1e-6), using design space center instead\n\n"
            )
            f.write("- **Score**: Center Point (Std ≈ 0)\n")
        else:
            f.write(
                "- **Score**: Std = {:.3f} (Mean = {:.3f})\n".format(
                    key_points["max_std"], key_points.get("max_std_mean", 0.0)
                )
            )
        f.write("- **Coordinates**: {}\n".format(max_std_coord_list))
        f.write("- **Detailed**: ")
        f.write(", ".join([f"{name}={max_std_coords[name]}" for name in factor_names]))
        f.write("\n\n")
        f.write("\n## 🧪 使用示例\n")
        f.write(
            "```python\nimport torch, json, gpytorch\nfrom phase1_step3_base_gp import _MaternARDGP\n# 加载 state_dict\nstate = torch.load('base_gp_state.pth', map_location='cpu')\n# 重建模型 (需知道输入维度)\nD = {d}\nlikelihood = gpytorch.likelihoods.GaussianLikelihood()\nmodel = _MaternARDGP(torch.zeros(1, D), torch.zeros(1), likelihood)\nmodel.load_state_dict(state['model'])\nlikelihood.load_state_dict(state['likelihood'])\nmodel.eval(); likelihood.eval()\n# 预测\nwith torch.no_grad():\n    x = torch.randn(5, D)\n    pred = likelihood(model(x))\n    print(pred.mean, pred.stddev)\n```\n".format(
                d=len(factor_names)
            )
        )
        f.write("\n*自动生成*\n")


def run_step3_interactive():  # pragma: no cover - 交互主入口
    print("=" * 80)
    print("Phase 1 Step3: Base GP 构建与扫描")
    print("=" * 80)
    data_csv = (
        input("Phase1 数据CSV路径 (含响应) [default warmup_data.csv]: ").strip()
        or "warmup_data.csv"
    )
    design_csv = (
        input("设计空间CSV路径 [default design_space.csv]: ").strip()
        or "design_space.csv"
    )
    subject_col = input("被试列名 [default subject_id]: ").strip() or "subject_id"
    response_col = input("响应列名 [default response]: ").strip() or "response"
    output_dir = (
        input("输出目录 [default base_gp_output]: ").strip() or "base_gp_output"
    )
    max_iters_str = input("训练迭代数 [default 300]: ").strip() or "300"
    lr_str = input("学习率 [default 0.05]: ").strip() or "0.05"
    use_cuda = input("使用CUDA? (Y/n): ").strip().lower() != "n"
    try:
        max_iters = int(max_iters_str)
        lr = float(lr_str)
    except ValueError:
        print("[错误] 参数格式不正确")
        sys.exit(1)
    process_step3(
        data_csv_path=data_csv,
        design_space_csv=design_csv,
        subject_col=subject_col,
        response_col=response_col,
        output_dir=output_dir,
        max_iters=max_iters,
        lr=lr,
        use_cuda=use_cuda,
    )


def process_step3(
    data_csv_path: str,
    design_space_csv: str,
    subject_col: str,
    response_col: str,
    output_dir: str,
    max_iters: int = 300,
    lr: float = 0.05,
    use_cuda: bool = True,
    ensure_diversity: bool = True,
) -> Dict[str, Any]:
    """核心流程 (供 quick_start 调用)。

    Args:
        data_csv_path: Phase1 数据路径
                      - 如果是文件: 直接读取（需包含subject_col和response_col）
                      - 如果是目录: 读取所有subject_*.csv，每个文件代表一个被试
        ensure_diversity: 若True，若Sample 3与Sample 1/2重复，则选Std第二高的点。
    """
    data_path = Path(data_csv_path)
    design_path = Path(design_space_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Phase1 数据路径不存在: {data_csv_path}")
    if not design_path.exists():
        raise FileNotFoundError(f"设计空间文件不存在: {design_space_csv}")

    # 检查是文件还是目录
    if data_path.is_dir():
        # 目录模式：读取所有 subject_*.csv
        print(f"[Step3] 从目录读取被试数据: {data_csv_path}")
        subject_csvs = sorted(data_path.glob("subject_*.csv"))

        if not subject_csvs:
            raise FileNotFoundError(f"目录中未找到 subject_*.csv 文件: {data_csv_path}")

        print(f"  找到 {len(subject_csvs)} 个被试文件")

        # 读取每个被试文件并添加subject列
        all_dfs = []
        for csv_path in subject_csvs:
            df_subject = pd.read_csv(csv_path)

            # 验证响应列存在
            if response_col not in df_subject.columns:
                raise ValueError(f"文件 {csv_path.name} 中未找到响应列: '{response_col}'")

            # 添加被试列（如果不存在）
            if subject_col not in df_subject.columns:
                subject_id = csv_path.stem  # "subject_1"
                df_subject.insert(0, subject_col, subject_id)

            all_dfs.append(df_subject)
            print(f"    - {csv_path.name}: {len(df_subject)} 行")

        # 合并所有数据
        df_phase1 = pd.concat(all_dfs, ignore_index=True)
        print(f"  合并后总计: {len(df_phase1)} 行")
    else:
        # 文件模式：直接读取
        print(f"[Step3] 读取数据文件: {data_csv_path}")
        df_phase1 = pd.read_csv(data_path)

    if subject_col not in df_phase1.columns or response_col not in df_phase1.columns:
        raise ValueError("Phase1 数据缺少必要列")

    factor_cols = [c for c in df_phase1.columns if c not in (subject_col, response_col)]
    factor_df = df_phase1[factor_cols]
    encoded_factors, encodings = _encode_factor_df(factor_df)

    # 如果是目录模式，尝试从采样方案推断额外的编码（用于设计空间）
    if data_path.is_dir():
        print("\n[推断] 从采样方案推断分类变量编码...")
        inferred_encodings = _infer_encoding_from_sampling(data_path, factor_cols)
        # 合并推断的编码（优先使用推断的，因为它包含完整的categorical->numeric映射）
        for col, mapping in inferred_encodings.items():
            if col not in encodings or not encodings[col]:
                encodings[col] = mapping
                print(f"  使用推断编码: {col}")
            else:
                print(f"  列 {col} 已有编码，跳过推断")

    # 标准化 (使用原始未编码因子, 但我们只需要 y_norm 与 X 编码后的数值)
    X_numeric = encoded_factors
    df_for_std = df_phase1[[subject_col, response_col] + factor_cols]
    y_norm, subject_stats = _standardize_subject_wise(
        df_phase1[[subject_col, response_col]], subject_col, response_col
    )
    X_train = X_numeric.values.astype(float)

    model, likelihood, train_meta = train_base_gp(
        X_train, y_norm, max_iters=max_iters, lr=lr, use_cuda=use_cuda
    )
    lengthscales = (
        model.covar_module.base_kernel.lengthscale.detach()
        .cpu()
        .numpy()
        .ravel()
        .tolist()
    )

    # 扫描设计空间
    design_df_raw = pd.read_csv(design_path)
    # 只取与训练相同的因子列, 丢弃其它列
    missing_cols = set(factor_cols) - set(design_df_raw.columns)
    if missing_cols:
        raise ValueError(f"设计空间缺少因子列: {missing_cols}")
    design_df_aligned = design_df_raw[factor_cols]

    # Debug: 检查编码前的数据类型
    print("\n[Debug] 设计空间编码前:")
    for col in design_df_aligned.columns:
        print(f"  {col}: dtype={design_df_aligned[col].dtype}, "
              f"in_encodings={col in encodings}, "
              f"sample_values={design_df_aligned[col].head(3).tolist()}")

    design_df_encoded = _apply_encodings(design_df_aligned, encodings)

    # Debug: 检查编码后的数据类型
    print("\n[Debug] 设计空间编码后:")
    for col in design_df_encoded.columns:
        print(f"  {col}: dtype={design_df_encoded[col].dtype}, "
              f"sample_values={design_df_encoded[col].head(3).tolist()}")

    means, stds = scan_design_space(
        model, likelihood, design_df_encoded.values.astype(float)
    )
    key_points = select_key_points(
        design_df_encoded, means, stds, ensure_diversity=ensure_diversity
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 保存 state_dict
    torch.save(
        {"model": model.state_dict(), "likelihood": likelihood.state_dict()},
        out_dir / "base_gp_state.pth",
    )
    # 其它 JSON
    (out_dir / "base_gp_lengthscales.json").write_text(
        json.dumps(
            {"factor_names": factor_cols, "lengthscales": lengthscales},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "base_gp_subject_stats.json").write_text(
        json.dumps(subject_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "base_gp_encodings.json").write_text(
        json.dumps(encodings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "base_gp_key_points.json").write_text(
        json.dumps(key_points, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 设计空间扫描 CSV
    scan_df = design_df_encoded.copy()
    scan_df["pred_mean"] = means
    scan_df["pred_std"] = stds
    scan_df.to_csv(out_dir / "design_space_scan.csv", index=False)
    # 报告
    write_report(
        out_dir / "base_gp_report.md",
        factor_cols,
        lengthscales,
        subject_stats,
        key_points,
        train_meta,
    )

    return {
        "output_dir": str(out_dir),
        "lengthscales": lengthscales,
        "key_points": key_points,
        "n_design_points": int(design_df_encoded.shape[0]),
    }


def main():  # pragma: no cover
    run_step3_interactive()


if __name__ == "__main__":  # pragma: no cover
    main()
