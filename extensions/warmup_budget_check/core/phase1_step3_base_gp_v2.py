#!/usr/bin/env python3
"""
Phase 1 Step3: Base GP 构建与设计空间扫描 (支持连续型/序数型)

功能概述:
1. 读取 Phase1 数据集 (含因子与响应)
2. 根据模型类型进行数据预处理:
   - 连续型 (continuous): 被试内 Z-score 标准化
   - 序数型 (ordinal): 转换为 0-indexed 类别标签 (例如 Likert 1-5 -> 0-4)
3. 训练 GP 模型:
   - 连续型: Matern ν=2.5 Kernel + ARD + GaussianLikelihood (精确推断)
   - 序数型: AEPsych OrdinalGPModel (RBF + ARD + OrdinalLikelihood, 变分推断)
4. 扫描用户给定的设计空间 CSV, 计算预测均值/标准差
5. 选出: 全局最高点 x_best_prior, 全局最低点 x_worst_prior, 最不确定点 x_max_std
6. 导出模型 state_dict, 长度尺度, 关键点, 设计空间扫描结果, 报告

模型类型选择:
  - model_type='continuous': 适用于连续响应变量 (如真实值测量)
  - model_type='ordinal': 适用于序数响应变量 (如 Likert 量表 1-5)

使用方式(配置调用, 推荐结合 quick_start.py):
  在 quick_start.py 中设置 MODE='step3' 并填写 STEP3_CONFIG

文件输出(默认 output_dir=base_gp_output):
  base_gp_state.pth               模型与likelihood state_dict (连续型) 或完整模型 (序数型)
  base_gp_lengthscales.json       长度尺度与敏感度排序
  base_gp_subject_stats.json      被试标准化统计 (连续型) / 类别映射 (序数型)
  base_gp_encodings.json          分类变量编码映射
  base_gp_key_points.json         三个关键点及预测值
  design_space_scan.csv           设计空间逐点预测 (mean,std)
  base_gp_report.md               报告摘要

依赖: torch, gpytorch, botorch, aepsych (在当前 pixi 环境中已满足)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Literal, Union

import numpy as np
import pandas as pd

try:
    import torch
    from torch import Tensor
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.optim.fit import fit_gpytorch_mll_torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from aepsych.models import OrdinalGPModel
    from aepsych.likelihoods import OrdinalLikelihood
except Exception as e:  # pragma: no cover
    print(f"[错误] 需要安装 torch/gpytorch/botorch/aepsych: {e}")
    sys.exit(1)


# ============================================================================
# Utility Functions (共用)
# ============================================================================

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
    """将已存在的编码映射应用到新的 DataFrame (设计空间)."""
    df_new = df.copy()
    for col, mapping in encodings.items():
        if col not in df_new.columns:
            raise ValueError(f"设计空间缺失因子列: {col}")
        if df_new[col].dtype == "object":
            unknown = set(df_new[col].dropna().unique()) - set(mapping.keys())
            if unknown:
                raise ValueError(f"设计空间列 {col} 出现未知类别: {unknown}")
            df_new[col] = df_new[col].map(mapping)
        elif df_new[col].dtype == "bool":
            df_new[col] = df_new[col].astype(int)
    return df_new


# ============================================================================
# Continuous GP (原有实现)
# ============================================================================

def _standardize_subject_wise(
    df: pd.DataFrame, subject_col: str, response_col: str
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """被试内 Z-score 标准化 (仅用于连续型模型)."""
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
    """自定义 Matern 2.5 + ARD 精确 GP (连续型)."""

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


def train_continuous_gp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    max_iters: int = 300,
    lr: float = 0.05,
    use_cuda: bool = True,
) -> Tuple[_MaternARDGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    """训练连续型 Matern2.5+ARD GP."""
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


def scan_design_space_continuous(
    model: _MaternARDGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    design_x: np.ndarray,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """批量预测设计空间 (连续型 GP)."""
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


# ============================================================================
# Ordinal GP (新增实现)
# ============================================================================

def convert_to_ordinal_labels(
    y_raw: np.ndarray, min_level: int = 1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """将 Likert 量表值转换为 0-indexed 类别标签.

    例如: Likert 1-5 -> 0-4

    Args:
        y_raw: 原始响应值 (如 1,2,3,4,5)
        min_level: 最小级别值 (默认 1)

    Returns:
        y_ordinal: 0-indexed 标签 (如 0,1,2,3,4)
        mapping: 映射信息字典
    """
    unique_levels = np.sort(np.unique(y_raw))
    n_levels = len(unique_levels)

    # 创建映射: 原始值 -> 0-indexed
    level_to_index = {level: i for i, level in enumerate(unique_levels)}
    y_ordinal = np.array([level_to_index[val] for val in y_raw])

    mapping = {
        "n_levels": n_levels,
        "unique_levels": unique_levels.tolist(),
        "level_to_index": {int(k): int(v) for k, v in level_to_index.items()},
        "index_to_level": {int(v): int(k) for k, v in level_to_index.items()},
        "min_level": int(unique_levels[0]),
        "max_level": int(unique_levels[-1]),
    }

    return y_ordinal, mapping


def train_ordinal_gp(
    train_x: np.ndarray,
    train_y_ordinal: np.ndarray,
    n_levels: int,
    inducing_size: int = 100,
    use_cuda: bool = True,
) -> Tuple[OrdinalGPModel, Dict[str, Any]]:
    """训练序数型 GP (使用 AEPsych OrdinalGPModel).

    Args:
        train_x: 输入特征 (n_samples, n_features)
        train_y_ordinal: 0-indexed 序数标签 (n_samples,)
        n_levels: 序数级别数量
        inducing_size: 诱导点数量
        use_cuda: 是否使用 CUDA

    Returns:
        model: 训练好的序数 GP 模型
        train_info: 训练信息字典
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    # 确保 dtype 一致性：使用 double (float64)
    X = torch.from_numpy(train_x.astype(np.float64)).double()
    y = torch.from_numpy(train_y_ordinal.astype(np.int64)).long()

    # 创建 OrdinalLikelihood
    likelihood = OrdinalLikelihood(n_levels=n_levels)

    # 创建模型
    model = OrdinalGPModel(
        dim=train_x.shape[1],
        likelihood=likelihood,
        inducing_size=min(inducing_size, train_x.shape[0]),
    )

    # 训练
    print(f"[INFO] 训练序数 GP: {train_x.shape[0]} 样本, {train_x.shape[1]} 维, {n_levels} 个类别")
    model.fit(X, y)

    # 提取训练信息
    # 获取 lengthscales (尝试多个可能的路径)
    try:
        lengthscales = (
            model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        )
    except:
        try:
            # OrdinalGPModel 的 covar_module 可能直接就是 RBF kernel
            lengthscales = (
                model.covar_module.lengthscale.detach().cpu().numpy().ravel()
            )
        except:
            lengthscales = np.array([])

    # 获取 cutpoints
    try:
        cutpoints = model.likelihood.cutpoints.detach().cpu().numpy()
    except:
        cutpoints = np.array([])

    train_info = {
        "device": str(device),
        "n_inducing": inducing_size,
        "lengthscales": lengthscales.tolist() if len(lengthscales) > 0 else [],
        "cutpoints": cutpoints.tolist() if len(cutpoints) > 0 else [],
    }

    return model, train_info


def scan_design_space_ordinal(
    model: OrdinalGPModel,
    design_x: np.ndarray,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """批量预测设计空间 (序数型 GP).

    返回潜在函数的均值和标准差（而非概率分布）。
    """
    means: List[np.ndarray] = []
    stds: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, design_x.shape[0], batch_size):
            end = start + batch_size
            # 确保 dtype 一致性：使用 double (float64)
            Xbatch = torch.from_numpy(design_x[start:end].astype(np.float64)).double()

            # 使用 predict 方法获取潜在函数的均值和方差
            fmean, fvar = model.predict(Xbatch)
            fstd = torch.sqrt(fvar)

            means.append(fmean.cpu().numpy())
            stds.append(fstd.cpu().numpy())

    mean_all = np.concatenate(means, axis=0)
    std_all = np.concatenate(stds, axis=0)
    return mean_all, std_all


# ============================================================================
# Common: Key Points Selection & Report
# ============================================================================

def select_key_points(
    design_df_encoded: pd.DataFrame,
    means: np.ndarray,
    stds: np.ndarray,
    ensure_diversity: bool = True,
) -> Dict[str, Any]:
    """选择三个关键点."""
    idx_best = int(np.argmax(means))
    idx_worst = int(np.argmin(means))
    max_std = float(np.max(stds))
    center_point = design_df_encoded.median(numeric_only=True).to_dict()

    if max_std < 1e-6:
        idx_std = -1
        max_std_mean = None
    else:
        idx_std = int(np.argmax(stds))
        max_std_mean = float(means[idx_std])

        if ensure_diversity and idx_std in (idx_best, idx_worst):
            sorted_indices = np.argsort(-stds)
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
    metadata: Dict[str, Any],
    key_points: Dict[str, Any],
    train_meta: Dict[str, Any],
    model_type: str,
):
    """生成 Markdown 报告."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Base GP 报告 (模型类型: {model_type.upper()})\n\n")

        f.write("## 📐 模型结构\n")
        if model_type == "continuous":
            f.write("- **类型**: 连续型 GP (Exact Inference)\n")
            f.write("- **Kernel**: Matern(ν=2.5) + ARD + Scale\n")
            f.write("- **Likelihood**: GaussianLikelihood\n")
        else:  # ordinal
            f.write("- **类型**: 序数型 GP (Variational Inference)\n")
            f.write("- **Kernel**: RBF + ARD\n")
            f.write("- **Likelihood**: OrdinalLikelihood\n")
            if "n_levels" in metadata:
                f.write(f"- **序数级别**: {metadata['n_levels']} 个类别\n")
                if "unique_levels" in metadata:
                    f.write(f"- **原始值范围**: {metadata['unique_levels']}\n")

        f.write(f"- **输入维度**: {len(factor_names)}\n")
        f.write(f"- **设备**: {train_meta.get('device')}\n")

        f.write("\n## 🔧 训练摘要\n")
        if model_type == "continuous":
            hist = train_meta.get("history", [])
            if hist:
                f.write("| Iter | Loss | Noise | Mean Lengthscale |\n")
                f.write("|------|------|-------|------------------|\n")
                for row in hist:
                    f.write(
                        f"| {row['iter']} | {row['loss']:.3f} | {row['noise']:.3e} | {row['lengthscale_mean']:.3f} |\n"
                    )
        else:  # ordinal
            f.write(f"- **诱导点数量**: {train_meta.get('n_inducing', 'N/A')}\n")
            if "cutpoints" in train_meta and train_meta["cutpoints"]:
                f.write(f"- **Cutpoints**: {train_meta['cutpoints']}\n")

        f.write("\n## 🎛️ 长度尺度 (Sensitivity)\n")
        if lengthscales:
            ranked = sorted(zip(factor_names, lengthscales), key=lambda x: x[1])
            f.write("| Rank | Factor | Lengthscale | Interpretation |\n")
            f.write("|------|--------|------------:|---------------|\n")
            for rank, (name, ls) in enumerate(ranked, 1):
                interp = (
                    "高敏感 (变化小即影响大)"
                    if rank <= max(1, len(ranked) // 3)
                    else ("中等" if rank <= 2 * len(ranked) // 3 else "低敏感")
                )
                f.write(f"| {rank} | {name} | {ls:.4f} | {interp} |\n")
        else:
            f.write("*长度尺度信息不可用*\n")

        f.write("\n## 📍 关键点 (设计空间)\n")
        for i, (label, key_prefix) in enumerate([
            ("Best Prior", "best"),
            ("Worst Prior", "worst"),
            ("Max Uncertainty", "uncertain"),
        ], 1):
            if key_prefix == "uncertain":
                coords = key_points["x_max_std"]
                mean_val = key_points.get("max_std_mean", 0.0)
                std_val = key_points["max_std"]
                is_center = key_points.get("used_center_point", False)
            else:
                coords = key_points[f"x_{key_prefix}_prior"]
                mean_val = key_points[f"{key_prefix}_mean"]
                std_val = key_points[f"{key_prefix}_std"]
                is_center = False

            coord_list = [coords[f] for f in factor_names]
            f.write(f"\n### {i} Sample {i} ({label})\n")
            if is_center:
                f.write("Note: Using center point (low variance)\n")
            f.write(f"- **Score**: Mean={mean_val:.3f}, Std={std_val:.3f}\n")
            f.write(f"- **Coordinates**: {coord_list}\n")

        f.write("\n*自动生成*\n")


# ============================================================================
# Main Processing Function
# ============================================================================

def process_step3(
    data_csv_path: str,
    design_space_csv: str,
    subject_col: str,
    response_col: str,
    output_dir: str,
    model_type: Literal["continuous", "ordinal"] = "continuous",
    max_iters: int = 300,
    lr: float = 0.05,
    use_cuda: bool = True,
    ensure_diversity: bool = True,
    inducing_size: int = 100,
    ordinal_min_level: int = 1,
) -> Dict[str, Any]:
    """核心流程: 支持连续型/序数型 GP.

    Args:
        data_csv_path: Phase1 数据 CSV 路径
        design_space_csv: 设计空间 CSV 路径
        subject_col: 被试列名
        response_col: 响应列名
        output_dir: 输出目录
        model_type: 模型类型 ('continuous' | 'ordinal')
        max_iters: 训练迭代数 (仅连续型)
        lr: 学习率 (仅连续型)
        use_cuda: 是否使用 CUDA
        ensure_diversity: 确保关键点多样性
        inducing_size: 诱导点数量 (仅序数型)
        ordinal_min_level: 序数最小级别值 (如 Likert 1-5 则为 1)
    """
    data_path = Path(data_csv_path)
    design_path = Path(design_space_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Phase1 数据文件不存在: {data_csv_path}")
    if not design_path.exists():
        raise FileNotFoundError(f"设计空间文件不存在: {design_space_csv}")

    df_phase1 = pd.read_csv(data_path)
    if subject_col not in df_phase1.columns or response_col not in df_phase1.columns:
        raise ValueError("Phase1 数据缺少必要列")

    factor_cols = [c for c in df_phase1.columns if c not in (subject_col, response_col)]
    factor_df = df_phase1[factor_cols]
    encoded_factors, encodings = _encode_factor_df(factor_df)
    X_train = encoded_factors.values.astype(float)

    print(f"\n{'='*80}")
    print(f"训练模型类型: {model_type.upper()}")
    print(f"{'='*80}")

    # ========== 根据模型类型选择不同的训练路径 ==========
    if model_type == "continuous":
        # 连续型: Z-score 标准化
        y_norm, subject_stats = _standardize_subject_wise(
            df_phase1[[subject_col, response_col]], subject_col, response_col
        )
        metadata = {"subject_stats": subject_stats}

        # 训练连续型 GP
        model, likelihood, train_meta = train_continuous_gp(
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
        missing_cols = set(factor_cols) - set(design_df_raw.columns)
        if missing_cols:
            raise ValueError(f"设计空间缺少因子列: {missing_cols}")
        design_df_aligned = design_df_raw[factor_cols]
        design_df_encoded = _apply_encodings(design_df_aligned, encodings)

        means, stds = scan_design_space_continuous(
            model, likelihood, design_df_encoded.values.astype(float)
        )

        # 保存模型
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model": model.state_dict(), "likelihood": likelihood.state_dict()},
            out_dir / "base_gp_state.pth",
        )

    else:  # ordinal
        # 序数型: 转换为 0-indexed
        y_raw = df_phase1[response_col].values
        y_ordinal, ordinal_mapping = convert_to_ordinal_labels(y_raw, ordinal_min_level)
        metadata = {"ordinal_mapping": ordinal_mapping, "n_levels": ordinal_mapping["n_levels"]}

        # 训练序数型 GP
        model, train_meta = train_ordinal_gp(
            X_train,
            y_ordinal,
            n_levels=ordinal_mapping["n_levels"],
            inducing_size=inducing_size,
            use_cuda=use_cuda,
        )

        lengthscales = train_meta.get("lengthscales", [])

        # 扫描设计空间
        design_df_raw = pd.read_csv(design_path)
        missing_cols = set(factor_cols) - set(design_df_raw.columns)
        if missing_cols:
            raise ValueError(f"设计空间缺少因子列: {missing_cols}")
        design_df_aligned = design_df_raw[factor_cols]
        design_df_encoded = _apply_encodings(design_df_aligned, encodings)

        means, stds = scan_design_space_ordinal(
            model, design_df_encoded.values.astype(float)
        )

        # 保存模型 (序数型保存整个模型)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model, out_dir / "base_gp_state.pth")
        metadata["unique_levels"] = ordinal_mapping["unique_levels"]

    # ========== 通用后处理 ==========
    key_points = select_key_points(
        design_df_encoded, means, stds, ensure_diversity=ensure_diversity
    )

    # 保存其他输出
    (out_dir / "base_gp_lengthscales.json").write_text(
        json.dumps(
            {"factor_names": factor_cols, "lengthscales": lengthscales},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out_dir / "base_gp_subject_stats.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "base_gp_encodings.json").write_text(
        json.dumps(encodings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "base_gp_key_points.json").write_text(
        json.dumps(key_points, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    scan_df = design_df_encoded.copy()
    scan_df["pred_mean"] = means
    scan_df["pred_std"] = stds
    scan_df.to_csv(out_dir / "design_space_scan.csv", index=False)

    write_report(
        out_dir / "base_gp_report.md",
        factor_cols,
        lengthscales,
        metadata,
        key_points,
        train_meta,
        model_type,
    )

    print(f"\n[OK] Output saved to: {out_dir}")
    return {
        "output_dir": str(out_dir),
        "model_type": model_type,
        "lengthscales": lengthscales,
        "key_points": key_points,
        "n_design_points": int(design_df_encoded.shape[0]),
    }


def main():  # pragma: no cover
    """交互式主入口 (示例)."""
    print("="*80)
    print("Phase 1 Step3: Base GP 构建与扫描 (v2 - 支持序数/连续)")
    print("="*80)
    data_csv = input("Phase1 数据CSV路径: ").strip() or "warmup_data.csv"
    design_csv = input("设计空间CSV路径: ").strip() or "design_space.csv"
    subject_col = input("被试列名 [subject_id]: ").strip() or "subject_id"
    response_col = input("响应列名 [response]: ").strip() or "response"
    model_type = input("模型类型 (continuous/ordinal) [continuous]: ").strip() or "continuous"
    output_dir = input("输出目录 [base_gp_output]: ").strip() or "base_gp_output"

    if model_type not in ("continuous", "ordinal"):
        print(f"[错误] 模型类型必须是 'continuous' 或 'ordinal', 得到: {model_type}")
        sys.exit(1)

    process_step3(
        data_csv_path=data_csv,
        design_space_csv=design_csv,
        subject_col=subject_col,
        response_col=response_col,
        output_dir=output_dir,
        model_type=model_type,  # type: ignore
    )


if __name__ == "__main__":  # pragma: no cover
    main()
