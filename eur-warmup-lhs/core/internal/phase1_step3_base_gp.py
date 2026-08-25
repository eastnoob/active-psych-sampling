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
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

# 必须在导入torch之前设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import numpy as np
import pandas as pd
from loguru import logger

try:
    import torch
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    from torch import Tensor
    import gpytorch
    from botorch.models import SingleTaskGP
    from botorch.optim.fit import fit_gpytorch_mll_torch
    from gpytorch.mlls import ExactMarginalLogLikelihood
except Exception as e:  # pragma: no cover - 环境导入失败时的提示
    logger.error(f"torch/gpytorch/botorch required: {e}")
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
    result_file = data_dir / "subject_1.csv"
    
    # 尝试多种可能的采样方案路径
    sampling_candidates = [
        data_dir.parent / "step1" / "subject_1.csv", # 标准结构
        data_dir.parent / "subject_1.csv",           # 扁平结构
        data_dir / "subject_1_raw.csv"               # 同目录备份
    ]
    
    sampling_file = None
    for cand in sampling_candidates:
        if cand.exists():
            sampling_file = cand
            break

    if not sampling_file or not result_file.exists():
        logger.warning(f"Unable to infer encoding: reference files not found at {data_dir}")
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
                    logger.warning(f"Inconsistent encoding for column {col}: {cat_val} -> {mapping[cat_val]} vs {num_val}")

            if mapping:
                encodings[col] = mapping
                logger.info(f"Inferred encoding for {col}: {mapping}")

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
                    raise ValueError(f"Unknown category in design space column {col}: {unknown}")
                df_new[col] = df_new[col].map(mapping)
            elif df_new[col].dtype == "bool":
                df_new[col] = df_new[col].astype(int)
        else:
            # 列不在编码字典中 - 检查是否为分类变量（这是错误）
            if df_new[col].dtype == "object":
                raise ValueError(
                    f"Design space column '{col}' is categorical, but training data for this column is numeric. "
                    f"Please ensure consistent column types. "
                    f"Example values: {df_new[col].head().tolist()}"
                )
            elif df_new[col].dtype == "bool":
                # 布尔型也需要转换
                df_new[col] = df_new[col].astype(int)

    return df_new


def _standardize_subject_wise(
    df: pd.DataFrame, subject_col: str, response_col: str, mode: str = "zscore"
) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    """被试内标准化。
    
    Args:
        mode: "zscore" (减均值除标准差) 或 "mean_centering" (仅减均值)
    """
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
        
        if mode == "zscore":
            adj_std = std_subj if std_subj > 1e-8 else global_std
            y_norm[mask] = (y_subj - mean_subj) / (adj_std + 1e-12)
        else:  # mean_centering
            y_norm[mask] = y_subj - mean_subj
            adj_std = 1.0
            
        subject_stats[subj] = {
            "mean": mean_subj,
            "std": std_subj,
            "adjusted_std_used": adj_std,
            "n": int(mask.sum()),
            "mode": mode
        }
    return y_norm.astype(float), subject_stats


def _get_x_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算 X 的 min 和 max 用于归一化。"""
    x_min = np.min(X, axis=0)
    x_max = np.max(X, axis=0)
    # 防止除零
    x_range = x_max - x_min
    x_range[x_range < 1e-8] = 1.0
    return x_min, x_max


def _apply_x_normalization(X: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """将 X 归一化到 [0, 1]。"""
    x_range = x_max - x_min
    x_range[x_range < 1e-8] = 1.0
    return (X - x_min) / x_range


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


class _FactoryGP(gpytorch.models.ExactGP):
    """使用 Factory 构建的通用 GP."""

    def __init__(
        self,
        train_x: Tensor,
        train_y: Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        factory: Any,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = factory._make_mean_module()
        self.covar_module = factory._make_covar_module()

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _extract_hyperparameters(model: gpytorch.models.ExactGP, n_dims: int) -> Dict[str, Any]:
    """从 GP 模型中鲁棒地提取真实超参数。"""
    lengthscales = [0.5] * n_dims
    kernel_types = ["Matern"] * n_dims
    outputscales = {"main": [1.0] * n_dims, "interactions": {}, "global": 1.0}
    noise = 0.01
    
    # 0. 提取噪声
    if hasattr(model, "likelihood") and hasattr(model.likelihood, "noise"):
        try:
            noise = float(model.likelihood.noise.detach().cpu().item())
        except Exception:
            pass

    # 1. 提取全局 Outputscale
    if hasattr(model.covar_module, "outputscale"):
        try:
            outputscales["global"] = float(model.covar_module.outputscale.detach().cpu().item())
        except Exception:
            pass

    # 2. 深度遍历寻找各维度的真实现状
    # 我们不仅看名字，还根据模块类型判断
    from gpytorch.kernels import MaternKernel, RBFKernel
    from botorch.models.kernels import CategoricalKernel

    for _, module in model.named_modules():
        # 长度尺度提取
        if hasattr(module, "lengthscale") and module.lengthscale is not None:
            try:
                ls_tensor = module.lengthscale.detach().cpu().numpy().ravel()
                active_dims = getattr(module, "active_dims", None)
                
                # 判定核函数类型
                if isinstance(module, CategoricalKernel):
                    k_name = "Categorical"
                elif isinstance(module, MaternKernel):
                    k_name = f"Matern(v={int(module.nu*2)/2 if hasattr(module, 'nu') else '??'})"
                elif isinstance(module, RBFKernel):
                    k_name = "RBF"
                else:
                    k_name = type(module).__name__

                if active_dims is not None:
                    dims = active_dims.tolist() if torch.is_tensor(active_dims) else list(active_dims)
                    if len(ls_tensor) == len(dims):
                        for i, dim_idx in enumerate(dims):
                            if dim_idx < n_dims:
                                lengthscales[dim_idx] = float(ls_tensor[i])
                                kernel_types[dim_idx] = k_name
                    elif len(ls_tensor) == 1:
                        for dim_idx in dims:
                            if dim_idx < n_dims:
                                lengthscales[dim_idx] = float(ls_tensor[0])
                                kernel_types[dim_idx] = k_name
                elif len(ls_tensor) == n_dims:
                    for i, ls in enumerate(ls_tensor):
                        lengthscales[i] = float(ls)
                        kernel_types[i] = k_name
            except Exception:
                pass

        # 提取输出尺度 (针对 ANOVA 结构)
        if hasattr(module, "outputscale") and module != model.covar_module:
            try:
                os_val = float(module.outputscale.detach().cpu().item())
                # 尝试判断这是主效应还是交互项
                base = getattr(module, "base_kernel", None)
                if base is not None:
                    # 递归获取所有子核的 active_dims
                    def get_all_active_dims(k):
                        if hasattr(k, "active_dims") and k.active_dims is not None:
                            return set(k.active_dims.tolist() if torch.is_tensor(k.active_dims) else k.active_dims)
                        if hasattr(k, "kernels"):
                            res = set()
                            for sub in k.kernels:
                                res.update(get_all_active_dims(sub))
                            return res
                        return set()

                    dims = sorted(list(get_all_active_dims(base)))
                    if len(dims) == 1:
                        outputscales["main"][dims[0]] = os_val
                    elif len(dims) > 1:
                        outputscales["interactions"][str(tuple(dims))] = os_val
            except Exception:
                pass

    return {
        "lengthscales": lengthscales,
        "kernel_types": kernel_types,
        "outputscales": outputscales,
        "noise": noise
    }


def train_base_gp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    max_iters: int = 300,
    lr: float = 0.05,
    use_cuda: bool = True,
    factory: Any = None,
    model_spec: Dict[str, Any] = None,
    factor_names: List[str] = None,
) -> Tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.GaussianLikelihood, Dict[str, Any]]:
    """训练 GP。支持默认 Matern 或通过 factory 自定义。"""
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(train_x).double().to(device)
    y = torch.from_numpy(train_y).double().to(device)
    
    # 提取噪声下界
    noise_lower_bound = 1e-4
    if model_spec and "noise_lower_bound" in model_spec:
        noise_lower_bound = model_spec["noise_lower_bound"]
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(noise_lower_bound)
    ).to(device).double()
    
    if factory is not None:
        model = _FactoryGP(X, y, likelihood, factory).to(device).double()
    else:
        model = _MaternARDGP(X, y, likelihood).to(device).double()
    
    # 应用模型规格 (Priors)
    if model_spec:
        # 1. 噪声初值 (如果指定了 noise_floor)
        if "noise_floor" in model_spec:
            likelihood.noise = model_spec["noise_floor"]
        elif "noise_lower_bound" in model_spec:
            # 如果没指定初值但指定了下界，初值设为下界的2倍
            likelihood.noise = model_spec["noise_lower_bound"] * 2.0
        
        # 2. 长度尺度初值
        if "lengthscale_priors" in model_spec and factor_names:
            ls_priors = model_spec["lengthscale_priors"]
            ls_values = []
            for name in factor_names:
                ls_values.append(ls_priors.get(name, 0.5))

            # 尝试设置长度尺度
            try:
                covar = model.covar_module
                # 解包 ScaleKernel
                if hasattr(covar, "base_kernel"):
                    covar = covar.base_kernel

                # 情况1: Mixed Factory 结构 (mult/add/full 模式)
                if hasattr(covar, "kernels") and len(covar.kernels) >= 2:
                    # 遍历所有子核寻找 Matern (主效应)
                    for k in covar.kernels:
                        # 跳过 ProductKernel (交互项)
                        if hasattr(k, "kernels"):
                            continue
                        # 设置 Matern lengthscale
                        if hasattr(k, "lengthscale") and k.lengthscale is not None:
                            target = k.lengthscale
                            new_ls = torch.tensor(ls_values).to(device).double()
                            if new_ls.numel() == target.numel():
                                k.lengthscale = new_ls.view(target.shape)

                # 情况2: 标准 ARD
                elif hasattr(covar, "lengthscale") and covar.lengthscale is not None:
                    target = covar.lengthscale
                    new_ls = torch.tensor(ls_values).to(device).double()
                    if new_ls.numel() == target.numel():
                        covar.lengthscale = new_ls.view(target.shape)

                # 情况3: AdditiveKernel (ANOVA)
                elif hasattr(covar, "kernels"):
                    # 遍历所有子核，寻找对应的维度并设置初值
                    for sub_k in covar.kernels:
                        # 解包 ScaleKernel
                        target_k = sub_k.base_kernel if hasattr(sub_k, "base_kernel") else sub_k
                        
                        active_dims = getattr(target_k, "active_dims", None)
                        if active_dims is not None and len(active_dims) == 1:
                            dim_idx = int(active_dims[0])
                            if dim_idx < len(ls_values) and hasattr(target_k, "lengthscale"):
                                try:
                                    val = ls_values[dim_idx]
                                    target_k.lengthscale = torch.tensor([[val]]).to(device).double()
                                except Exception:
                                    pass

            except Exception as e:
                logger.warning(f"无法应用长度尺度先验: {e}")

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
            # 使用新定义的鲁棒提取函数
            params = _extract_hyperparameters(model, X.shape[1])
            lengthscales = params["lengthscales"]

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
            Xbatch = torch.from_numpy(design_x[start:end]).double().to(device)
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
    outputscales: Dict[str, Any],
    subject_stats: Dict[str, Dict[str, float]],
    key_points: Dict[str, Any],
    train_meta: Dict[str, Any],
    model_spec: Dict[str, Any] = None,
    interactions_data: Dict[str, Any] = None,
    noise: float = 0.01,
    subject_std: float = 0.0,
    kernel_types: List[str] = None,
):
    """Generate Markdown report."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Base GP Training Report\n\n")
        
        f.write("## 🚀 Active Learning Initialization Parameters\n")
        f.write("这些参数将用于初始化后续的主动学习阶段：\n\n")
        f.write(f"- **Estimated Noise ($\hat{{\sigma}}_n$)**: `{noise:.4f}` (来自重复点)\n")
        f.write(f"- **Subject Variance ($\hat{{\sigma}}_u$)**: `{subject_std:.4f}` (来自锚点跨被试差异)\n")
        f.write("- **Main Effect Importance**: 见下方 Outputscales 部分\n\n")

        f.write("## 📐 Model Structure\n")
        
        if model_spec:
            k_type = model_spec.get("kernel_type", "default")
            if k_type == "custom_anova":
                f.write("- **Kernel Type**: Custom ANOVA Decomposition\n")
                pairs = model_spec.get("interaction_pairs") or []
                f.write(f"- **Interactions**: {len(pairs)} pairs identified in Step 2\n")
                if pairs:
                    f.write("  - Pairs: " + ", ".join([str(p) for p in pairs]) + "\n")
            elif k_type == "custom_basegp_residual_mixed":
                comb = model_spec.get("kernel_combination", "mult")
                f.write(f"- **Kernel Type**: Custom Mixed Factory ({comb})\n")
                f.write("- **Structure**: Matern 2.5 (Cont) & Categorical (Disc)\n")
            else:
                f.write("- **Kernel Type**: Standard Matern 2.5 + ARD\n")
        else:
            f.write("- **Kernel Type**: Standard Matern 2.5 + ARD\n")
            
        f.write("- **Input Dimensions**: {}\n".format(len(factor_names)))
        f.write("- **Device**: {}\n".format(train_meta.get("device")))
        f.write("\n## 🔧 Training Summary\n")
        hist = train_meta.get("history", [])
        if hist:
            f.write(
                "| Iter | Loss | Noise | Mean Lengthscale |\n|------|------|-------|------------------|\n"
            )
            for row in hist:
                f.write(
                    f"| {row['iter']} | {row['loss']:.3f} | {row['noise']:.3e} | {row['lengthscale_mean']:.3f} |\n"
                )
        f.write("\n## 🎛️ Lengthscales (Sensitivity)\n")
        
        # 准备数据，包含真实的核函数类型
        if kernel_types and len(kernel_types) == len(lengthscales):
            report_data = zip(factor_names, lengthscales, kernel_types)
        else:
            report_data = zip(factor_names, lengthscales, ["Matern"] * len(lengthscales))
            
        ranked = sorted(report_data, key=lambda x: x[1])
        
        f.write(
            "| Rank | Factor | Kernel | Lengthscale | Interpretation |\n|------|--------|--------|------------:|---------------|\n"
        )
        for rank, (name, ls, k_type) in enumerate(ranked, 1):
            # 基于数值的解释逻辑
            if ls < 0.3:
                interp = "High Sensitivity"
            elif ls > 0.7:
                interp = "Low Sensitivity"
            else:
                interp = "Medium"
            f.write(f"| {rank} | {name} | {k_type} | {ls:.4f} | {interp} |\n")

        f.write("\n## 🔊 Outputscales (Amplitude)\n")
        f.write("- **Global/Base Scale**: {:.4f}\n".format(outputscales.get("global", 1.0)))
        if "main" in outputscales:
            f.write("\n### Main Effect Scales\n")
            f.write("| Factor | Outputscale |\n|--------|------------:|\n")
            for i, name in enumerate(factor_names):
                f.write(f"| {name} | {outputscales['main'][i]:.4f} |\n")
        
        if outputscales.get("interactions"):
            f.write("\n### Interaction Scales\n")
            f.write("| Interaction | Outputscale |\n|-------------|------------:|\n")
            for dims_str, val in outputscales["interactions"].items():
                # Convert dims_str "(0, 1)" to names
                try:
                    dims = eval(dims_str)
                    names = [factor_names[d] for d in dims]
                    f.write(f"| {' x '.join(names)} | {val:.4f} |\n")
                except:
                    f.write(f"| {dims_str} | {val:.4f} |\n")

        # --- Interaction Recovery Analysis ---
        if model_spec and model_spec.get("interaction_pairs"):
            f.write("\n## 🎯 Interaction Recovery Analysis\n")
            pairs = model_spec.get("interaction_pairs")
            inter_dims = set()
            for p in pairs:
                if isinstance(p, (list, tuple)):
                    for d in p:
                        if isinstance(d, int): inter_dims.add(d)
                        elif isinstance(d, str) and d in factor_names: 
                            try: inter_dims.add(factor_names.index(d))
                            except ValueError: pass
            
            if inter_dims:
                inter_ls = [lengthscales[i] for i in inter_dims if i < len(lengthscales)]
                other_ls = [lengthscales[i] for i in range(len(lengthscales)) if i not in inter_dims]
                
                if inter_ls and other_ls:
                    avg_inter = np.mean(inter_ls)
                    avg_other = np.mean(other_ls)
                    sci = avg_other / avg_inter
                    f.write(f"- **Interacting Dims**: {', '.join([factor_names[i] for i in inter_dims if i < len(factor_names)])}\n")
                    f.write(f"- **Avg Lengthscale (Interacting)**: {avg_inter:.4f}\n")
                    f.write(f"- **Avg Lengthscale (Others)**: {avg_other:.4f}\n")
                    f.write(f"- **Sensitivity Contrast Index (SCI)**: {sci:.2f}\n")
                    if sci > 1.2:
                        f.write("  - ✅ **Result**: Model successfully identified higher sensitivity in interaction dimensions.\n")
                    elif sci < 0.8:
                        f.write("  - ❌ **Result**: Model is less sensitive to interaction dimensions than others.\n")
                    else:
                        f.write("  - ⚠️ **Result**: No significant sensitivity difference detected.\n")
        
        # --- Pairwise Interaction Analysis (Non-ANOVA) ---
        if interactions_data and interactions_data.get("interactions"):
            f.write("\n## 🔗 Pairwise Interaction Analysis\n")
            f.write("*Detected via residual pattern & uncertainty analysis (non-parametric)*\n\n")
            f.write("| Rank | Interaction | Score | Factors |\n|------|-------------|-------|----------|\n")
            for rank, ((i, j), score) in enumerate(interactions_data["interactions"][:10], 1):
                f.write(f"| {rank} | {factor_names[i]} ⨯ {factor_names[j]} | {score:.4f} | ({i}, {j}) |\n")
            f.write("\n**Interpretation**: Higher scores indicate stronger evidence of pairwise interaction effects.\n")

        f.write("\n## 👥 Subject Standardization Stats\n")
        f.write(
            "| Subject | Mean | Std | Adjusted_Std_Used | N |\n|---------|------|-----|-------------------|---|\n"
        )
        for subj, stats in subject_stats.items():
            f.write(
                f"| {subj} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['adjusted_std_used']:.3f} | {stats['n']} |\n"
            )
        f.write("\n## 📍 Key Points (Design Space) - Three Sampling Points\n")
        f.write("*Three key parameter recipes for direct use in Phase 2*\n\n")

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
        f.write("\n## 🧪 Usage Example\n")
        f.write("### 1. Python API\n")
        f.write("```python\nimport torch, json, gpytorch\n")
        
        if model_spec and model_spec.get("kernel_type") == "custom_basegp_residual_mixed":
            f.write("from extensions.custom_factory.custom_basegp_residual_mixed_factory import CustomBaseGPResidualMixedFactory\n")
            f.write("# Load state_dict\nstate = torch.load('base_gp_state.pth', map_location='cpu')\n")
            f.write(f"# Reconstruct model using factory\nfactory = CustomBaseGPResidualMixedFactory(dim={len(factor_names)}, ...)\n")
            f.write("model = _FactoryGP(torch.zeros(1, D), torch.zeros(1), likelihood, factory)\n")
        else:
            f.write("from phase1_step3_base_gp import _MaternARDGP\n")
            f.write("# Load state_dict\nstate = torch.load('base_gp_state.pth', map_location='cpu')\n")
            f.write(f"# Reconstruct model (requires input dimension)\nD = {len(factor_names)}\n")
            f.write("likelihood = gpytorch.likelihoods.GaussianLikelihood()\n")
            f.write("model = _MaternARDGP(torch.zeros(1, D), torch.zeros(1), likelihood)\n")
            
        f.write("model.load_state_dict(state['model'])\nlikelihood.load_state_dict(state['likelihood'])\nmodel.eval(); likelihood.eval()\n")
        f.write("```\n\n")

        f.write("### 2. Phase 2 (Active Learning) Config Snippet\n")
        f.write("Copy this into your Phase 2 `.ini` or `.toml` configuration:\n\n")
        f.write("```ini\n[CustomBaseGPResidualFactory]\n")
        f.write(f"basegp_scan_csv = {path.parent.absolute()}/design_space_scan.csv\n")
        f.write("mean_type = pure_residual\n")
        f.write("lengthscale_prior = lognormal\n")
        
        # Calculate ls_loc for LogNormal prior (mode = L => mu = ln(L) + sigma^2)
        # Assume sigma = 0.5
        sigma = 0.5
        ls_locs = [np.log(max(1e-4, ls)) + sigma**2 for ls in lengthscales]
        f.write(f"ls_loc = [{', '.join([f'{l:.4f}' for l in ls_locs])}]\n")
        f.write(f"ls_scale = [{', '.join(['0.5'] * len(lengthscales))}]\n")

        # Add outputscale info
        if outputscales.get("global"):
            f.write(f"outputscale_prior = gamma\n")
            # Simple heuristic: if global scale is S, set prior to match
            f.write(f"# Population Outputscale: {outputscales['global']:.4f}\n")
        
        f.write("```\n")
        
        f.write("\n*Automatically generated*\n")


def run_step3_interactive():  # pragma: no cover - 交互主入口
    logger.info("=" * 40)
    logger.info("Phase 1 Step 3: Base GP Construction and Scanning")
    logger.info("=" * 40)
    data_csv = (
        input("Phase 1 data CSV path (with response) [default warmup_data.csv]: ").strip()
        or "warmup_data.csv"
    )
    design_csv = (
        input("Design space CSV path [default design_space.csv]: ").strip()
        or "design_space.csv"
    )
    subject_col = input("Subject column name [default subject_id]: ").strip() or "subject_id"
    response_col = input("Response column name [default response]: ").strip() or "response"
    output_dir = (
        input("Output directory [default base_gp_output]: ").strip() or "base_gp_output"
    )
    max_iters_str = input("Training iterations [default 300]: ").strip() or "300"
    lr_str = input("Learning rate [default 0.05]: ").strip() or "0.05"
    use_cuda = input("Use CUDA? (Y/n): ").strip().lower() != "n"
    try:
        max_iters = int(max_iters_str)
        lr = float(lr_str)
    except ValueError:
        logger.error("参数格式不正确")
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
    factory: Any = None,
    model_spec: Dict[str, Any] = None,
    y_normalization_type: str = "zscore",
    noise_lower_bound: float = 1e-4,
) -> Dict[str, Any]:
    """核心流程 (供 quick_start 调用)。

    Args:
        data_csv_path: Phase1 数据路径
                      - 如果是文件: 直接读取（需包含subject_col和response_col）
                      - 如果是目录: 读取所有subject_*.csv，每个文件代表一个被试
        ensure_diversity: 若True，若Sample 3与Sample 1/2重复，则选Std第二高的点。
        y_normalization_type: "zscore" 或 "mean_centering"
        noise_lower_bound: 噪声下界
    """
    if design_space_csv is None:
        raise ValueError("design_space_csv cannot be None. Step 3 requires a design space to scan.")
        
    # 更新 model_spec 包含 noise_lower_bound
    if model_spec is None:
        model_spec = {}
    model_spec["noise_lower_bound"] = noise_lower_bound

    data_path = Path(data_csv_path)
    design_path = Path(design_space_csv)
    if not data_path.exists():
        raise FileNotFoundError(f"Phase1 数据路径不存在: {data_csv_path}")
    if not design_path.exists():
        raise FileNotFoundError(f"设计空间文件不存在: {design_space_csv}")

    # 检查是文件还是目录
    if data_path.is_dir():
        # Directory mode: read all subject_*.csv
        logger.info(f"[Step3] Reading subject data from directory: {data_csv_path}")
        subject_csvs = sorted(data_path.glob("subject_*.csv"))

        if not subject_csvs:
            raise FileNotFoundError(f"No subject_*.csv files found in directory: {data_csv_path}")

        logger.info(f"  Found {len(subject_csvs)} subject files")

        # 读取每个被试文件并添加subject列
        all_dfs = []
        for csv_path in subject_csvs:
            df_subject = pd.read_csv(csv_path)

            # 验证响应列存在
            if response_col not in df_subject.columns:
                raise ValueError(f"Response column '{response_col}' not found in file: {csv_path.name}")

            # 添加被试列（如果不存在）
            if subject_col not in df_subject.columns:
                subject_id = csv_path.stem  # "subject_1"
                df_subject.insert(0, subject_col, subject_id)

            all_dfs.append(df_subject)
            logger.info(f"    - {csv_path.name}: {len(df_subject)} rows")

        # 合并所有数据
        df_phase1 = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"  Total after merging: {len(df_phase1)} rows")
    else:
        # File mode: direct read
        logger.info(f"[Step3] Reading data file: {data_csv_path}")
        df_phase1 = pd.read_csv(data_path)

    if subject_col not in df_phase1.columns or response_col not in df_phase1.columns:
        raise ValueError("Phase1 数据缺少必要列")

    # 智能选择因子列: 仅选择 x1, x2, ... xN 格式的列 (x + 数字开头)
    all_cols = [c for c in df_phase1.columns if c not in (subject_col, response_col)]
    factor_cols = [c for c in all_cols if re.match(r'^x\d+', c)]

    if not factor_cols:
        # 如果没有找到 x1, x2 格式的列，则回退到使用所有非subject/response列
        logger.warning("未找到 x1, x2, ... 格式的因子列，使用所有列")
        factor_cols = all_cols
    else:
        excluded_cols = set(all_cols) - set(factor_cols)
        if excluded_cols:
            logger.info(f"智能过滤: 使用 {len(factor_cols)} 个因子列 (x1, x2, ...)")
            logger.info(f"排除的列: {', '.join(sorted(excluded_cols))}")

    factor_df = df_phase1[factor_cols]
    encoded_factors, encodings = _encode_factor_df(factor_df)

    # 如果是目录模式，尝试从采样方案推断额外的编码（用于设计空间）
    if data_path.is_dir():
        logger.info("Inferring categorical encodings from sampling plan...")
        inferred_encodings = _infer_encoding_from_sampling(data_path, factor_cols)
        # 合并推断的编码（优先使用推断的，因为它包含完整的categorical->numeric映射）
        for col, mapping in inferred_encodings.items():
            if col not in encodings or not encodings[col]:
                encodings[col] = mapping
                logger.info(f"  Using inferred encoding: {col}")
            else:
                logger.info(f"  Column {col} already has encoding, skipping inference")

    # 标准化 (使用原始未编码因子, 但我们只需要 y_norm 与 X 编码后的数值)
    X_numeric = encoded_factors
    df_for_std = df_phase1[[subject_col, response_col] + factor_cols]
    y_norm, subject_stats = _standardize_subject_wise(
        df_phase1[[subject_col, response_col]], subject_col, response_col, mode=y_normalization_type
    )
    X_raw = X_numeric.values.astype(float)
    
    # X 归一化
    x_min, x_max = _get_x_stats(X_raw)
    X_train = _apply_x_normalization(X_raw, x_min, x_max)
    logger.info(f"X normalization applied. Min: {x_min}, Max: {x_max}")

    model, likelihood, train_meta = train_base_gp(
        X_train, 
        y_norm, 
        max_iters=max_iters, 
        lr=lr, 
        use_cuda=use_cuda, 
        factory=factory,
        model_spec=model_spec,
        factor_names=factor_cols
    )
    
    # 提取 Hyperparameters (使用新定义的鲁棒提取函数)
    params = _extract_hyperparameters(model, len(factor_cols))
    lengthscales = params["lengthscales"]
    outputscales = params["outputscales"]

    # 扫描设计空间
    design_df_raw = pd.read_csv(design_path)
    # 只取与训练相同的因子列, 丢弃其它列
    missing_cols = set(factor_cols) - set(design_df_raw.columns)
    if missing_cols:
        raise ValueError(f"设计空间缺少因子列: {missing_cols}")
    design_df_aligned = design_df_raw[factor_cols]

    # Debug: Check data types before encoding
    logger.debug("Design space before encoding:")
    for col in design_df_aligned.columns:
        logger.debug(f"  {col}: dtype={design_df_aligned[col].dtype}, "
                     f"in_encodings={col in encodings}, "
                     f"sample_values={design_df_aligned[col].head(3).tolist()}")

    design_df_encoded = _apply_encodings(design_df_aligned, encodings)

    # X 归一化 (使用训练集的 stats)
    design_x_raw = design_df_encoded.values.astype(float)
    design_x_norm = _apply_x_normalization(design_x_raw, x_min, x_max)

    # Debug: Check data types after encoding
    logger.debug("Design space after encoding:")
    for col in design_df_encoded.columns:
        logger.debug(f"  {col}: dtype={design_df_encoded[col].dtype}, "
                     f"sample_values={design_df_encoded[col].head(3).tolist()}")

    means, stds = scan_design_space(
        model, likelihood, design_x_norm
    )
    key_points = select_key_points(
        design_df_encoded, means, stds, ensure_diversity=ensure_diversity
    )

    # 如果不使用ANOVA核心，自动启用交互敏感度分析
    interactions_data = {}
    kernel_type = model_spec.get("kernel_type") if model_spec else "default"
    
    if kernel_type != "custom_anova":
        logger.info(f"[Step3] Analyzing pairwise interactions (kernel_type={kernel_type}, non-ANOVA mode)...")
        try:
            from .interaction_analyzer import analyze_interactions_for_step3
            
            # 设计空间没有真实因变量y，所以无法计算真实残差
            # 只使用基于预测值的方法（四象限方差 + 不确定性）
            interactions_data = analyze_interactions_for_step3(
                design_df=design_df_encoded,
                means=means,
                stds=stds,
                residuals=None,  # 设计空间无真实残差，不使用伪残差
                factor_names=factor_cols,
                k=6,
            )
            logger.info(f"[Step3] Detected {len(interactions_data['interactions'])} significant interactions")
        except Exception as e:
            logger.warning(f"[Step3] Interaction analysis failed: {e}", exc_info=True)
            interactions_data = {"interactions": [], "interaction_dict": {}}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 保存 state_dict
    torch.save(
        {"model": model.state_dict(), "likelihood": likelihood.state_dict()},
        out_dir / "base_gp_state.pth",
    )
    # 计算被试差异尺度 (Subject Variance Scale)
    subject_means = [stats['mean'] for stats in subject_stats.values()]
    subject_variance_scale = float(np.std(subject_means)) if len(subject_means) > 1 else 0.0

    # 其它 JSON - 包含交互信息
    lengthscales_output = {
        "factor_names": factor_cols, 
        "lengthscales": lengthscales,
        "outputscales": outputscales,
        "noise": params["noise"],
        "subject_variance": subject_variance_scale
    }
    
    # 添加交互对信息（如果存在）
    if interactions_data.get("interaction_dict"):
        lengthscales_output["interactions"] = interactions_data["interaction_dict"]
    
    (out_dir / "base_gp_lengthscales.json").write_text(
        json.dumps(
            lengthscales_output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    
    # 独立输出交互对文件（供后续步骤使用）
    if interactions_data.get("interactions"):
        interactions_output = {
            "kernel_type": kernel_type,
            "detected_method": "non-parametric (quadrant variance + uncertainty analysis)",
            "interactions": [
                {
                    "pair": f"({i}, {j})",
                    "factors": (factor_cols[i], factor_cols[j]),
                    "score": float(score),
                }
                for (i, j), score in interactions_data["interactions"]
            ],
            "summary": {
                "total_detected": len(interactions_data["interactions"]),
                "method_note": "Higher scores indicate stronger evidence of pairwise interaction effects.",
            }
        }
        (out_dir / "base_gp_interactions.json").write_text(
            json.dumps(
                interactions_output,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        logger.info(f"[Step3] Saved interaction analysis to base_gp_interactions.json")
    
    (out_dir / "base_gp_subject_stats.json").write_text(
        json.dumps(subject_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "base_gp_encodings.json").write_text(
        json.dumps(encodings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    key_points["noise"] = params["noise"]
    key_points["subject_variance"] = subject_variance_scale
    
    (out_dir / "base_gp_key_points.json").write_text(
        json.dumps(key_points, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 设计空间扫描 CSV
    scan_df = design_df_encoded.copy()
    scan_df["pred_mean"] = means
    scan_df["pred_std"] = stds
    scan_df.to_csv(out_dir / "design_space_scan.csv", index=False)
    
    # 准备主动 learning 初始化配置 (用于后续主动学习阶段)
    al_init_config = {
        "noise_std": params["noise"],
        "subject_std": subject_variance_scale,
        "main_effect_scales": {name: val for name, val in zip(factor_cols, outputscales["main"])},
        "lengthscales": {name: val for name, val in zip(factor_cols, lengthscales)},
        "global_scale": outputscales.get("global", 1.0),
        "interactions": outputscales.get("interactions", {}),
        "normalization": {
            "y_type": y_normalization_type,
            "x_min": x_min.tolist(),
            "x_max": x_max.tolist()
        }
    }
    (out_dir / "active_learning_init.json").write_text(
        json.dumps(al_init_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 报告
    write_report(
        out_dir / "base_gp_report.md",
        factor_cols,
        lengthscales,
        outputscales,
        subject_stats,
        key_points,
        train_meta,
        model_spec=model_spec,
        interactions_data=interactions_data,
        noise=params["noise"],
        subject_std=subject_variance_scale,
        kernel_types=params.get("kernel_types")
    )

    # 打印简要报告到控制台
    print("\n" + "=" * 40)
    print("Base GP Training Summary")
    print("=" * 40)
    print(f"Model: {'Custom Factory' if factory else 'Default Matern'}")
    print(f"Factors: {len(factor_cols)}")
    print(f"Training Samples: {len(df_phase1)}")
    print(f"Iterations: {max_iters}")
    if train_meta.get("history"):
        print(f"Final Loss: {train_meta['history'][-1]['loss']:.4f}")
    print(f"Estimated Noise: {params['noise']:.4f}")
    print(f"Subject Variance: {subject_variance_scale:.4f}")
    print("-" * 40)
    print("Key Points Selected:")
    print(f"  1. Best Prior: Index {key_points['x_best_prior_index']}")
    print(f"  2. Worst Prior: Index {key_points['x_worst_prior_index']}")
    print(f"  3. Max Uncertainty: Index {key_points['x_max_std_index']}")
    print("=" * 40 + "\n")

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
