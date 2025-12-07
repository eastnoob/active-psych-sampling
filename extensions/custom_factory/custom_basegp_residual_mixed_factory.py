#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CustomBaseGPResidualMixedFactory - 支持混合参数的残差学习工厂

支持：
- 连续参数: MaternKernel (ARD)
- 离散参数: CategoricalKernel (ARD)
- 核组合: ProductKernel (乘法组合)
- Mean 模式: pure_residual (固定) 或 learned_offset (可学习)

核心设计决策（基于交接文档）：
1. 使用 ProductKernel 组合连续和离散核（乘法，非加法）
2. 每种参数类型使用单个 Kernel + ARD（而非多个 Kernel）
3. CategoricalKernel 支持每维独立 ARD lengthscales
4. 向后兼容 CustomBaseGPResidualFactory
"""

import torch
import gpytorch
import botorch
from aepsych.factory.default import MeanCovarFactory
from aepsych.config import ConfigurableMixin
from extensions.custom_mean.custom_basegp_prior_mean import CustomBaseGPPriorMean, CustomMeanWithOffsetPrior


class CustomBaseGPResidualMixedFactory(MeanCovarFactory, ConfigurableMixin):
    """
    支持混合参数（连续+离散）的残差学习工厂

    核心思想：
    - Mean: 使用 BaseGP 预测（固定或可学习 offset）
    - Covariance: ProductKernel(MaternKernel × CategoricalKernel)
    - 数学表示: y ~ BaseGP_mean(x) + GP(0, K_continuous × K_discrete)

    Args:
        dim (int): 总维度数 (continuous + discrete)
        continuous_params (list[str]): 连续参数名称列表，默认 []
        discrete_params (dict[str, int]): 离散参数字典 {name: n_categories}，默认 {}
        basegp_scan_csv (str): BaseGP预测CSV路径（可选，如果不使用 BaseGP mean 则不需要）
        mean_type (str): Mean function类型
            - "pure_residual": 固定 BaseGP mean (默认)
            - "learned_offset": BaseGP mean + 可学习 offset
        offset_prior_std (float): Offset 先验标准差（learned_offset 模式）
        lengthscale_prior (str): Lengthscale prior 类型 ("lognormal", "gamma", "invgamma")
        ls_loc (list): 连续参数的 LogNormal prior loc（仅连续参数）
        ls_scale (list): 连续参数的 LogNormal prior scale（仅连续参数）
        fixed_kernel_amplitude (bool): 是否固定 kernel amplitude
        outputscale_prior (str): Output scale prior 类型 ("gamma", "box")

    维度映射约定（CRITICAL）：
        train_X: [batch_size, n_continuous + n_discrete]
        - 前 n_continuous 维: 连续参数
        - 后 n_discrete 维: 离散参数（0-indexed 整数，范围 [0, n_cat-1]）

    Example:
        在 INI 配置中使用：
        ```ini
        [GPRegressionModel]
        mean_covar_factory = CustomBaseGPResidualMixedFactory

        [CustomBaseGPResidualMixedFactory]
        # 参数定义
        continuous_params = [dur, freq]
        discrete_params = {intensity: 3, color: 2}  # intensity: 3类, color: 2类

        # Mean 配置
        basegp_scan_csv = path/to/design_space_scan.csv
        mean_type = pure_residual

        # Kernel 配置（仅连续参数）
        lengthscale_prior = lognormal
        ls_loc = [0.0, -0.3]  # 对应 dur, freq
        ls_scale = [0.5, 0.5]
        fixed_kernel_amplitude = False
        outputscale_prior = gamma
        ```

    参数计数：
        - pure_residual 模式: n_continuous_ard + n_discrete_ard + 1_outputscale
        - learned_offset 模式: n_continuous_ard + n_discrete_ard + 1_offset + 1_outputscale
    """

    def __init__(
        self,
        dim: int,
        continuous_params: list = None,
        discrete_params: dict = None,
        basegp_scan_csv: str = None,
        mean_type: str = "pure_residual",
        offset_prior_std: float = 0.10,
        lengthscale_prior: str = "lognormal",
        ls_loc: list = None,
        ls_scale: list = None,
        fixed_kernel_amplitude: bool = False,
        outputscale_prior: str = "gamma",
        stimuli_per_trial: int = 1,
    ):
        # 参数配置
        self.continuous_params = continuous_params or []
        self.discrete_params = discrete_params or {}

        # 验证维度一致性
        n_continuous = len(self.continuous_params)
        n_discrete = len(self.discrete_params)

        if n_continuous + n_discrete != dim:
            raise ValueError(
                f"Dimension mismatch: "
                f"continuous({n_continuous}) + discrete({n_discrete}) = {n_continuous + n_discrete} "
                f"!= dim({dim})"
            )

        if n_continuous == 0 and n_discrete == 0:
            raise ValueError(
                "Must specify at least one continuous or discrete parameter"
            )

        # Mean 配置
        self.basegp_scan_csv = basegp_scan_csv
        self.mean_type = mean_type
        self.offset_prior_std = offset_prior_std

        # 验证 mean_type
        if mean_type not in ["pure_residual", "learned_offset"]:
            raise ValueError(
                f"mean_type must be 'pure_residual' or 'learned_offset', "
                f"got '{mean_type}'"
            )

        # Covariance 配置
        self.lengthscale_prior = lengthscale_prior
        self.ls_loc = torch.tensor(ls_loc, dtype=torch.float64) if ls_loc else None
        self.ls_scale = torch.tensor(ls_scale, dtype=torch.float64) if ls_scale else None
        self.fixed_kernel_amplitude = fixed_kernel_amplitude
        self.outputscale_prior = outputscale_prior

        # 验证 ls_loc/ls_scale 与连续参数维度匹配
        if self.ls_loc is not None and len(self.ls_loc) != n_continuous:
            raise ValueError(
                f"ls_loc length ({len(self.ls_loc)}) must match "
                f"number of continuous params ({n_continuous})"
            )
        if self.ls_scale is not None and len(self.ls_scale) != n_continuous:
            raise ValueError(
                f"ls_scale length ({len(self.ls_scale)}) must match "
                f"number of continuous params ({n_continuous})"
            )

        # 调用父类初始化
        super().__init__(dim, stimuli_per_trial)

        print(f"[CustomBaseGPResidualMixedFactory] Initialized")
        print(f"  Continuous params: {self.continuous_params} (n={n_continuous})")
        print(f"  Discrete params: {self.discrete_params} (n={n_discrete})")
        print(f"  Mean type: {self.mean_type}")
        print(f"  Total dim: {dim}")

    def _make_mean_module(self):
        """
        根据 mean_type 创建不同的 mean module

        Returns:
            CustomBaseGPPriorMean (pure_residual 模式, 0参数) 或
            CustomMeanWithOffsetPrior (learned_offset 模式, 1参数)
        """
        if self.basegp_scan_csv is None:
            # 如果没有提供 BaseGP CSV，使用零均值
            print("[CustomBaseGPResidualMixedFactory] No BaseGP CSV provided, using ZeroMean")
            return gpytorch.means.ZeroMean()

        if self.mean_type == "pure_residual":
            return CustomBaseGPPriorMean(basegp_scan_csv=self.basegp_scan_csv)
        elif self.mean_type == "learned_offset":
            return CustomMeanWithOffsetPrior(
                basegp_scan_csv=self.basegp_scan_csv,
                offset_prior_std=self.offset_prior_std
            )
        else:
            raise ValueError(f"Unknown mean_type: {self.mean_type}")

    def _make_covar_module(self, active_dims=None):
        """
        创建协方差模块：ProductKernel(MaternKernel × CategoricalKernel)

        架构：
        1. 连续参数 → MaternKernel(nu=2.5, ard_num_dims=n_continuous)
        2. 离散参数 → CategoricalKernel(ard_num_dims=n_discrete)
        3. 组合 → ProductKernel(*kernels)
        4. 缩放 → ScaleKernel(base_kernel) [如果 !fixed_kernel_amplitude]

        Returns:
            gpytorch.kernels.Kernel: 协方差核
        """
        import math

        kernels = []

        # 1. 连续参数核
        if self.continuous_params:
            continuous_dims = self._get_active_dims_continuous()

            # 设置默认 lengthscale prior 参数
            if self.ls_loc is None:
                self.ls_loc = torch.tensor(
                    [math.sqrt(2.0)] * len(self.continuous_params),
                    dtype=torch.float64
                )
            if self.ls_scale is None:
                self.ls_scale = torch.tensor(
                    [math.sqrt(3.0)] * len(self.continuous_params),
                    dtype=torch.float64
                )

            # 构建 lengthscale prior
            if self.lengthscale_prior == "lognormal":
                # LogNormal prior with dimension adjustment
                ls_prior = gpytorch.priors.LogNormalPrior(
                    self.ls_loc + math.log(len(self.continuous_params)) / 2,
                    self.ls_scale
                )
                ls_prior_mode = torch.exp(self.ls_loc - self.ls_scale**2)
            elif self.lengthscale_prior == "gamma":
                ls_prior = gpytorch.priors.GammaPrior(concentration=3.0, rate=6.0)
                ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
            elif self.lengthscale_prior == "invgamma":
                ls_prior = gpytorch.priors.GammaPrior(
                    concentration=3.0, rate=1.0, transform=lambda x: 1 / x
                )
                ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)
            else:
                raise ValueError(
                    f"lengthscale_prior must be 'lognormal', 'gamma', or 'invgamma', "
                    f"got '{self.lengthscale_prior}'"
                )

            # Lengthscale constraint
            ls_constraint = gpytorch.constraints.GreaterThan(
                lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
            )

            # 创建 Matern 核（与 BaseGP 保持一致）
            kernel_continuous = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=len(self.continuous_params),
                active_dims=continuous_dims,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
            )

            kernels.append(kernel_continuous)
            print(f"[CustomBaseGPResidualMixedFactory] Added MaternKernel for {len(self.continuous_params)} continuous params")

        # 2. 离散参数核
        if self.discrete_params:
            discrete_dims = self._get_active_dims_discrete()

            # 使用 BoTorch CategoricalKernel 支持每维独立 ARD
            ls_constraint = gpytorch.constraints.GreaterThan(lower_bound=1e-4)
            kernel_discrete = botorch.models.kernels.CategoricalKernel(
                active_dims=tuple(discrete_dims),
                ard_num_dims=len(self.discrete_params),
                lengthscale_constraint=ls_constraint
            )

            kernels.append(kernel_discrete)
            print(f"[CustomBaseGPResidualMixedFactory] Added CategoricalKernel for {len(self.discrete_params)} discrete params")

        # 3. 组合核
        if len(kernels) == 0:
            raise ValueError("No kernels created - this should not happen")
        elif len(kernels) == 1:
            base_kernel = kernels[0]
        else:
            # ProductKernel: 乘法组合（标准做法）
            base_kernel = gpytorch.kernels.ProductKernel(*kernels)
            print("[CustomBaseGPResidualMixedFactory] Combined kernels with ProductKernel (multiplicative)")

        # 4. 添加 ScaleKernel
        if not self.fixed_kernel_amplitude:
            if self.outputscale_prior == "gamma":
                os_prior = gpytorch.priors.GammaPrior(concentration=2.0, rate=0.15)
            elif self.outputscale_prior == "box":
                os_prior = gpytorch.priors.SmoothedBoxPrior(a=1, b=4)
            else:
                raise ValueError(
                    f"outputscale_prior must be 'gamma' or 'box', "
                    f"got '{self.outputscale_prior}'"
                )

            covar = gpytorch.kernels.ScaleKernel(
                base_kernel,
                outputscale_prior=os_prior,
                outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
            )
            print("[CustomBaseGPResidualMixedFactory] Added ScaleKernel")
            return covar
        else:
            return base_kernel

    def _get_active_dims_continuous(self):
        """
        获取连续参数的维度索引

        Returns:
            list[int]: 连续参数在 train_X 中的列索引
        """
        return list(range(len(self.continuous_params)))

    def _get_active_dims_discrete(self):
        """
        获取离散参数的维度索引

        Returns:
            list[int]: 离散参数在 train_X 中的列索引
        """
        n_cont = len(self.continuous_params)
        n_disc = len(self.discrete_params)
        return list(range(n_cont, n_cont + n_disc))

    @classmethod
    def get_config_args(cls, config, name=None):
        """从 Config 对象获取初始化参数"""
        import ast

        # 获取维度
        dim = config.getint("common", "n_params", fallback=None)
        if dim is None:
            # 尝试从 parnames 推断
            parnames = config.get("common", "parnames", fallback=None)
            if parnames:
                dim = len(ast.literal_eval(parnames))
            else:
                raise ValueError("Cannot determine dimension from config")

        # 解析连续参数
        continuous_params_str = config.get(name, "continuous_params", fallback=None)
        continuous_params = ast.literal_eval(continuous_params_str) if continuous_params_str else []

        # 解析离散参数
        discrete_params_str = config.get(name, "discrete_params", fallback=None)
        discrete_params = ast.literal_eval(discrete_params_str) if discrete_params_str else {}

        # 解析 ls_loc 和 ls_scale（仅连续参数）
        ls_loc_str = config.get(name, "ls_loc", fallback=None)
        ls_scale_str = config.get(name, "ls_scale", fallback=None)
        ls_loc = ast.literal_eval(ls_loc_str) if ls_loc_str else None
        ls_scale = ast.literal_eval(ls_scale_str) if ls_scale_str else None

        return {
            "dim": dim,
            "continuous_params": continuous_params,
            "discrete_params": discrete_params,
            "basegp_scan_csv": config.get(name, "basegp_scan_csv", fallback=None),
            "mean_type": config.get(name, "mean_type", fallback="pure_residual"),
            "offset_prior_std": config.getfloat(name, "offset_prior_std", fallback=0.10),
            "lengthscale_prior": config.get(name, "lengthscale_prior", fallback="lognormal"),
            "ls_loc": ls_loc,
            "ls_scale": ls_scale,
            "fixed_kernel_amplitude": config.getboolean(name, "fixed_kernel_amplitude", fallback=False),
            "outputscale_prior": config.get(name, "outputscale_prior", fallback="gamma"),
            "stimuli_per_trial": config.getint("common", "stimuli_per_trial", fallback=1),
        }
