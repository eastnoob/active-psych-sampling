#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义MeanCovarFactory - 残差学习方案

包含：
- CustomBaseGPResidualFactory: 使用BaseGP mean + lengthscale prior的Factory
"""

import torch
import gpytorch
from aepsych.factory.default import MeanCovarFactory
from aepsych.config import ConfigurableMixin
from extensions.custom_mean.custom_basegp_prior_mean import CustomBaseGPPriorMean, CustomMeanWithOffsetPrior


class CustomBaseGPResidualFactory(MeanCovarFactory, ConfigurableMixin):
    """
    使用BaseGP作为mean function的MeanCovarFactory（残差学习方案）。

    核心思想：
    - Mean function: 使用BaseGP预测值（固定，不学习）
    - Covariance: 新GP只需要学习残差的协方差结构
    - 数学表示: y ~ BaseGP_mean(x) + GP(0, K(x,x'))

    优势：
    - 样本效率更高：30个点足以刻画个体差异
    - 收敛更快：只需学习偏差而非绝对值
    - 不确定性更准确：初期不确定性较小

    Args:
        dim (int): 参数空间维度
        basegp_scan_csv (str): BaseGP预测结果CSV路径
        mean_type (str): Mean function类型 ("pure_residual" or "learned_offset")
            - "pure_residual": 固定BaseGP mean，0个可学习参数（默认，推荐）
            - "learned_offset": BaseGP mean + 可学习offset，1个额外参数
        offset_prior_std (float): Offset先验标准差（仅learned_offset模式），默认0.10
        lengthscale_prior (str): Lengthscale prior类型 ("lognormal", "gamma", "invgamma")
        ls_loc (list): LogNormal prior的loc参数（修正后应让prior mode = BaseGP lengthscales）
        ls_scale (list): LogNormal prior的scale参数
        fixed_kernel_amplitude (bool): 是否固定kernel amplitude
        outputscale_prior (str): Output scale prior类型 ("gamma", "box")

    Example:
        在INI配置中使用：
        ```ini
        [GPRegressionModel]
        mean_covar_factory = CustomBaseGPResidualFactory

        [CustomBaseGPResidualFactory]
        basegp_scan_csv = extensions/.../design_space_scan.csv
        mean_type = pure_residual  # 或 learned_offset
        offset_prior_std = 0.10    # 仅learned_offset模式需要
        lengthscale_prior = lognormal
        # 修正后的ls_loc，让prior mode = BaseGP lengthscales
        ls_loc = [0.0166, -0.2634, 0.7133, -1.4744, 0.7983, 0.6391]
        ls_scale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        fixed_kernel_amplitude = False
        outputscale_prior = gamma
        ```
    """

    def __init__(
        self,
        dim: int,
        basegp_scan_csv: str,
        mean_type: str = "pure_residual",
        offset_prior_std: float = 0.10,
        lengthscale_prior: str = "lognormal",
        ls_loc: list = None,
        ls_scale: list = None,
        fixed_kernel_amplitude: bool = False,
        outputscale_prior: str = "gamma",
        stimuli_per_trial: int = 1,
    ):
        # Mean配置
        self.basegp_scan_csv = basegp_scan_csv
        self.mean_type = mean_type
        self.offset_prior_std = offset_prior_std

        # 验证mean_type
        if mean_type not in ["pure_residual", "learned_offset"]:
            raise ValueError(
                f"mean_type must be 'pure_residual' or 'learned_offset', "
                f"got '{mean_type}'"
            )

        # Covariance配置
        self.lengthscale_prior = lengthscale_prior
        self.ls_loc = torch.tensor(ls_loc, dtype=torch.float64) if ls_loc else None
        self.ls_scale = (
            torch.tensor(ls_scale, dtype=torch.float64) if ls_scale else None
        )
        self.fixed_kernel_amplitude = fixed_kernel_amplitude
        self.outputscale_prior = outputscale_prior

        # 调用父类初始化
        super().__init__(dim, stimuli_per_trial)

    def _make_mean_module(self):
        """
        根据mean_type创建不同的mean module

        Returns:
            CustomBaseGPPriorMean (pure_residual模式, 0参数) 或
            CustomMeanWithOffsetPrior (learned_offset模式, 1参数)
        """
        if self.mean_type == "pure_residual":
            return CustomBaseGPPriorMean(basegp_scan_csv=self.basegp_scan_csv)
        elif self.mean_type == "learned_offset":
            return CustomMeanWithOffsetPrior(
                basegp_scan_csv=self.basegp_scan_csv,
                offset_prior_std=self.offset_prior_std
            )
        else:
            # 这个分支理论上不会到达（__init__中已经验证）
            raise ValueError(f"Unknown mean_type: {self.mean_type}")

    def _make_covar_module(self):
        """
        创建协方差模块（使用BaseGP lengthscale prior）。

        注意：这里需要考虑DefaultMeanCovarFactory在构建LogNormal prior时
        会加上 log(dim)/2 的调整项。因此ls_loc应该是修正后的值。
        """
        import math

        # 设置默认值
        if self.ls_loc is None:
            self.ls_loc = torch.tensor(math.sqrt(2.0), dtype=torch.float64)
        if self.ls_scale is None:
            self.ls_scale = torch.tensor(math.sqrt(3.0), dtype=torch.float64)

        # 构建lengthscale prior
        if self.lengthscale_prior == "lognormal":
            # LogNormal prior with dimension adjustment
            ls_prior = gpytorch.priors.LogNormalPrior(
                self.ls_loc + math.log(self.dim) / 2, self.ls_scale
            )
            # Prior mode = exp(loc - scale²)
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
                f"got {self.lengthscale_prior}"
            )

        # Lengthscale constraint
        ls_constraint = gpytorch.constraints.GreaterThan(
            lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
        )

        # 创建基础kernel（使用Matern 2.5以匹配BaseGP）
        covar = gpytorch.kernels.MaternKernel(
            nu=2.5,  # 与BaseGP保持一致
            ard_num_dims=self.dim,
            lengthscale_prior=ls_prior,
            lengthscale_constraint=ls_constraint,
        )

        # 添加ScaleKernel（如果需要）
        if not self.fixed_kernel_amplitude:
            if self.outputscale_prior == "gamma":
                os_prior = gpytorch.priors.GammaPrior(concentration=2.0, rate=0.15)
            elif self.outputscale_prior == "box":
                os_prior = gpytorch.priors.SmoothedBoxPrior(a=1, b=4)
            else:
                raise ValueError(
                    f"outputscale_prior must be 'gamma' or 'box', "
                    f"got {self.outputscale_prior}"
                )

            covar = gpytorch.kernels.ScaleKernel(
                covar,
                outputscale_prior=os_prior,
                outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
            )

        return covar

    @classmethod
    def get_config_args(cls, config, name=None):
        """从Config对象获取初始化参数"""
        # 获取维度
        dim = config.getint("common", "n_params", fallback=None)
        if dim is None:
            # 尝试从parnames推断
            parnames = config.get("common", "parnames", fallback=None)
            if parnames:
                import ast

                dim = len(ast.literal_eval(parnames))
            else:
                raise ValueError("Cannot determine dimension from config")

        # 解析ls_loc和ls_scale
        ls_loc_str = config.get(name, "ls_loc", fallback=None)
        ls_scale_str = config.get(name, "ls_scale", fallback=None)

        import ast

        ls_loc = ast.literal_eval(ls_loc_str) if ls_loc_str else None
        ls_scale = ast.literal_eval(ls_scale_str) if ls_scale_str else None

        return {
            "dim": dim,
            "basegp_scan_csv": config.get(name, "basegp_scan_csv"),
            "mean_type": config.get(name, "mean_type", fallback="pure_residual"),
            "offset_prior_std": config.getfloat(
                name, "offset_prior_std", fallback=0.10
            ),
            "lengthscale_prior": config.get(
                name, "lengthscale_prior", fallback="lognormal"
            ),
            "ls_loc": ls_loc,
            "ls_scale": ls_scale,
            "fixed_kernel_amplitude": config.getboolean(
                name, "fixed_kernel_amplitude", fallback=False
            ),
            "outputscale_prior": config.get(
                name, "outputscale_prior", fallback="gamma"
            ),
            "stimuli_per_trial": config.getint(
                "common", "stimuli_per_trial", fallback=1
            ),
        }
