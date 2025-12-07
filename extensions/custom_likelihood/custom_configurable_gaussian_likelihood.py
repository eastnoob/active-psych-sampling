#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义Likelihood - 支持配置noise prior

包含：
- ConfigurableGaussianLikelihood: 可配置noise variance prior的GaussianLikelihood
"""

import gpytorch
from aepsych.config import ConfigurableMixin


class ConfigurableGaussianLikelihood(
    gpytorch.likelihoods.GaussianLikelihood, ConfigurableMixin
):
    """
    GaussianLikelihood with configurable noise prior from BaseGP.

    允许通过INI配置文件设置noise variance的prior分布和初始值。
    这使得可以引入BaseGP Phase1学到的noise variance作为先验。

    Args:
        noise_prior_concentration (float): GammaPrior的concentration参数
        noise_prior_rate (float): GammaPrior的rate参数
        noise_init (float): Noise variance的初始值

    Example:
        在INI配置中使用：
        ```ini
        [ConfigurableGaussianLikelihood]
        noise_prior_concentration = 2.0
        noise_prior_rate = 1.228
        noise_init = 0.814  # 从BaseGP学习得到
        ```

        GammaPrior的mode = (concentration - 1) / rate
        例如：(2.0 - 1) / 1.228 ≈ 0.814
    """

    def __init__(
        self,
        noise_prior_concentration: float = 2.0,
        noise_prior_rate: float = 1.228,
        noise_init: float = 0.814,
        **kwargs
    ):
        # 创建noise variance的Gamma prior
        noise_prior = gpytorch.priors.GammaPrior(
            concentration=noise_prior_concentration, rate=noise_prior_rate
        )

        # 设置noise constraint和初始值
        noise_constraint = gpytorch.constraints.GreaterThan(
            1e-4, initial_value=noise_init
        )

        # 调用父类初始化
        super().__init__(
            noise_prior=noise_prior, noise_constraint=noise_constraint, **kwargs
        )

    @classmethod
    def get_config_args(cls, config, name=None):
        """从Config对象获取初始化参数"""
        return {
            "noise_prior_concentration": config.getfloat(
                name, "noise_prior_concentration", fallback=2.0
            ),
            "noise_prior_rate": config.getfloat(
                name, "noise_prior_rate", fallback=1.228
            ),
            "noise_init": config.getfloat(name, "noise_init", fallback=0.814),
        }
