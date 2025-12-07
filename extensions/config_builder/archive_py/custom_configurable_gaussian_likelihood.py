# ConfigurableGaussianLikelihood - 可配置高斯似然（带Gamma先验约束）
#
# 作用：对观测噪声添加Gamma先验约束，使噪声参数可配置
# 特点：支持自定义噪声先验参数和初始值
#
# 配置文件：ConfigurableGaussianLikelihood.ini

from extensions.custom_likelihood.custom_configurable_gaussian_likelihood import (
    ConfigurableGaussianLikelihood,
)

__all__ = ["ConfigurableGaussianLikelihood"]
