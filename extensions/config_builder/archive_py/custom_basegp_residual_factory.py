# CustomBaseGPResidualFactory - 残差学习工厂（连续参数）
#
# 作用：整合BaseGP均值+可配置的lengthscale先验，用于残差学习
# 特点：
#   - 支持pure_residual（固定均值）和learned_offset（可学习偏移）两种mean模式
#   - 支持三种lengthscale先验：LogNormal, Gamma, InvGamma
#   - 支持两种outputscale先验：Gamma, Box
#   - 使用Matern(nu=2.5)核函数
#
# 配置文件：CustomBaseGPResidualFactory.ini

from extensions.custom_factory.custom_basegp_residual_factory import (
    CustomBaseGPResidualFactory,
)

__all__ = ["CustomBaseGPResidualFactory"]
