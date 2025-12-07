# CustomBaseGPResidualMixedFactory - 残差学习工厂（混合参数）
#
# 作用：支持混合连续+离散参数的残差学习
# 特点：
#   - 连续参数使用Matern(nu=2.5)核函数
#   - 离散参数使用CategoricalKernel
#   - 两个核函数通过ProductKernel组合
#   - 支持纯残差和可学习偏移两种mean模式
#   - 灵活的lengthscale和outputscale先验配置
#
# 配置文件：CustomBaseGPResidualMixedFactory.ini

from extensions.custom_factory.custom_basegp_residual_mixed_factory import (
    CustomBaseGPResidualMixedFactory,
)

__all__ = ["CustomBaseGPResidualMixedFactory"]
