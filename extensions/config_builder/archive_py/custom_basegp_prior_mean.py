# CustomBaseGPPriorMean - 固定均值函数（使用BaseGP预测）
#
# 作用：使用BaseGP的预测结果作为固定均值函数
# 特点：无可学习参数，参数完全固定
#
# 配置文件：CustomBaseGPPriorMean.ini

from extensions.custom_mean.custom_basegp_prior_mean import CustomBaseGPPriorMean

__all__ = ["CustomBaseGPPriorMean"]
