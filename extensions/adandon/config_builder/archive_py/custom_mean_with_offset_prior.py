# CustomMeanWithOffsetPrior - BaseGP均值+可学习偏移
#
# 作用：在BaseGP均值基础上添加可学习的全局偏移参数
# 特点：增加1个可学习参数，可建模个体的全局偏移
#
# 配置文件：CustomMeanWithOffsetPrior.ini

from extensions.custom_mean.custom_basegp_prior_mean import CustomMeanWithOffsetPrior

__all__ = ["CustomMeanWithOffsetPrior"]
