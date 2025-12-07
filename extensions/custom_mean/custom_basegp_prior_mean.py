#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自定义Mean Function - 使用BaseGP预测作为先验

包含：
- CustomBaseGPPriorMean: 使用BaseGP预计算的预测值作为mean function（残差学习）
- CustomMeanWithOffsetPrior: BaseGP mean + learnable offset（残差学习增强版）
"""

import numpy as np
import pandas as pd
import torch
import gpytorch
from scipy.spatial import cKDTree
from aepsych.config import ConfigurableMixin
from pathlib import Path


class CustomBaseGPPriorMean(gpytorch.means.Mean, ConfigurableMixin):
    """
    使用BaseGP预计算的预测值作为固定mean function。

    这实现了残差学习方案：
    - 新GP模型学习: y - BaseGP_mean(x) ~ GP(0, K(x,x'))
    - BaseGP的预测作为先验知识，新模型只需要学习个体偏差
    - 优势：30个点足以刻画"与平均水平的差异"

    实现方式：
    - 使用预计算的design_space_scan.csv作为查找表
    - KD树最近邻查找，O(log N)复杂度
    - 自动检测以 x1, x2, x3... 开头的特征列，无需硬编码列名

    Args:
        basegp_scan_csv (str): BaseGP预测结果CSV文件路径
            必须包含:
            - 特征列: x1_*, x2_*, x3_*, ... (任意后缀名,自动检测)
            - 预测列: pred_mean

    Example:
        在INI配置中使用：
        ```ini
        [BaseGPResidualFactory]
        basegp_scan_csv = extensions/warmup_budget_check/.../design_space_scan.csv
        ```

        CSV 格式示例:
        ```csv
        x1_CeilingHeight,x2_GridModule,x3_OuterFurniture,...,pred_mean,pred_std
        2.8,6.5,2,...,0.99,0.55
        ```
    """

    def __init__(self, basegp_scan_csv: str):
        super().__init__()

        # 加载BaseGP预测结果
        csv_path = Path(basegp_scan_csv)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"BaseGP scan CSV not found: {basegp_scan_csv}\n"
                f"Expected path: {csv_path.absolute()}"
            )

        df = pd.read_csv(csv_path)

        # 自动检测特征列: 以 x1, x2, x3 等开头的列 (按数字排序)
        import re
        feature_cols = sorted(
            [col for col in df.columns if re.match(r'^x\d+', col)],
            key=lambda x: int(re.match(r'^x(\d+)', x).group(1))
        )

        if not feature_cols:
            raise ValueError(
                f"No feature columns found (expected columns starting with x1, x2, etc.)\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # 验证必须有 pred_mean 列
        if "pred_mean" not in df.columns:
            raise ValueError(
                f"Missing required column 'pred_mean'\n"
                f"Available columns: {df.columns.tolist()}"
            )

        # 保存特征列名 (用于日志)
        self.feature_cols = feature_cols
        self.n_features = len(feature_cols)

        # 构建查找表（BaseGP空间）
        self.design_points_basegp = torch.tensor(
            df[feature_cols].values,
            dtype=torch.float32,
        )
        self.pred_means = torch.tensor(df["pred_mean"].values, dtype=torch.float32)

        # 构建KD树用于快速最近邻查找
        self.kdtree = cKDTree(self.design_points_basegp.numpy())

        print(f"[CustomBaseGPPriorMean] 加载 {len(df)} 个BaseGP预测点")
        print(f"  检测到 {self.n_features} 个特征列: {feature_cols}")
        print(
            f"  pred_mean范围: [{self.pred_means.min():.3f}, {self.pred_means.max():.3f}]"
        )
        print(f"  pred_mean均值: {self.pred_means.mean():.3f}")

    def forward(self, x):
        """
        根据输入点返回BaseGP的预测均值。

        Args:
            x: 输入点 (INI空间: x0-x5), shape [N, 6]

        Returns:
            BaseGP的预测均值, shape [N]
        """
        # 1. 转换：INI空间 → BaseGP空间
        x_basegp = self._convert_ini_to_basegp(x)

        # 2. 最近邻查找
        dists, indices = self.kdtree.query(x_basegp.cpu().numpy(), k=1)

        # 3. 返回对应的预测均值
        mean_values = self.pred_means[indices]

        return mean_values.to(x.device)

    def _convert_ini_to_basegp(self, x_ini):
        """
        INI空间 → BaseGP空间的转换。

        注意: 如果 BaseGP 的 CSV 已经使用与 INI 相同的参数空间,
        则直接返回原始输入 (无需转换)。

        如果需要特殊转换 (例如旧版本的参数空间差异),
        可以在这里添加转换逻辑。

        Args:
            x_ini: 输入点, shape [N, n_features]

        Returns:
            转换后的点, shape [N, n_features]
        """
        # 默认: 直接返回 (假设参数空间一致)
        return x_ini.clone()

    @classmethod
    def get_config_args(cls, config, name=None):
        """从Config对象获取初始化参数"""
        return {
            "basegp_scan_csv": config.get(name, "basegp_scan_csv"),
        }


class CustomMeanWithOffsetPrior(gpytorch.means.Mean, ConfigurableMixin):
    """
    BaseGP mean + learnable offset (残差学习的可选增强版本)

    数学模型: mean(x) = BaseGP_mean(x) + offset
    其中 offset ~ N(0, offset_prior_std²) 是可学习参数

    使用场景：
    - 当认为个体与BaseGP有系统性的全局偏移时使用
    - 例如：某些个体对所有刺激的响应普遍偏高或偏低

    注意：
    - 这会增加1个可学习参数
    - 在小样本情况下（如30点），fixed mean（BaseGPPriorMean）通常表现更好
    - 推荐在确认需要全局偏移时再使用此选项

    Args:
        basegp_scan_csv (str): BaseGP预测结果CSV文件路径
        offset_prior_std (float): Offset参数的先验标准差，默认0.10
            - 较小值（如0.05）表示强先验，认为偏移不会太大
            - 较大值（如0.20）表示弱先验，允许较大的全局偏移

    Example:
        在INI配置中使用：
        ```ini
        [BaseGPResidualFactory]
        basegp_scan_csv = path/to/design_space_scan.csv
        mean_type = learned_offset
        offset_prior_std = 0.10
        ```
    """

    def __init__(
        self,
        basegp_scan_csv: str,
        offset_prior_std: float = 0.10
    ):
        super().__init__()

        # 复用 CustomBaseGPPriorMean 的查找表逻辑
        self.base_mean = CustomBaseGPPriorMean(basegp_scan_csv)

        # 添加可学习的 offset 参数（初始化为0）
        self.register_parameter(
            "offset",
            torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        )

        # 设置 N(0, offset_prior_std²) 先验
        self.register_prior(
            "offset_prior",
            gpytorch.priors.NormalPrior(0.0, offset_prior_std),
            "offset"
        )

        self.offset_prior_std = offset_prior_std

        print(f"[CustomMeanWithOffsetPrior] Initialized with offset_prior_std={offset_prior_std:.3f}")
        print(f"  offset初始值: {self.offset.item():.3f}")
        print(f"  offset先验: N(0, {offset_prior_std**2:.4f})")

    def forward(self, x):
        """
        返回 BaseGP mean + offset

        Args:
            x: 输入点, shape [N, dim]

        Returns:
            mean值, shape [N]
        """
        base_mean_values = self.base_mean(x)

        # offset是标量，广播到所有点
        return base_mean_values + self.offset

    @classmethod
    def get_config_args(cls, config, name=None):
        """从Config对象获取初始化参数"""
        return {
            "basegp_scan_csv": config.get(name, "basegp_scan_csv"),
            "offset_prior_std": config.getfloat(name, "offset_prior_std", fallback=0.10),
        }
