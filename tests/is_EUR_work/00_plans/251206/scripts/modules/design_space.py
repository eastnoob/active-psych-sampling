"""
设计空间加载和转换模块
"""

import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path


def load_design_space(design_space_path: str) -> pd.DataFrame:
    """加载设计空间CSV文件.

    Args:
        design_space_path: 设计空间CSV文件路径

    Returns:
        设计空间DataFrame
    """
    return pd.read_csv(design_space_path)


def transform_to_numeric(df: pd.DataFrame) -> np.ndarray:
    """将设计空间DataFrame转换为数值数组.

    Args:
        df: 设计空间DataFrame

    Returns:
        数值化的numpy数组
    """
    df_numeric = df.copy()

    # 处理分类变量
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            # 尝试映射为数值
            unique_vals = df_numeric[col].unique()
            mapping = {val: i / (len(unique_vals) - 1) if len(unique_vals) > 1 else 0.0
                      for i, val in enumerate(sorted(unique_vals))}
            df_numeric[col] = df_numeric[col].map(mapping)

    return df_numeric.astype(float).values


def generate_fallback_design_space(n_dims: int = 3, n_points: int = 100) -> np.ndarray:
    """生成后备设计空间(Sobol序列或均匀随机).

    Args:
        n_dims: 维度数
        n_points: 点数

    Returns:
        设计空间数组
    """
    # 简单的均匀随机采样
    return np.random.rand(n_points, n_dims)
