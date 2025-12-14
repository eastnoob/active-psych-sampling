#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设计空间加载和转换模块

提供设计空间加载、数值转换、后备生成等功能。
支持分类变量自动映射和自定义映射。
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from pathlib import Path
import logging


def load_design_space(
    design_space_path: str,
    categorical_mappings: Optional[Dict[str, Dict]] = None
) -> pd.DataFrame:
    """加载设计空间CSV文件。

    Args:
        design_space_path: 设计空间CSV文件路径
        categorical_mappings: 自定义分类变量映射，如 {'x4': {'low': 0.0, 'medium': 0.5, 'high': 1.0}}

    Returns:
        设计空间DataFrame
    """
    df = pd.read_csv(design_space_path)
    logging.info(f"[DesignSpace] Loaded from {design_space_path}: {df.shape}")

    # 应用自定义分类映射
    if categorical_mappings:
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                logging.info(f"[DesignSpace] Applied mapping to {col}: {mapping}")

    return df


def transform_to_numeric(
    df: pd.DataFrame,
    categorical_mappings: Optional[Dict[str, Dict]] = None
) -> np.ndarray:
    """将设计空间DataFrame转换为数值数组。

    自动处理分类变量，将其映射为数值。
    对于未指定映射的分类变量，使用均匀分布映射。

    Args:
        df: 设计空间DataFrame
        categorical_mappings: 自定义分类变量映射

    Returns:
        数值化的numpy数组
    """
    df_numeric = df.copy()

    # 处理分类变量
    for col in df_numeric.columns:
        if df_numeric[col].dtype == 'object':
            # 检查是否有自定义映射
            if categorical_mappings and col in categorical_mappings:
                mapping = categorical_mappings[col]
            else:
                # 自动生成均匀映射
                unique_vals = sorted(df_numeric[col].unique())
                if len(unique_vals) > 1:
                    mapping = {
                        val: i / (len(unique_vals) - 1)
                        for i, val in enumerate(unique_vals)
                    }
                else:
                    mapping = {unique_vals[0]: 0.0}
                logging.info(f"[DesignSpace] Auto-mapping {col}: {mapping}")

            df_numeric[col] = df_numeric[col].map(mapping)

    return df_numeric.astype(float).values


def generate_fallback_design_space(
    n_dims: int = 3,
    n_points: int = 100,
    method: str = 'sobol'
) -> np.ndarray:
    """生成后备设计空间。

    Args:
        n_dims: 维度数
        n_points: 点数
        method: 生成方法 ('sobol', 'lhs', 'random')

    Returns:
        设计空间数组 (n_points, n_dims)
    """
    if method == 'sobol':
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n_dims, scramble=True)
            design_space = sampler.random(n_points)
            logging.info(f"[DesignSpace] Generated Sobol design: {design_space.shape}")
        except ImportError:
            logging.warning("[DesignSpace] scipy not available, falling back to random")
            design_space = np.random.rand(n_points, n_dims)
    elif method == 'lhs':
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=n_dims)
            design_space = sampler.random(n_points)
            logging.info(f"[DesignSpace] Generated LHS design: {design_space.shape}")
        except ImportError:
            logging.warning("[DesignSpace] scipy not available, falling back to random")
            design_space = np.random.rand(n_points, n_dims)
    else:
        design_space = np.random.rand(n_points, n_dims)
        logging.info(f"[DesignSpace] Generated random design: {design_space.shape}")

    return design_space


def create_x4_4level_mapping() -> Dict[str, float]:
    """创建x4_4level分类变量的标准映射。

    Returns:
        {'low': 0.0, 'medium-low': 0.33, 'medium-high': 0.67, 'high': 1.0}
    """
    return {
        'low': 0.0,
        'medium-low': 0.33,
        'medium-high': 0.67,
        'high': 1.0
    }


def load_and_transform_design_space(
    design_space_path: str,
    categorical_mappings: Optional[Dict[str, Dict]] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """加载设计空间并转换为数值数组。

    便捷函数，组合load_design_space和transform_to_numeric。

    Args:
        design_space_path: 设计空间CSV文件路径
        categorical_mappings: 自定义分类变量映射

    Returns:
        (设计空间DataFrame, 数值化数组)
    """
    df = load_design_space(design_space_path, categorical_mappings)
    arr = transform_to_numeric(df, categorical_mappings)
    return df, arr
