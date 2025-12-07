"""
Response distribution validators
"""

from typing import List, Dict
import numpy as np
from collections import Counter


def check_normality(
    responses: List[int],
    min_coverage: int = 3,
    max_single_ratio: float = 0.6,
    mean_range: tuple = (2.0, 4.0)
) -> Dict[str, any]:
    """
    检查响应分布是否满足正态性要求

    Args:
        responses: 响应列表
        min_coverage: 最少覆盖的Likert等级数
        max_single_ratio: 单个等级的最大占比
        mean_range: 均值的合理范围 (min, max)

    Returns:
        {
            "passed": bool,
            "reason": str,  # 不通过时的原因
            "coverage": int,  # 实际覆盖的等级数
            "max_ratio": float,  # 最大单一等级占比
            "mean": float,  # 均值
        }
    """
    if len(responses) == 0:
        return {
            "passed": False,
            "reason": "Empty responses",
            "coverage": 0,
            "max_ratio": 0.0,
            "mean": 0.0,
        }

    # 统计
    counter = Counter(responses)
    unique_levels = len(counter)
    max_count = max(counter.values())
    max_ratio = max_count / len(responses)
    mean_response = np.mean(responses)

    # 检查1：覆盖度
    if unique_levels < min_coverage:
        return {
            "passed": False,
            "reason": f"Coverage too low: {unique_levels} < {min_coverage}",
            "coverage": unique_levels,
            "max_ratio": max_ratio,
            "mean": mean_response,
        }

    # 检查2：单一等级占比
    if max_ratio > max_single_ratio:
        return {
            "passed": False,
            "reason": f"Single level ratio too high: {max_ratio:.1%} > {max_single_ratio:.0%}",
            "coverage": unique_levels,
            "max_ratio": max_ratio,
            "mean": mean_response,
        }

    # 检查3：均值范围
    if mean_response < mean_range[0] or mean_response > mean_range[1]:
        return {
            "passed": False,
            "reason": f"Mean out of range: {mean_response:.2f} not in {mean_range}",
            "coverage": unique_levels,
            "max_ratio": max_ratio,
            "mean": mean_response,
        }

    # 通过
    return {
        "passed": True,
        "reason": "OK",
        "coverage": unique_levels,
        "max_ratio": max_ratio,
        "mean": mean_response,
    }


def get_distribution_stats(responses: List[int]) -> Dict[str, any]:
    """
    获取响应分布统计

    Args:
        responses: 响应列表

    Returns:
        统计信息字典
    """
    if len(responses) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0,
            "max": 0,
            "distribution": {},
        }

    counter = Counter(responses)
    return {
        "count": len(responses),
        "mean": float(np.mean(responses)),
        "std": float(np.std(responses)),
        "median": float(np.median(responses)),
        "min": int(np.min(responses)),
        "max": int(np.max(responses)),
        "distribution": dict(counter),
    }
