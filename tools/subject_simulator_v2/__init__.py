"""
Subject Simulator V2

简洁、可靠、易用的被试模拟器

主要功能：
- LinearSubject: 线性被试模型
- ClusterGenerator: 被试集群生成器
- load_subject: 从JSON加载被试

示例：
    >>> from subject_simulator_v2 import LinearSubject, ClusterGenerator, load_subject
    >>>
    >>> # 场景1：生成集群
    >>> gen = ClusterGenerator(design_space=np.array(...), n_subjects=5)
    >>> cluster = gen.generate_cluster("output/cluster_001")
    >>>
    >>> # 场景2：加载被试
    >>> subject = load_subject("output/cluster_001/subject_1_spec.json")
    >>> y = subject(x)
    >>>
    >>> # 场景3：创建单个被试
    >>> subject = LinearSubject(weights=np.array([0.2, -0.3, 0.5]))
    >>> subject.save("my_subject.json")
"""

__version__ = "1.0.0"
__author__ = "Claude & AEPsych Team"

from .base import BaseSubject
from .linear import LinearSubject
from .cluster import ClusterGenerator
from .validators import check_normality, get_distribution_stats

# 便捷函数
def load_subject(filepath: str) -> BaseSubject:
    """
    从JSON文件加载被试

    Args:
        filepath: JSON文件路径

    Returns:
        被试实例

    示例:
        >>> subject = load_subject("subject_1_spec.json")
        >>> y = subject(x)
    """
    import json

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    model_type = data.get("model_type", "linear")

    if model_type == "linear":
        return LinearSubject.from_dict(data)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


__all__ = [
    "BaseSubject",
    "LinearSubject",
    "ClusterGenerator",
    "load_subject",
    "check_normality",
    "get_distribution_stats",
]
