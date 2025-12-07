"""
Base classes for subject simulators
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np


class BaseSubject(ABC):
    """
    抽象基类：被试模拟器

    所有具体模型（线性、非线性等）都应继承此类
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """
        被试作答

        Args:
            x: 输入特征向量 (n_features,)

        Returns:
            响应值（连续值或Likert评分）
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        导出为完整参数字典（用于保存）

        Returns:
            包含所有参数的字典
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseSubject":
        """
        从参数字典创建实例（用于加载）

        Args:
            data: 参数字典

        Returns:
            被试实例
        """
        pass

    def save(self, filepath: str):
        """
        保存到JSON文件

        Args:
            filepath: 文件路径
        """
        import json
        from pathlib import Path

        data = self.to_dict()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> "BaseSubject":
        """
        从JSON文件加载

        Args:
            filepath: 文件路径

        Returns:
            被试实例
        """
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls.from_dict(data)
