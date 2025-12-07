"""
Linear subject model
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
from .base import BaseSubject


class LinearSubject(BaseSubject):
    """
    线性被试模型

    公式：
        y_continuous = bias + Σ(weights[i] * x[i]) + Σ(interaction_weights * x[i] * x[j]) + noise
        y_likert = likert_transform(y_continuous)

    参数：
        weights: 主效应权重 (n_features,)
        interaction_weights: 交互效应权重 dict{(i,j): weight}
        bias: 截距
        noise_std: 试次内噪声标准差
        likert_levels: Likert量表等级数（None表示连续输出）
        likert_sensitivity: Likert转换灵敏度
        seed: 随机种子（用于噪声）

    示例：
        >>> subject = LinearSubject(
        ...     weights=np.array([0.2, -0.3, 0.5]),
        ...     interaction_weights={(0, 1): -0.15},
        ...     bias=0.0,
        ...     noise_std=0.1,
        ...     likert_levels=5,
        ...     likert_sensitivity=2.0
        ... )
        >>> x = np.array([1.0, 2.0, 0.5])
        >>> y = subject(x)  # 返回Likert评分
    """

    def __init__(
        self,
        weights: np.ndarray,
        interaction_weights: Optional[Dict[Tuple[int, int], float]] = None,
        bias: float = 0.0,
        noise_std: float = 0.0,
        likert_levels: Optional[int] = None,
        likert_sensitivity: float = 2.0,
        seed: Optional[int] = None
    ):
        """
        初始化线性被试模型

        Args:
            weights: 主效应权重 (n_features,)
            interaction_weights: 交互效应权重 {(i,j): weight}
            bias: 截距
            noise_std: 试次内噪声标准差（0表示确定性）
            likert_levels: Likert等级数（None表示连续输出）
            likert_sensitivity: Likert转换灵敏度（值越大，分布越集中）
            seed: 随机种子
        """
        self.weights = np.array(weights, dtype=float)
        self.interaction_weights = interaction_weights or {}
        self.bias = float(bias)
        self.noise_std = max(0.0, float(noise_std))
        self.likert_levels = int(likert_levels) if likert_levels is not None else None
        self.likert_sensitivity = max(1e-6, float(likert_sensitivity))
        self.seed = int(seed) if seed is not None else None

        # 初始化随机数生成器
        if self.seed is not None:
            np.random.seed(self.seed)

    def __call__(self, x: np.ndarray) -> Union[float, int]:
        """
        被试作答

        Args:
            x: 输入特征向量 (n_features,)

        Returns:
            Likert评分（如果likert_levels不为None）或连续值

        Raises:
            ValueError: 输入维度与weights不匹配
        """
        x = np.array(x, dtype=float)

        if len(x) != len(self.weights):
            raise ValueError(
                f"Input dimension mismatch: expected {len(self.weights)}, got {len(x)}"
            )

        # 主效应
        y = self.bias + np.dot(self.weights, x)

        # 交互效应
        for (i, j), weight in self.interaction_weights.items():
            y += weight * x[i] * x[j]

        # 试次内噪声
        if self.noise_std > 0:
            y += np.random.normal(0, self.noise_std)

        # Likert转换
        if self.likert_levels is not None:
            return self._to_likert(y)
        else:
            return float(y)

    def _to_likert(self, value: float) -> int:
        """
        连续值转Likert评分

        使用tanh函数进行非线性转换，使分布更接近真实人类响应

        Args:
            value: 连续值

        Returns:
            Likert评分 [1, likert_levels]
        """
        levels = self.likert_levels
        sensitivity = self.likert_sensitivity

        # tanh转换：将实数映射到(-1, 1)
        tanh_val = np.tanh(value * sensitivity)

        # 映射到Likert范围：(-1, 1) → [1, levels]
        # formula: tanh_val * (levels-1)/2 + (levels+1)/2
        likert_float = tanh_val * (levels - 1) / 2 + (levels + 1) / 2

        # 四舍五入并限制范围
        likert = int(np.round(likert_float))
        likert = max(1, min(levels, likert))

        return likert

    def to_dict(self) -> Dict[str, Any]:
        """
        导出为完整参数字典

        Returns:
            包含所有参数的字典
        """
        return {
            "model_type": "linear",
            "weights": self.weights.tolist(),
            "interaction_weights": {
                f"{i},{j}": float(w) for (i, j), w in self.interaction_weights.items()
            },
            "bias": self.bias,
            "noise_std": self.noise_std,
            "likert_levels": self.likert_levels,
            "likert_sensitivity": self.likert_sensitivity,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LinearSubject":
        """
        从参数字典创建实例

        Args:
            data: 参数字典

        Returns:
            LinearSubject实例

        Raises:
            ValueError: 如果model_type不是"linear"
        """
        if data.get("model_type") != "linear":
            raise ValueError(f"Expected model_type='linear', got '{data.get('model_type')}'")

        # 解析interaction_weights
        interaction_weights = {}
        for key, value in data.get("interaction_weights", {}).items():
            i, j = map(int, key.split(','))
            interaction_weights[(i, j)] = float(value)

        return cls(
            weights=np.array(data["weights"]),
            interaction_weights=interaction_weights,
            bias=data.get("bias", 0.0),
            noise_std=data.get("noise_std", 0.0),
            likert_levels=data.get("likert_levels"),
            likert_sensitivity=data.get("likert_sensitivity", 2.0),
            seed=data.get("seed"),
        )

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"LinearSubject("
            f"n_features={len(self.weights)}, "
            f"n_interactions={len(self.interaction_weights)}, "
            f"likert_levels={self.likert_levels})"
        )
