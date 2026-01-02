#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle模型创建和管理模块
负责创建模拟被试的Oracle模型
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any


def load_fixed_weights_from_json(json_path: Path) -> np.ndarray:
    """
    从JSON文件加载固定权重

    Args:
        json_path: JSON文件路径（如 fixed_weights_auto.json）

    Returns:
        fixed_weights: 权重矩阵 (num_outputs, num_features)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # 提取global权重（假设格式为 {"global": [[w0, w1, ..., w5]]}）
    if "global" in data:
        weights = np.array(data["global"], dtype=np.float64)
    else:
        raise ValueError(f"JSON文件格式错误，缺少'global'键: {json_path}")

    print(f"✓ 从 {json_path.name} 加载固定权重: {weights.shape}")
    print(f"  权重值: {weights[0]}")

    return weights


def create_oracle(
    oracle_class,
    seed: int = 123,
    bias: float = 2.93,
    weight_std: float = 0.7,
    noise_std: float = 0.77,
    interaction_pairs: list = None,
    interaction_scale: float = 0.45,
    likert_levels: int = 5,
    likert_sensitivity: float = 1.0,
    strategy_seed: Optional[int] = None,
    fixed_weights_path: Optional[Path] = None,
    use_latent: bool = True,
):
    """
    创建Oracle模型实例

    Args:
        oracle_class: Oracle类（如SingleSubject）
        seed: Oracle随机种子
        bias: 截距（群体均值）
        weight_std: 主效应权重标准差（仅当fixed_weights_path=None时使用）
        noise_std: 测量噪声标准差
        interaction_pairs: 交互对列表，如[(1,2), (3,4), (0,5)]
        interaction_scale: 交互效应缩放系数
        likert_levels: Likert量表级别数
        likert_sensitivity: Sigmoid灵敏度
        strategy_seed: 策略随机种子（影响采样顺序）
        fixed_weights_path: 固定权重JSON文件路径（如提供，将使用固定权重而非随机生成）
        use_latent: 是否使用潜变量结构（与fixed_weights配合使用）

    Returns:
        oracle: Oracle实例
    """
    if interaction_pairs is None:
        interaction_pairs = [(1, 2), (3, 4), (0, 5)]

    # 设置策略随机种子
    if strategy_seed is not None:
        print(f"策略种子: {strategy_seed}")
        np.random.seed(strategy_seed)

    # 加载固定权重（如果提供）
    fixed_weights = None
    if fixed_weights_path is not None:
        fixed_weights = load_fixed_weights_from_json(fixed_weights_path)
        print(f"✓ 使用固定权重模式（从JSON加载）")

    # 创建Oracle参数
    oracle_kwargs = {
        "seed": seed,
        "bias": bias,
        "noise_std": noise_std,
        "interaction_pairs": interaction_pairs,
        "interaction_scale": interaction_scale,
        "likert_levels": likert_levels,
        "likert_sensitivity": likert_sensitivity,
    }

    # 如果提供了固定权重，必须使用SingleOutputLatentSubject
    if fixed_weights is not None:
        from single_output_subject import SingleOutputLatentSubject

        oracle_kwargs["use_latent"] = use_latent
        oracle_kwargs["fixed_weights"] = fixed_weights
        oracle_kwargs["num_features"] = fixed_weights.shape[1]  # 从权重矩阵推断特征数
        # 不再需要weight_std参数（使用固定权重）

        oracle = SingleOutputLatentSubject(**oracle_kwargs)
        print(f"✓ Oracle 类型: SingleOutputLatentSubject (固定权重)")
    else:
        # 使用随机生成权重模式
        oracle_kwargs["weight_std"] = weight_std

        oracle = oracle_class(**oracle_kwargs)
        print(f"✓ Oracle 类型: {type(oracle).__name__} (随机权重)")

    return oracle


def print_oracle_spec(oracle, save_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    打印并保存Oracle模型规格

    Args:
        oracle: Oracle实例
        save_path: 保存路径（可选）

    Returns:
        model_spec: 模型规格字典
    """
    try:
        model_spec = oracle.get_model_spec()

        print(f"\n{'='*60}")
        print(f"【被试模型规格】")
        print(f"{'='*60}")
        print(f"  模型类型: {model_spec.get('model_type', 'unknown')}")
        print(f"  特征数量: {model_spec.get('num_features', 0)}")
        print(f"  Likert级别: {model_spec.get('likert_levels', 0)}")
        print(f"  噪声标准差: {model_spec.get('noise_std', 0):.4f}")
        print(f"  权重标准差: {model_spec.get('weight_std', 0):.4f}")

        print(f"\n  主效应权重:")
        weights = model_spec.get("weights", [])
        for i, w in enumerate(weights):
            print(f"    x{i}: {w:+.6f}")

        print(f"\n  交互项权重:")
        int_weights = model_spec.get("interaction_terms", {})
        if not int_weights:
            int_weights = model_spec.get("interaction_weights", {})

        if int_weights:
            for term_name, weight in int_weights.items():
                print(f"    {term_name}: {weight:+.6f}")
        else:
            print(f"    (无交互项)")

        print(f"\n  模型公式:")
        print(f"    y = bias + ∑(w_i · x_i) + ∑(w_ij · x_i · x_j) + ε")
        print(f"    其中 ε ~ N(0, {model_spec.get('noise_std', 0):.4f}²)")
        print(f"    bias = {model_spec.get('bias', 0):.6f}")
        print(f"{'='*60}\n")

        # 保存模型规格
        if save_path is not None:
            save_oracle_spec(model_spec, save_path)

        return model_spec

    except Exception as e:
        print(f"⚠ 警告: 无法获取模型规格: {e}")
        return {}


def save_oracle_spec(model_spec: dict, save_path: Path):
    """
    保存Oracle模型规格为JSON文件

    Args:
        model_spec: 模型规格字典
        save_path: 保存路径
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为可JSON序列化的格式
    oracle_spec_serializable = {
        "model_type": model_spec.get("model_type"),
        "num_features": model_spec.get("num_features"),
        "likert_levels": model_spec.get("likert_levels"),
        "noise_std": float(model_spec.get("noise_std", 0)),
        "weight_std": float(model_spec.get("weight_std", 0)),
        "bias": float(model_spec.get("bias", 0)),
        "weights": [float(w) for w in model_spec.get("weights", [])],
        "interaction_terms": {
            term: float(weight)
            for term, weight in model_spec.get("interaction_terms", {}).items()
        },
        "formula": "y = bias + ∑(w_i · x_i) + ∑(w_ij · x_i · x_j) + ε",
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(oracle_spec_serializable, f, indent=2, ensure_ascii=False)

    print(f"✓ Oracle模型规格已保存至: {save_path}")
