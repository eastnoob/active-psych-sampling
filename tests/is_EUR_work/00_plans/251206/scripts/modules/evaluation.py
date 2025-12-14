"""
效应识别和预测质量评估模块
"""

import torch
import numpy as np
from typing import Optional, Dict, Set, List, Tuple


def identify_effects_from_model(
    server,
    oracle,
    ordinal_helper_class=None,
    configured_pairs: Optional[List[Tuple[int, int]]] = None,
    threshold_percentile: int = 75,
    oracle_spec: Optional[Dict] = None
) -> Dict:
    """
    从训练好的模型识别主效应和交互效应 (ANOVA风格分析)

    Args:
        server: AEPsych server实例
        oracle: Oracle实例
        ordinal_helper_class: Ordinal辅助类
        configured_pairs: 预配置的交互对
        threshold_percentile: 阈值百分位
        oracle_spec: Oracle规格

    Returns:
        包含识别的效应的字典
    """
    results = {
        'main_effects': [],
        'interaction_effects': [],
        'configured_pairs': configured_pairs or []
    }

    try:
        strat = server._strats[0]
        model = strat.model

        # 提取权重(如果可用)
        if hasattr(model, 'weights'):
            weights = model.weights
            # 识别主效应
            for i, w in enumerate(weights):
                if abs(w) > 0.1:  # 简单阈值
                    results['main_effects'].append({'index': i, 'weight': float(w)})

    except Exception as e:
        print(f"Effect identification error: {e}")

    return results


def evaluate_prediction_quality(
    server,
    test_x: torch.Tensor,
    test_y: torch.Tensor
) -> Dict[str, float]:
    """
    评估模型预测质量

    Args:
        server: AEPsych server实例
        test_x: 测试输入
        test_y: 测试标签

    Returns:
        评估指标字典
    """
    with torch.no_grad():
        try:
            strat = server._strats[0]
            model = strat.model

            # 预测
            pred = model.predict(test_x)

            # 计算MSE
            mse = float(torch.mean((pred - test_y) ** 2))

            return {
                'mse': mse,
                'n_test': len(test_y)
            }
        except Exception as e:
            print(f"Evaluation error: {e}")
            return {'error': str(e)}
