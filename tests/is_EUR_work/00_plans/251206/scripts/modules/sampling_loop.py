"""
采样循环执行模块
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional
import logging


def run_sampling_loop(
    server,
    oracle,
    budget: int,
    sobol_init: int = 0,
    parnames: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    运行采样循环

    Args:
        server: AEPsych server实例
        oracle: Oracle实例
        budget: 总采样预算
        sobol_init: Sobol初始化样本数
        parnames: 参数名称列表

    Returns:
        包含采样历史的字典
    """
    sampling_history = []
    interaction_logs = []

    logging.info(f"Starting sampling loop: budget={budget}, sobol_init={sobol_init}")

    for trial_idx in range(budget):
        # Ask for next configuration
        x_config = server.ask()

        # Convert to array
        if parnames is None:
            parnames = list(x_config.keys())

        x_array = np.array([x_config[name] for name in parnames], dtype=np.float64)

        # Query oracle
        y = oracle.query(x_array)

        # Tell server
        server.tell(x_config, y)

        # Log
        sampling_history.append(x_array.tolist())
        interaction_logs.append({
            'trial': trial_idx,
            'x': x_array.tolist(),
            'y': y,
            'x_config': x_config
        })

        if (trial_idx + 1) % 10 == 0:
            logging.info(f"  Trial {trial_idx + 1}/{budget} completed")

    return {
        'sampling_history': sampling_history,
        'interaction_logs': interaction_logs
    }
