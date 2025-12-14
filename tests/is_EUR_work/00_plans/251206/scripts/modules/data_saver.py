"""
数据保存模块
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import logging


def save_sampling_data(
    result_dir: Path,
    sampling_history: List,
    interaction_logs: List[Dict]
) -> None:
    """
    保存采样数据

    Args:
        result_dir: 结果目录
        sampling_history: 采样历史
        interaction_logs: 交互日志
    """
    data_dir = result_dir / 'data_files'
    data_dir.mkdir(parents=True, exist_ok=True)

    # 保存采样历史
    history_file = data_dir / 'sampling_history.npy'
    np.save(history_file, np.array(sampling_history))
    logging.info(f"Saved sampling history to {history_file}")

    # 保存交互日志
    log_file = data_dir / 'interaction_log.json'
    with open(log_file, 'w') as f:
        json.dump(interaction_logs, f, indent=2)
    logging.info(f"Saved interaction log to {log_file}")


def update_summary_with_results(
    summary: Dict[str, Any],
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """
    用结果更新摘要字典

    Args:
        summary: 现有摘要字典
        results: 结果字典

    Returns:
        更新后的摘要
    """
    summary.update({
        'n_trials': len(results.get('sampling_history', [])),
        'sampling_completed': True
    })

    if 'effects' in results:
        summary['identified_effects'] = results['effects']

    return summary
