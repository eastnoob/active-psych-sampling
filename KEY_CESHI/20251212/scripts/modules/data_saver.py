#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据保存模块

提供标准化的实验数据保存接口，支持：
- 采样历史和交互日志
- EUR诊断数据（lambda_t, gamma_t, r_t等）
- Oracle规格和迭代详情CSV
- 增强的实验摘要（含效应恢复评估）
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging


def save_sampling_data(
    result_dir: Path,
    sampling_history: List,
    interaction_logs: List[Dict],
    eur_diagnostics: Optional[Dict[str, List]] = None,
    model_checkpoints: Optional[List] = None,
    basegp_keypoints: Optional[np.ndarray] = None
) -> None:
    """保存采样数据和诊断信息。

    Args:
        result_dir: 结果目录
        sampling_history: 采样历史 [(x1, x2, ..., xn), ...]
        interaction_logs: 交互日志 [{trial, x, y, x_config}, ...]
        eur_diagnostics: EUR诊断数据，包含：
            - lambda_t: EUR动态权重
            - gamma_t: 不确定性权重
            - r_t: 差异权重
            - model_entropy: 模型熵
            - acqf_mean: 采集函数均值
            - acqf_std: 采集函数标准差
            - min_distance: 最小距离
        model_checkpoints: 模型检查点列表
        basegp_keypoints: BaseGP关键点 (3, n_dims)
    """
    data_dir = result_dir / 'data_files'
    data_dir.mkdir(parents=True, exist_ok=True)

    # 保存采样历史
    history_file = data_dir / 'sampling_history.npy'
    np.save(history_file, np.array(sampling_history))
    logging.info(f"[DataSaver] Saved sampling history: {history_file}")

    # 保存交互日志
    log_file = data_dir / 'interaction_log.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(interaction_logs, f, indent=2, ensure_ascii=False)
    logging.info(f"[DataSaver] Saved interaction log: {log_file}")

    # 保存EUR诊断数据
    if eur_diagnostics:
        diagnostics_dir = data_dir / 'eur_diagnostics'
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        for key, values in eur_diagnostics.items():
            diag_file = diagnostics_dir / f'{key}.npy'
            np.save(diag_file, np.array(values))
            logging.info(f"[DataSaver] Saved EUR diagnostic '{key}': {diag_file}")

    # 保存模型检查点
    if model_checkpoints:
        checkpoint_file = data_dir / 'model_checkpoints.pt'
        torch.save(model_checkpoints, checkpoint_file)
        logging.info(f"[DataSaver] Saved model checkpoints: {checkpoint_file}")

    # 保存BaseGP关键点
    if basegp_keypoints is not None:
        keypoints_file = data_dir / 'basegp_keypoints.npy'
        np.save(keypoints_file, basegp_keypoints)
        logging.info(f"[DataSaver] Saved BaseGP keypoints: {keypoints_file}")


def update_summary_with_results(
    summary: Dict[str, Any],
    results: Dict[str, Any]
) -> Dict[str, Any]:
    """用结果更新摘要字典。

    Args:
        summary: 现有摘要字典
        results: 结果字典，可包含：
            - sampling_history: 采样历史
            - effects: 识别的效应
            - spearman_correlation: Spearman相关系数
            - eur_diagnostics: EUR诊断数据

    Returns:
        更新后的摘要
    """
    summary = summary.copy()

    # 基础信息
    summary.update({
        'n_trials': len(results.get('sampling_history', [])),
        'sampling_completed': True
    })

    # 效应识别结果
    if 'effects' in results:
        summary['identified_effects'] = results['effects']

    # Spearman相关
    if 'spearman_correlation' in results:
        summary['spearman_correlation'] = results['spearman_correlation']

    # EUR诊断统计
    if 'eur_diagnostics' in results:
        diagnostics = results['eur_diagnostics']
        summary['eur_stats'] = {
            key: {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            for key, values in diagnostics.items()
            if len(values) > 0
        }

    return summary


def save_experiment_summary(
    result_dir: Path,
    summary: Dict[str, Any]
) -> None:
    """保存实验摘要。

    Args:
        result_dir: 结果目录
        summary: 摘要字典
    """
    summary_file = result_dir / 'experiment_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info(f"[DataSaver] Saved experiment summary: {summary_file}")


def load_sampling_history(data_dir: Path) -> np.ndarray:
    """加载采样历史。

    Args:
        data_dir: 数据目录

    Returns:
        采样历史数组
    """
    history_file = data_dir / 'sampling_history.npy'
    if history_file.exists():
        return np.load(history_file)
    else:
        raise FileNotFoundError(f"Sampling history not found: {history_file}")


def load_interaction_log(data_dir: Path) -> List[Dict]:
    """加载交互日志。

    Args:
        data_dir: 数据目录

    Returns:
        交互日志列表
    """
    log_file = data_dir / 'interaction_log.json'
    if log_file.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Interaction log not found: {log_file}")


def load_eur_diagnostics(data_dir: Path) -> Dict[str, np.ndarray]:
    """加载EUR诊断数据。

    Args:
        data_dir: 数据目录

    Returns:
        诊断数据字典 {key: array}
    """
    diagnostics_dir = data_dir / 'eur_diagnostics'
    if not diagnostics_dir.exists():
        return {}

    diagnostics = {}
    for diag_file in diagnostics_dir.glob('*.npy'):
        key = diag_file.stem
        diagnostics[key] = np.load(diag_file)

    return diagnostics


# ========================================
# 新增函数：增强的输出格式
# ========================================

def save_oracle_spec(
    result_dir: Path,
    oracle: Any,
    source: str = "SimpleOracle"
) -> None:
    """保存Oracle模型规格到oracle_spec.json。

    Args:
        result_dir: 结果目录
        oracle: Oracle实例(需实现get_model_spec())
        source: Oracle来源描述
    """
    data_dir = result_dir / 'data_files'
    data_dir.mkdir(parents=True, exist_ok=True)

    # 获取Oracle规格
    spec = oracle.get_model_spec() if hasattr(oracle, 'get_model_spec') else {}

    # 添加来源信息
    spec['source'] = source

    # 保存
    spec_file = data_dir / 'oracle_spec.json'
    with open(spec_file, 'w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    logging.info(f"[DataSaver] Saved Oracle spec: {spec_file}")


def save_iterations_csv(
    result_dir: Path,
    interaction_logs: List[Dict],
    eur_diagnostics: Optional[Dict[str, List]] = None,
    warmup_budget: int = 3,
    model: Optional[object] = None,
    design_space: Optional[np.ndarray] = None,
    oracle: Optional[object] = None
) -> None:
    """生成iterations.csv,记录每次迭代的详细信息。

    Args:
        result_dir: 结果目录
        interaction_logs: 交互日志 [{trial, x_array, x_config, y}, ...]
        eur_diagnostics: EUR诊断数据 {lambda_t: [...], gamma_t: [...], r_t: [...], ...}
        warmup_budget: warmup阶段预算数(用于判断phase)
        model: 当前模型实例(用于计算learning curves)
        design_space: 完整设计空间 (N, d)
        oracle: Oracle实例(用于获取真实y值)
    """
    data_dir = result_dir / 'data_files'
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log in interaction_logs:
        trial = log['trial']
        x_array = log['x_array']
        y_value = log['y']

        # 确定phase
        phase = 'warmup' if trial < warmup_budget else 'eur'

        # 基础行
        row = {
            'iteration': trial + 1,  # 1-indexed
            'phase': phase,
            'y_value': y_value
        }

        # EUR诊断数据(仅在eur阶段有效)
        if eur_diagnostics and trial >= warmup_budget:
            eur_idx = trial - warmup_budget
            for key in ['lambda_t', 'gamma_t', 'r_t', 'acqf_mean', 'acqf_std', 'min_distance']:
                if key in eur_diagnostics and eur_idx < len(eur_diagnostics[key]):
                    row[key] = eur_diagnostics[key][eur_idx]
                else:
                    row[key] = None
            # n_train = warmup + eur已完成
            row['n_train'] = warmup_budget + eur_idx
        else:
            # warmup阶段没有EUR诊断数据
            for key in ['lambda_t', 'gamma_t', 'r_t', 'n_train', 'acqf_mean', 'acqf_std', 'min_distance']:
                row[key] = None

        # === 新增：Learning curves ===
        if phase == 'eur' and model is not None and design_space is not None and oracle is not None:
            # 计算累积性能
            with torch.no_grad():
                y_oracle = np.array([oracle.query(x) for x in design_space])
                # Handle ordinal models
                if hasattr(model.likelihood, "cutpoints"):
                    probs = model.predict_probs(torch.from_numpy(design_space).to(torch.float64))
                    y_pred = torch.argmax(probs, dim=1).cpu().numpy()
                else:
                    posterior = model.posterior(torch.from_numpy(design_space).to(torch.float64))
                    y_pred = posterior.mean.squeeze(-1).cpu().numpy()

                rmse = np.sqrt(np.mean((y_oracle - y_pred)**2))
                ss_res = np.sum((y_oracle - y_pred)**2)
                ss_tot = np.sum((y_oracle - y_oracle.mean())**2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                mae = np.mean(np.abs(y_oracle - y_pred))

                row['cumulative_rmse'] = rmse
                row['cumulative_r2'] = r2
                row['cumulative_mae'] = mae
        else:
            row['cumulative_rmse'] = None
            row['cumulative_r2'] = None
            row['cumulative_mae'] = None

        # 参数值(使用实际值,x_array)
        for i, x_val in enumerate(x_array):
            row[f'x{i}'] = x_val

        rows.append(row)

    # 构建DataFrame并保存
    df = pd.DataFrame(rows)

    # 列顺序
    base_cols = ['iteration', 'phase', 'y_value', 'lambda_t', 'gamma_t', 'r_t', 'n_train',
                 'acqf_mean', 'acqf_std', 'min_distance',
                 'cumulative_rmse', 'cumulative_r2', 'cumulative_mae']
    x_cols = [f'x{i}' for i in range(len(interaction_logs[0]['x_array']))]
    df = df[base_cols + x_cols]

    csv_file = data_dir / 'iterations.csv'
    df.to_csv(csv_file, index=False)
    logging.info(f"[DataSaver] Saved iterations CSV: {csv_file}")


def save_enhanced_summary(
    result_dir: Path,
    config: Dict[str, Any],
    oracle_spec: Dict[str, Any],
    interaction_logs: List[Dict],
    eur_diagnostics: Optional[Dict[str, List]],
    eval_results: Optional[Dict[str, Any]],
    warmup_budget: int,
    seed: Optional[int] = None,
    eval_results_v3: Optional[Dict[str, Any]] = None,
    eval_results_v4: Optional[Dict[str, Any]] = None
) -> None:
    """保存增强的实验摘要(summary.json)。

    Args:
        result_dir: 结果目录
        config: 配置信息
        oracle_spec: Oracle规格
        interaction_logs: 交互日志
        eur_diagnostics: EUR诊断数据
        eval_results: 评估结果(来自evaluate_effect_recovery_v2)
        warmup_budget: warmup预算
        seed: 随机种子
        eval_results_v3: 效应捕捉评估结果(来自evaluate_effect_capture)
        eval_results_v4: 自动模型对比评估结果(来自evaluate_effect_capture_v4)
    """
    data_dir = result_dir / 'data_files'
    timestamp = result_dir.name

    # 构建summary结构
    summary = {
        # 基础配置
        "experiment": {
            "timestamp": timestamp,
            "config": config.get('config_name', 'eur_residual_test.ini'),
            "tag": config.get('tag', None),
            "result_dir": str(result_dir),
            "config_path": config.get('config_path', '')
        },

        "parameters": {
            "total_budget": len(interaction_logs),
            "warmup_budget": warmup_budget,
            "eur_budget": len(interaction_logs) - warmup_budget,
            "seed": seed,
            "interaction_pairs": config.get('interaction_pairs', []),
            "parnames": config.get('parnames', [])
        },

        # Oracle规格
        "oracle": {
            "model_type": oracle_spec.get('model_type', 'linear'),
            "bias": oracle_spec.get('bias', 0.0),
            "noise_std": oracle_spec.get('noise_std', 0.0),
            "main_weights": oracle_spec.get('weights', []),
            "interaction_weights": oracle_spec.get('interaction_terms', {})
        }
    }

    # 采样轨迹统计
    if eur_diagnostics:
        def compute_stats(values: List) -> Dict:
            """Compute statistics, filtering out None values"""
            if not values:
                return {}
            # Filter out None values
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                return {}
            arr = np.array(valid_values)
            return {
                'initial': float(arr[0]),
                'final': float(arr[-1]),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr))
            }

        summary['sampling_trajectory'] = {
            'lambda_t': compute_stats(eur_diagnostics.get('lambda_t', [])),
            'gamma_t': compute_stats(eur_diagnostics.get('gamma_t', [])),
            'r_t': compute_stats(eur_diagnostics.get('r_t', []))
        }

        # 采样多样性
        min_distances = eur_diagnostics.get('min_distance', [])
        valid_min_distances = [d for d in min_distances if d is not None]
        if valid_min_distances:
            summary['sampling_trajectory']['diversity'] = {
                'min_distance_mean': float(np.mean(valid_min_distances)),
                'unique_samples': len(interaction_logs)
            }

    # 效应恢复评估
    if eval_results:
        summary['effect_recovery'] = {
            'main_effects': {
                'correlation': eval_results.get('main_correlation', {}),
                'rmse': eval_results.get('main_effects', {}).get('rmse', None),
                'mae': eval_results.get('main_effects', {}).get('mae', None)
            },
            'interaction_effects': {
                'correlation': eval_results.get('interaction_correlation', {}),
                'rmse': eval_results.get('interaction_effects', {}).get('rmse', None),
                'mae': eval_results.get('interaction_effects', {}).get('mae', None)
            },
            'model_fit': eval_results.get('model_fit', {})
        }

        # 预测质量
        pred_quality = eval_results.get('prediction_quality', {})
        summary['prediction_quality'] = pred_quality

        # 效应对比(精简版,不存完整数组)
        if 'main_effects' in eval_results:
            main_eff = eval_results['main_effects']
            oracle_main = main_eff.get('oracle', [])
            estimated_main = main_eff.get('estimated', [])
            if oracle_main and estimated_main:
                errors = [abs(o - e) for o, e in zip(oracle_main, estimated_main)]
                summary['effect_comparison'] = {
                    'main_effects': {
                        'oracle': oracle_main,
                        'estimated': estimated_main,
                        'error': errors
                    }
                }

        if 'interaction_effects' in eval_results:
            int_eff = eval_results['interaction_effects']
            oracle_int = int_eff.get('oracle', [])
            estimated_int = int_eff.get('estimated', [])
            if oracle_int and estimated_int:
                errors = [abs(o - e) for o, e in zip(oracle_int, estimated_int)]
                if 'effect_comparison' not in summary:
                    summary['effect_comparison'] = {}
                summary['effect_comparison']['interaction_effects'] = {
                    'pairs': int_eff.get('pairs', []),
                    'oracle': oracle_int,
                    'estimated': estimated_int,
                    'error': errors
                }

    # 效应捕捉评估 (v3)
    if eval_results_v3:
        summary['effect_capture_v3'] = {
            'core_metrics': eval_results_v3.get('core_metrics', {}),
            'quality_metrics': eval_results_v3.get('quality_metrics', {}),
            'reference_metrics': eval_results_v3.get('reference_metrics', {}),
            'model_fit': eval_results_v3.get('model_fit', {})
        }

    # 自动模型对比评估 (v4)
    if eval_results_v4:
        summary['effect_capture_v4'] = {
            'evaluation_mode': eval_results_v4.get('evaluation_mode', 'auto_discovery'),
            'structure_discovery': eval_results_v4.get('structure_discovery', {}),
            'best_model': eval_results_v4.get('best_model', {}),
            'model_comparison_table': eval_results_v4.get('model_comparison_table', []),
            'statistical_power': eval_results_v4.get('statistical_power', {}),
            'effect_size_comparison': eval_results_v4.get('effect_size_comparison', {}),
            'effect_discovery_timeline': eval_results_v4.get('effect_discovery_timeline', {})
        }

    # 保存
    summary_file = data_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logging.info(f"[DataSaver] Saved enhanced summary: {summary_file}")
