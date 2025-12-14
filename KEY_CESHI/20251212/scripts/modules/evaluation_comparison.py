#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_comparison.py - EUR vs Random Comparison Evaluation Module

Provides metrics for comparing EUR and Random sampling strategies:
1. Sample efficiency ratio
2. Top-k model stability across runs
3. Cross-run stability (effect discovery timeline)
4. Win rate statistics
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from itertools import combinations
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


def calculate_sample_efficiency(
    eur_iterations_csv: pd.DataFrame,
    random_iterations_csv: pd.DataFrame,
    criteria: Optional[Dict[str, float]] = None
) -> Dict:
    """
    计算样本效率比。

    Args:
        eur_iterations_csv: EUR的iterations.csv数据
        random_iterations_csv: Random的iterations.csv数据
        criteria: 达标标准 {'structure_f1': 0.8, 'spearman_corr': 0.85, 'cumulative_r2': 0.75}

    Returns:
        效率比指标字典
    """
    if criteria is None:
        criteria = {
            'cumulative_r2': 0.75
        }

    logger.info(f"Calculating sample efficiency with criteria: {criteria}")

    # 查找EUR达标点
    eur_达标_iter = None
    达标_metric = None
    for metric_name, threshold in criteria.items():
        if metric_name in eur_iterations_csv.columns:
            达标_rows = eur_iterations_csv[eur_iterations_csv[metric_name] >= threshold]
            if not 达标_rows.empty:
                eur_达标_iter = int(达标_rows.iloc[0]['iteration'])
                达标_metric = metric_name
                logger.info(f"  EUR达标 at iteration {eur_达标_iter} ({metric_name} >= {threshold})")
                break

    # 查找Random达标点
    random_达标_iter = None
    for metric_name, threshold in criteria.items():
        if metric_name in random_iterations_csv.columns:
            达标_rows = random_iterations_csv[random_iterations_csv[metric_name] >= threshold]
            if not 达标_rows.empty:
                random_达标_iter = int(达标_rows.iloc[0]['iteration'])
                logger.info(f"  Random达标 at iteration {random_达标_iter} ({metric_name} >= {threshold})")
                break

    # 计算效率比
    budget = len(eur_iterations_csv)
    eur_iter = eur_达标_iter if eur_达标_iter else budget + 1
    random_iter = random_达标_iter if random_达标_iter else budget + 1

    efficiency_ratio = eur_iter / random_iter
    absolute_savings = random_iter - eur_iter

    logger.info(f"  Efficiency ratio: {efficiency_ratio:.3f} (EUR {eur_iter} vs Random {random_iter})")

    return {
        'criteria': criteria,
        'eur_达标_iteration': eur_达标_iter,
        'random_达标_iteration': random_达标_iter,
        'efficiency_ratio': float(efficiency_ratio),
        'absolute_savings': int(absolute_savings),
        '达标_metric': 达标_metric
    }


def calculate_topk_stability(
    run_results: List[Dict],
    k: int = 5
) -> Dict:
    """
    计算Top-k模型稳定性。

    Args:
        run_results: 多次run的evaluation_v4结果 (summary.json)
        k: Top-k的k值

    Returns:
        稳定性指标字典
    """
    logger.info(f"Calculating Top-{k} model stability across {len(run_results)} runs")

    # 提取每次run的Top-k模型结构
    topk_signatures = []
    for result in run_results:
        models = result['effect_capture_v4']['model_comparison_table'][:k]
        signatures = [
            frozenset(tuple(pair) for pair in m['interactions'])  # Convert to frozenset of tuples
            for m in models
        ]
        topk_signatures.append(set(signatures))

    # 计算pairwise Jaccard similarity
    jaccard_scores = []
    for sig_a, sig_b in combinations(topk_signatures, 2):
        intersection = len(sig_a & sig_b)
        union = len(sig_a | sig_b)
        jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)

    # 统计最频繁的模型结构
    all_structures = [sig for sigs in topk_signatures for sig in sigs]
    structure_counts = Counter(all_structures)
    most_frequent = [
        {
            'structure': [f"x{i}*x{j}" for i, j in sorted(sig)],
            'frequency': count
        }
        for sig, count in structure_counts.most_common(5)
    ]

    mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 0.0
    std_jaccard = float(np.std(jaccard_scores)) if jaccard_scores else 0.0

    logger.info(f"  Mean Jaccard: {mean_jaccard:.3f} ± {std_jaccard:.3f}")

    return {
        'k': k,
        'n_runs': len(run_results),
        'mean_jaccard': mean_jaccard,
        'std_jaccard': std_jaccard,
        'pairwise_similarities': [float(x) for x in jaccard_scores],
        'most_frequent_models': most_frequent
    }


def calculate_discovery_stability(
    run_results: List[Dict]
) -> Dict:
    """
    计算效应发现时间点稳定性。

    Args:
        run_results: 多次run的evaluation结果 (summary.json)，包含effect_discovery_timeline字段

    Returns:
        稳定性指标字典
    """
    logger.info(f"Calculating discovery timeline stability across {len(run_results)} runs")

    # 收集所有效应的发现时间
    discovery_times_per_effect = defaultdict(list)

    for result in run_results:
        timeline = result['effect_capture_v4']['effect_discovery_timeline']
        for effect_name, iteration in timeline.items():
            if iteration is not None:  # 过滤未发现的效应
                discovery_times_per_effect[effect_name].append(iteration)

    # 计算每个效应的稳定性指标
    per_effect = {}
    cvs = []
    mads = []

    for effect_name, times in discovery_times_per_effect.items():
        if len(times) < 2:  # 至少需要2次数据
            continue

        times_arr = np.array(times)
        mean_time = np.mean(times_arr)
        std_time = np.std(times_arr)
        cv = std_time / mean_time if mean_time > 0 else np.inf

        median_time = np.median(times_arr)
        mad = np.median(np.abs(times_arr - median_time))

        per_effect[effect_name] = {
            'mean_discovery_iter': float(mean_time),
            'std_discovery_iter': float(std_time),
            'cv': float(cv),
            'mad': float(mad),
            'discovery_times': [int(t) for t in times]
        }

        if not np.isinf(cv):
            cvs.append(cv)
        mads.append(mad)

    mean_cv = float(np.mean(cvs)) if cvs else None
    mean_mad = float(np.mean(mads)) if mads else None

    logger.info(f"  Mean CV: {mean_cv:.3f}" if mean_cv else "  Mean CV: None")
    logger.info(f"  Mean MAD: {mean_mad:.3f}" if mean_mad else "  Mean MAD: None")

    return {
        'n_runs': len(run_results),
        'per_effect': per_effect,
        'overall': {
            'mean_cv': mean_cv,
            'mean_mad': mean_mad
        }
    }


def calculate_win_rate(
    paired_results: List[Dict],
    criteria: Optional[List[str]] = None
) -> Dict:
    """
    计算EUR vs Random的胜率。

    Args:
        paired_results: 配对实验结果列表 [{'eur': {...}, 'random': {...}, 'sample_efficiency': {...}}, ...]
        criteria: 评判标准列表

    Returns:
        胜率统计字典
    """
    if criteria is None:
        criteria = ['cumulative_r2', 'sample_efficiency']

    logger.info(f"Calculating win rate for {len(paired_results)} paired runs")
    logger.info(f"  Criteria: {criteria}")

    # 初始化计数器
    per_criterion = {c: {'eur_better': 0, 'random_better': 0} for c in criteria}
    outcomes = []

    for pair_id, pair in enumerate(paired_results):
        # 比较各维度
        scores = []
        for criterion in criteria:
            if criterion == 'cumulative_r2':
                # 从iterations.csv获取最终R²
                eur_val = pair['eur'].get('final_r2', 0)
                random_val = pair['random'].get('final_r2', 0)
                eur_better = eur_val > random_val
            elif criterion == 'sample_efficiency':
                # 从sample_efficiency字段获取效率比
                eur_val = pair.get('sample_efficiency', {}).get('efficiency_ratio', 1.0)
                random_val = 1.0  # Random作为baseline
                # EUR越小越好
                eur_better = eur_val < random_val
            else:
                # 从effect_capture_v4获取其他指标
                eur_result = pair['eur'].get('effect_capture_v4', {})
                random_result = pair['random'].get('effect_capture_v4', {})

                if criterion == 'structure_f1':
                    eur_val = eur_result.get('structure_discovery', {}).get('f1_score', 0)
                    random_val = random_result.get('structure_discovery', {}).get('f1_score', 0)
                else:
                    eur_val = eur_result.get(criterion, 0)
                    random_val = random_result.get(criterion, 0)

                eur_better = eur_val > random_val

            scores.append(1 if eur_better else 0)
            if eur_better:
                per_criterion[criterion]['eur_better'] += 1
            else:
                per_criterion[criterion]['random_better'] += 1

        # 判定胜负
        eur_wins = sum(scores)
        if eur_wins == len(criteria):
            outcome = 'strong_win'
        elif eur_wins >= len(criteria) // 2 + 1:
            outcome = 'win'
        elif eur_wins == len(criteria) // 2:
            outcome = 'tie'
        else:
            outcome = 'loss'

        outcomes.append({'pair_id': pair_id, 'outcome': outcome, 'scores': scores})

    # 统计总体胜率
    strong_wins = sum(1 for o in outcomes if o['outcome'] == 'strong_win')
    wins = sum(1 for o in outcomes if o['outcome'] == 'win')
    ties = sum(1 for o in outcomes if o['outcome'] == 'tie')
    losses = sum(1 for o in outcomes if o['outcome'] == 'loss')

    win_rate = (strong_wins + wins) / len(paired_results) if paired_results else 0.0

    logger.info(f"  Overall win rate: {win_rate:.2%} (strong: {strong_wins}, win: {wins}, tie: {ties}, loss: {losses})")

    # 计算各维度胜率
    for criterion in criteria:
        total = per_criterion[criterion]['eur_better'] + per_criterion[criterion]['random_better']
        per_criterion[criterion]['win_rate'] = (
            per_criterion[criterion]['eur_better'] / total if total > 0 else 0.0
        )

    return {
        'n_pairs': len(paired_results),
        'criteria': criteria,
        'overall': {
            'strong_wins': strong_wins,
            'wins': wins,
            'ties': ties,
            'losses': losses,
            'win_rate': float(win_rate)
        },
        'per_criterion': per_criterion,
        'detailed_outcomes': outcomes
    }


def compare_paired_runs(
    eur_result_dir: Path,
    random_result_dir: Path,
    output_dir: Path
) -> Dict:
    """
    对比单对EUR vs Random实验结果。

    Args:
        eur_result_dir: EUR实验结果目录
        random_result_dir: Random实验结果目录
        output_dir: 对比结果输出目录

    Returns:
        对比结果字典
    """
    logger.info(f"Comparing paired runs:")
    logger.info(f"  EUR: {eur_result_dir}")
    logger.info(f"  Random: {random_result_dir}")

    # 加载两个实验的summary.json和iterations.csv
    eur_summary = json.load(open(eur_result_dir / 'data_files' / 'summary.json'))
    random_summary = json.load(open(random_result_dir / 'data_files' / 'summary.json'))

    eur_iterations = pd.read_csv(eur_result_dir / 'data_files' / 'iterations.csv')
    random_iterations = pd.read_csv(random_result_dir / 'data_files' / 'iterations.csv')

    # 计算样本效率
    sample_efficiency = calculate_sample_efficiency(eur_iterations, random_iterations)

    # 构建对比结果
    comparison = {
        'eur_result_dir': str(eur_result_dir),
        'random_result_dir': str(random_result_dir),
        'sample_efficiency': sample_efficiency,
        'structure_discovery': {
            'eur': eur_summary.get('effect_capture_v4', {}).get('structure_discovery', {}),
            'random': random_summary.get('effect_capture_v4', {}).get('structure_discovery', {})
        },
        'final_performance': {
            'eur': {
                'r2': float(eur_iterations['cumulative_r2'].iloc[-1]) if 'cumulative_r2' in eur_iterations.columns else None,
                'rmse': float(eur_iterations['cumulative_rmse'].iloc[-1]) if 'cumulative_rmse' in eur_iterations.columns else None
            },
            'random': {
                'r2': float(random_iterations['cumulative_r2'].iloc[-1]) if 'cumulative_r2' in random_iterations.columns else None,
                'rmse': float(random_iterations['cumulative_rmse'].iloc[-1]) if 'cumulative_rmse' in random_iterations.columns else None
            }
        }
    }

    # 保存对比结果
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    logger.info(f"Comparison saved to {output_dir / 'comparison.json'}")
    return comparison


def aggregate_multiple_comparisons(
    comparison_dirs: List[Path],
    output_dir: Path
) -> Dict:
    """
    汇总多次对比实验结果。

    Args:
        comparison_dirs: 多次对比实验的结果目录列表
        output_dir: 汇总结果输出目录

    Returns:
        汇总统计字典
    """
    logger.info(f"Aggregating {len(comparison_dirs)} comparisons")

    # 加载所有对比结果
    paired_results = []
    for comp_dir in comparison_dirs:
        comp_file = comp_dir / 'comparison.json'
        if comp_file.exists():
            with open(comp_file, 'r', encoding='utf-8') as f:
                paired_results.append(json.load(f))

    # 重新加载每个run的详细数据用于稳定性计算
    eur_results = []
    random_results = []
    for comp in paired_results:
        eur_dir = Path(comp['eur_result_dir'])
        random_dir = Path(comp['random_result_dir'])

        with open(eur_dir / 'data_files' / 'summary.json', 'r', encoding='utf-8') as f:
            eur_summary = json.load(f)
        with open(random_dir / 'data_files' / 'summary.json', 'r', encoding='utf-8') as f:
            random_summary = json.load(f)

        # 添加final_r2用于win rate计算
        eur_iterations = pd.read_csv(eur_dir / 'data_files' / 'iterations.csv')
        random_iterations = pd.read_csv(random_dir / 'data_files' / 'iterations.csv')

        eur_summary['final_r2'] = float(eur_iterations['cumulative_r2'].iloc[-1]) if 'cumulative_r2' in eur_iterations.columns else 0.0
        random_summary['final_r2'] = float(random_iterations['cumulative_r2'].iloc[-1]) if 'cumulative_r2' in random_iterations.columns else 0.0

        eur_results.append(eur_summary)
        random_results.append(random_summary)

    # 计算稳定性指标
    topk_stability = {
        'eur': calculate_topk_stability(eur_results),
        'random': calculate_topk_stability(random_results),
        'stability_advantage': None
    }
    topk_stability['stability_advantage'] = (
        topk_stability['eur']['mean_jaccard'] - topk_stability['random']['mean_jaccard']
    )

    discovery_stability = {
        'eur': calculate_discovery_stability(eur_results),
        'random': calculate_discovery_stability(random_results)
    }

    # 计算稳定性优势
    eur_cv = discovery_stability['eur']['overall']['mean_cv']
    random_cv = discovery_stability['random']['overall']['mean_cv']
    if eur_cv is not None and random_cv is not None:
        discovery_stability['stability_advantage'] = {
            'cv_reduction': (random_cv - eur_cv) / random_cv
        }
    else:
        discovery_stability['stability_advantage'] = None

    # 计算胜率
    # 为每个pair添加summary数据
    for i, comp in enumerate(paired_results):
        comp['eur'] = eur_results[i]
        comp['random'] = random_results[i]

    win_rate = calculate_win_rate(paired_results)

    # 汇总结果
    aggregate = {
        'n_comparisons': len(paired_results),
        'topk_model_stability': topk_stability,
        'discovery_timeline_stability': discovery_stability,
        'win_rate': win_rate
    }

    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'aggregate_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    logger.info(f"Aggregate comparison saved to {output_dir / 'aggregate_comparison.json'}")
    return aggregate
