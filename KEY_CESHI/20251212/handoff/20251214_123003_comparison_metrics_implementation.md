# 对比阶段评估指标实施计划

**日期**: 2025-12-14
**目标**: 实现EUR vs Random的系统性对比评估（对比阶段4核心指标）
**前置条件**: 孤立阶段指标已实施并验证通过
**状态**: 待实施

---

## 一、对比实验设计原则

### 1.1 实验配对原则

**Paired-run设计**:
- EUR run 和 Random run 必须使用**相同的Oracle**（相同的权重配置）
- 使用**相同的random seed**确保warmup阶段相同
- 使用**相同的budget**（总样本数）
- 使用**相同的设计空间**

**对比维度**:
1. **固定预算对比**: 给定N个样本，谁的效应识别更准确
2. **固定精度对比**: 达到R²=0.8，谁用的样本更少（Sample efficiency）
3. **稳定性对比**: 多次运行，谁的结果更稳定
4. **胜率统计**: 在M次对比中，谁赢得更多

### 1.2 Random Baseline实现

**采样策略**:
- Warmup阶段：与EUR相同（Sobol quasi-random初始化）
- 主采样阶段：Sobol序列继续采样（不使用模型指导）

**实现方式**:
```python
# run_eur_residual.py 添加 --strategy 参数
if strategy == 'random':
    # 使用 SobolQMCNormalGenerator 替代 EUR acquisition
    from botorch.sampling import SobolQMCNormalGenerator
    sobol_gen = SobolQMCNormalGenerator(seed=seed)
    next_x = sobol_gen.draw(1)
else:  # strategy == 'eur'
    # 现有EUR逻辑
    next_x = server.ask()
```

---

## 二、对比阶段4核心指标

### 2.1 指标概览

| 指标 | 含义 | 评估维度 | 输出格式 |
|------|------|---------|----------|
| **Sample Efficiency Ratio** | EUR vs Random达到相同精度的样本数比 | 效率 | float (0-1区间，越小越好) |
| **Top-k Model Stability** | 跨run的Top-k模型一致性 | 稳定性 | Jaccard similarity (0-1) |
| **Cross-run Stability** | 效应发现时间点的稳定性 | 稳定性 | CV (变异系数) |
| **Win Rate** | EUR在多次对比中的胜率 | 综合优势 | percentage (0-100%) |

---

## 三、指标详细设计

### 3.1 Sample Efficiency Ratio

**定义**:
```
efficiency_ratio = N_EUR(精度达标) / N_Random(精度达标)
```

**精度达标标准** (可配置):
- **结构发现**: F1-score >= 0.8 (交互效应识别)
- **效应恢复**: Spearman相关性 >= 0.85 (主效应+交互)
- **预测质量**: R² >= 0.75

**计算逻辑**:
1. 对于EUR run，找到首次满足精度标准的迭代次数 `t_EUR`
2. 对于Random run，找到首次满足精度标准的迭代次数 `t_Random`
3. 计算比值: `efficiency_ratio = t_EUR / t_Random`

**特殊情况处理**:
- 如果某个策略在预算内未达标: 记为 `达标iteration = budget + 1`
- 如果两者都未达标: efficiency_ratio = 1.0 (无差异)
- 如果仅EUR达标: efficiency_ratio = t_EUR / (budget + 1) < 1.0
- 如果仅Random达标: efficiency_ratio = (budget + 1) / t_Random > 1.0

**输出示例**:
```json
"sample_efficiency": {
  "criteria": {
    "structure_f1": 0.8,
    "spearman_corr": 0.85,
    "r2": 0.75
  },
  "eur_达标_iteration": 45,
  "random_达标_iteration": 78,
  "efficiency_ratio": 0.577,  // EUR用57.7%的样本达标
  "absolute_savings": 33,      // 节省33个样本
  "达标_metric": "structure_f1" // 首个达标的指标
}
```

**实现函数**:
```python
# modules/evaluation_comparison.py (新建)

def calculate_sample_efficiency(
    eur_iterations_csv: pd.DataFrame,
    random_iterations_csv: pd.DataFrame,
    criteria: Dict[str, float] = None
) -> Dict:
    """
    计算样本效率比。

    Args:
        eur_iterations_csv: EUR的iterations.csv数据
        random_iterations_csv: Random的iterations.csv数据
        criteria: 达标标准 {'structure_f1': 0.8, 'spearman_corr': 0.85, 'r2': 0.75}

    Returns:
        效率比指标字典
    """
    if criteria is None:
        criteria = {
            'structure_f1': 0.8,
            'spearman_corr': 0.85,
            'cumulative_r2': 0.75
        }

    # 查找EUR达标点
    eur_达标_iter = None
    for metric_name, threshold in criteria.items():
        if metric_name in eur_iterations_csv.columns:
            达标_rows = eur_iterations_csv[eur_iterations_csv[metric_name] >= threshold]
            if not 达标_rows.empty:
                eur_达标_iter = int(达标_rows.iloc[0]['iteration'])
                达标_metric = metric_name
                break

    # 查找Random达标点
    random_达标_iter = None
    for metric_name, threshold in criteria.items():
        if metric_name in random_iterations_csv.columns:
            达标_rows = random_iterations_csv[random_iterations_csv[metric_name] >= threshold]
            if not 达标_rows.empty:
                random_达标_iter = int(达标_rows.iloc[0]['iteration'])
                break

    # 计算效率比
    budget = len(eur_iterations_csv)
    eur_iter = eur_达标_iter if eur_达标_iter else budget + 1
    random_iter = random_达标_iter if random_达标_iter else budget + 1

    efficiency_ratio = eur_iter / random_iter
    absolute_savings = random_iter - eur_iter

    return {
        'criteria': criteria,
        'eur_达标_iteration': eur_达标_iter,
        'random_达标_iteration': random_达标_iter,
        'efficiency_ratio': float(efficiency_ratio),
        'absolute_savings': int(absolute_savings),
        '达标_metric': 达标_metric if eur_达标_iter else None
    }
```

---

### 3.2 Top-k Model Stability Across Runs

**定义**: 多次运行中，Top-k模型结构的重叠程度

**度量指标**:
1. **Jaccard Similarity**:
   ```
   J(A, B) = |A ∩ B| / |A ∪ B|
   ```
   其中 A, B 是两次run的Top-k模型集合（按模型结构识别）

2. **Rank Correlation** (Spearman):
   计算两次run中相同模型的排名相关性

**模型结构识别**:
使用交互项集合作为唯一标识：
```python
model_signature = frozenset(m['interactions'])  # e.g., frozenset({(0,5), (1,2)})
```

**计算逻辑**:
1. 对于EUR: 运行N次 (N=10)，提取每次run的Top-k模型结构
2. 对于Random: 运行N次，提取Top-k模型结构
3. 计算pairwise Jaccard similarity:
   ```
   J_eur = mean([J(run_i, run_j) for all pairs i,j in EUR runs])
   J_random = mean([J(run_i, run_j) for all pairs i,j in Random runs])
   ```
4. 比较: `stability_advantage = J_eur - J_random`

**输出示例**:
```json
"topk_model_stability": {
  "k": 5,
  "n_runs": 10,
  "eur": {
    "mean_jaccard": 0.78,
    "std_jaccard": 0.12,
    "pairwise_similarities": [0.8, 0.75, 0.82, ...],  // C(10,2)=45个
    "most_frequent_models": [
      {"structure": ["x0*x5", "x1*x2"], "frequency": 9},
      {"structure": ["x0*x5", "x1*x2", "x3*x4"], "frequency": 7}
    ]
  },
  "random": {
    "mean_jaccard": 0.42,
    "std_jaccard": 0.18,
    "pairwise_similarities": [...],
    "most_frequent_models": [...]
  },
  "stability_advantage": 0.36  // EUR稳定性优势
}
```

**实现函数**:
```python
# modules/evaluation_comparison.py

def calculate_topk_stability(
    run_results: List[Dict],  # 多次run的evaluation_v4结果
    k: int = 5
) -> Dict:
    """
    计算Top-k模型稳定性。

    Args:
        run_results: 多次run的model_comparison_table列表
        k: Top-k的k值

    Returns:
        稳定性指标字典
    """
    # 提取每次run的Top-k模型结构
    topk_signatures = []
    for result in run_results:
        models = result['effect_capture_v4']['model_comparison_table'][:k]
        signatures = [
            frozenset(m['interactions']) for m in models
        ]
        topk_signatures.append(set(signatures))

    # 计算pairwise Jaccard similarity
    from itertools import combinations
    jaccard_scores = []
    for sig_a, sig_b in combinations(topk_signatures, 2):
        intersection = len(sig_a & sig_b)
        union = len(sig_a | sig_b)
        jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)

    # 统计最频繁的模型结构
    from collections import Counter
    all_structures = [sig for sigs in topk_signatures for sig in sigs]
    structure_counts = Counter(all_structures)
    most_frequent = [
        {
            'structure': [f"x{i}*x{j}" for i, j in sorted(sig)],
            'frequency': count
        }
        for sig, count in structure_counts.most_common(5)
    ]

    return {
        'k': k,
        'n_runs': len(run_results),
        'mean_jaccard': float(np.mean(jaccard_scores)),
        'std_jaccard': float(np.std(jaccard_scores)),
        'pairwise_similarities': [float(x) for x in jaccard_scores],
        'most_frequent_models': most_frequent
    }
```

---

### 3.3 Cross-run Stability (Effect Discovery Timeline)

**定义**: 效应发现时间点在多次运行中的稳定性

**度量指标**:
1. **变异系数 (CV)**:
   ```
   CV = std(discovery_times) / mean(discovery_times)
   ```
   CV越小，稳定性越好

2. **中位数绝对偏差 (MAD)**:
   ```
   MAD = median(|discovery_times - median(discovery_times)|)
   ```
   对异常值更鲁棒

**计算逻辑**:
1. 对于每个效应 (e.g., x0, x1*x2)，收集N次run的发现时间点
2. 计算该效应的CV和MAD
3. 汇总所有效应的稳定性指标

**输出示例**:
```json
"discovery_timeline_stability": {
  "n_runs": 10,
  "eur": {
    "per_effect": {
      "x0": {
        "mean_discovery_iter": 5.2,
        "std_discovery_iter": 1.3,
        "cv": 0.25,
        "mad": 1.0,
        "discovery_times": [4, 5, 6, 5, 4, 6, 5, 5, 7, 5]
      },
      "x1*x2": {
        "mean_discovery_iter": 12.8,
        "std_discovery_iter": 3.1,
        "cv": 0.24,
        "mad": 2.5,
        "discovery_times": [10, 12, 15, 13, 11, 14, 12, 13, 12, 16]
      }
    },
    "overall": {
      "mean_cv": 0.23,  // 所有效应的平均CV
      "mean_mad": 1.8
    }
  },
  "random": {
    "per_effect": { ... },
    "overall": {
      "mean_cv": 0.45,
      "mean_mad": 4.2
    }
  },
  "stability_advantage": {
    "cv_reduction": 0.49,  // (0.45-0.23)/0.45 = 49% CV降低
    "mad_reduction": 0.57
  }
}
```

**实现函数**:
```python
# modules/evaluation_comparison.py

def calculate_discovery_stability(
    run_results: List[Dict]  # 多次run的effect_discovery_timeline
) -> Dict:
    """
    计算效应发现时间点稳定性。

    Args:
        run_results: 多次run的evaluation结果，包含effect_discovery_timeline字段

    Returns:
        稳定性指标字典
    """
    # 收集所有效应的发现时间
    from collections import defaultdict
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

    return {
        'n_runs': len(run_results),
        'per_effect': per_effect,
        'overall': {
            'mean_cv': float(np.mean(cvs)) if cvs else None,
            'mean_mad': float(np.mean(mads)) if mads else None
        }
    }
```

---

### 3.4 Win Rate

**定义**: 在M次配对实验中，EUR优于Random的比例

**胜负判定标准** (多维度):
1. **结构发现**: F1-score (EUR > Random)
2. **效应恢复**: Spearman相关性 (EUR > Random)
3. **预测质量**: R² (EUR > Random)
4. **样本效率**: efficiency_ratio < 1.0

**综合判定**:
- **Strong Win**: 4个维度都胜出
- **Win**: 至少3个维度胜出
- **Tie**: 2:2平局
- **Loss**: ≤1个维度胜出

**计算逻辑**:
```python
win_count = sum([
    1 for pair in paired_runs
    if is_eur_better(pair['eur'], pair['random'], criteria)
])
win_rate = win_count / len(paired_runs)
```

**输出示例**:
```json
"win_rate": {
  "n_pairs": 20,
  "criteria": [
    "structure_f1",
    "spearman_corr",
    "r2",
    "sample_efficiency"
  ],
  "overall": {
    "strong_wins": 12,  // 60%
    "wins": 6,          // 30%
    "ties": 2,          // 10%
    "losses": 0,        // 0%
    "win_rate": 0.90    // (12+6)/20
  },
  "per_criterion": {
    "structure_f1": {
      "eur_better": 18,
      "random_better": 2,
      "win_rate": 0.90
    },
    "spearman_corr": {
      "eur_better": 17,
      "random_better": 3,
      "win_rate": 0.85
    },
    "r2": {
      "eur_better": 16,
      "random_better": 4,
      "win_rate": 0.80
    },
    "sample_efficiency": {
      "eur_better": 19,
      "random_better": 1,
      "win_rate": 0.95
    }
  },
  "detailed_outcomes": [
    {"pair_id": 0, "outcome": "strong_win", "scores": [1,1,1,1]},
    {"pair_id": 1, "outcome": "win", "scores": [1,1,1,0]},
    ...
  ]
}
```

**实现函数**:
```python
# modules/evaluation_comparison.py

def calculate_win_rate(
    paired_results: List[Dict],  # [{"eur": {...}, "random": {...}}, ...]
    criteria: List[str] = None
) -> Dict:
    """
    计算EUR vs Random的胜率。

    Args:
        paired_results: 配对实验结果列表
        criteria: 评判标准列表

    Returns:
        胜率统计字典
    """
    if criteria is None:
        criteria = ['structure_f1', 'spearman_corr', 'cumulative_r2', 'sample_efficiency']

    # 初始化计数器
    per_criterion = {c: {'eur_better': 0, 'random_better': 0} for c in criteria}
    outcomes = []

    for pair_id, pair in enumerate(paired_results):
        eur_result = pair['eur']['effect_capture_v4']
        random_result = pair['random']['effect_capture_v4']

        # 比较各维度
        scores = []
        for criterion in criteria:
            if criterion == 'structure_f1':
                eur_val = eur_result['structure_discovery']['f1_score']
                random_val = random_result['structure_discovery']['f1_score']
            elif criterion == 'spearman_corr':
                eur_val = eur_result.get('spearman_correlation', 0)
                random_val = random_result.get('spearman_correlation', 0)
            elif criterion == 'cumulative_r2':
                eur_val = eur_result.get('final_r2', 0)
                random_val = random_result.get('final_r2', 0)
            elif criterion == 'sample_efficiency':
                eur_val = pair.get('sample_efficiency', {}).get('efficiency_ratio', 1.0)
                random_val = 1.0  # Random作为baseline
                # EUR越小越好
                eur_better = eur_val < random_val
            else:
                continue

            if criterion != 'sample_efficiency':
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
```

---

## 四、实施方案

### 4.1 新增模块: evaluation_comparison.py

**位置**: [modules/evaluation_comparison.py](KEY_CESHI/20251212/scripts/modules/evaluation_comparison.py)

**内容结构**:
```python
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
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_sample_efficiency(...) -> Dict:
    """见3.1节"""
    pass


def calculate_topk_stability(...) -> Dict:
    """见3.2节"""
    pass


def calculate_discovery_stability(...) -> Dict:
    """见3.3节"""
    pass


def calculate_win_rate(...) -> Dict:
    """见3.4节"""
    pass


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
    # 加载两个实验的summary.json和iterations.csv
    eur_summary = json.load(open(eur_result_dir / 'data_files/summary.json'))
    random_summary = json.load(open(random_result_dir / 'data_files/summary.json'))

    eur_iterations = pd.read_csv(eur_result_dir / 'data_files/iterations.csv')
    random_iterations = pd.read_csv(random_result_dir / 'data_files/iterations.csv')

    # 计算各项指标
    comparison = {
        'eur_result_dir': str(eur_result_dir),
        'random_result_dir': str(random_result_dir),
        'sample_efficiency': calculate_sample_efficiency(eur_iterations, random_iterations),
        # 其他指标需要多次run数据，此处仅记录单次
        'structure_discovery': {
            'eur': eur_summary['effect_capture_v4']['structure_discovery'],
            'random': random_summary['effect_capture_v4']['structure_discovery']
        },
        'final_performance': {
            'eur': {
                'r2': eur_iterations['cumulative_r2'].iloc[-1],
                'rmse': eur_iterations['cumulative_rmse'].iloc[-1]
            },
            'random': {
                'r2': random_iterations['cumulative_r2'].iloc[-1],
                'rmse': random_iterations['cumulative_rmse'].iloc[-1]
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
    # 加载所有对比结果
    paired_results = []
    for comp_dir in comparison_dirs:
        comp_file = comp_dir / 'comparison.json'
        if comp_file.exists():
            paired_results.append(json.load(open(comp_file)))

    # 计算跨run的稳定性和胜率
    # 需要重新加载每个run的详细数据
    eur_results = []
    random_results = []
    for comp in paired_results:
        eur_dir = Path(comp['eur_result_dir'])
        random_dir = Path(comp['random_result_dir'])

        eur_summary = json.load(open(eur_dir / 'data_files/summary.json'))
        random_summary = json.load(open(random_dir / 'data_files/summary.json'))

        eur_results.append(eur_summary)
        random_results.append(random_summary)

    # 计算稳定性指标
    topk_stability = {
        'eur': calculate_topk_stability(eur_results),
        'random': calculate_topk_stability(random_results),
        'stability_advantage': None  # 计算差值
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

    # 计算胜率
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
```

---

### 4.2 对比实验运行脚本

**位置**: [scripts/run_comparison_experiment.py](KEY_CESHI/20251212/scripts/run_comparison_experiment.py)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_comparison_experiment.py - EUR vs Random Paired Comparison Runner

Runs paired EUR and Random experiments with identical Oracle and seed.
"""

import argparse
import logging
from pathlib import Path
import subprocess
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def run_single_pair(
    config_path: Path,
    budget: int,
    seed: int,
    output_base: Path,
    pair_id: int
) -> Dict[str, Path]:
    """
    运行单对EUR vs Random实验。

    Args:
        config_path: 配置文件路径
        budget: 总样本预算
        seed: 随机种子
        output_base: 输出基础目录
        pair_id: 配对ID

    Returns:
        {'eur': eur_result_dir, 'random': random_result_dir}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 运行EUR实验
    eur_tag = f"pair{pair_id}_eur_seed{seed}"
    eur_cmd = [
        "pixi", "run", "python", "run_eur_residual.py",
        "--config", str(config_path),
        "--budget", str(budget),
        "--seed", str(seed),
        "--strategy", "eur",
        "--tag", eur_tag
    ]

    logger.info(f"Running EUR experiment (pair {pair_id}, seed {seed})...")
    subprocess.run(eur_cmd, check=True)

    # 查找EUR结果目录（最新的包含eur_tag的目录）
    result_dirs = sorted(
        (output_base / 'results').glob(f"*{eur_tag}*"),
        key=lambda p: p.stat().st_mtime
    )
    eur_result_dir = result_dirs[-1] if result_dirs else None

    # 运行Random实验（相同配置和seed）
    random_tag = f"pair{pair_id}_random_seed{seed}"
    random_cmd = [
        "pixi", "run", "python", "run_eur_residual.py",
        "--config", str(config_path),
        "--budget", str(budget),
        "--seed", str(seed),
        "--strategy", "random",
        "--tag", random_tag
    ]

    logger.info(f"Running Random baseline (pair {pair_id}, seed {seed})...")
    subprocess.run(random_cmd, check=True)

    # 查找Random结果目录
    result_dirs = sorted(
        (output_base / 'results').glob(f"*{random_tag}*"),
        key=lambda p: p.stat().st_mtime
    )
    random_result_dir = result_dirs[-1] if result_dirs else None

    return {
        'eur': eur_result_dir,
        'random': random_result_dir
    }


def main():
    parser = argparse.ArgumentParser(description="EUR vs Random Paired Comparison")
    parser.add_argument('--config', type=Path, required=True, help='Config file')
    parser.add_argument('--budget', type=int, default=50, help='Total sample budget')
    parser.add_argument('--n-pairs', type=int, default=10, help='Number of paired runs')
    parser.add_argument('--seed-start', type=int, default=42, help='Starting seed')
    parser.add_argument('--output-dir', type=Path, default=Path('comparison_results'),
                        help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 运行多对实验
    paired_dirs = []
    for pair_id in range(args.n_pairs):
        seed = args.seed_start + pair_id

        logger.info(f"\n{'='*80}")
        logger.info(f"Running paired experiment {pair_id + 1}/{args.n_pairs} (seed={seed})")
        logger.info(f"{'='*80}\n")

        pair_dirs = run_single_pair(
            config_path=args.config,
            budget=args.budget,
            seed=seed,
            output_base=Path.cwd(),
            pair_id=pair_id
        )

        # 运行对比分析
        from modules.evaluation_comparison import compare_paired_runs

        comparison_dir = output_dir / f"pair{pair_id}_seed{seed}"
        compare_paired_runs(
            eur_result_dir=pair_dirs['eur'],
            random_result_dir=pair_dirs['random'],
            output_dir=comparison_dir
        )

        paired_dirs.append(comparison_dir)

    # 汇总所有对比结果
    logger.info(f"\n{'='*80}")
    logger.info(f"Aggregating {len(paired_dirs)} paired comparisons...")
    logger.info(f"{'='*80}\n")

    from modules.evaluation_comparison import aggregate_multiple_comparisons

    aggregate_multiple_comparisons(
        comparison_dirs=paired_dirs,
        output_dir=output_dir
    )

    logger.info(f"\nComparison experiment completed! Results in: {output_dir}")


if __name__ == '__main__':
    main()
```

---

### 4.3 run_eur_residual.py 修改

**添加 --strategy 参数**:

```python
# run_eur_residual.py

def main():
    parser = argparse.ArgumentParser()
    # ... 现有参数 ...
    parser.add_argument(
        '--strategy',
        type=str,
        default='eur',
        choices=['eur', 'random'],
        help='Sampling strategy: eur or random (Sobol baseline)'
    )
    args = parser.parse_args()

    # 在采样循环中:
    for trial_idx in range(warmup_budget, total_budget):
        if args.strategy == 'random':
            # Random baseline: 使用Sobol继续采样
            from torch.quasirandom import SobolEngine
            sobol_engine = SobolEngine(dimension=6, scramble=True, seed=seed)
            # 跳过已用的warmup样本
            sobol_engine.fast_forward(trial_idx + 1)
            next_x = sobol_engine.draw(1).numpy()[0]

            # 转换到配置空间
            x_config = {f'x{i}': float(next_x[i]) for i in range(6)}
            x_array = next_x
        else:
            # EUR策略: 现有逻辑
            x_config, x_array = server.ask()

        # 后续tell和评估逻辑相同
        ...
```

---

## 五、输出格式

### 5.1 单次对比结果 (comparison.json)

```json
{
  "eur_result_dir": "results/20251214_120000_pair0_eur_seed42",
  "random_result_dir": "results/20251214_120500_pair0_random_seed42",
  "sample_efficiency": {
    "criteria": {"structure_f1": 0.8, "spearman_corr": 0.85, "r2": 0.75},
    "eur_达标_iteration": 45,
    "random_达标_iteration": 78,
    "efficiency_ratio": 0.577,
    "absolute_savings": 33,
    "达标_metric": "structure_f1"
  },
  "structure_discovery": {
    "eur": {"precision": 1.0, "recall": 1.0, "f1_score": 1.0},
    "random": {"precision": 0.67, "recall": 0.67, "f1_score": 0.67}
  },
  "final_performance": {
    "eur": {"r2": 0.82, "rmse": 0.38},
    "random": {"r2": 0.71, "rmse": 0.51}
  }
}
```

### 5.2 汇总结果 (aggregate_comparison.json)

```json
{
  "n_comparisons": 10,
  "topk_model_stability": {
    "eur": {
      "mean_jaccard": 0.78,
      "std_jaccard": 0.12,
      "most_frequent_models": [
        {"structure": ["x0*x5", "x1*x2"], "frequency": 9}
      ]
    },
    "random": {
      "mean_jaccard": 0.42,
      "std_jaccard": 0.18
    },
    "stability_advantage": 0.36
  },
  "discovery_timeline_stability": {
    "eur": {
      "overall": {"mean_cv": 0.23, "mean_mad": 1.8}
    },
    "random": {
      "overall": {"mean_cv": 0.45, "mean_mad": 4.2}
    },
    "stability_advantage": {"cv_reduction": 0.49}
  },
  "win_rate": {
    "n_pairs": 10,
    "overall": {
      "strong_wins": 7,
      "wins": 2,
      "ties": 1,
      "losses": 0,
      "win_rate": 0.90
    },
    "per_criterion": {
      "structure_f1": {"eur_better": 9, "win_rate": 0.90},
      "spearman_corr": {"eur_better": 8, "win_rate": 0.80},
      "r2": {"eur_better": 8, "win_rate": 0.80},
      "sample_efficiency": {"eur_better": 10, "win_rate": 1.00}
    }
  }
}
```

---

## 六、实施步骤

### Phase 1: Random Baseline实现

1. **修改 run_eur_residual.py**:
   - 添加 `--strategy` 参数
   - 在主循环中实现Random采样分支

2. **测试Random策略**:
   ```bash
   pixi run python run_eur_residual.py --budget 20 --strategy random
   ```

### Phase 2: 对比指标实现

1. **创建 evaluation_comparison.py**:
   - 实现4个核心函数
   - 单元测试每个函数

2. **创建 run_comparison_experiment.py**:
   - 实现paired run逻辑
   - 实现结果汇总逻辑

### Phase 3: 对比实验运行

1. **单对测试**:
   ```bash
   pixi run python run_comparison_experiment.py \
     --config configs/eur_residual_test.ini \
     --budget 50 \
     --n-pairs 1
   ```

2. **多对运行**:
   ```bash
   pixi run python run_comparison_experiment.py \
     --config configs/eur_residual_test.ini \
     --budget 50 \
     --n-pairs 10
   ```

### Phase 4: 结果可视化 (可选)

创建可视化脚本生成：
- Learning curves对比图
- Win rate统计图
- Stability box plots

---

## 七、文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| [modules/evaluation_comparison.py](KEY_CESHI/20251212/scripts/modules/evaluation_comparison.py) | 🆕 新建 | 对比评估指标计算模块 |
| [scripts/run_comparison_experiment.py](KEY_CESHI/20251212/scripts/run_comparison_experiment.py) | 🆕 新建 | 对比实验运行脚本 |
| [scripts/run_eur_residual.py](KEY_CESHI/20251212/scripts/run_eur_residual.py) | ✏️ 修改 | 添加 --strategy 参数 |

---

## 八、风险与注意事项

### 8.1 实验配对一致性

**风险**: EUR和Random使用不同的Oracle会导致对比失效

**缓解**:
- 在run_comparison_experiment.py中确保相同seed
- 记录Oracle规格到comparison.json中进行事后验证
- 添加自动检查: 比较两个summary.json中的oracle字段

### 8.2 计算开销

**风险**: 10对×2策略×50 budget = 1000次迭代，耗时可能较长

**缓解**:
- 提供 `--n-pairs` 参数灵活控制
- 支持分批运行和结果合并
- 优先测试小budget (如20)

### 8.3 统计显著性

**风险**: 样本量不足可能导致结论不可靠

**建议**:
- 至少10对实验 (n=10)
- 报告置信区间和p-value
- 使用bootstrap重采样验证稳定性

---

## 九、与孤立阶段的关系

**依赖关系**:
- 对比阶段**必须**在孤立阶段完成后实施
- 依赖孤立阶段的输出: iterations.csv中的cumulative_r2, effect_discovery_timeline等

**数据流**:
```
孤立阶段输出 (summary.json, iterations.csv)
    ↓
对比阶段输入 (EUR run, Random run)
    ↓
evaluation_comparison.py 计算对比指标
    ↓
对比阶段输出 (comparison.json, aggregate_comparison.json)
```

---

## 十、Checklist

- [ ] Phase 1: Random Baseline
  - [ ] 修改 run_eur_residual.py 添加 --strategy
  - [ ] 测试 Random 策略运行
  - [ ] 验证输出格式一致性

- [ ] Phase 2: 对比指标实现
  - [ ] 创建 evaluation_comparison.py
  - [ ] 实现 calculate_sample_efficiency()
  - [ ] 实现 calculate_topk_stability()
  - [ ] 实现 calculate_discovery_stability()
  - [ ] 实现 calculate_win_rate()
  - [ ] 单元测试各函数

- [ ] Phase 3: 对比实验运行
  - [ ] 创建 run_comparison_experiment.py
  - [ ] 实现 run_single_pair()
  - [ ] 实现 aggregate_multiple_comparisons()
  - [ ] 单对测试 (n=1)
  - [ ] 多对测试 (n=10)

- [ ] Phase 4: 结果验证
  - [ ] 检查 comparison.json 格式
  - [ ] 检查 aggregate_comparison.json 内容
  - [ ] 验证胜率计算逻辑
  - [ ] 生成可视化 (可选)

---

**实施优先级**: Phase 1 > Phase 2 > Phase 3 > Phase 4 (可选)

**前置条件**: 孤立阶段实施完成并验证通过

**预计工作量**: 4-6小时
