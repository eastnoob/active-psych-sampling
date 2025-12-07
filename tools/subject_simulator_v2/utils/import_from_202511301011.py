#!/usr/bin/env python3
"""
从202511301011格式的数据导入被试到Subject Simulator V2

功能：
1. 读取 result/subject_X_model.md 提取weights
2. 读取 result/combined_results.csv 验证响应分布
3. 生成V2格式的JSON spec文件
4. 可选：重新生成CSV（使用V2被试）

使用方法：
    python import_from_202511301011.py \\
        --input extensions/warmup_budget_check/sample/202511301011 \\
        --output tools/subject_simulator_v2/examples/output/imported_cluster
"""

import argparse
from pathlib import Path
import re
import json
import sys
import pandas as pd
import numpy as np

# 添加subject_simulator_v2到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subject_simulator_v2 import LinearSubject


def parse_subject_model_md(filepath: Path) -> dict:
    """
    从subject_X_model.md文件解析参数

    Returns:
        {
            "seed": int,
            "weights": list[float],
            "num_features": int
        }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取seed
    seed_match = re.search(r'seed:\s*(\d+)', content)
    seed = int(seed_match.group(1)) if seed_match else None

    # 提取num_features
    features_match = re.search(r'num_features:\s*(\d+)', content)
    num_features = int(features_match.group(1)) if features_match else 6

    # 提取fixed weights
    weights = []
    for i in range(1, num_features + 1):
        pattern = rf'x{i}:\s*([-\d.]+)'
        match = re.search(pattern, content)
        if match:
            weights.append(float(match.group(1)))

    if len(weights) != num_features:
        raise ValueError(f"Expected {num_features} weights, found {len(weights)}")

    return {
        "seed": seed,
        "weights": weights,
        "num_features": num_features
    }


def analyze_response_distribution(csv_path: Path, subject_id: str) -> dict:
    """
    分析被试响应分布

    Returns:
        {
            "n_responses": int,
            "distribution": {level: count},
            "mean": float,
            "std": float
        }
    """
    df = pd.read_csv(csv_path)
    subject_data = df[df['subject'] == subject_id]

    if len(subject_data) == 0:
        raise ValueError(f"No data found for {subject_id}")

    responses = subject_data['y'].values

    from collections import Counter
    dist = Counter(responses)

    return {
        "n_responses": len(responses),
        "distribution": {int(k): int(v) for k, v in dist.items()},
        "mean": float(np.mean(responses)),
        "std": float(np.std(responses))
    }


def import_cluster(input_dir: Path, output_dir: Path,
                   interaction_pairs=None,
                   bias=0.0,
                   noise_std=0.0,
                   likert_sensitivity=2.0):
    """
    从202511301011格式导入被试集群

    Args:
        input_dir: 输入目录（包含result/子目录）
        output_dir: 输出目录（V2格式）
        interaction_pairs: 交互效应对，如[(3,4), (0,1)]
        bias: 截距（默认0）
        noise_std: 噪声标准差（默认0）
        likert_sensitivity: Likert灵敏度（默认2.0）
    """
    result_dir = input_dir / "result"

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取combined_results.csv
    combined_csv = result_dir / "combined_results.csv"
    if not combined_csv.exists():
        raise FileNotFoundError(f"combined_results.csv not found: {combined_csv}")

    df = pd.read_csv(combined_csv)
    subject_ids = df['subject'].unique()
    n_subjects = len(subject_ids)

    print(f"Found {n_subjects} subjects in combined_results.csv")
    print(f"Subject IDs: {', '.join(subject_ids)}\n")

    # 导入每个被试
    subjects = []
    subject_specs = []

    for i, subject_id in enumerate(sorted(subject_ids), start=1):
        model_file = result_dir / f"subject_{i}_model.md"

        if not model_file.exists():
            print(f"Warning: {model_file.name} not found, skipping...")
            continue

        # 解析参数
        params = parse_subject_model_md(model_file)

        # 分析响应分布
        stats = analyze_response_distribution(combined_csv, subject_id)

        # 创建LinearSubject
        subject = LinearSubject(
            weights=np.array(params['weights']),
            interaction_weights={pair: 0.0 for pair in (interaction_pairs or [])},  # 占位符
            bias=bias,
            noise_std=noise_std,
            likert_levels=5,
            likert_sensitivity=likert_sensitivity,
            seed=params['seed']
        )

        subjects.append(subject)

        # 生成spec
        spec = subject.to_dict()
        spec["subject_id"] = subject_id
        spec["response_statistics"] = stats
        spec["imported_from"] = str(model_file.relative_to(input_dir))

        subject_specs.append(spec)

        # 保存被试JSON
        output_file = output_dir / f"subject_{i}_spec.json"
        subject.save(str(output_file))

        print(f"[OK] {subject_id} imported")
        print(f"  Weights: {params['weights']}")
        print(f"  Response distribution: {stats['distribution']}")
        print(f"  Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
        print(f"  Saved to: {output_file.name}\n")

    # 计算群体平均权重
    all_weights = np.array([s.weights for s in subjects])
    population_weights = np.mean(all_weights, axis=0)

    # 保存集群摘要
    summary = {
        "n_subjects": len(subjects),
        "population_weights": population_weights.tolist(),
        "interaction_pairs": interaction_pairs or [],
        "bias": bias,
        "noise_std": noise_std,
        "likert_levels": 5,
        "likert_sensitivity": likert_sensitivity,
        "imported_from": str(input_dir.resolve()),
        "overall_response_distribution": {}
    }

    # 合并所有响应分布
    for spec in subject_specs:
        for level, count in spec["response_statistics"]["distribution"].items():
            level_str = str(level)
            summary["overall_response_distribution"][level_str] = \
                summary["overall_response_distribution"].get(level_str, 0) + count

    summary_file = output_dir / "cluster_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] Cluster import completed!")
    print(f"  Population weights: {population_weights}")
    print(f"  Files saved in: {output_dir}")

    return {
        "subjects": subjects,
        "subject_specs": subject_specs,
        "population_weights": population_weights,
        "output_dir": output_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description="Import subjects from 202511301011 format to Subject Simulator V2"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory (e.g., extensions/warmup_budget_check/sample/202511301011)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for V2 format files"
    )
    parser.add_argument(
        "--interaction-pairs",
        type=str,
        default="3,4;0,1",
        help="Interaction pairs (e.g., '3,4;0,1' for [(3,4), (0,1)])"
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Bias/intercept (default: 0.0)"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Noise standard deviation (default: 0.0)"
    )
    parser.add_argument(
        "--likert-sensitivity",
        type=float,
        default=2.0,
        help="Likert sensitivity (default: 2.0)"
    )

    args = parser.parse_args()

    # 解析interaction_pairs
    interaction_pairs = []
    if args.interaction_pairs:
        for pair_str in args.interaction_pairs.split(';'):
            i, j = map(int, pair_str.split(','))
            interaction_pairs.append((i, j))

    # 导入
    import_cluster(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        interaction_pairs=interaction_pairs,
        bias=args.bias,
        noise_std=args.noise_std,
        likert_sensitivity=args.likert_sensitivity
    )


if __name__ == "__main__":
    main()
