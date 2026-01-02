#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_comparison_experiment.py - EUR vs Random Paired Comparison Runner

Runs paired EUR and Random experiments with identical Oracle and seed.
"""

import sys
import io
import argparse
import logging
from pathlib import Path
import subprocess
import json
from datetime import datetime

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup logging
from loguru import logger
logger.remove()
logger.add(sys.stderr, level='INFO')

# Import comparison functions
sys.path.insert(0, str(Path(__file__).parent / 'modules'))
from evaluation_comparison import compare_paired_runs, aggregate_multiple_comparisons


def run_single_pair(
    config_path: Path,
    budget: int,
    seed: int,
    scripts_dir: Path,
    pair_id: int
) -> Dict[str, Path]:
    """
    运行单对EUR vs Random实验。

    Args:
        config_path: 配置文件路径
        budget: 总样本预算
        seed: 随机种子
        scripts_dir: scripts目录路径
        pair_id: 配对ID

    Returns:
        {'eur': eur_result_dir, 'random': random_result_dir}
    """
    # 运行EUR实验
    eur_tag = f"pair{pair_id}_eur_seed{seed}"
    eur_cmd = [
        "pixi", "run", "python", str(scripts_dir / "run_eur_residual.py"),
        "--config", str(config_path),
        "--budget", str(budget),
        "--seed", str(seed),
        "--strategy", "eur",
        "--tag", eur_tag
    ]

    logger.info(f"Running EUR experiment (pair {pair_id}, seed {seed})...")
    logger.info(f"  Command: {' '.join(eur_cmd)}")
    result = subprocess.run(eur_cmd, check=True, cwd=scripts_dir)

    # 查找EUR结果目录（最新的包含eur_tag的目录）
    results_dir = scripts_dir / 'results'
    result_dirs = sorted(
        results_dir.glob(f"*{eur_tag}*"),
        key=lambda p: p.stat().st_mtime
    )
    eur_result_dir = result_dirs[-1] if result_dirs else None

    if not eur_result_dir:
        raise RuntimeError(f"Could not find EUR result directory for tag: {eur_tag}")

    logger.info(f"  EUR result: {eur_result_dir}")

    # 运行Random实验（相同配置和seed）
    random_tag = f"pair{pair_id}_random_seed{seed}"
    random_cmd = [
        "pixi", "run", "python", str(scripts_dir / "run_eur_residual.py"),
        "--config", str(config_path),
        "--budget", str(budget),
        "--seed", str(seed),
        "--strategy", "random",
        "--tag", random_tag
    ]

    logger.info(f"Running Random baseline (pair {pair_id}, seed {seed})...")
    logger.info(f"  Command: {' '.join(random_cmd)}")
    result = subprocess.run(random_cmd, check=True, cwd=scripts_dir)

    # 查找Random结果目录
    result_dirs = sorted(
        results_dir.glob(f"*{random_tag}*"),
        key=lambda p: p.stat().st_mtime
    )
    random_result_dir = result_dirs[-1] if result_dirs else None

    if not random_result_dir:
        raise RuntimeError(f"Could not find Random result directory for tag: {random_tag}")

    logger.info(f"  Random result: {random_result_dir}")

    return {
        'eur': eur_result_dir,
        'random': random_result_dir
    }


def main():
    parser = argparse.ArgumentParser(
        description="EUR vs Random Paired Comparison Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to config file (e.g., configs/eur_residual_test.ini)'
    )
    parser.add_argument(
        '--budget',
        type=int,
        default=50,
        help='Total sample budget for each run'
    )
    parser.add_argument(
        '--n-pairs',
        type=int,
        default=10,
        help='Number of paired runs to execute'
    )
    parser.add_argument(
        '--seed-start',
        type=int,
        default=42,
        help='Starting seed (incremented for each pair)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for comparison results (default: comparison_results/TIMESTAMP)'
    )
    args = parser.parse_args()

    # 确定scripts目录
    scripts_dir = Path(__file__).parent
    logger.info(f"Scripts directory: {scripts_dir}")

    # 创建输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = scripts_dir / 'comparison_results' / timestamp
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # 保存实验配置
    experiment_config = {
        'timestamp': datetime.now().isoformat(),
        'config_file': str(args.config),
        'budget': args.budget,
        'n_pairs': args.n_pairs,
        'seed_start': args.seed_start
    }
    with open(output_dir / 'experiment_config.json', 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)

    logger.info("="*80)
    logger.info("EUR vs Random Paired Comparison Experiment")
    logger.info("="*80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Budget: {args.budget}")
    logger.info(f"N pairs: {args.n_pairs}")
    logger.info(f"Seed range: {args.seed_start} - {args.seed_start + args.n_pairs - 1}")
    logger.info("="*80)

    # 运行多对实验
    paired_dirs = []
    for pair_id in range(args.n_pairs):
        seed = args.seed_start + pair_id

        logger.info("\n" + "="*80)
        logger.info(f"Running paired experiment {pair_id + 1}/{args.n_pairs} (seed={seed})")
        logger.info("="*80 + "\n")

        try:
            pair_dirs = run_single_pair(
                config_path=args.config,
                budget=args.budget,
                seed=seed,
                scripts_dir=scripts_dir,
                pair_id=pair_id
            )

            # 运行对比分析
            logger.info(f"\nComparing results for pair {pair_id}...")
            comparison_dir = output_dir / f"pair{pair_id}_seed{seed}"
            compare_paired_runs(
                eur_result_dir=pair_dirs['eur'],
                random_result_dir=pair_dirs['random'],
                output_dir=comparison_dir
            )

            paired_dirs.append(comparison_dir)
            logger.info(f"Pair {pair_id} comparison saved to: {comparison_dir}")

        except Exception as e:
            logger.error(f"Failed to run pair {pair_id}: {e}")
            raise

    # 汇总所有对比结果
    logger.info("\n" + "="*80)
    logger.info(f"Aggregating {len(paired_dirs)} paired comparisons...")
    logger.info("="*80 + "\n")

    aggregate_multiple_comparisons(
        comparison_dirs=paired_dirs,
        output_dir=output_dir
    )

    logger.info("\n" + "="*80)
    logger.info("Comparison experiment completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*80)
    logger.info("\nOutput files:")
    logger.info(f"  - experiment_config.json: Experiment configuration")
    logger.info(f"  - pair*/comparison.json: Individual pair comparisons")
    logger.info(f"  - aggregate_comparison.json: Aggregated statistics")


if __name__ == '__main__':
    main()
