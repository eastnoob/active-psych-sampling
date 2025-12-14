#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Oracle模型在整个设计空间的响应分布

目的：
1. 使用与run_eur_simple.py完全相同的Oracle配置（202511301011固定权重）
2. 对整个i9csy设计空间（324个点）进行全采样
3. 统计Likert响应的分布，验证EUR是否只采样到了中间区域
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 设置UTF-8编码（避免Windows控制台编码问题）
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加模块路径
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR / "modules"))

# 导入依赖
from single_output_subject import SingleOutputLatentSubject
import json

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[5]


def load_design_space_inline(design_space_path: Path):
    """
    内联实现设计空间加载（避免调用有中文输出的模块）
    """
    df_design = pd.read_csv(design_space_path)
    print(f"Design space loaded: {df_design.shape}")

    # 提取数值列
    X_np = df_design.values

    # 转换为规范化格式（0-based索引）
    X_canonical = np.zeros_like(X_np, dtype=float)

    # CeilingHeight: 转换为0,1,2
    height_map = {2.8: 0, 4.0: 1, 8.5: 2}
    X_canonical[:, 0] = df_design.iloc[:, 0].map(height_map).values

    # GridModule: 转换为0,1
    grid_map = {6.5: 0, 8.0: 1}
    X_canonical[:, 1] = df_design.iloc[:, 1].map(grid_map).values

    # 对分类变量进行转换（从字符串到索引）
    for col_idx in [2, 3, 5]:  # OuterFurniture, VisualBoundary, InnerFurniture
        col_name = df_design.columns[col_idx]
        unique_vals = df_design[col_name].unique()
        val_to_idx = {val: float(idx) for idx, val in enumerate(sorted(unique_vals))}
        X_canonical[:, col_idx] = df_design[col_name].map(val_to_idx).values

    # PhysicalBoundary: 转换为0,1
    boundary_vals = df_design.iloc[:, 4].unique()
    boundary_map = {val: float(idx) for idx, val in enumerate(sorted(boundary_vals))}
    X_canonical[:, 4] = df_design.iloc[:, 4].map(boundary_map).values

    print(f"Design space transformed: {X_canonical.shape}")
    print(f"Value ranges:")
    for i in range(X_canonical.shape[1]):
        print(f"  x{i}: [{X_canonical[:, i].min():.1f}, {X_canonical[:, i].max():.1f}], {len(np.unique(X_canonical[:, i]))} unique values")

    return df_design, X_canonical


def load_fixed_weights_from_json(json_path: Path) -> np.ndarray:
    """Load fixed weights from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if "global" in data:
        weights = np.array(data["global"], dtype=np.float64)
    else:
        raise ValueError(f"JSON format error, missing 'global' key: {json_path}")

    print(f"Loaded fixed weights from {json_path.name}: {weights.shape}")
    print(f"  Weight values: {weights[0]}")

    return weights


def create_oracle_same_as_run_eur():
    """
    Create the exact same Oracle model as run_eur_simple.py

    Config source: run_eur_simple.py L189-208
    Using fixed_weights_auto.json from 202511301011
    """
    # Load fixed weights (consistent with run_eur_simple.py)
    fixed_weights_path = (
        PROJECT_ROOT
        / "extensions/warmup_budget_check/sample/202511301011/result/fixed_weights_auto.json"
    )

    fixed_weights = load_fixed_weights_from_json(fixed_weights_path)

    # Create Oracle (exactly matching i9csy config in run_eur_simple.py)
    oracle_kwargs = {
        "seed": 42,  # Consistent with quick_start.py
        "bias": 0.0,  # Population mean
        "noise_std": 0.1,  # Latent layer noise
        "interaction_pairs": [(3, 4), (0, 1)],  # Consistent with quick_start.py
        "interaction_scale": 0.25,  # Consistent with quick_start.py
        "likert_levels": 5,
        "likert_sensitivity": 2.0,  # Consistent with quick_start.py
        "use_latent": False,  # Using fixed weights, no latent variables
        "fixed_weights": fixed_weights,
        "num_features": fixed_weights.shape[1],
    }

    oracle = SingleOutputLatentSubject(**oracle_kwargs)
    print(f"Oracle type: SingleOutputLatentSubject (fixed weights)")

    return oracle


def sample_full_design_space(oracle, design_space_np: np.ndarray):
    """
    Sample the entire design space

    Args:
        oracle: Oracle model
        design_space_np: Design space array (N, 6)

    Returns:
        responses: Likert response array (N,)
        continuous_responses: Continuous response array (before Likert conversion) (N,)
    """
    N = design_space_np.shape[0]
    responses = np.zeros(N, dtype=int)
    continuous_responses = np.zeros(N, dtype=float)

    print(f"\nStarting full sampling for {N} design points...")

    for i in range(N):
        x = design_space_np[i]
        y_likert = oracle(x)
        responses[i] = y_likert

        # Get continuous value (if Oracle supports it)
        if hasattr(oracle, 'get_continuous_response'):
            y_cont = oracle.get_continuous_response(x)
            continuous_responses[i] = y_cont

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{N}")

    print(f"Full sampling completed")

    return responses, continuous_responses


def analyze_distribution(responses: np.ndarray, design_space_df: pd.DataFrame, save_dir: Path):
    """
    分析并可视化响应分布

    Args:
        responses: Likert响应数组
        design_space_df: 设计空间DataFrame（用于分析）
        save_dir: 保存目录
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # 统计分布
    counter = Counter(responses)
    total = len(responses)

    print(f"\n{'='*60}")
    print(f"Oracle Response Distribution Statistics (Full 324 points)")
    print(f"{'='*60}")

    for likert_val in range(1, 6):
        count = counter.get(likert_val, 0)
        percentage = (count / total) * 100
        print(f"  Likert={likert_val}: {count:3d} points ({percentage:5.1f}%)")

    print(f"\n  Mean: {np.mean(responses):.2f}")
    print(f"  Std: {np.std(responses):.2f}")
    print(f"  Median: {np.median(responses):.1f}")
    print(f"  Range: [{np.min(responses)}, {np.max(responses)}]")
    print(f"{'='*60}\n")

    # Save CSV
    result_df = design_space_df.copy()
    result_df['y_likert'] = responses

    csv_path = save_dir / "full_sampling_results.csv"
    result_df.to_csv(csv_path, index=False)
    print(f"Full sampling results saved: {csv_path}")

    # 可视化分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 直方图
    ax1 = axes[0]
    likert_values = list(range(1, 6))
    counts = [counter.get(v, 0) for v in likert_values]

    bars = ax1.bar(likert_values, counts, color='steelblue', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Likert Response', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Oracle Response Distribution (Full Design Space)', fontsize=14, fontweight='bold')
    ax1.set_xticks(likert_values)
    ax1.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # 子图2: 饼图
    ax2 = axes[1]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 5))
    wedges, texts, autotexts = ax2.pie(
        counts,
        labels=[f'Likert {v}' for v in likert_values],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )
    ax2.set_title('Response Proportion', fontsize=14, fontweight='bold')

    plt.tight_layout()

    fig_path = save_dir / "oracle_response_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved: {fig_path}")

    plt.close()

    return counter


def compare_with_eur_sampling(oracle_distribution: Counter, save_dir: Path):
    """
    Compare Oracle full sampling vs EUR sampling distribution

    Args:
        oracle_distribution: Oracle full sampling distribution statistics
        save_dir: Save directory
    """
    # EUR sampling distribution (from test_data.csv)
    eur_data_path = (
        PROJECT_ROOT
        / "test/is_EUR_work/00_plans/20251130/scripts/results/20251130_121609/data_files/test_data.csv"
    )

    if not eur_data_path.exists():
        print(f"Warning: EUR sampling data not found, skipping comparison")
        return

    eur_df = pd.read_csv(eur_data_path)
    eur_responses = eur_df['y_value'].values
    eur_counter = Counter(eur_responses)

    print(f"\n{'='*60}")
    print(f"EUR Sampling vs Oracle Full Sampling Comparison")
    print(f"{'='*60}")
    print(f"{'Likert':<8} {'Oracle (324)':<20} {'EUR (50)':<20} {'Diff'}")
    print(f"{'-'*60}")

    for likert_val in range(1, 6):
        oracle_count = oracle_distribution.get(likert_val, 0)
        eur_count = eur_counter.get(likert_val, 0)
        oracle_pct = (oracle_count / 324) * 100
        eur_pct = (eur_count / 50) * 100 if 50 > 0 else 0
        diff = eur_pct - oracle_pct

        print(f"  {likert_val:<6} {oracle_count:3d} ({oracle_pct:5.1f}%)       "
              f"{eur_count:3d} ({eur_pct:5.1f}%)       {diff:+6.1f}%")

    print(f"{'='*60}\n")

    # 可视化对比
    fig, ax = plt.subplots(figsize=(10, 6))

    likert_values = list(range(1, 6))
    oracle_pcts = [(oracle_distribution.get(v, 0) / 324) * 100 for v in likert_values]
    eur_pcts = [(eur_counter.get(v, 0) / 50) * 100 for v in likert_values]

    x = np.arange(len(likert_values))
    width = 0.35

    bars1 = ax.bar(x - width/2, oracle_pcts, width, label='Oracle (Full 324)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, eur_pcts, width, label='EUR Sampling (50)',
                   color='coral', alpha=0.8)

    ax.set_xlabel('Likert Response', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Response Distribution: Oracle vs EUR Sampling',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Likert {v}' for v in likert_values])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    fig_path = save_dir / "oracle_vs_eur_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved: {fig_path}")

    plt.close()


def main():
    """Main function"""
    # Save directory
    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Oracle Model Full Sampling Test")
    print(f"{'='*60}\n")

    # 1. Load design space
    print("Step 1: Loading design space")
    design_space_csv = (
        PROJECT_ROOT
        / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
    )

    design_space_df, design_space_np = load_design_space_inline(design_space_csv)
    print(f"Design space loaded: {design_space_np.shape}")

    # 2. Create Oracle (exactly same as run_eur_simple.py)
    print("\nStep 2: Creating Oracle model")
    oracle = create_oracle_same_as_run_eur()

    # 3. Full sampling
    print("\nStep 3: Full sampling")
    responses, continuous_responses = sample_full_design_space(oracle, design_space_np)

    # 4. Analyze distribution
    print("\nStep 4: Analyzing distribution")
    distribution = analyze_distribution(responses, design_space_df, save_dir)

    # 5. Compare with EUR sampling
    print("\nStep 5: Comparing with EUR sampling")
    compare_with_eur_sampling(distribution, save_dir)

    print(f"\n{'='*60}")
    print(f"Test completed! Results saved in: {save_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
