#!/usr/bin/env python3
"""
示例1：生成被试集群

演示如何：
1. 定义设计空间
2. 生成5个被试的集群
3. 验证每个被试响应的正态性
4. 保存所有结果
"""

import numpy as np
from pathlib import Path
import sys

# 添加路径 (tools/ 目录)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subject_simulator_v2 import ClusterGenerator

def main():
    # 1. 定义设计空间（与i9csy数据集一致）
    # 6个特征，每个特征的取值范围
    feature_levels = [
        [0, 1, 2],  # feature 0: 3 levels
        [0, 1, 2],  # feature 1: 3 levels
        [0, 1, 2],  # feature 2: 3 levels
        [0, 1, 2, 3],  # feature 3: 4 levels
        [0, 1, 2],  # feature 4: 3 levels
        [0, 1, 2, 3],  # feature 5: 4 levels
    ]

    # 生成完整设计空间（笛卡尔积）
    from itertools import product
    design_space = np.array(list(product(*feature_levels)))

    print(f"设计空间大小: {len(design_space)} 个点")
    print(f"特征数量: {design_space.shape[1]}")
    print(f"前3个点示例:\n{design_space[:3]}\n")

    # 2. 创建集群生成器
    generator = ClusterGenerator(
        design_space=design_space,
        n_subjects=5,  # 生成5个被试
        population_mean=0.0,  # 群体权重均值
        population_std=0.15,  # 群体权重标准差（较小，避免过度偏斜）
        individual_std=0.08,  # 个体偏差标准差
        interaction_pairs=[(3, 4), (0, 1)],  # 交互效应对
        interaction_scale=0.15,  # 交互权重标准差（较小）
        bias=-0.3,  # 负截距，使响应中心化到Likert=3附近
        noise_std=0.0,  # 无试次内噪声（确定性）
        likert_levels=5,  # 5级Likert量表
        likert_sensitivity=1.5,  # Likert灵敏度（较小，扩散响应）
        ensure_normality=True,  # 确保正态性
        max_retries=20,  # 最大重试次数
        seed=42  # 随机种子（可复现）
    )

    # 3. 生成集群
    output_dir = Path(__file__).parent / "output" / "cluster_example"
    cluster = generator.generate_cluster(str(output_dir))

    # 4. 查看结果
    print(f"\n{'='*60}")
    print("生成结果:")
    print(f"{'='*60}")
    print(f"输出目录: {cluster['output_dir']}")
    print(f"\n群体权重:\n{cluster['population_weights']}")
    print(f"\n交互效应权重:")
    for pair, weight in cluster['interaction_weights'].items():
        print(f"  {pair}: {weight:.6f}")

    # 5. 验证每个被试
    print(f"\n{'='*60}")
    print("被试响应分布验证:")
    print(f"{'='*60}")

    for i, subject in enumerate(cluster['subjects'], start=1):
        # 采样整个设计空间
        responses = [subject(x) for x in design_space]

        # 计算分布
        from collections import Counter
        dist = Counter(responses)

        print(f"\nSubject {i}:")
        print(f"  权重: {subject.weights}")
        print(f"  响应分布:")
        for level in sorted(dist.keys()):
            ratio = dist[level] / len(responses)
            print(f"    Likert={level}: {dist[level]:3d} ({ratio*100:5.1f}%)")
        print(f"  均值: {np.mean(responses):.2f}, 标准差: {np.std(responses):.2f}")

    # 6. 生成的文件说明
    print(f"\n{'='*60}")
    print("生成的文件:")
    print(f"{'='*60}")
    print(f"  cluster_summary.json     - 集群参数摘要")
    print(f"  subject_1_spec.json      - 被试1完整参数")
    print(f"  subject_2_spec.json      - 被试2完整参数")
    print(f"  ...                      - ...")
    print(f"  subject_1.csv            - 被试1响应数据")
    print(f"  subject_2.csv            - 被试2响应数据")
    print(f"  ...                      - ...")
    print(f"  combined_results.csv     - 所有被试合并数据")

    print(f"\n[OK] Example completed!")

if __name__ == "__main__":
    main()
