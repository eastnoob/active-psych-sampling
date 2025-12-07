#!/usr/bin/env python3
"""
示例2：加载已有被试并进行实时作答

演示如何：
1. 从JSON文件加载已保存的被试
2. 使用被试进行实时作答
3. 验证加载的参数
"""

import numpy as np
from pathlib import Path
import sys
import json

# 添加路径 (tools/ 目录)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subject_simulator_v2 import load_subject

def main():
    # 1. 指定被试文件路径
    # 假设已运行 example_cluster_generation.py 生成了集群
    subject_file = Path(__file__).parent / "output" / "cluster_example" / "subject_1_spec.json"

    if not subject_file.exists():
        print(f"错误: 找不到文件 {subject_file}")
        print(f"请先运行 example_cluster_generation.py 生成集群")
        return

    # 2. 加载被试
    print("Loading subject...")
    subject = load_subject(str(subject_file))
    print(f"[OK] Successfully loaded: {subject_file.name}\n")

    # 3. 查看被试参数
    print(f"{'='*60}")
    print("被试参数:")
    print(f"{'='*60}")

    with open(subject_file, 'r', encoding='utf-8') as f:
        spec = json.load(f)

    print(f"模型类型: {spec['model_type']}")
    print(f"权重: {spec['weights']}")
    print(f"交互效应权重: {spec['interaction_weights']}")
    print(f"截距: {spec['bias']}")
    print(f"噪声标准差: {spec['noise_std']}")
    print(f"Likert等级数: {spec['likert_levels']}")
    print(f"Likert灵敏度: {spec['likert_sensitivity']}")

    # 4. 实时作答示例
    print(f"\n{'='*60}")
    print("实时作答示例:")
    print(f"{'='*60}\n")

    # 示例刺激
    test_stimuli = [
        np.array([0, 0, 0, 0, 0, 0]),
        np.array([1, 1, 1, 1, 1, 1]),
        np.array([2, 2, 2, 3, 2, 3]),
        np.array([0, 2, 1, 3, 0, 2]),
    ]

    for i, x in enumerate(test_stimuli, start=1):
        y = subject(x)
        print(f"试次 {i}:")
        print(f"  输入: {x}")
        print(f"  响应: {y}")
        print()

    # 5. 批量作答（整个设计空间）
    print(f"{'='*60}")
    print("批量作答（设计空间采样）:")
    print(f"{'='*60}\n")

    # 定义设计空间（与生成时一致）
    from itertools import product
    feature_levels = [
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 2],
        [0, 1, 2, 3],
    ]
    design_space = np.array(list(product(*feature_levels)))

    # 批量预测
    responses = [subject(x) for x in design_space]

    # 统计分布
    from collections import Counter
    dist = Counter(responses)

    print(f"设计空间大小: {len(design_space)} 个点")
    print(f"响应分布:")
    for level in sorted(dist.keys()):
        ratio = dist[level] / len(responses)
        print(f"  Likert={level}: {dist[level]:3d} ({ratio*100:5.1f}%)")
    print(f"\n均值: {np.mean(responses):.2f}")
    print(f"标准差: {np.std(responses):.2f}")

    # 6. 重新保存（演示to_dict功能）
    print(f"\n{'='*60}")
    print("重新保存被试:")
    print(f"{'='*60}\n")

    output_file = Path(__file__).parent / "output" / "reloaded_subject.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    subject.save(str(output_file))
    print(f"[OK] Saved to: {output_file}")

    print(f"\n[OK] Example completed!")

if __name__ == "__main__":
    main()
