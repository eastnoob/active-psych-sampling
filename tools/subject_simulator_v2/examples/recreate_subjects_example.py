#!/usr/bin/env python3
"""
示例：如何使用fixed_weights_auto.json在其他地方重建类似的被试

使用场景：
1. 在Step 1.5生成了5个被试
2. 保存了fixed_weights_auto.json（包含weights, interactions, bias）
3. 现在想在其他实验或脚本中重建类似的被试

方法1：使用warmup_adapter（推荐）
方法2：直接使用LinearSubject类（灵活）
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from subject_simulator_v2 import LinearSubject

print("=" * 80)
print("方法1：使用warmup_adapter重建被试（最简单）")
print("=" * 80)
print()

print("代码示例：")
print("""
from subject_simulator_v2.adapters.warmup_adapter import run

# 使用fixed_weights_auto.json重建被试
run(
    input_dir="path/to/new/sampling/plan",  # 新的采样方案目录
    seed=99,  # 保持相同的seed
    output_mode="combined",
    clean=True,
    interaction_pairs=[(3, 4), (0, 1)],  # 必须与原来相同
    interaction_scale=0.25,  # 必须与原来相同
    output_type="likert",
    likert_levels=5,
    likert_mode="tanh",
    likert_sensitivity=2.0,
    population_mean=0.0,
    population_std=0.3,
    individual_std_percent=0.3,
    ensure_normality=False,  # 重建时关闭正态性检查
    bias=0.0,  # 会自动从fixed_weights中加载
    noise_std=0.0,
    fixed_weights_file="path/to/fixed_weights_auto.json",  # 指定这个！
    design_space_csv="path/to/full_design_space.csv",
)
""")

print()
print("=" * 80)
print("方法2：直接从fixed_weights_auto.json创建单个被试（最灵活）")
print("=" * 80)
print()

# 读取fixed_weights_auto.json
import json

fixed_weights_file = Path("../../extensions/warmup_budget_check/sample/202511301011/result/fixed_weights_auto.json")

if fixed_weights_file.exists():
    with open(fixed_weights_file, 'r') as f:
        fixed_data = json.load(f)

    print(f"从 {fixed_weights_file.name} 加载参数：")
    print(f"  weights: {fixed_data['global'][0]}")
    print(f"  interactions: {fixed_data['interactions']}")
    print(f"  bias: {fixed_data['bias']}")
    print()

    # 创建单个被试
    weights = np.array(fixed_data['global'][0])

    # 转换interactions格式
    interaction_weights = {}
    for key, value in fixed_data['interactions'].items():
        i, j = map(int, key.split(','))
        interaction_weights[(i, j)] = value

    bias = fixed_data['bias']

    # 创建被试（与原来5个人具有相同的群体特征）
    subject = LinearSubject(
        weights=weights,
        interaction_weights=interaction_weights,
        bias=bias,
        noise_std=0.0,
        likert_levels=5,
        likert_sensitivity=2.0,
        seed=42  # 可以使用不同的seed产生不同的噪声
    )

    print("创建的被试：")
    print(f"  weights: {subject.weights}")
    print(f"  interaction_weights: {subject.interaction_weights}")
    print(f"  bias: {subject.bias}")
    print()

    # 测试：在一些点上作答
    test_data = np.array([
        [2.8, 6.5, 2.0, 0.0, 1.0, 0.0],  # 转换后的分类变量
        [4.0, 8.0, 1.0, 1.0, 0.0, 2.0],
        [8.5, 6.5, 0.0, 2.0, 1.0, 1.0],
    ])

    print("测试作答（3个点）：")
    for i, x in enumerate(test_data):
        response = subject(x)
        print(f"  点{i+1}: {x} -> Likert {response}")

print()
print("=" * 80)
print("方法3：创建具有个体差异的新被试（扩展）")
print("=" * 80)
print()

if fixed_weights_file.exists():
    print("如果想创建与原来5人\"类似但不完全相同\"的新被试：")
    print()

    # 群体权重（固定）
    population_weights = np.array(fixed_data['global'][0])

    # 添加个体偏差
    individual_std = 0.3 * 0.3  # population_std * individual_std_percent

    print(f"群体权重（固定）: {population_weights}")
    print(f"个体偏差标准差: {individual_std}")
    print()

    # 创建3个新被试，每个有不同的个体偏差
    for subject_id in range(1, 4):
        np.random.seed(100 + subject_id)  # 不同的seed

        # 个体权重 = 群体权重 + 随机偏差
        individual_deviation = np.random.normal(0, individual_std, size=len(population_weights))
        individual_weights = population_weights + individual_deviation

        new_subject = LinearSubject(
            weights=individual_weights,
            interaction_weights=interaction_weights,
            bias=bias,
            noise_std=0.0,
            likert_levels=5,
            likert_sensitivity=2.0,
            seed=100 + subject_id
        )

        print(f"新被试{subject_id}:")
        print(f"  个体权重: {individual_weights}")
        print(f"  偏差: {individual_deviation}")

        # 测试
        response = new_subject(test_data[0])
        print(f"  测试响应: {response}")
        print()

print()
print("=" * 80)
print("总结：3种方法的使用场景")
print("=" * 80)
print()
print("方法1 (warmup_adapter):")
print("  适用：需要在新的采样方案上重建完整的5人集群")
print("  优点：简单、自动化、完全兼容")
print("  示例：在新实验中使用相同的被试模型")
print()
print("方法2 (LinearSubject):")
print("  适用：需要单个被试实时作答")
print("  优点：灵活、可集成到任何代码")
print("  示例：Oracle模拟、实时交互实验")
print()
print("方法3 (个体差异):")
print("  适用：需要生成更多\"相似\"的被试")
print("  优点：保持群体特征，但有个体变化")
print("  示例：扩展被试池、蒙特卡洛模拟")
print()
print("=" * 80)
print("文件说明")
print("=" * 80)
print()
print("fixed_weights_auto.json格式（V2扩展）：")
print("""
{
  "global": [[w1, w2, ..., w6]],      # 群体权重（主效应）
  "interactions": {                    # 交互项权重
    "3,4": 0.189,                      # x3*x4的权重
    "0,1": 0.206                       # x0*x1的权重
  },
  "bias": -7.65                        # bias（所有被试共享）
}
""")
print()
print("subject_X_spec.json格式（V2完整参数）：")
print("""
{
  "model_type": "linear",
  "weights": [w1, w2, ..., w6],        # 个体权重
  "interaction_weights": {             # 交互项权重
    "3,4": 0.189,
    "0,1": 0.206
  },
  "bias": -7.65,                       # bias
  "noise_std": 0.0,                    # 噪声
  "likert_levels": 5,                  # Likert级别
  "likert_sensitivity": 2.0,           # 灵敏度
  "seed": 99,                          # 随机种子
  "subject_id": "subject_1",           # 被试ID
  "response_statistics": {...}         # 响应统计
}
""")
print()
print("=" * 80)
