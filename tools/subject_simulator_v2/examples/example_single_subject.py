#!/usr/bin/env python3
"""
示例3：创建单个被试

演示如何：
1. 手动创建一个LinearSubject
2. 设置主效应权重和交互效应权重
3. 保存被试参数
4. 验证响应
"""

import numpy as np
from pathlib import Path
import sys

# 添加路径 (tools/ 目录)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subject_simulator_v2 import LinearSubject

def main():
    print(f"{'='*60}")
    print("示例3：创建单个自定义被试")
    print(f"{'='*60}\n")

    # 1. 创建一个简单的线性被试（无交互效应）
    print("场景1：简单线性模型（无交互）")
    print("-" * 60)

    subject_simple = LinearSubject(
        weights=np.array([0.2, -0.3, 0.5, 0.1, -0.2, 0.4]),
        bias=0.0,
        noise_std=0.0,  # 确定性
        likert_levels=5,
        likert_sensitivity=2.0,
        seed=42
    )

    # 测试
    x_test = np.array([1, 1, 1, 1, 1, 1])
    y_simple = subject_simple(x_test)
    print(f"输入: {x_test}")
    print(f"输出: {y_simple}")
    print(f"连续值计算: 0.2 - 0.3 + 0.5 + 0.1 - 0.2 + 0.4 = {0.2-0.3+0.5+0.1-0.2+0.4:.1f}\n")

    # 2. 创建带交互效应的被试
    print("场景2：线性模型 + 交互效应")
    print("-" * 60)

    subject_interaction = LinearSubject(
        weights=np.array([0.2, -0.3, 0.5, 0.1, -0.2, 0.4]),
        interaction_weights={
            (0, 1): 0.15,  # feature 0 × feature 1
            (3, 4): -0.10,  # feature 3 × feature 4
        },
        bias=0.5,
        noise_std=0.0,
        likert_levels=5,
        likert_sensitivity=2.0,
        seed=42
    )

    # 测试
    x_test = np.array([2, 1, 0, 3, 2, 1])
    y_interaction = subject_interaction(x_test)

    # 手动计算
    main_effects = np.dot(subject_interaction.weights, x_test)
    interaction_01 = 0.15 * x_test[0] * x_test[1]  # 0.15 * 2 * 1 = 0.30
    interaction_34 = -0.10 * x_test[3] * x_test[4]  # -0.10 * 3 * 2 = -0.60
    total = 0.5 + main_effects + interaction_01 + interaction_34

    print(f"输入: {x_test}")
    print(f"主效应: {main_effects:.2f}")
    print(f"交互(0,1): {interaction_01:.2f}")
    print(f"交互(3,4): {interaction_34:.2f}")
    print(f"截距: 0.5")
    print(f"总和: {total:.2f}")
    print(f"Likert输出: {y_interaction}\n")

    # 3. 创建带噪声的被试（随机性）
    print("场景3：线性模型 + 噪声（非确定性）")
    print("-" * 60)

    subject_noisy = LinearSubject(
        weights=np.array([0.3, -0.2, 0.4, 0.1, -0.3, 0.2]),
        bias=0.0,
        noise_std=0.2,  # 每次试次添加随机噪声
        likert_levels=5,
        likert_sensitivity=2.0,
        seed=123
    )

    # 同一输入，多次测试
    x_test = np.array([1, 1, 1, 1, 1, 1])
    print(f"输入: {x_test}")
    print("多次响应（由于噪声，可能不同）:")
    for i in range(5):
        y = subject_noisy(x_test)
        print(f"  试次 {i+1}: {y}")
    print()

    # 4. 保存所有被试
    print(f"{'='*60}")
    print("保存被试:")
    print(f"{'='*60}\n")

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_simple.save(str(output_dir / "subject_simple.json"))
    print(f"[OK] Simple model saved to: subject_simple.json")

    subject_interaction.save(str(output_dir / "subject_interaction.json"))
    print(f"[OK] Interaction model saved to: subject_interaction.json")

    subject_noisy.save(str(output_dir / "subject_noisy.json"))
    print(f"[OK] Noisy model saved to: subject_noisy.json")

    # 5. 在完整设计空间上验证正态性
    print(f"\n{'='*60}")
    print("正态性验证（完整设计空间）:")
    print(f"{'='*60}\n")

    # 定义设计空间
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

    # 测试每个被试
    subjects = [
        ("简单模型", subject_simple),
        ("交互模型", subject_interaction),
    ]

    for name, subject in subjects:
        responses = [subject(x) for x in design_space]

        from collections import Counter
        dist = Counter(responses)

        print(f"{name}:")
        print(f"  响应分布:")
        for level in sorted(dist.keys()):
            ratio = dist[level] / len(responses)
            bar = "█" * int(ratio * 40)
            print(f"    Likert={level}: {dist[level]:3d} ({ratio*100:5.1f}%) {bar}")

        # 检查正态性
        from subject_simulator_v2 import check_normality
        validation = check_normality(responses)

        print(f"  Normality check: {'[PASS]' if validation['passed'] else '[FAIL]'}")
        if not validation['passed']:
            print(f"    Reason: {validation['reason']}")
        print()

    print(f"[OK] Example completed!")

if __name__ == "__main__":
    main()
