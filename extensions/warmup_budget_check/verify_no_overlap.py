"""
验证多被试采样时配置无重叠
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from warmup_sampler import WarmupSampler

# 使用真实设计空间
design_csv = str(Path(__file__).parent.parent.parent / 'data' / 'i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv')
sampler = WarmupSampler(design_csv)

print("=" * 70)
print("验证多被试采样时配置无重叠")
print("=" * 70)
print()

# 5个被试，每人25次
n_subjects = 5
trials_per_subject = 25

adequacy, budget = sampler.evaluate_budget(n_subjects, trials_per_subject, skip_interaction=False)

print(f"预算分配:")
print(f"  Core-1: {budget['core1_configs']}")
print(f"  Boundary: {budget['boundary_configs']}")
print(f"  Core-2a: {budget['core2a_configs']}")
print(f"  Core-2b: {budget['core2b_configs']}")
print(f"  LHS: {budget['lhs_configs']}")
print(f"  总计: {sum(budget.values())}")
print()

# 生成采样
samples = sampler.generate_samples(
    budget=budget,
    interaction_mode='hybrid',
    interaction_pairs_to_explore=[(0, 1), (3, 4)],
)

print()
print("=" * 70)
print("检查配置重叠")
print("=" * 70)
print()

import pandas as pd

# 读取所有CSV文件并收集配置索引
all_configs = []
for i, csv_path in enumerate(samples):
    if isinstance(csv_path, str) and csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path)
        configs = df.index.tolist()
        all_configs.extend(configs)
        print(f"被试 {i+1}: {len(configs)} 个配置 (首配置索引: {configs[0] if configs else 'N/A'})")

print()
print(f"总配置数: {len(all_configs)}")
print(f"唯一配置数: {len(set(all_configs))}")

if len(all_configs) == len(set(all_configs)):
    print()
    print("✅ SUCCESS: 所有配置完全不重复！")
    print("   每个被试看到的配置都是独特的")
else:
    duplicates = len(all_configs) - len(set(all_configs))
    print()
    print(f"❌ FAIL: 发现 {duplicates} 个重复配置！")

    # 找出重复的配置
    from collections import Counter
    config_counts = Counter(all_configs)
    duplicated_configs = [cfg for cfg, count in config_counts.items() if count > 1]
    print(f"   重复的配置索引: {duplicated_configs[:10]}...")

print()
print("=" * 70)
