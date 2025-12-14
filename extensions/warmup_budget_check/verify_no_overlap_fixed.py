"""
验证多被试采样时配置无重叠（修复版）
正确方法：比较配置内容的哈希值，而非DataFrame索引
"""
import sys
from pathlib import Path
import pandas as pd
import hashlib

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from warmup_sampler import WarmupSampler

# 使用真实设计空间
design_csv = str(Path(__file__).parent.parent.parent / 'data' / 'i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv')
sampler = WarmupSampler(design_csv)

print("=" * 70)
print("验证多被试采样时配置无重叠（修复版）")
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
print("检查配置重叠（使用配置哈希）")
print("=" * 70)
print()

def config_hash(row):
    """计算配置的哈希值"""
    # 排除subject_id，只用因子值
    factor_cols = [col for col in row.index if col != 'subject_id']
    config_str = '_'.join(str(row[col]) for col in sorted(factor_cols))
    return hashlib.md5(config_str.encode()).hexdigest()

# 收集所有配置的哈希
all_hashes = []
for i, csv_path in enumerate(samples):
    if isinstance(csv_path, str) and csv_path.endswith('.csv'):
        df = pd.read_csv(csv_path)
        hashes = df.apply(config_hash, axis=1).tolist()
        all_hashes.extend(hashes)
        print(f"被试 {i+1}: {len(hashes)} 个配置")
        print(f"  唯一哈希: {len(set(hashes))}")
        print(f"  前3个哈希: {hashes[:3]}")

print()
print(f"总配置数: {len(all_hashes)}")
print(f"唯一配置数: {len(set(all_hashes))}")

if len(all_hashes) == len(set(all_hashes)):
    print()
    print("[SUCCESS] 所有配置完全不重复！")
    print("   每个被试看到的配置都是独特的")
else:
    duplicates = len(all_hashes) - len(set(all_hashes))
    print()
    print(f"[FAIL] 发现 {duplicates} 个重复配置！")

    # 找出重复的配置
    from collections import Counter
    hash_counts = Counter(all_hashes)
    duplicated_hashes = [h for h, count in hash_counts.items() if count > 1]
    print(f"   重复的配置数: {len(duplicated_hashes)}")
    print(f"   重复次数: {[hash_counts[h] for h in duplicated_hashes[:5]]}")

print()
print("=" * 70)
