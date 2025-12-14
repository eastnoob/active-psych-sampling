"""
测试真实设计空间的D-optimal实现
使用 i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv (324个配置)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from warmup_sampler import WarmupSampler

print()
print('=' * 70)
print('Testing D-optimal with Real Design Space (324 configs)')
print('=' * 70)
print()

# 使用真实设计空间（绝对路径）
design_csv = str(Path(__file__).parent.parent.parent / 'data' / 'i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv')

# 创建采样器
sampler = WarmupSampler(design_csv)

print()
print('Evaluating budget for realistic scenario...')
print()

# 评估预算：5个被试，每人25次
n_subjects = 5
trials_per_subject = 25
adequacy, budget = sampler.evaluate_budget(n_subjects, trials_per_subject, skip_interaction=False)

print()
print('=' * 70)
print('Generating samples with D-optimal Core-2a...')
print('=' * 70)
print()

# 生成采样
samples = sampler.generate_samples(
    budget=budget,
    output_dir='sample_test_real',
    interaction_mode='hybrid',
    interaction_pairs_to_explore=[(0, 1), (3, 4)],  # CeilingHeight×GridModule, VisualBoundary×PhysicalBoundary
    min_config_per_pair=3,
)

print()
print('=' * 70)
print('SUCCESS: Real Design Space Test Complete!')
print('=' * 70)
print()
print(f'Generated samples for {len(samples)} subjects')
for subject_id, df_sample in samples.items():
    print(f'  Subject {subject_id}: {len(df_sample)} trials')
print()
print('Check the D-efficiency value above for Core-2a!')
print('Expected: 25-50% for a well-structured design space')
