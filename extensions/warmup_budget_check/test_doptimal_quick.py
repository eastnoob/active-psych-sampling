"""
快速验证D-optimal Core-2a实现
"""
import pandas as pd
import numpy as np
import tempfile
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent / 'core'))

from warmup_sampler import WarmupSampler

# 创建测试设计空间
np.random.seed(42)
n = 100
df = pd.DataFrame({
    'x1': np.random.uniform(0, 1, n),
    'x2': np.random.uniform(0, 1, n),
    'x3': np.random.choice([0, 1, 2], n),
    'x4': np.random.choice([0, 1, 2], n),
    'x5': np.random.uniform(-1, 1, n),
    'x6': np.random.uniform(-1, 1, n),
})

# 保存为临时文件
tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
df.to_csv(tmp_file.name, index=False)
tmp_file.close()

print()
print('=' * 70)
print('Testing D-optimal Core-2a Implementation')
print('=' * 70)
print()

# 创建采样器
sampler = WarmupSampler(tmp_file.name)

# 首先评估预算
n_subjects = 3
trials_per_subject = 25
budget_result = sampler.evaluate_budget(n_subjects, trials_per_subject, skip_interaction=False)

print()
print('Budget evaluation:')
print(f"  Core-1: {budget_result['core1_configs']} configs")
print(f"  Core-2a: {budget_result['core2a_configs']} configs")
print(f"  Core-2b: {budget_result['core2b_configs']} configs")
print()

# 生成采样
samples = sampler.generate_samples(
    budget=budget_result,
    interaction_mode='hybrid',
    interaction_pairs_to_explore=[(0,1), (3,4)],
)

print()
print('=' * 70)
print('SUCCESS: D-optimal Core-2a Implementation Working!')
print('=' * 70)
print(f'Generated samples for {len(samples)} subjects')
for subject_id, df_sample in samples.items():
    print(f'  Subject {subject_id}: {len(df_sample)} trials')

import os
os.unlink(tmp_file.name)
