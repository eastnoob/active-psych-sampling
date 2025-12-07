"""快速测试Phase1采样器"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scout_warmup_251113 import Phase1WarmupSampler

print("创建测试数据...")
np.random.seed(42)
design_data = {f"f{i+1}": np.random.rand(1200) for i in range(5)}
design_df = pd.DataFrame(design_data)

print("初始化采样器...")
sampler = Phase1WarmupSampler(
    design_df=design_df,
    n_subjects=7,
    trials_per_subject=25,
    interaction_selection="auto",
    seed=42,
)

print("执行采样...")
results = sampler.run_sampling()

print("\n" + "=" * 60)
print("采样结果:")
print("=" * 60)
print(f"总试验数: {len(results['trials'])}")
print(f"Core-1点数: {len(results['core1_points'])}")
print(f"唯一配置: {results['quality']['n_unique_configs']}")
print(f"覆盖率: {results['quality']['coverage_rate']:.2%}")
print(f"最小距离: {results['quality'].get('min_dist', 0):.4f}")

print("\n预算分配:")
for block, count in results["trials"]["block_type"].value_counts().items():
    print(f"  {block}: {count}")

print("\n✅ 测试成功!")
