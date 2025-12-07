#!/usr/bin/env python3
"""
测试自动bias计算功能
使用202511301011的数据验证
"""

from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2.adapters.warmup_adapter import run

# 测试参数（使用202511301011数据 + 自动bias）
input_dir = Path(__file__).parent.parent.parent / "extensions/warmup_budget_check/sample/202511301011"

print("=" * 80)
print("Test Auto-Bias Calculation")
print("=" * 80)
print()

if not input_dir.exists():
    print(f"[Error] Input directory not found: {input_dir}")
    sys.exit(1)

# 使用fixed_weights + 自动bias（bias=0.0默认值会触发自动计算）
fixed_weights_file = input_dir / "result/fixed_weights_auto.json"

print(f"Input directory: {input_dir}")
print(f"Fixed weights file: {fixed_weights_file}")
print()
print("[Note] bias=0.0 (default) will trigger auto-calculation")
print()

# 运行适配器（bias=0.0会触发自动计算）
run(
    input_dir=input_dir,
    seed=99,
    output_mode="combined",
    clean=True,  # 清理旧结果
    interaction_pairs=[(3, 4), (0, 1)],
    interaction_scale=0.25,
    output_type="likert",
    likert_levels=5,
    likert_mode="tanh",
    likert_sensitivity=2.0,
    population_mean=0.0,
    population_std=0.3,
    individual_std_percent=0.3,
    ensure_normality=True,
    bias=0.0,  # Use default value to trigger auto-calculation
    noise_std=0.0,
    fixed_weights_file=str(fixed_weights_file),
    print_model=True,
    save_model_summary=True,
    model_summary_format="both"
)

# 读取并分析结果
import pandas as pd

result_csv = input_dir / "result/combined_results.csv"
df = pd.read_csv(result_csv)

print()
print("=" * 80)
print("Response Distribution Analysis")
print("=" * 80)
print(f"Total responses: {len(df)}")
print()
print("Response distribution:")
from collections import Counter
counter = Counter(df['y'])
for level in range(1, 6):
    count = counter.get(level, 0)
    pct = count / len(df) * 100
    print(f"  Likert {level}: {count:4d} ({pct:5.1f}%)")
print()
print(f"Mean: {df['y'].mean():.2f}")
print()

# 检查正态性
coverage = len([l for l in counter if counter[l] > 0])
max_ratio = max(counter.values()) / len(df)
mean = df['y'].mean()

print("Normality Check:")
print(f"  Coverage (>=3 required): {coverage}")
print(f"  Max single level ratio (<=60% required): {max_ratio*100:.1f}%")
print(f"  Mean (2-4 preferred): {mean:.2f}")
print()

if coverage >= 3 and max_ratio <= 0.6 and 2.0 <= mean <= 4.0:
    print("[OK] Normality check passed!")
else:
    print("[FAIL] Normality check failed")
    if coverage < 3:
        print(f"  - Coverage too low: {coverage} < 3")
    if max_ratio > 0.6:
        print(f"  - Single level ratio too high: {max_ratio*100:.1f}% > 60%")
    if not (2.0 <= mean <= 4.0):
        print(f"  - Mean out of range: {mean:.2f} not in [2.0, 4.0]")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
