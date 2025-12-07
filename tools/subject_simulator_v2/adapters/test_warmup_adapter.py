#!/usr/bin/env python3
"""
测试warmup_adapter是否正常工作
"""

from pathlib import Path
import sys

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from subject_simulator_v2.adapters.warmup_adapter import run

# 测试参数（使用202511301011数据）
input_dir = Path(__file__).parent.parent.parent.parent / "extensions/warmup_budget_check/sample/202511301011"

print(f"Testing warmup_adapter with: {input_dir}")
print()

if not input_dir.exists():
    print(f"[Error] Input directory not found: {input_dir}")
    print("Please ensure 202511301011 sample exists")
    sys.exit(1)

# 运行适配器
run(
    input_dir=input_dir,
    seed=99,  # 使用不同的seed以避免覆盖原result
    output_mode="combined",
    clean=False,
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
    bias=-0.3,
    print_model=True,
    save_model_summary=True,
    model_summary_format="both"
)

print("\n[Test completed successfully!]")
