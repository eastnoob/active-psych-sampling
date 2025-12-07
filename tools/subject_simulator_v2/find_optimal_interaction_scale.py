#!/usr/bin/env python3
"""
找到合适的interaction_scale值，使得在324点上产生正态分布
保留交互项，只调整scale
"""

from pathlib import Path
import sys
import shutil
import pandas as pd
from collections import Counter

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))
from subject_simulator_v2.adapters.warmup_adapter import run

print("=" * 80)
print("Finding optimal interaction_scale for normal distribution")
print("=" * 80)
print()

# 设计空间CSV
design_space_csv = Path(__file__).parent.parent.parent / "data/only_independences/data/only_independences/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"

if not design_space_csv.exists():
    print(f"[Error] Design space CSV not found: {design_space_csv}")
    sys.exit(1)

# 测试不同的interaction_scale值
test_scales = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

print(f"Testing {len(test_scales)} different interaction_scale values:")
print(f"  {test_scales}")
print()
print("Interaction pairs: [(3, 4), (0, 1)]")
print()

results = []

for scale in test_scales:
    print("=" * 80)
    print(f"Testing interaction_scale = {scale}")
    print("=" * 80)

    # 创建临时测试目录
    test_dir = Path(__file__).parent / f"temp_test_scale_{scale}"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)

    # 复制完整设计空间作为subject_1.csv
    df_full = pd.read_csv(design_space_csv)
    subject_1_csv = test_dir / "subject_1.csv"
    df_full.to_csv(subject_1_csv, index=False)

    # 运行warmup_adapter
    try:
        run(
            input_dir=test_dir,
            seed=99,
            output_mode="combined",
            clean=True,
            interaction_pairs=[(3, 4), (0, 1)],  # Keep interactions
            interaction_scale=scale,  # Adjust scale
            output_type="likert",
            likert_levels=5,
            likert_mode="tanh",
            likert_sensitivity=2.0,
            population_mean=0.0,
            population_std=0.3,
            individual_std_percent=0.3,
            ensure_normality=False,  # Disable to test actual distribution
            bias=0.0,  # Auto-calculate
            noise_std=0.0,
            design_space_csv=str(design_space_csv),
            print_model=False,
            save_model_summary=False
        )

        # 读取结果
        result_csv = test_dir / "result/subject_1.csv"
        df_result = pd.read_csv(result_csv)

        counter = Counter(df_result['y'])

        # 计算统计
        coverage = len([l for l in counter if counter[l] > 0])
        max_ratio = max(counter.values()) / len(df_result)
        mean = df_result['y'].mean()

        # 正态性判断
        is_normal = coverage >= 3 and max_ratio <= 0.6 and 2.0 <= mean <= 4.0

        # 保存结果
        result = {
            'scale': scale,
            'coverage': coverage,
            'max_ratio': max_ratio,
            'mean': mean,
            'is_normal': is_normal,
            'distribution': dict(counter)
        }
        results.append(result)

        # 打印分布
        print()
        for level in range(1, 6):
            count = counter.get(level, 0)
            pct = count / len(df_result) * 100
            bar = '#' * int(pct / 5)  # Scale down for readability
            print(f"  Likert {level}: {count:4d} ({pct:5.1f}%) {bar}")

        print(f"\n  Mean: {mean:.2f}")
        print(f"  Coverage: {coverage}")
        print(f"  Max ratio: {max_ratio*100:.1f}%")

        if is_normal:
            print(f"\n  [OK] NORMAL DISTRIBUTION!")
        else:
            print(f"\n  [FAIL] Not normal")

        print()

    except Exception as e:
        print(f"[Error] Failed with scale={scale}: {e}")
        results.append({
            'scale': scale,
            'error': str(e)
        })

    finally:
        # 清理
        if test_dir.exists():
            shutil.rmtree(test_dir)

# 总结
print()
print("=" * 80)
print("Summary of Results")
print("=" * 80)
print()

print(f"{'Scale':<8} {'Coverage':<10} {'Max Ratio':<12} {'Mean':<8} {'Normal?':<10}")
print("-" * 60)

for r in results:
    if 'error' in r:
        print(f"{r['scale']:<8} ERROR: {r['error']}")
    else:
        normal_mark = "[OK] YES" if r['is_normal'] else "[FAIL] NO"
        print(f"{r['scale']:<8} {r['coverage']:<10} {r['max_ratio']*100:>5.1f}%{' '*6} {r['mean']:<8.2f} {normal_mark}")

# 找到最佳值
normal_results = [r for r in results if r.get('is_normal', False)]

if normal_results:
    print()
    print("=" * 80)
    print("Recommended interaction_scale values:")
    print("=" * 80)
    print()

    for r in normal_results:
        print(f"  interaction_scale = {r['scale']}")
        print(f"    Distribution: {r['distribution']}")
        print(f"    Mean: {r['mean']:.2f}, Coverage: {r['coverage']}, Max ratio: {r['max_ratio']*100:.1f}%")
        print()
else:
    print()
    print("[Warning] No scale produced a normal distribution!")
    print("Try even smaller values (< 0.01) or adjust other parameters.")

print()
print("=" * 80)
print("Test completed!")
print("=" * 80)
