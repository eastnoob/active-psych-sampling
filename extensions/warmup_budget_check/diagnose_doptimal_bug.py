"""
诊断D-optimal实现的bug
检查pyDOE3返回值的真实结构
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from warmup_sampler import WarmupSampler

# 加载真实设计空间
design_csv = str(Path(__file__).parent.parent.parent / 'data' / 'i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv')
sampler = WarmupSampler(design_csv)

print("=" * 70)
print("诊断pyDOE3的optimal_design返回值")
print("=" * 70)
print()

# 获取candidates
available = list(range(100))  # 使用前100个配置
candidates = sampler._normalize_design_to_candidates(available)

print(f"Candidates shape: {candidates.shape}")
print(f"Candidates前3行:\n{candidates[:3]}")
print()

# 调用pyDOE3
from pyDOE3.doe_optimal import optimal_design

n_configs = 20
design, info = optimal_design(
    candidates,
    n_points=n_configs,
    degree=1,
    criterion="D",
    method="detmax",
)

print(f"Design shape: {design.shape}")
print(f"Info keys: {info.keys()}")
print(f"D_eff: {info.get('D_eff', 'N/A')}")
print()

print(f"Design前3行:\n{design[:3]}")
print()

# 检查design是否是candidates的子集
print("检查design是否完全匹配candidates中的某些行...")
matches = []
for i, design_row in enumerate(design):
    distances = np.linalg.norm(candidates - design_row, axis=1)
    closest_idx = np.argmin(distances)
    closest_dist = distances[closest_idx]
    matches.append((i, closest_idx, closest_dist))
    if i < 5:
        print(f"  Design行{i} -> Candidates行{closest_idx}, 距离={closest_dist:.6f}")

print()
print(f"平均距离: {np.mean([m[2] for m in matches]):.6f}")
print(f"最大距离: {np.max([m[2] for m in matches]):.6f}")

# 检查是否有重复选择
selected_candidate_idx = [m[1] for m in matches]
print(f"\n选中的候选索引: {selected_candidate_idx[:10]}...")
print(f"唯一索引数: {len(set(selected_candidate_idx))}")
print(f"是否有重复: {len(selected_candidate_idx) != len(set(selected_candidate_idx))}")

print()
print("=" * 70)
print("BUG分析")
print("=" * 70)

# 检查原代码的错误逻辑
selected_mask = np.any(design != 0, axis=1)
print(f"\n原代码的逻辑: selected_mask = np.any(design != 0, axis=1)")
print(f"  selected_mask.sum() = {selected_mask.sum()} (应该等于 {n_configs})")
print(f"  selected_mask[:5] = {selected_mask[:5]}")

if selected_mask.sum() != n_configs:
    print("\n❌ BUG确认: selected_mask不等于n_configs!")
    print("  原因: design所有行都非零，mask无意义")
else:
    print("\n✅ 这个逻辑没问题（但可能是巧合）")

# 检查D-efficiency计算是否正确
print(f"\n检查D-efficiency是否由pyDOE3正确返回:")
print(f"  info['D_eff'] = {info.get('D_eff', 'N/A')}")

if 'D_eff' in info:
    if info['D_eff'] < 1.0:
        print(f"  ⚠️ D-efficiency非常低 ({info['D_eff']:.6f}%)")
        print(f"  可能原因:")
        print(f"    1. 候选集高度结构化/规则网格")
        print(f"    2. 特征维度过高（one-hot爆炸）")
        print(f"    3. 候选集本身D-optimal空间有限")
