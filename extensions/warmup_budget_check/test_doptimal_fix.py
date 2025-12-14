"""
测试D-optimal修复方案
关键改进：
1. 使用原始候选集而非采样后的子集
2. 仅在候选集过大时（>1000）才采样
3. 调整degree参数或使用完整候选集
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from warmup_sampler import WarmupSampler

# 加载真实设计空间
design_csv = str(Path(__file__).parent.parent.parent / 'data' / 'i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv')
sampler = WarmupSampler(design_csv)

print("=" * 70)
print("测试不同候选集大小对D-efficiency的影响")
print("=" * 70)
print()

from pyDOE3.doe_optimal import optimal_design

# 测试1: 使用全部324个配置
print("测试1: 使用全部324个候选配置")
available_all = list(range(324))
candidates_all = sampler._normalize_design_to_candidates(available_all)
print(f"  Candidates shape: {candidates_all.shape}")

design1, info1 = optimal_design(
    candidates_all,
    n_points=26,  # 实际Core-2a需要的数量
    degree=1,
    criterion="D",
    method="detmax",
)
print(f"  D_eff: {info1.get('D_eff', 'N/A'):.4f}%")
print()

# 测试2: 仅使用100个配置
print("测试2: 仅使用前100个候选配置")
available_100 = list(range(100))
candidates_100 = sampler._normalize_design_to_candidates(available_100)
print(f"  Candidates shape: {candidates_100.shape}")

design2, info2 = optimal_design(
    candidates_100,
    n_points=26,
    degree=1,
    criterion="D",
    method="detmax",
)
print(f"  D_eff: {info2.get('D_eff', 'N/A'):.4f}%")
print()

# 测试3: 检查特征数vs样本数比例
print("测试3: 分析特征数/样本数比例")
n_features = candidates_all.shape[1]
n_samples = 26
print(f"  特征数: {n_features}")
print(f"  样本数: {n_samples}")
print(f"  比例: {n_samples / n_features:.2f}")
print(f"  线性模型参数数: {n_features + 1} (含截距)")
print(f"  自由度: {n_samples - (n_features + 1)}")

if n_samples < 2 * n_features:
    print(f"  [WARNING] samples < 2*features, D-optimal may not work well")
print()

# 测试4: 使用更简化的特征表示（不用one-hot）
print("测试4: 测试序数编码（避免one-hot爆炸）")
df_sub = sampler.design_df.iloc[available_all]
X_list = []

for col in df_sub.columns:
    col_data = df_sub[col]

    if col_data.dtype in ["float64", "int64", "int32", "float32"]:
        col_min, col_max = col_data.min(), col_data.max()
        if col_max > col_min:
            normalized = (col_data.values - col_min) / (col_max - col_min)
            X_list.append(normalized)
    elif col_data.dtype == "bool":
        X_list.append(col_data.astype(int).values)
    else:
        # 分类变量：强制使用序数编码
        unique_cats = col_data.unique()
        cat_to_idx = {cat: idx for idx, cat in enumerate(unique_cats)}
        encoded = col_data.map(cat_to_idx).values
        X_list.append(encoded / (len(unique_cats) - 1) if len(unique_cats) > 1 else encoded)

candidates_ordinal = np.column_stack(X_list)
print(f"  序数编码后的shape: {candidates_ordinal.shape}")

design4, info4 = optimal_design(
    candidates_ordinal,
    n_points=26,
    degree=1,
    criterion="D",
    method="detmax",
)
print(f"  D_eff: {info4.get('D_eff', 'N/A'):.4f}%")
print()

print("=" * 70)
print("结论")
print("=" * 70)
print()
print("比较D-efficiency:")
print(f"  全324配置(one-hot): {info1.get('D_eff', 0):.4f}%")
print(f"  前100配置(one-hot): {info2.get('D_eff', 0):.4f}%")
print(f"  全324配置(序数):   {info4.get('D_eff', 0):.4f}%")
print()

if info4.get('D_eff', 0) > info1.get('D_eff', 0):
    print("[OK] Recommend: Use ordinal encoding, D-efficiency improved")
else:
    print("[INFO] Ordinal encoding did not help much, likely design space limit")
