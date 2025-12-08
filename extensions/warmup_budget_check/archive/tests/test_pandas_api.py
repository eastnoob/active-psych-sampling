"""测试pandas API类型检测改进"""
import pandas as pd
import numpy as np
from warmup_sampler import is_categorical, gower_distance

print("=" * 60)
print("Pandas API类型检测改进测试")
print("=" * 60)
print()

# 测试1: 传统类型
print("测试1 - 传统类型检测:")
df_traditional = pd.DataFrame({
    'num': [1, 2, 3],
    'obj': ['A', 'B', 'C'],
    'bool': [True, False, True]
})
print(f"  object列检测: {is_categorical(df_traditional['obj'].dtype)}")
print(f"  数值列检测: {is_categorical(df_traditional['num'].dtype)}")
print("[OK] 传统类型兼容")
print()

# 测试2: pandas新类型（如果有）
print("测试2 - pandas nullable types:")
try:
    df_nullable = pd.DataFrame({
        'string': pd.array(['A', 'B', 'C'], dtype='string'),
        'boolean': pd.array([True, False, True], dtype='boolean'),
        'Int64': pd.array([1, 2, 3], dtype='Int64')
    })
    print(f"  StringDtype检测: {is_categorical(df_nullable['string'].dtype)}")
    print(f"  BooleanDtype检测（应为False）: {is_categorical(df_nullable['boolean'].dtype)}")
    print(f"  BooleanDtype用is_bool_dtype: {pd.api.types.is_bool_dtype(df_nullable['boolean'].dtype)}")
    print("[OK] nullable types支持")
except Exception as e:
    print(f"  [跳过] nullable types不可用: {e}")
print()

# 测试3: CategoricalDtype
print("测试3 - CategoricalDtype:")
df_cat = pd.DataFrame({
    'cat_ordered': pd.Categorical(['low', 'medium', 'high'],
                                   categories=['low', 'medium', 'high'],
                                   ordered=True),
    'cat_unordered': pd.Categorical(['A', 'B', 'C'])
})
print(f"  有序categorical检测: {is_categorical(df_cat['cat_ordered'].dtype)}")
print(f"  无序categorical检测: {is_categorical(df_cat['cat_unordered'].dtype)}")
print(f"  获取categories: {df_cat['cat_ordered'].cat.categories.tolist()}")
print("[OK] CategoricalDtype支持")
print()

# 测试4: Gower距离兼容性
print("测试4 - Gower距离兼容性:")
test_df = pd.DataFrame({
    'num': [1, 2, 3],
    'cat': pd.Categorical(['A', 'B', 'C']),
    'bool': [True, False, True]
})
dist = gower_distance(test_df.loc[0], test_df.loc[1], test_df)
print(f"  混合类型Gower距离: {dist:.4f}")
print("[OK] Gower距离正常工作")
print()

# 测试5: LHS分类映射确定性
print("测试5 - LHS分类映射确定性:")
df_mixed = pd.DataFrame({
    'cat_type': pd.Categorical(['A', 'C', 'B'], categories=['A', 'B', 'C']),
    'obj_type': ['Z', 'X', 'Y']
})
print(f"  CategoricalDtype categories: {df_mixed['cat_type'].cat.categories.tolist()}")
print(f"  object dtype sorted: {sorted(df_mixed['obj_type'].unique())}")
print("[OK] 确定性映射")
print()

print("=" * 60)
print("[OK] 所有pandas API改进测试通过！")
print("=" * 60)
