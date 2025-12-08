# Step 3 分类变量编码问题修复说明

## 问题描述

**错误信息**:
```
ValueError: could not convert string to float: 'Strict'
```

**发生位置**: `phase1_step3_base_gp.py` 第 483 行（设计空间扫描时）

## 根本原因

### 数据流程中的编码不一致

```
Step 1 采样方案 (subject_1.csv)
  ↓ 分类变量: 'Strict', 'Rotated', 'Chaos', ...

Step 1.5 模拟
  ↓ warmup_adapter 自动编码为数值

训练数据 (result/subject_1.csv)
  → 数值变量: 0, 1, 2, ...

----- Step 3 训练 -----

训练数据读取 → 数值型，_encode_factor_df 无法识别为分类变量
  → encodings 字典为空！

设计空间 (design_space.csv)
  → 分类变量: 'Strict', 'Solid', 'Closed', ...

_apply_encodings(设计空间, encodings={})
  → 无编码映射 → 分类值保留 → astype(float) 失败！
```

## 解决方案

### 1. 新增函数：`_infer_encoding_from_sampling`

**功能**: 从采样方案和模拟结果推断编码映射

**原理**:
- 读取 `subject_1.csv` (采样方案，分类)
- 读取 `result/subject_1.csv` (模拟结果，数值)
- 逐行对比相同位置的值，推断映射关系

**示例输出**:
```
[推断编码] x3_OuterFurniture: {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
[推断编码] x4_VisualBoundary: {'Color': 0, 'Solid': 1}
[推断编码] x5_PhysicalBoundary: {'Closed': 0, 'Open': 1}
[推断编码] x6_InnerFurniture: {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
```

### 2. 改进函数：`_apply_encodings`

**变更**:
```python
# 修改前：只处理 encodings 字典中的列
for col, mapping in encodings.items():
    ...

# 修改后：遍历所有设计空间列
for col in df_new.columns:
    if col in encodings:
        # 应用编码
    else:
        # 检查是否为未编码的分类列（报错）
        if df_new[col].dtype == "object":
            raise ValueError(f"列 '{col}' 是分类变量但无编码映射")
```

**优势**:
- 捕获所有未编码的分类列
- 提供清晰的错误信息（包含列名和样本值）

### 3. 集成到 `process_step3`

**目录模式下的完整流程**:
```python
# 1. 读取训练数据 (numeric)
df_phase1 = pd.concat(subject_csvs)

# 2. 尝试编码（但因为是numeric，encodings为空）
encoded_factors, encodings = _encode_factor_df(factor_df)

# 3. 【新增】推断编码（仅目录模式）
if data_path.is_dir():
    inferred_encodings = _infer_encoding_from_sampling(data_path, factor_cols)
    # 合并到 encodings
    encodings.update(inferred_encodings)

# 4. 读取设计空间 (categorical)
design_df_raw = pd.read_csv(design_space_csv)

# 5. 应用编码（现在有映射了！）
design_df_encoded = _apply_encodings(design_df_raw, encodings)

# 6. 转换为浮点数 ✅ 成功
design_df_encoded.values.astype(float)
```

## 修改的文件

### `phase1_step3_base_gp.py`

#### 新增函数（第 33-81 行）
```python
def _infer_encoding_from_sampling(
    data_dir: Path, factor_cols: List[str]
) -> Dict[str, Dict[Any, int]]:
    """从采样方案和模拟结果推断编码映射"""
```

#### 改进函数（第 104-134 行）
```python
def _apply_encodings(
    df: pd.DataFrame, encodings: Dict[str, Dict[Any, int]]
) -> pd.DataFrame:
    """遍历所有列，确保没有遗漏的分类变量"""
```

#### 集成调用（第 522-532 行）
```python
if data_path.is_dir():
    print("\n[推断] 从采样方案推断分类变量编码...")
    inferred_encodings = _infer_encoding_from_sampling(data_path, factor_cols)
    for col, mapping in inferred_encodings.items():
        if col not in encodings or not encodings[col]:
            encodings[col] = mapping
```

#### 调试输出（第 547-560 行）
```python
# Debug: 检查编码前后的数据类型
print("\n[Debug] 设计空间编码前:")
for col in design_df_aligned.columns:
    print(f"  {col}: dtype={...}, in_encodings={...}, sample_values={...}")

print("\n[Debug] 设计空间编码后:")
for col in design_df_encoded.columns:
    print(f"  {col}: dtype={...}, sample_values={...}")
```

## 使用说明

### 运行 Step 3（目录模式）

```python
# quick_start.py
MODE = "step3"

STEP3_CONFIG = {
    # 【方式1】目录模式（推荐）- 会自动推断编码
    "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result",

    # 设计空间（分类变量）
    "design_space_csv": "data\\only_independences\\data\\only_independences\\i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv",

    # 其他参数...
}
```

### 预期输出

```
[Step3] 从目录读取被试数据: extensions\warmup_budget_check\sample\202511302204\result
  找到 5 个被试文件
    - subject_1.csv: 30 行
    - subject_2.csv: 30 行
    ...
  合并后总计: 150 行

[推断] 从采样方案推断分类变量编码...
[推断编码] x3_OuterFurniture: {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
[推断编码] x4_VisualBoundary: {'Color': 0, 'Solid': 1}
[推断编码] x5_PhysicalBoundary: {'Closed': 0, 'Open': 1}
[推断编码] x6_InnerFurniture: {'Strict': 0, 'Rotated': 1, 'Chaos': 2}
  使用推断编码: x3_OuterFurniture
  使用推断编码: x4_VisualBoundary
  使用推断编码: x5_PhysicalBoundary
  使用推断编码: x6_InnerFurniture

[Debug] 设计空间编码前:
  x1_CeilingHeight: dtype=float64, in_encodings=False, sample_values=[2.8, 2.8, 2.8]
  x2_GridModule: dtype=float64, in_encodings=False, sample_values=[6.5, 6.5, 6.5]
  x3_OuterFurniture: dtype=object, in_encodings=True, sample_values=['Strict', 'Strict', 'Strict']
  x4_VisualBoundary: dtype=object, in_encodings=True, sample_values=['Solid', 'Solid', 'Solid']
  ...

[Debug] 设计空间编码后:
  x1_CeilingHeight: dtype=float64, sample_values=[2.8, 2.8, 2.8]
  x2_GridModule: dtype=float64, sample_values=[6.5, 6.5, 6.5]
  x3_OuterFurniture: dtype=int64, sample_values=[0, 0, 0]  ✅ 已编码
  x4_VisualBoundary: dtype=int64, sample_values=[1, 1, 1]  ✅ 已编码
  ...

[训练] 开始训练 Base GP...
[扫描] 扫描设计空间 (xxxx 个点)...
✅ 成功！
```

## 兼容性

### 文件模式（不受影响）

如果使用文件模式（combined_results.csv），编码推断会被跳过：
```python
STEP3_CONFIG = {
    # 【方式2】文件模式
    "data_csv_path": "extensions\\...\\combined_results.csv",
    ...
}
```

**行为**:
- 不调用 `_infer_encoding_from_sampling`（因为 `data_path.is_file()`）
- 如果文件中已是数值，正常工作
- 如果文件中有分类变量，`_encode_factor_df` 会正常编码

### 旧版本数据

如果 `subject_1.csv` 不存在于采样目录，会打印警告但不中断：
```
[Warning] 无法推断编码：找不到采样文件或结果文件
```

## 测试建议

### 1. 验证编码推断
```bash
# 运行 Step 3，检查日志中的 [推断编码] 输出
python quick_start.py
```

### 2. 检查关键点
```python
# 查看生成的 base_gp_key_points.json
import json
with open('base_gp_output/base_gp_key_points.json') as f:
    points = json.load(f)

# Sample 1 (Best Prior) 应该有正确的因子值
print(points['x_best_prior'])
# 输出示例: {'x1_CeilingHeight': 4.0, 'x3_OuterFurniture': 1, ...}
```

### 3. 验证设计空间扫描
```python
# 查看 design_space_scan.csv
import pandas as pd
df = pd.read_csv('base_gp_output/design_space_scan.csv')

# 所有因子列应该是数值型
print(df.dtypes)
# x1_CeilingHeight    float64
# x3_OuterFurniture   int64  ✅
# pred_mean           float64
# pred_std            float64
```

## 注意事项

1. **编码一致性**: 推断的编码基于 subject_1.csv 的前 N 行，确保采样方案覆盖所有分类水平

2. **多文件一致性**: 假设所有 subject_*.csv 使用相同的编码（由同一个 warmup_adapter 生成）

3. **设计空间完整性**: 设计空间中出现的分类值必须在采样方案中出现过，否则会报错"未知类别"

## 未来改进方向

1. **保存编码映射**: 在 Step 1.5 时保存编码到 JSON 文件
2. **显式编码参数**: 允许用户手动提供编码映射
3. **双向推断**: 支持从设计空间推断训练数据的编码

---

**修复日期**: 2025-11-30
**修复版本**: warmup_budget_check v1.2
**相关问题**: Step 3 设计空间扫描编码错误
