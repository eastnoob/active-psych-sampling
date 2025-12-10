# CustomPoolBasedGenerator Transform 修复方案

## 问题诊断

### 症状

- Generator返回的点不在pool中
- param_validator一直在修正值: x1: 1.0 → 2.8, x2: 0.5 → 6.5
- AEPsych返回normalized值而不是raw categorical values

### 根本原因

经过深入分析,发现问题的根源是:**design_space.py错误地将numeric categorical values映射为indices**

```python
# ❌ 错误: 将2.8, 4.0, 8.5映射为0, 1, 2
df_numeric["x1_CeilingHeight"] = df_design["x1_CeilingHeight"].map({
    2.8: 0, 4.0: 1, 8.5: 2
})
```

### AEPsych Categorical 处理机制

AEPsych对categorical参数的处理:

1. **Config中定义choices**:

   ```ini
   [x1_CeilingHeight]
   par_type = categorical
   choices = [2.8, 4.0, 8.5]  # raw values
   ```

2. **Generator返回raw values**:

   ```python
   # ManualGenerator example
   points = [[2.8, 6.5, 2, 2, 0, 0]]  # x1=2.8, x2=6.5 (numeric), x3=2 (index for "Strict")
   ```

3. **Numeric categorical vs String categorical**:
   - **Numeric** (x1, x2): 使用actual values (2.8, 4.0, 8.5, 6.5, 8.0)
   - **String** (x3-x6): 使用indices (0, 1, 2 for "Chaos", "Rotated", "Strict")

4. **ParameterTransformedGenerator**:
   - 只负责continuous parameters的normalization
   - **不负责categorical → indices的映射**

## 修复方案

### 步骤 1: 修改 `design_space.py`

**文件**: `tests/is_EUR_work/00_plans/251206/scripts/modules/design_space.py`

修改`transform_to_numeric()`函数:

```python
def transform_to_numeric(df_design: pd.DataFrame) -> np.ndarray:
    """
    将DataFrame转换为数值格式（全categorical）
    
    变量映射规则：
    - x1_CeilingHeight: 2.8, 4.0, 8.5 → 保持原值（numeric categorical）
    - x2_GridModule: 6.5, 8.0 → 保持原值（numeric categorical）
    - x3_OuterFurniture: Chaos, Rotated, Strict → 0, 1, 2 (string categorical)
    - x4_VisualBoundary: Color, Solid, Translucent → 0, 1, 2
    - x5_PhysicalBoundary: Closed, Open → 0, 1
    - x6_InnerFurniture: Chaos, Rotated, Strict → 0, 1, 2
    """
    df_numeric = df_design.copy()

    # x1: CeilingHeight - 保持原始numeric values
    df_numeric["x1_CeilingHeight"] = df_design["x1_CeilingHeight"].astype(float)
    
    # x2: GridModule - 保持原始numeric values
    df_numeric["x2_GridModule"] = df_design["x2_GridModule"].astype(float)

    # x3: OuterFurniture - string categorical → indices
    df_numeric["x3_OuterFurniture"] = df_design["x3_OuterFurniture"].map({
        "Chaos": 0, "Rotated": 1, "Strict": 2
    }).astype(float)

    # x4: VisualBoundary - string categorical → indices
    df_numeric["x4_VisualBoundary"] = df_design["x4_VisualBoundary"].map({
        "Color": 0, "Solid": 1, "Translucent": 2
    }).astype(float)

    # x5: PhysicalBoundary - string categorical → indices
    df_numeric["x5_PhysicalBoundary"] = df_design["x5_PhysicalBoundary"].map({
        "Closed": 0, "Open": 1
    }).astype(float)

    # x6: InnerFurniture - string categorical → indices
    df_numeric["x6_InnerFurniture"] = df_design["x6_InnerFurniture"].map({
        "Chaos": 0, "Rotated": 1, "Strict": 2
    }).astype(float)

    # 按正确顺序选择列
    cols_order = [
        "x1_CeilingHeight", "x2_GridModule", "x3_OuterFurniture",
        "x4_VisualBoundary", "x5_PhysicalBoundary", "x6_InnerFurniture"
    ]
    design_space = df_numeric[cols_order].values.astype(np.float64)

    # 防御性检查
    allowed = [
        {2.8, 4.0, 8.5},  # x1 numeric categorical values
        {6.5, 8.0},        # x2 numeric categorical values
        {0, 1, 2},         # x3 string categorical indices
        {0, 1, 2},         # x4 string categorical indices
        {0, 1},            # x5 string categorical indices
        {0, 1, 2},         # x6 string categorical indices
    ]
    for col_idx, allowed_set in enumerate(allowed):
        col = design_space[:, col_idx]

        # x3-x6 are integer indices
        if col_idx >= 2 and not np.all(col == col.astype(int)):
            raise ValueError(f"column {col_idx} not integral: {col}")

        if not set(col.tolist()).issubset(allowed_set):
            raise ValueError(f"column {col_idx} out of range: {col}")

    return design_space
```

同时修改`validate_design_space()`:

```python
def validate_design_space(design_space: np.ndarray, column_names: list):
    """验证设计空间的值范围（全categorical）"""
    print(f"\n✓ 设计空间转换完成: {design_space.shape}")
    print(f"  值范围验证:")

    # 预期范围
    expected_ranges = [
        (2.8, 8.5, "x1_CeilingHeight"),      # numeric: 2.8, 4.0, 8.5
        (6.5, 8.0, "x2_GridModule"),         # numeric: 6.5, 8.0
        (0, 2, "x3_OuterFurniture"),         # indices: 0,1,2
        (0, 2, "x4_VisualBoundary"),         # indices: 0,1,2
        (0, 1, "x5_PhysicalBoundary"),       # indices: 0,1
        (0, 2, "x6_InnerFurniture"),         # indices: 0,1,2
    ]

    for i, (min_expected, max_expected, name) in enumerate(expected_ranges):
        min_val = design_space[:, i].min()
        max_val = design_space[:, i].max()
        unique_count = len(np.unique(design_space[:, i]))

        # x1和x2显示浮点数，x3-x6显示整数
        if i < 2:
            print(f"    x{i} ({name}): [{min_val:.1f}, {max_val:.1f}], {unique_count} 个唯一值")
        else:
            print(f"    x{i} ({name}): [{min_val:.0f}, {max_val:.0f}], {unique_count} 个唯一值")

        assert design_space[:, i].min() >= min_expected, f"{name} 最小值错误: {min_val} < {min_expected}"
        assert design_space[:, i].max() <= max_expected, f"{name} 最大值错误: {max_val} > {max_expected}"

    assert design_space.shape[1] == 6, f"设计空间维度错误: {design_space.shape}"
    print(f"✓ 所有值范围验证通过 (x1/x2使用numeric values，x3-x6使用indices)")
```

### 步骤 2: 撤销 `CustomPoolBasedGenerator` 的normalize修改

**文件**: `extensions/custom_generators/custom_pool_based_generator.py`

#### 2.1 撤销__init__中的修改

恢复原注释:

```python
# For categorical parameters, pool_points should contain RAW CATEGORICAL VALUES:
# - Numeric categorical (x1, x2): actual values (2.8, 4.0, 8.5, 6.5, 8.0)
# - String categorical (x3-x6): indices (0, 1, 2)
# 
# AEPsych will handle categorical mapping internally. DO NOT manually normalize.

# Store pool points
self.raw_pool_points = pool_points.clone()

logger.info(f"[PoolGen] Pool points loaded: {pool_points.shape}, containing raw categorical values")

# Optionally shuffle the pool
if shuffle:
    if seed is not None:
        torch.manual_seed(seed)
    perm = torch.randperm(len(pool_points))
    pool_points = pool_points[perm]

self.pool_points = pool_points  # Raw categorical values
```

#### 2.2 删除normalize方法和调用

删除 `_normalize_pool_points()` 方法。

在`gen()`方法中,恢复直接返回:

```python
# Mark as used
self._used_indices.update(selected_pool_indices.tolist())

# Store the last selected indices for external access
self.last_selected_indices = selected_pool_indices.tolist()

# 【新增】记录选中的点到去重数据库
self._record_points_to_dedup_db(selected_points)

# Return raw categorical values directly
# AEPsych will handle categorical mapping (numeric values stay as-is, 
# string categorical indices are mapped to choice strings)
return selected_points
```

### 步骤 3: 更新config中的bounds

**文件**: `tests/is_EUR_work/00_plans/251206/scripts/eur_config_residual.ini`

修改common section的bounds以匹配raw categorical values:

```ini
[common]
parnames = ['x1_CeilingHeight', 'x2_GridModule', 'x3_OuterFurniture', 'x4_VisualBoundary', 'x5_PhysicalBoundary', 'x6_InnerFurniture']
stimuli_per_trial = 1
outcome_types = [continuous]
strategy_names = [init_strat, opt_strat]
# lb/ub for numeric categorical use actual min/max values
# lb/ub for string categorical use index ranges
lb = [2.8, 6.5, 0, 0, 0, 0]
ub = [8.5, 8.0, 2, 2, 1, 2]
```

保持各参数的individual lb/ub定义不变(这些会被common覆盖)。

## 预期效果

修复后:

1. **Pool_points包含raw categorical values**:

   ```python
   pool_points = torch.tensor([
       [2.8, 6.5, 0, 0, 0, 0],  # x1=2.8, x2=6.5, x3="Chaos"(0), ...
       [4.0, 6.5, 1, 1, 0, 1],  # x1=4.0, x2=6.5, x3="Rotated"(1), ...
       [8.5, 8.0, 2, 2, 1, 2],  # x1=8.5, x2=8.0, x3="Strict"(2), ...
   ])
   ```

2. **Generator直接返回这些raw values**

3. **AEPsych Server返回的x_dict包含raw categorical values**:

   ```python
   {
       "'x1_CeilingHeight'": [2.8],  # ✅ not 0.0
       "'x2_GridModule'": [6.5],      # ✅ not 0.0
       "'x3_OuterFurniture'": [0],    # ✅ index for "Chaos"
       ...
   }
   ```

4. **param_validator不再需要修正值**

## 测试验证

修复后运行测试:

```powershell
cd d:\ENVS\active-psych-sampling
pixi run python tests\is_EUR_work\00_plans\251206\scripts\run_eur_residual.py --budget 5 --tag transform_fix_v2
```

检查日志中**不应该再有** `[参数修正]` 的WARNING。

## 相关文件清单

修改的文件:

1. `tests/is_EUR_work/00_plans/251206/scripts/modules/design_space.py` - 保持numeric categorical values
2. `extensions/custom_generators/custom_pool_based_generator.py` - 撤销normalize修改
3. `tests/is_EUR_work/00_plans/251206/scripts/eur_config_residual.ini` - 更新lb/ub bounds

## 注意事项

1. **数据类型一致性**: 确保pool_points中x1, x2使用float类型存储numeric values (2.8, 4.0, 8.5, 6.5, 8.0)

2. **Config parser**: ManualGenerator的points定义已经正确使用raw values,这是验证我们方案正确性的证据

3. **Backward compatibility**: 如果有其他使用indices的代码,需要同步更新

## 下一步

如果修复后问题仍然存在,需要检查:

1. AEPsych Server的categorical parameter transform chain
2. ParameterTransformedGenerator是否有额外的transform逻辑
3. Config parsing是否正确解析numeric categorical choices
