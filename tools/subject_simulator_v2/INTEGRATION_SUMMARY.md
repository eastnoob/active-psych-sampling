# Warmup Adapter V3 Integration Summary

## 集成完成 ✅

成功将 **Config #1 (Interaction-as-Features 方法)** 集成到 `warmup_adapter.py` 作为默认选项。

## 关键成果

### 1. 完美的分布 (Perfect Distribution)
- **Mean**: 3.01 (理想中点)
- **Coverage**: 全部 5 个等级
- **Max ratio**: 28.7% (非常平衡)
- **分布细节**:
  - Likert 1: 93 (28.7%)
  - Likert 2: 45 (13.9%)
  - Likert 3: 41 (12.7%)
  - Likert 4: 56 (17.3%)
  - Likert 5: 89 (27.5%)

### 2. 可探测的交互效应 (Detectable Interaction)
- **x3 × x4 (categorical)**: 权重 = 0.12 (强，可探测)
  - 贡献范围: 0.00 - 0.48 (足以改变 ~0.5 Likert 等级)
- **x0 × x1 (continuous)**: 权重 = -0.02 (弱，平衡分布)

### 3. 技术实现

#### 方法：交互作为显式特征
```python
# 扩展设计空间: 6 个基础特征 + 2 个交互特征 = 8 个总特征
X_extended = [x0, x1, x2, x3, x4, x5, x3*x4, x0*x1]

# 权重采样
main_weights = N(0.0, 0.3)  # 6 个主效应
interaction_weights = [0.12, -0.02]  # 2 个交互项（固定）

# 组合
population_weights = [main_weights, interaction_weights]  # 8 个权重
```

## 使用方法

### 默认使用 (V3 方法)
```python
from subject_simulator_v2.adapters.warmup_adapter import run

# 直接调用，自动使用 V3 方法
run(
    input_dir="path/to/sampling/plan",
    seed=99,
    design_space_csv="path/to/full_design_space.csv",
    # 其他参数使用默认值即可
)
```

**默认参数**:
- `interaction_as_features=True` (启用 V3)
- `interaction_x3x4_weight=0.12` (强交互)
- `interaction_x0x1_weight=-0.02` (弱交互)
- `population_mean=0.0`
- `population_std=0.3`

### 可选：使用旧的 V2 方法
```python
run(
    input_dir="path/to/sampling/plan",
    interaction_as_features=False,  # 禁用 V3，使用 V2
    interaction_pairs=[(3, 4), (0, 1)],
    interaction_scale=0.05,
    # ...
)
```

## 文件结构

### 新增文件
1. **warmup_adapter_v3.py**: V3 方法的独立实现
2. **test_adapter_v3.py**: V3 单元测试
3. **test_integrated_adapter.py**: 集成测试
4. **test_asymmetric_interactions.py**: 参数优化测试

### 修改文件
1. **warmup_adapter.py**:
   - 添加 V3 相关参数
   - 添加分支逻辑 (默认调用 V3)

## 验证结果

### 集成测试 (test_integrated_adapter.py)
```
✅ Exact match: YES
✅ V3 method used: YES
✅ Interaction weights correct: YES
✅ Distribution perfect: YES
```

### 输出文件格式
`fixed_weights_auto.json` 扩展格式:
```json
{
  "global": [[w0, w1, w2, w3, w4, w5]],
  "interactions": {
    "3,4": 0.12,
    "0,1": -0.02
  },
  "bias": -4.08,
  "method": "interaction_as_features_v3"
}
```

## 技术优势

### 相比 V2 方法
1. **分布质量**: V2 容易产生极端偏斜 (85-100% Likert 5)，V3 完美平衡
2. **可探测性**: V2 交互权重太小难以探测，V3 保证至少一个强交互
3. **可控性**: V3 使用固定交互权重，行为可预测
4. **简洁性**: 直接矩阵乘法，无需 LinearSubject

### 关键洞察
- **分类交互** (x3×x4, 范围 0-4): 需要较大权重 (0.12) 才可探测
- **连续交互** (x0×x1, 范围 18-68): 需要极小权重 (-0.02) 避免支配分布
- **非对称策略**: 一强一弱，既保证可探测性又维持平衡

## 演进历史

1. **V1**: 基础 ClusterGenerator
2. **V2**: 添加 interaction_pairs 参数
   - 问题: interaction_scale 难以调优 (任何值都导致偏斜)
3. **V3** (当前): Interaction-as-features 方法
   - 解决方案: 交互作为显式特征 + 非对称固定权重

## 下一步

集成已完成并验证通过。用户可以直接使用默认参数运行 Step 1.5，将获得：
- 平衡的 Likert 分布 (Mean ≈ 3.0)
- 至少一个可探测的交互效应 (x3×x4)
- 真实的心理学响应模式

---

**状态**: ✅ 集成完成，测试通过
**日期**: 2025-11-30
**方法**: Config #1 (Interaction-as-Features V3)
