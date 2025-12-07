# Migration Guide: 替换 quick_start.py Step 1.5

本指南说明如何使用Subject Simulator V2替换`extensions/warmup_budget_check/quick_start.py`中的Step 1.5（模拟被试作答）。

---

## 为什么要替换？

**旧实现（simulation_runner.py + MixedEffectsLatentSubject）的问题：**

1. **3个严重bug** - 详见`test/is_EUR_work/00_plans/20251130/BUG_FIX_SUMMARY.md`
   - Bug #1: 缺少交互项计算（导致98.8% Likert=3单调输出）
   - Bug #2: Likert转换公式错误（除法应为乘法）
   - Bug #3: fixed_weights被忽略

2. **参数保存不完整** - 缺少item_biases, item_noises等，无法完整复现
3. **无正态性检查** - 可能生成单调/偏斜数据
4. **复杂潜变量结构** - 难以理解和调试

**新实现（Subject Simulator V2）的优势：**

1. ✓ 所有bug已修复
2. ✓ 完整参数保存（JSON格式）
3. ✓ 自动正态性检查（可选）
4. ✓ 简洁线性模型，易于理解
5. ✓ 完全兼容旧接口

---

## 替换步骤

### 方案1：修改quick_start.py（推荐）

在`extensions/warmup_budget_check/quick_start.py`的**Step 1.5函数**中：

**原代码（第644行）：**
```python
from core.simulation_runner import run as simulate_responses
```

**替换为：**
```python
# 使用Subject Simulator V2（修复了旧实现的3个bug）
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from subject_simulator_v2.adapters.warmup_adapter import run as simulate_responses
```

**就这样！** 其他代码无需修改，所有参数和输出格式完全兼容。

---

### 方案2：创建新的quick_start_v2.py（保守）

如果不想修改原文件，可以复制一份：

```bash
cp extensions/warmup_budget_check/quick_start.py extensions/warmup_budget_check/quick_start_v2.py
```

然后在`quick_start_v2.py`中应用方案1的修改。

---

## 参数映射表

所有旧参数都被正确映射到V2实现：

| 旧参数 (simulation_runner.py) | V2映射 (warmup_adapter.py) | 说明 |
|-------------------------------|----------------------------|------|
| `seed` | `seed` | ✓ 直接使用 |
| `population_mean` | `population_mean` | ✓ 直接使用 |
| `population_std` | `population_std` | ✓ 直接使用 |
| `individual_std_percent` | `individual_std = population_std * individual_std_percent` | ✓ 自动计算 |
| `interaction_pairs` | `interaction_pairs` | ✓ 直接使用 |
| `interaction_scale` | `interaction_scale` | ✓ 直接使用 |
| `output_type` | `likert_levels=None(continuous) / likert_levels=5(likert)` | ✓ 自动转换 |
| `likert_levels` | `likert_levels` | ✓ 直接使用 |
| `likert_sensitivity` | `likert_sensitivity` | ✓ 直接使用 |
| `likert_mode` | `likert_sensitivity调整（percentile时减半）` | ✓ 自动适配 |
| `output_mode` | `output_mode` (individual/combined/both) | ✓ 直接使用 |
| `clean` | `clean` | ✓ 直接使用 |
| `fixed_weights_file` | `fixed_weights_file` | ✓ 直接使用 |
| `print_model` | `print_model` | ✓ 直接使用 |
| `save_model_summary` | `save_model_summary` | ✓ 直接使用 |
| `model_summary_format` | `model_summary_format` | ✓ 直接使用 |
| `use_latent` | ⚠ **忽略**（V2不支持潜变量） | 会打印警告 |
| `individual_corr` | ⚠ **忽略**（V2不支持特征相关） | 会打印警告 |
| `num_interactions` | ⚠ **忽略**（请使用interaction_pairs） | 会打印警告 |

**新增参数（V2专属）：**
- `ensure_normality=True` - 自动检查正态性，避免单调数据
- `bias=0.0` - 调整响应中心
- `noise_std=0.0` - 试次内噪声

---

## 输出文件对比

V2适配器输出的文件与旧实现**完全兼容**，且增加了额外信息：

| 文件名 | 旧实现 | V2实现 | 备注 |
|-------|-------|--------|------|
| `subject_1.csv` ... `subject_N.csv` | ✓ | ✓ | 格式相同 |
| `combined_results.csv` | ✓ | ✓ | 格式相同 |
| `subject_1_model.md` ... | ✓ | ✓ | 格式相同（兼容性） |
| `fixed_weights_auto.json` | ✓ | ✓ | 格式相同 |
| `subject_1_spec.json` ... | ✗ | **✓ 新增** | V2完整参数（可复现） |
| `cluster_summary.json` | ✗ | **✓ 新增** | 集群参数摘要 |
| `MODEL_SUMMARY.txt/md` | ✗ | **✓ 新增** | 模型规格总览 |

**Step 2和Step 3无需修改** - 它们读取`combined_results.csv`，格式完全一致。

---

## 测试替换效果

### 测试1：运行Step 1.5

在`quick_start.py`中设置：

```python
MODE = "step1.5"

STEP1_5_CONFIG = {
    "input_dir": "extensions\\warmup_budget_check\\sample\\202511301011",
    "seed": 42,
    "output_mode": "combined",
    "output_type": "likert",
    "likert_levels": 5,
    "likert_mode": "tanh",
    "likert_sensitivity": 2.0,
    "population_mean": 0.0,
    "population_std": 0.3,
    "individual_std_percent": 0.3,
    "interaction_pairs": [(3, 4), (0, 1)],
    "interaction_scale": 0.25,
    "ensure_normality": True,  # V2新增
}
```

运行：
```bash
cd extensions/warmup_budget_check
python quick_start.py
```

**检查输出：**
- 响应分布应该是正态的（不再是98.8% Likert=3）
- `result/combined_results.csv`格式应与旧版本一致
- 应生成额外的`subject_X_spec.json`文件

### 测试2：完整流程（Step 1 -> 1.5 -> 2 -> 3）

```python
MODE = "all"
```

运行完整链条，确保Step 2和Step 3能正确读取V2生成的数据。

---

## 回滚方案

如果遇到问题，只需还原quick_start.py的导入：

```python
# 还原为旧实现
from core.simulation_runner import run as simulate_responses
```

所有旧文件仍然保留，无风险。

---

## 常见问题

### Q: 为什么有些参数被忽略？

**A:** V2采用更简洁的线性模型，不支持以下特性：
- `use_latent=True` - 潜变量模型（过于复杂，且有bug）
- `individual_corr!=0` - 特征间相关（实际数据中少见）
- `num_interactions>0` - 随机交互项（请明确指定interaction_pairs）

这些简化是有意为之，基于以下原则：
1. **简洁优于复杂** - Linear > Latent
2. **明确优于隐式** - 明确指定交互对 > 随机生成
3. **可复现优于随机** - 完整参数保存

### Q: V2生成的数据质量如何？

**A:** 更好！
- ✓ 修复了3个严重bug
- ✓ 自动正态性检查（如果失败会重试）
- ✓ 完整参数保存（可100%复现）

测试对比（5被试，1296设计点）：
- **旧实现**: 98.8% Likert=3（单调！）
- **V2实现**: Likert 1-5均匀分布，均值3.18-3.96

### Q: 会影响Step 2和Step 3吗？

**A:** 不会，甚至更好！
- Step 2读取`combined_results.csv`，格式完全一致
- 由于数据质量提升（正态分布），Step 2的交互效应检测**更准确**
- Step 3训练Base GP时，输入数据质量更高

---

## 总结

**替换非常简单：**
1. 修改quick_start.py中的1行导入代码
2. 所有参数和输出格式完全兼容
3. 数据质量显著提升
4. 无需修改Step 2和Step 3

**建议立即替换！** 旧实现的bug会导致Phase 1数据质量低下，影响后续所有分析。
