# Core-2b 三模式交互对探索系统

## 概述

Core-2b现已支持**三种交互对探索模式**，用户可以在`quick_start.py`中灵活配置，以在假设驱动和无偏探索之间找到最佳平衡。

---

## 三种模式详解

### 模式1: FREE（自由探索）- 默认行为

**配置:**

```python
ALL_CONFIG = {
    "interaction_mode": "free",
    # 其他参数被忽略
}
```

**行为:**

- Core-2b从所有可用配置中**随机选择**
- 不受用户指定交互对的影响
- 所有15个交互对（C(6,2)）被**等概率探索**

**适用场景:**

- ✅ 无先验知识，希望无偏探索
- ✅ 设计空间相对简单（因子少）
- ✅ 预算充足，可以覆盖所有对

**缺点:**

- ❌ 预算紧张时，某些对可能被严重欠采样
- ❌ 无法集中资源于"可疑"交互对

---

### 模式2: SPECIFIED_ONLY（仅指定对）

**配置:**

```python
ALL_CONFIG = {
    "interaction_mode": "specified_only",
    "interaction_pairs_to_explore": [(3, 4), (0, 1)],
    "min_config_per_pair": 2,
}
```

**行为:**

- Core-2b**仅为指定的交互对分配配置**
- 将预算均分给这些对
- 完全忽略其他交互对

**适用场景:**

- ✅ 对交互对有明确的先验假设
- ✅ 需要高效利用有限预算
- ✅ 大因子空间，只关心少数关键对

**缺点:**

- ❌ 可能遗漏重要的未指定交互对
- ❌ 无偏探索性差，可能导致模型偏差

---

### 模式3: HYBRID（混合模式）- **推荐** ⭐

**配置:**

```python
ALL_CONFIG = {
    "interaction_mode": "hybrid",
    "interaction_pairs_to_explore": [(3, 4), (0, 1)],
    "min_config_per_pair": 2,
}
```

**行为:**

1. **保护分配**: 为每个指定对至少分配`min_config_per_pair`个配置
   - 保证关键对的探索深度
   - 避免被随机淹没

2. **自由探索**: 剩余预算从所有可用配置中随机选择
   - 可以探索其他交互对
   - 减少遗漏意外交互的风险

3. **去重**: 保护配置和自由探索不重复

**适用场景:**

- ✅ **最通用的选择** - 平衡假设驱动和无偏探索
- ✅ 预算相对有限（如5人×20次）
- ✅ 既有先验知识，又想发现意外效应
- ✅ 实际研究中最常见的场景

**优点:**

- ✔️ 高可靠性：指定对必然被充分探索
- ✔️ 开放性：仍可发现未指定的重要交互
- ✔️ 灵活性：通过`min_config_per_pair`调整保护强度

---

## 实际配置示例

### 示例1: 预算充足，无明确假设

```python
ALL_CONFIG = {
    "n_subjects": 8,
    "trials_per_subject": 25,
    "interaction_mode": "free",  # 所有交互对等概率探索
}
```

**预期结果:**

- Core-2b分配~40个配置
- 每个交互对平均2-3个配置
- 广泛但轻微的覆盖

---

### 示例2: 预算紧张，高度聚焦

```python
ALL_CONFIG = {
    "n_subjects": 5,
    "trials_per_subject": 20,
    "interaction_mode": "specified_only",
    "interaction_pairs_to_explore": [
        (1, 3),  # GridModule × VisualBoundary (高风险交互)
        (0, 4),  # CeilingHeight × PhysicalBoundary (已知关键)
    ],
    "min_config_per_pair": 4,  # 每对至少4个
}
```

**预期结果:**

- Core-2b仅分配~8-10个配置（2对 × 4个）
- 高度聚焦于已知关键交互
- 零风险遗漏指定对

---

### 示例3: 平衡方案（推荐）✨

```python
ALL_CONFIG = {
    "n_subjects": 5,
    "trials_per_subject": 20,
    "interaction_mode": "hybrid",  # ⭐ 推荐模式
    "interaction_pairs_to_explore": [
        (3, 4),  # VisualBoundary × PhysicalBoundary
        (0, 1),  # CeilingHeight × GridModule
    ],
    "min_config_per_pair": 2,  # 每对最少2个
}
```

**预期分配:**

1. 保护分配: 2对 × 2个 = 4个配置（保证基本覆盖）
2. 自由探索: ~16-17个配置（从所有可用配置中随机抽样）
3. 总计: ~20-21个Core-2b配置

**特点:**

- 指定对的(3,4)和(0,1)必然被探索
- 其他交互对(如1-2, 2-5等)仍有机会被探索
- 发现意外交互的同时，聚焦已知关键对

---

## 参数参考

### ALL_CONFIG 中的新参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interaction_mode` | str | `"free"` | `"free"` \| `"specified_only"` \| `"hybrid"` |
| `interaction_pairs_to_explore` | list[tuple] | `None` | 交互对列表，如`[(3,4), (0,1)]` |
| `min_config_per_pair` | int | `2` | 每个指定对的最少配置数（hybrid模式生效） |

### 参数类型说明

**interaction_pairs_to_explore:**

- 使用**0-based索引**指代因子
- 例如：`[(3, 4)]` 表示第4个因子与第5个因子的交互
- 自动标准化为`(min, max)`形式

**min_config_per_pair:**

- 仅在`hybrid`模式下生效
- 在`specified_only`模式下被忽略（所有预算均分）
- 建议值: 2-3个（确保基本统计显著性）

---

## 使用流程

### 步骤1: 编辑 quick_start.py

在`ALL_CONFIG`中设置交互对参数：

```python
ALL_CONFIG = {
    # ... 其他参数 ...
    
    # Core-2b 三模式配置
    "interaction_mode": "hybrid",              # 选择模式
    "interaction_pairs_to_explore": [(3, 4)],  # 指定交互对
    "min_config_per_pair": 2,                  # 最少配置数
}
```

### 步骤2: 运行 Step 1

```bash
python quick_start.py  # MODE='step1'
```

系统输出示例：

```
================================================================================
Core-2b模式: HYBRID
  指定交互对: [(3, 4), (0, 1)]
  每对最少配置: 2个
================================================================================
```

### 步骤3: 查看采样结果

检查生成的CSV文件：

- 查看Core-2b配置是否合理分布
- 检查保护对是否都被包含
- 验证其他因子组合的多样性

---

## 测试脚本

已提供`test_three_modes.py`脚本，可对比三种模式的行为：

```bash
cd extensions/warmup_budget_check
pixi run python test_three_modes.py
```

输出会在三个目录中生成样本：

- `test_output_mode_free/` - 自由模式
- `test_output_mode_specified/` - 指定对模式
- `test_output_mode_hybrid/` - 混合模式

---

## 常见问题

### Q1: 如何确定交互对列表？

**方法A: 基于理论**

- 查阅相关领域文献
- 咨询领域专家
- 基于已知的机制假设

**方法B: 基于探索**

- 先用`interaction_mode="free"`做初步探索
- 检查Phase 1的交互分析结果
- 在Phase 2中使用发现的重要对

**方法C: 混合**

- hybrid模式：既聚焦理论预测，又开放于数据驱动发现

### Q2: min_config_per_pair 应该设多少？

推荐值：**2-3个**

| 预算 | 建议 | 理由 |
|------|------|------|
| 紧张(5人×20) | 2 | 最低统计保障 |
| 中等(8人×20) | 2-3 | 平衡资源 |
| 充足(10人×25) | 3-4 | 提高可靠性 |

### Q3: 能否动态调整模式？

**能**。在对同一设计空间的多个样本中使用不同模式，可对比：

- 模式1结果 vs 模式3结果 → 了解无偏探索的价值
- 模式2结果 vs 模式3结果 → 了解开放探索的收益

---

## 实现细节

### Core-2b采样函数签名

```python
def _select_interaction_aware_configs(
    self,
    n_configs: int,
    interaction_mode: str = "free",
    interaction_pairs_to_explore: List[Tuple[int, int]] = None,
    min_config_per_pair: int = 2,
    used_indices: set = None,
) -> List[int]:
    """交互对感知的配置选择"""
```

### 集成位置

- **配置传递**: `quick_start.py` → `run_step1()` → `sampler.generate_samples()`
- **采样调用**: `generate_samples()` → `_generate_five_step_samples()` → `_select_interaction_aware_configs()`
- **覆盖范围**: 仅影响Core-2b（Step 3中Core-2a不受影响）

---

## 总结建议

| 场景 | 推荐模式 | min_config_per_pair |
|------|---------|-------------------|
| 探索性研究，无假设 | **free** | - |
| 验证性研究，已有假设 | **specified_only** | 3-4 |
| 现实研究，平衡需求 | **hybrid** ⭐ | 2-3 |
| 预算极其紧张 | **specified_only** | 4-5 |
| 预算充足 | **hybrid** | 2 |

**最安全的选择**: 使用`hybrid`模式，让系统自动在假设驱动和无偏探索间找到平衡。
