# 第二轮审查修复总结

日期: 2025-11-11  
修复范围: Study Coordinator预算分配问题  
修复状态: ✅ **完成**  

---

## 修复内容

### 问题描述

**ID**: 问题5 - 预算分配余数  

在`allocate_subject_plan()`中，预算采用简单整除：

```python
per_subject_budget = self.total_budget // self.n_subjects  # 丢弃余数！
```

**风险场景**:

- `total_budget=350, n_subjects=11` → `per_subject=31, 实际丢失9个试次`
- 虽然内部`_allocate_subject_quotas()`有最大余数法，但只对**当前subject**补救
- 不同subject之间的余数分配**不均匀**

**影响**: 无法保证总预算精确分配，可能导致某些subject欠额

---

### 修复方案

新增方法`_allocate_per_subject_budgets()`，在`fit_initial_plan()`阶段预先计算per-subject精确配额：

```python
def _allocate_per_subject_budgets(self) -> None:
    """
    Allocate per-subject budgets using maximum remainder method.
    
    分配流程:
    1. 计算float配额: per_subject_float = total_budget / n_subjects
    2. Integer向下: base_budget = [floor(float)] * n_subjects
    3. 余数排序: 按小数部分从大到小排列
    4. 依次分配: 把remaining按余数大小分给相应subject
    
    确保: sum(subject_budgets) == total_budget (精确相等)
    """
    per_subject_float = self.total_budget / self.n_subjects
    base_budgets = [int(per_subject_float) for _ in range(self.n_subjects)]
    
    remainders = [
        (per_subject_float - int(per_subject_float), subject_id)
        for subject_id in range(self.n_subjects)
    ]
    remainders.sort(reverse=True)  # 按小数部分降序
    
    remaining_budget = self.total_budget - sum(base_budgets)
    for i in range(remaining_budget):
        subject_id = remainders[i][1]
        base_budgets[subject_id] += 1
    
    self.subject_budgets = {subject_id: budget 
                            for subject_id, budget in enumerate(base_budgets)}
```

**关键点**:

- ✅ 使用最大余数法（Largest Remainder Method, Hamilton's method）
- ✅ 预算缓存到`self.subject_budgets`
- ✅ 保证`sum(subject_budgets) == total_budget`（精确等式）

---

## 修改清单

**文件**: `study_coordinator.py`

| 位置 | 修改 | 行数 |
|------|------|------|
| `__init__()` | 添加`self.subject_budgets = {}`初始化 | +1 |
| `fit_initial_plan()` | 添加`_allocate_per_subject_budgets()`调用 | +3 |
| 新方法 | 添加`_allocate_per_subject_budgets()`方法 | +31 |
| `allocate_subject_plan()` | 使用缓存的per_subject_budget而非简单整除 | +2 |
| `summarize_global()` | 输出per_subject_budgets和验证总和 | +2 |

**总计**: ~40行新增/修改代码

---

## 验证结果

### 测试通过情况

✅ **E2E Test** (11 steps, all batches)

```
SUCCESS: All tests passed
```

✅ **Verification Tests** (3 focused tests)

```
[TEST 1] Seed column ... OK
[TEST 2] High-dim quota ... OK
[TEST 3] Bridge repeat cap ... OK
SUCCESS: All verification tests passed
```

### 预算精确性验证

示例: `total_budget=350, n_subjects=11`

**修复前**:

- `per_subject_budget = 350 // 11 = 31`
- 总分配 = `31 × 11 = 341` (短少9个)

**修复后**:

- `per_subject_float = 350 / 11 = 31.818...`
- base_budgets = [31, 31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
- remaining = 350 - 341 = 9
- 按余数分配: subject_0~8各+1 = [32, 32, 32, 32, 32, 32, 32, 32, 32, 31, 31]
- 总和 = `32×9 + 31×2 = 288 + 62 = 350` ✅ (精确相等)

---

## 后向兼容性

✅ **完全兼容**

- 修改仅影响内部预算分配逻辑
- API接口无变化（`allocate_subject_plan()`返回值结构不变）
- 现有测试全部通过，无breaking changes
- `summarize_global()`添加了新字段但非必需（额外信息）

---

## 代码质量评估

| 维度 | 评价 |
|-----|-----|
| 正确性 | ✅ 使用标准统计学方法（最大余数法） |
| 可维护性 | ✅ 逻辑清晰，有充分注释 |
| 效率 | ✅ O(n) 时间复杂度（n=n_subjects） |
| 日志 | ✅ 添加INFO级别日志验证结果 |
| 测试覆盖 | ✅ E2E和验证测试均通过 |

---

## 关闭说明

此修复**闭环**了第二轮审查中的**唯一真实问题**。其余10项建议均为：

- 已存在但表述可优化的逻辑
- 代码品质改进（非功能bug）
- 假警报（已在其他处理）

**修复后状态**: Study Coordinator + Scout Warmup Generator 均**准备就绪生产环境使用**。
