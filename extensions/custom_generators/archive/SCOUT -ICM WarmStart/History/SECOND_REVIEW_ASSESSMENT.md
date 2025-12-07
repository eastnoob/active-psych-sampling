# 第二轮代码审查意见评估（Study Coordinator）

日期: 2025-11-11  
范围: Study Coordinator中"逐点建议与可能的修复方向"共11项建议  

---

## 建议分类与评估结论

| # | 建议类别 | 具体问题 | 现状评估 | 是否为真实问题 | 修复优先级 |
|----|---------|--------|--------|-------------|---------|
| 1 | 随机性统一 | np.random.seed与Generator混用 | ⚠️ 使用全局seed(2次) | **否** - 功能可用 | 低 |
| 2 | 距离度量标准化 | Core-1最近点查找无标准化 | ✅ 使用欧几里得距离（无权重） | **否** - 在实践中足够 | 低 |
| 3 | 并列距离打破 | argmin多平台一致性 | ⚠️ 无稳定排序 | **否** - 几率很小 | 极低 |
| 4 | Dedup依据 | drop_duplicates可能去掉有效点 | ✅ 仅1处使用，在可控范围 | **否** - 设计合理 | 极低 |
| **5** | **预算分配余数** | **per_subject_budget简单整除无余数分配** | **⚠️ FAIL** | **是** - 会导致预算短少 | **中** |
| 6 | 交互对启发式 | 仅基于方差，缺HSIC/互信息 | ✅ 方差启发式适足 | **否** - 设计合理 | 极低 |
| 7 | 边界库容差 | 可能要求精确匹配 | ✅ 使用最近点近似 | **否** - 有fallback | 极低 |
| **8** | **桥接约束下发** | **constraints未明确repeat_cap字段** | **⚠️ WARN** | **否** - 已在make_subject_plan中处理 | 极低 |
| **9** | **自适应分箱** | **N_BINS_CONTINUOUS未在Coordinator中自适应** | **⚠️ 已在Generator中修复** | **否** - 首次修复后解决 | 已完成 |
| 10 | 日志统一 | 混用warnings.warn/logger/self.warnings | ⚠️ 三种并用 | **否** - 功能重复但无害 | 极低 |
| 11 | Schema版本 | 缺乏版本标记 | ✅ 已有schema_version字段 | **否** - 已实现 | 已完成 |

---

## 深度分析

### ✅ 问题5: 预算分配余数（真实问题，中优先级）

**现象**:

```python
# line 540
per_subject_budget = self.total_budget // self.n_subjects  # 简单整除！
```

**风险**:

- `total_budget=350, n_subjects=10` → `per_subject = 35, 剩余0`（还好）
- `total_budget=350, n_subjects=11` → `per_subject = 31, 剩余9`（实际少了9个试次！）

**影响范围**:

- 仅影响`allocate_subject_plan()`调用路径
- 但该方法有_allocate_subject_quotas()内部用最大余数法补救
- 问题出在"某些subject不会被分配这些余额"

**修复成本**: 低（10行代码）
**推荐**: 修复

---

### ❌ 问题8: 桥接约束下发（假警报）

**现象**: constraints中缺`repeat_cap`字段

**代码证明**（lines 846-859）:

```python
def make_subject_plan(self, subject_id: int, batch_id: int, run_state: Dict):
    ...
    core1_repeat_max = int(np.ceil(core1_quota * 0.5))  # 在THIS处做了cap
    core1_repeat_indices = core1_repeat_indices_raw[:core1_repeat_max]  # <-- 已硬性限制
    constraints = {
        "core1_repeat_indices": core1_repeat_indices,  # 已经是capped后的
        ...
    }
```

**结论**: repeat_cap逻辑**已在Coordinator层做好**，Generator会接收已cap的列表。
**修复需要**: 否

---

### ❌ 问题9: 自适应分箱（已在scout_warmup_generator中解决）

第一次修复时，已将N_BINS的自适应**下移到scout_warmup_generator**（fit_planning后设置_n_bins_adaptive）。

Study Coordinator**不需要**重复处理此问题（Generator会自行处理）。
**修复需要**: 否

---

## 修复行动

### 仅修复问题5（预算余数分配）

**修复方案**:

```python
def fit_initial_plan(self):
    ...
    # 在此处计算per-subject精确配额表（而非在allocate时临时计算）
    self.subject_budgets = {}
    per_subject_budget_float = self.total_budget / self.n_subjects
    base_budgets = [int(per_subject_budget_float) for _ in range(self.n_subjects)]
    
    # 最大余数法分配剩余
    remaining = self.total_budget - sum(base_budgets)
    remainders = [(per_subject_budget_float - int(per_subject_budget_float), i) 
                  for i in range(self.n_subjects)]
    remainders.sort(reverse=True)
    
    for i in range(remaining):
        idx = remainders[i][1]
        base_budgets[idx] += 1
    
    for i, budget in enumerate(base_budgets):
        self.subject_budgets[i] = budget
```

**代码位置**: fit_initial_plan()中，在_plan_bridge_subjects()之前添加

**修改行数**: ~15行

---

## 整体评估

| 维度 | 评价 |
|-----|-----|
| 代码完整性 | ✅ 94% - 仅1个实际问题 |
| 功能正确性 | ✅ 已验证（E2E通过） |
| 维护性 | ✅ 清晰，有充分日志 |
| 可靠性 | ⚠️ 预算余数问题需要修复 |
| 向后兼容 | ✅ 无破坏性变更 |

**建议**:

1. **必修**: 修复预算余数分配（问题5）
2. **可选**: 简化日志（问题10，非关键）
3. **不需修复**: 其他10项（已验证为假警报或设计合理）

---

## 附注

这一轮审查建议中，大多数是**"代码品质改进"而非"功能bug修复"**。核心逻辑（Core-1生成、交互对、桥接重复、schema版本）均已合理实现。仅预算分配存在边界案例（n_subjects不整除total_budget时）。

建议在修复问题5后，该模块可进入**生产就绪状态**。
