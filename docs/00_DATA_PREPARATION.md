# 00 数据准备规范

> **本文档定义了 Phase 1 分析整个工作流中的数据格式规则。** 严格遵守这些规则可避免后续分析中的数据清理问题。

---

## 核心规范

### 列名命名规则

| 列类型 | 命名格式 | 示例 | 说明 |
|--------|---------|------|------|
| **自变量** | `x[0-9]+` 或 `x[0-9]+_描述` | `x1`, `x2`, `x1_CeilingHeight`, `x10_Factor` | 必须以 `x` 开头，后跟 1+ 个数字 |
| **响应变量** | 任意（通常 `y` 或 `response`） | `y`, `response`, `rating` | 不能以 `x[0-9]+` 开头 |
| **被试编号** | 任意（通常 `subject` 或 `subject_id`） | `subject`, `subject_id`, `participant` | 用于识别不同被试 |
| **元数据** | ❌ **禁止** | ~~`Condition_ID`~~, ~~`ID`~~, ~~`metadata`~~ | 自动排除，不纳入分析 |

### 自动排除规则

以下列会被 **自动识别为元数据并排除**（不用手动删除）：

- 任何 **不符合** `x[0-9]+` 格式的列（除了指定的响应列和被试列）
- 常见元数据列示例：`Condition_ID`, `ID`, `metadata`, `timestamp`, 等

**示例：**

```
CSV 列: [Condition_ID, x1_Height, x2_Width, x3_Color, y, subject]
        ↓ 自动识别
因子列: [x1_Height, x2_Width, x3_Color]  ✓ 
排除列: [Condition_ID]  ✓（自动）
```

---

## 数据格式要求

### CSV 文件结构

```csv
x1_Factor1,x2_Factor2,x3_Factor3,...,y,subject
value1,value2,value3,...,response,1
value1,value2,value3,...,response,2
```

**要求：**

- ✅ 自变量列以 `x[0-9]+` 开头
- ✅ 每行一条数据记录
- ✅ 列用逗号分隔，无额外空格
- ❌ 不包含 `Condition_ID` 等元数据列（若有，会被自动排除）

### 数据类型

| 列类型 | 接受的数据类型 | 处理方式 |
|--------|----------------|---------|
| 自变量 | 数值、分类、布尔 | 分类变量自动 Label Encode；数值保持原样 |
| 响应变量 | 数值 | 保持原样（被试内 Z-score 标准化）|
| 被试列 | 任意 | 识别唯一被试 |

**编码示例：**

```python
# 分类变量自动编码
x3_Color: ['red', 'blue', 'green'] → [0, 1, 2]

# 数值变量保持不变
x1_Height: [1.5, 2.0, 1.8] → [1.5, 2.0, 1.8]

# 布尔变量转 0/1
x5_IsActive: [True, False, True] → [1, 0, 1]
```

---

## 工作流中的规则应用

### Phase 1 Step 1.5（模拟被试）

**输入格式：**

- 采样设计空间 CSV（包含所有因子列）

**输出格式：**

- 只包含 `x[0-9]+_...` 和 `y` 列
- **自动排除** `Condition_ID` 等元数据

```
输出: [x1_Height, x2_Width, ..., y]  ✓
```

### Phase 1 Step 2（交互对分析）

**输入识别：**

```python
# 从 CSV 中自动识别
factor_cols = [c for c in df.columns if re.match(r'^x[0-9]+', c)]
# → [x1_Height, x2_Width, x3_Color, ...]
```

**支持格式：**

- 目录模式：读取所有 `subject_*.csv`，自动合并
- 文件模式：直接读取单个合并 CSV

### Phase 1 Step 3（Base GP）

**自动因子识别：**

```python
# Base GP 只使用符合 x[0-9]+ 格式的列
valid_factors = [c for c in df.columns if re.match(r'^x[0-9]+', c)]

# 所有其他列自动排除
excluded_metadata = [c for c in df.columns if c not in valid_factors]
```

---

## 常见问题排查

### Q1: 数据中有 `Condition_ID`，会影响分析吗？

**A:** 不会。系统会自动排除它。

```
CSV: [Condition_ID, x1, x2, x3, y]
      ↓ 自动排除
因子: [x1, x2, x3]  ✓
```

### Q2: 我有 10+ 个因子，如何命名？

**A:** 用 `x1, x2, ..., x10, x11, ...` 的格式（支持任意位数）。

```
✓ x1, x2, x9, x10, x100, x1000
✓ x1_Name, x10_Name, x100_Name
```

### Q3: 因子名称中能有空格吗？

**A:** 不推荐。用下划线代替。

```
✗ x1 Ceiling Height
✓ x1_Ceiling_Height
```

### Q4: 能否混合使用 `x1` 和 `x1_Name` 两种格式？

**A:** 可以，系统都能识别。但建议统一为 `x[0-9]+_描述` 便于理解。

```
✓ [x1_Height, x2_Width, x3_Color, ...]  推荐
✓ [x1, x2, x3, ...]                     也可以
✗ [x1_Height, x2, x3_Color]            不推荐混用
```

### Q5: 被试列必须叫 `subject` 吗？

**A:** 不必须。在 `quick_start.py` 中指定即可。

```python
STEP2_CONFIG = {
    "subject_col": "participant_id",  # 自定义列名
    ...
}
```

---

## 快速检查清单

在开始 Phase 1 分析前，检查你的数据：

- [ ] **自变量列** 符合 `x[0-9]+` 或 `x[0-9]+_描述` 格式
- [ ] **响应列** 是数值型（如 `y`, `response`, `rating`）
- [ ] **被试列** 存在且能唯一识别每个被试
- [ ] **无元数据混入** （若有 `Condition_ID` 等，会自动排除，无需手动）
- [ ] **数据完整** （无 NaN、无空行）
- [ ] **编码一致** （同一分类变量值保持一致，如 'red' 不要写成 'Red'）

---

## 数据准备工作流

```
原始数据 CSV
    ↓
【规则检查】
  ├─ 自变量列名以 x[0-9]+ 开头？
  ├─ 响应列存在？
  ├─ 被试列存在？
  └─ 元数据将被自动排除 ✓
    ↓
【开始 Phase 1】
  ├─ Step 1.5: 模拟被试（自动排除元数据）
  ├─ Step 2: 分析交互对（自动识别因子）
  └─ Step 3: Base GP 训练（自动识别因子）
```

---

## 参考

- **正则表达式规则：** `^x[0-9]+` 匹配 x1, x2, ..., x10, x100 等
- **自动排除机制：** 在 `analyze_phase1.py` 和 `phase1_step3_base_gp.py` 中实现
- **被试内标准化：** 响应变量 Z-score 标准化，按被试进行

---

**更新日期：** 2025-12-07  
**适用版本：** Phase 1 完整分析流程（Step 1 - Step 3）
