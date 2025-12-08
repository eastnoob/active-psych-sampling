# 双模式数据读取指南

## 概述

`quick_start.py` 中的 **Step 2** 和 **Step 3** 现在支持两种数据读取模式：

1. **目录模式**（推荐）- 自动读取所有 `subject_*.csv` 文件
2. **文件模式**（兼容旧流程）- 读取单个合并 CSV 文件

## 使用方法

### Step 2: Phase 1 数据分析

#### 方式1: 目录模式（推荐）✅

```python
STEP2_CONFIG = {
    # 指向 Step 1.5 生成的 result 目录
    "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result",
    "subject_col": "subject",  # 会从文件名自动生成
    "response_col": "y",
    # ...
}
```

**优点**:
- 直接使用 Step 1.5 的输出，无需手动合并
- 自动理解每个 `subject_*.csv` 文件代表一个被试
- 自动从文件名生成 subject 列 (`subject_1`, `subject_2`, ...)

**工作流程**:
```
Step 1.5 输出:
  result/
    ├── subject_1.csv  (30行，包含 y 列)
    ├── subject_2.csv  (30行，包含 y 列)
    ├── subject_3.csv  (30行，包含 y 列)
    ├── subject_4.csv  (30行，包含 y 列)
    └── subject_5.csv  (30行，包含 y 列)

Step 2 读取:
  → 自动合并为 150 行
  → 添加 subject 列 (值: subject_1, subject_2, ...)
  → 进行分析
```

#### 方式2: 文件模式（兼容旧流程）

```python
STEP2_CONFIG = {
    # 指向已合并的 CSV 文件
    # "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result\\combined_results.csv",
    "subject_col": "subject",  # 必须已存在于文件中
    "response_col": "y",
    # ...
}
```

**适用场景**:
- 已经手动合并了所有被试数据
- 兼容旧版流程

---

### Step 3: Base GP 训练与设计空间扫描

#### 方式1: 目录模式（推荐）✅

```python
STEP3_CONFIG = {
    # 指向 Step 1.5 生成的 result 目录
    "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result",
    "subject_col": "subject",
    "response_col": "y",
    "design_space_csv": "data\\...",
    # ...
}
```

**优点**:
- 与 Step 2 一致的使用方式
- 自动读取所有被试文件

#### 方式2: 文件模式

```python
STEP3_CONFIG = {
    # 指向已合并的 CSV 文件
    # "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result\\combined_results.csv",
    "subject_col": "subject",
    "response_col": "y",
    "design_space_csv": "data\\...",
    # ...
}
```

---

## 配置切换

在 `quick_start.py` 中切换模式非常简单，只需注释/取消注释：

### 当前配置（目录模式）
```python
STEP2_CONFIG = {
    # 【方式1】目录模式 - 自动读取所有 subject_*.csv（推荐）
    "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result",

    # 【方式2】文件模式 - 读取单个合并CSV
    # "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result\\combined_results.csv",
}
```

### 切换到文件模式
```python
STEP2_CONFIG = {
    # 【方式1】目录模式 - 自动读取所有 subject_*.csv（推荐）
    # "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result",  # ← 注释掉

    # 【方式2】文件模式 - 读取单个合并CSV
    "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result\\combined_results.csv",  # ← 启用
}
```

---

## 实现细节

### 目录模式的自动处理

当 `data_csv_path` 指向目录时，系统会：

1. **自动查找**: 查找所有 `subject_*.csv` 文件
2. **读取每个文件**: 逐个读取被试数据
3. **添加被试列**: 如果文件中没有 subject 列，从文件名提取（`subject_1.csv` → `subject_1`）
4. **合并数据**: 使用 `pd.concat()` 合并所有数据
5. **验证**: 确保所有文件都包含响应列

### 示例输出

#### Step 2 目录模式输出:
```
[加载] 从目录读取被试数据: extensions\warmup_budget_check\sample\202511302204\result
  找到 5 个被试文件
    - subject_1.csv: 30 行
    - subject_2.csv: 30 行
    - subject_3.csv: 30 行
    - subject_4.csv: 30 行
    - subject_5.csv: 30 行
  合并后总计: 150 行
  样本数: 150
  被试数: 5
  因子数: 6
```

#### Step 3 目录模式输出:
```
[Step3] 从目录读取被试数据: extensions\warmup_budget_check\sample\202511302204\result
  找到 5 个被试文件
    - subject_1.csv: 30 行
    - subject_2.csv: 30 行
    - subject_3.csv: 30 行
    - subject_4.csv: 30 行
    - subject_5.csv: 30 行
  合并后总计: 150 行
```

---

## 两种模式对比

| 特性 | 目录模式 | 文件模式 |
|------|---------|---------|
| **输入路径** | 目录路径 (result/) | 文件路径 (combined_results.csv) |
| **自动合并** | ✅ 是 | ❌ 否（需预先合并） |
| **自动添加 subject 列** | ✅ 是（从文件名） | ❌ 否（需已存在） |
| **适用场景** | 新实验，直接使用 Step 1.5 输出 | 旧流程，已有合并 CSV |
| **推荐使用** | ✅ 推荐 | 兼容性 |

---

## 完整工作流程示例

### 推荐流程（目录模式）

1. **Step 1**: 生成采样方案
   ```python
   MODE = "step1"
   # 生成 subject_1.csv, subject_2.csv, ...
   ```

2. **Step 1.5**: 模拟被试作答
   ```python
   MODE = "step1.5"
   STEP1_5_CONFIG = {
       "input_dir": "extensions\\warmup_budget_check\\sample\\202511302204",
       # ...
   }
   # 输出: result/subject_1.csv, subject_2.csv, ... (带 y 列)
   ```

3. **Step 2**: 分析数据（目录模式）
   ```python
   MODE = "step2"
   STEP2_CONFIG = {
       "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result",  # ← 目录
       # ...
   }
   # 自动读取所有 subject_*.csv → 分析
   ```

4. **Step 3**: Base GP 训练（目录模式）
   ```python
   MODE = "step3"
   STEP3_CONFIG = {
       "data_csv_path": "extensions\\warmup_budget_check\\sample\\202511302204\\result",  # ← 目录
       # ...
   }
   # 自动读取所有 subject_*.csv → 训练 GP
   ```

### 旧流程（文件模式）

1. **Step 1**: 生成采样方案
2. **Step 1.5**: 模拟被试作答
3. **手动合并**: 合并所有 subject_*.csv → combined_results.csv
4. **Step 2**: 使用 combined_results.csv
5. **Step 3**: 使用 combined_results.csv

---

## 技术实现

### Phase1DataAnalyzer (Step 2)

位置: `extensions/warmup_budget_check/core/analyze_phase1.py`

```python
def __init__(self, data_csv_path: str, subject_col: str, response_col: str):
    data_path = Path(data_csv_path)

    if data_path.is_dir():
        # 目录模式
        subject_csvs = sorted(data_path.glob("subject_*.csv"))
        all_dfs = []
        for csv_path in subject_csvs:
            df_subject = pd.read_csv(csv_path)
            if subject_col not in df_subject.columns:
                df_subject.insert(0, subject_col, csv_path.stem)
            all_dfs.append(df_subject)
        self.df = pd.concat(all_dfs, ignore_index=True)
    else:
        # 文件模式
        self.df = pd.read_csv(data_path)
```

### process_step3 (Step 3)

位置: `extensions/warmup_budget_check/core/phase1_step3_base_gp.py`

```python
def process_step3(data_csv_path: str, ...):
    data_path = Path(data_csv_path)

    if data_path.is_dir():
        # 目录模式（逻辑与 Step 2 相同）
        subject_csvs = sorted(data_path.glob("subject_*.csv"))
        # ... 合并逻辑
        df_phase1 = pd.concat(all_dfs, ignore_index=True)
    else:
        # 文件模式
        df_phase1 = pd.read_csv(data_path)
```

---

## 常见问题

### Q1: 如何判断使用哪种模式？

**A**: 系统自动检测：
- 如果 `data_csv_path` 指向目录 → 目录模式
- 如果 `data_csv_path` 指向文件 → 文件模式

### Q2: 目录模式下，subject 列的值是什么？

**A**: 从文件名提取，例如：
- `subject_1.csv` → `subject_1`
- `subject_2.csv` → `subject_2`

### Q3: 如果目录中没有 subject_*.csv 文件会怎样？

**A**: 系统会报错:
```
FileNotFoundError: 目录中未找到 subject_*.csv 文件: ...
```

### Q4: 可以混合使用两种模式吗？

**A**: 可以，Step 2 和 Step 3 可以独立选择模式。例如：
- Step 2 使用目录模式
- Step 3 使用文件模式（如果你手动生成了 combined CSV）

### Q5: 目录模式会修改原始文件吗？

**A**: 不会。系统只读取文件，所有修改都在内存中进行。

### Q6: Step 3 报错 "could not convert string to float: 'Strict'" 怎么办？

**A**: 这是分类变量编码问题，已在最新版本修复（2025-11-30）。修复内容：
- Step 3 目录模式下会自动从采样方案推断分类变量的编码映射
- 将分类值（如 'Strict', 'Rotated'）自动转换为数值（0, 1, 2）
- 详见 [STEP3_ENCODING_FIX.md](STEP3_ENCODING_FIX.md)

**解决方案**:
1. 确保使用最新版本的 `phase1_step3_base_gp.py`
2. 目录模式下会看到 `[推断编码]` 日志输出
3. 如果仍有问题，检查采样方案目录是否包含 `subject_1.csv`

---

## 更新日志

**2025-11-30 (下午)**:
- ✅ **Step 3 编码修复**: 自动推断分类变量编码 (`_infer_encoding_from_sampling`)
- ✅ **改进错误提示**: 分类变量未编码时提供清晰的错误信息
- ✅ **调试输出**: 显示编码前后的数据类型和样本值
- 📄 新增文档: [STEP3_ENCODING_FIX.md](STEP3_ENCODING_FIX.md)

**2025-11-30 (上午)**:
- ✅ Step 2 (`Phase1DataAnalyzer`) 支持目录模式
- ✅ Step 3 (`process_step3`) 支持目录模式
- ✅ `quick_start.py` STEP2_CONFIG 和 STEP3_CONFIG 提供双模式注释
- ✅ 默认启用目录模式

---

生成日期: 2025-11-30
适用版本: AEPsych warmup_budget_check v1.2 (最新版)
相关文档: [STEP3_ENCODING_FIX.md](STEP3_ENCODING_FIX.md), [REPRODUCTION_GUIDE.md](sample/202511302204/result/REPRODUCTION_GUIDE.md)
