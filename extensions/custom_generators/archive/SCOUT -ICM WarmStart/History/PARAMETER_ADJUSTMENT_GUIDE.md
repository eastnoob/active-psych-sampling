# 参数调整指南

## 快速调整（推荐）

编辑 `scripts/quick_start.py`，修改第 28-31 行：

```python
"--n_subjects",
"5",              # <-- 修改被试数量
"--trials_per_subject",
"25",             # <-- 修改每被试试验数
```

**示例**：改成 10 个被试，每被试 30 试验

```python
"--n_subjects",
"10",
"--trials_per_subject",
"30",
```

然后运行：

```powershell
cd scripts
pixi run python quick_start.py
```

---

## 命令行直接调整（高级）

直接运行主脚本并指定参数：

```powershell
cd scripts
pixi run python run_warmup_sampling.py `
  --n_subjects 10 `
  --trials_per_subject 30 `
  --design_csv ../test/sample_design.csv
```

---

## 所有可用参数

| 参数 | 类型 | 默认值 | 说明 | 约束 |
|------|------|--------|------|------|
| `--n_subjects` | int | 5 | 被试数量 | ≥ 1 |
| `--trials_per_subject` | int | 25 | 每被试试验数 | ≥ 8 |
| `--design_csv` | str | `../test/sample_design.csv` | 设计空间CSV路径 | 文件必须存在 |
| `--output_dir` | str | `../results` | 输出目录（自动生成时间戳子目录） | 目录必须存在 |
| `--seed` | int | 42 | 随机种子 | 用于重现性 |

---

## 常见配置

### 配置1：小规模测试（1被试 × 10试验）

```powershell
python run_warmup_sampling.py --n_subjects 1 --trials_per_subject 10
```

### 配置2：标准配置（5被试 × 25试验 = 125总试验）

```powershell
python run_warmup_sampling.py --n_subjects 5 --trials_per_subject 25
```

### 配置3：大规模实验（20被试 × 50试验 = 1000总试验）

```powershell
python run_warmup_sampling.py --n_subjects 20 --trials_per_subject 50
```

### 配置4：自定义设计空间

```powershell
python run_warmup_sampling.py `
  --design_csv "path/to/your/design.csv" `
  --n_subjects 10 `
  --trials_per_subject 30
```

---

## 参数约束

### trials_per_subject 的最小值为 8

因为 SCOUT Phase-1 需要至少 6 个 Core-1 试验 + 其他试验类型。

如果设置 < 8，会报错：

```
ValueError: trials_per_subject must be ≥8 (for Core-1), got X
```

### 总试验数 = n_subjects × trials_per_subject

- 影响计算量和输出数据大小
- 更多试验 = 更好的覆盖率，但耗时更久

---

## 输出文件结构

运行后，结果保存在：

```
results/
└── YYMMDD_HHMMSS/              # 时间戳目录（自动创建）
    ├── trial_schedule.csv       # 综合试验表
    ├── subject_0.csv            # 被试0
    ├── subject_1.csv            # 被试1
    ├── ...
    ├── subject_N.csv            # 被试N
    ├── execution_summary.json   # 执行摘要
    ├── subject_summaries.json   # 被试摘要
    └── warmup_sampling.log      # 详细日志
```

每个被试 CSV 包含：

- `trial_number`: 本地试验编号 (1 到 trials_per_subject)
- `subject_id`: 被试ID (0 到 n_subjects-1)
- `block_type`: 试验类型 (core1, core2, individual, boundary, lhs)
- `f1, f2, f3, ...`: 因子值
- 其他元数据列 (seed, design_row_id, interaction_pair_id 等)

---

## 修改 quick_start.py 的具体步骤

1. 用文本编辑器打开 `scripts/quick_start.py`

2. 找到这一段（约第 28-31 行）：

```python
    cmd = [
        sys.executable,
        str(runner_script),
        "--design_csv",
        "../test/sample_design.csv",
        "--n_subjects",
        "5",                    # <-- 这里
        "--trials_per_subject",
        "25",                   # <-- 这里
        "--output_dir",
        "../results",
    ]
```

3. 修改数字后保存

4. 运行：

```powershell
cd scripts
pixi run python quick_start.py
```

---

## 环境变量方式（可选）

也可以通过环境变量设置（需要修改脚本）：

```powershell
$env:SCOUT_N_SUBJECTS=10
$env:SCOUT_TRIALS_PER_SUBJECT=30
pixi run python quick_start.py
```

---

## 性能提示

| 配置 | 总试验数 | 预计耗时 |
|------|---------|---------|
| 5 × 25 | 125 | ~30秒 |
| 10 × 25 | 250 | ~1分钟 |
| 20 × 25 | 500 | ~2分钟 |
| 10 × 50 | 500 | ~2分钟 |

*实际耗时取决于设计空间大小和计算机性能*

---

## 设计空间文件格式

设计空间 CSV 应为：

- **格式**：无表头，每行一个设计点
- **列数**：对应因子数 (f1, f2, f3, ...)
- **值**：数值或分类值
- **示例**：见 `test/sample_design.csv` (5个因子，100个设计点)

```
0.123,0.456,0.789,0.234,50
0.234,0.567,0.891,0.345,60
...
```

---

## 故障排除

### 错误：`FileNotFoundError: Design CSV not found`

**原因**：设计空间 CSV 路径不对
**解决**：检查 `--design_csv` 参数指向的文件是否存在

### 错误：`ValueError: trials_per_subject must be ≥8`

**原因**：设置的每被试试验数过少
**解决**：改成 ≥8 的值

### 生成速度慢

**原因**：被试数量或试验数太多
**解决**：先用小值测试，如 `--n_subjects 2 --trials_per_subject 10`

---

## 总结

**3种参数调整方式**：

1. ✅ **推荐**：编辑 `quick_start.py` 第28-31行 → `python quick_start.py`
2. 命令行：`python run_warmup_sampling.py --n_subjects X --trials_per_subject Y`
3. 代码：修改脚本硬编码值

**关键约束**：

- `n_subjects ≥ 1`
- `trials_per_subject ≥ 8`
- 总试验数 = n_subjects × trials_per_subject
