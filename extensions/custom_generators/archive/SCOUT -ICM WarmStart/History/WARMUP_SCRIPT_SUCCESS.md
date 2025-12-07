# SCOUT Phase-1 Warmup Sampling Script - SUCCESS

## ✓ 所有要求已满足

### 用户原始需求

```
生成一个脚本，生成5个被试，每个人25次试验
生成的结果放在results中，一个由时间戳命名的文件夹
```

### 验证结果

**✓ 被试数量**: 5个 (Subject 0-4)
**✓ 每被试试验数**: 恰好25次
**✓ 总试验数**: 125次 (5 × 25)
**✓ 输出目录**: `results/251112_004342/` (YYMMDD_HHMMSS 格式)

---

## 文件清单

### 主脚本

- `scripts/run_warmup_sampling.py` - 主要执行脚本 (496行)
- `scripts/generate_subject_csvs.py` - 被试CSV生成器 (200行)
- `scripts/quick_start.py` - 快速启动脚本

### 输出结果 (example: 251112_004342)

```
results/251112_004342/
├── trial_schedule.csv              # 综合试验表 (125行 × 19列)
├── subject_0.csv                   # 被试0试验 (25行)
├── subject_1.csv                   # 被试1试验 (25行)
├── subject_2.csv                   # 被试2试验 (25行)
├── subject_3.csv                   # 被试3试验 (25行)
├── subject_4.csv                   # 被试4试验 (25行)
├── execution_summary.json          # 执行摘要
├── subject_summaries.json          # 被试摘要
└── warmup_sampling.log            # 详细日志
```

---

## 数据质量检查

| 检查项 | 结果 |
|--------|------|
| 总试验数 | 125 ✓ |
| 被试0 | 25试验 ✓ |
| 被试1 | 25试验 ✓ |
| 被试2 | 25试验 ✓ |
| 被试3 | 25试验 ✓ |
| 被试4 | 25试验 ✓ |
| f-值缺失 | 0个 ✓ |
| design_row_id 类型 | int64 ✓ |
| 试验类型分布 | core1:30, core2:95 ✓ |

---

## 使用方法

### 运行脚本

```powershell
cd scripts
pixi run python run_warmup_sampling.py `
  --design_csv ..\test\sample_design.csv `
  --n_subjects 5 `
  --trials_per_subject 25
```

### 或使用快速启动

```powershell
cd scripts
pixi run python quick_start.py
```

---

## 输出文件格式

### trial_schedule.csv

综合试验表，包含所有125行试验：

```csv
trial_number,subject_id,batch_id,is_bridge,block_type,is_core1,is_core2,is_individual,is_boundary,is_lhs,is_core1_repeat,interaction_pair_id,design_row_id,f1,f2,f3,f4,f5,seed
1,0,0,False,core1,True,False,False,False,False,False,,1,0.472214,0.241852,0.590833,0.806835,89.0,42
2,0,0,False,core1,True,False,False,False,False,False,,5,0.706857,0.645173,0.511342,0.071189,74.0,42
...
```

### subject_N.csv

每个被试的25次独立试验，包含 `trial_number` 列（本地编号1-25）

---

## 功能特性

### 整数编码管道

- 输入: 混合数据类型设计空间 (CSV)
- 编码: 所有因子转换为整数 [0..99]
- 处理: StudyCoordinator + WarmupAEPsychGenerator（整数空间运行）
- 解码: 转换回原始因子值（数值/类别）
- 输出: 清洁的 CSV 文件

### 试验类型分布

- **Core-1** (30个): 确定性设计点，学习初始响应
- **Core-2** (95个): 主效应和交互效应探索

### 每被试分配

每个被试分配的试验类型（由StudyCoordinator自动优化）：

- Core-1: 6个
- Boundary: 1个  
- LHS (Latin Hypercube Sampling): 7个
- 其他: 11个

---

## 数据完整性

✓ 所有 f1-f5 值完整（无NaN）
✓ design_row_id 为整数类型
✓ 所有分类因子正确解码
✓ trial_number 列提供本地编号
✓ 种子(seed)值一致

---

## 日志示例

```
INFO:warmup_sampling:======================================================================
INFO:warmup_sampling:SCOUT Phase-1 Warmup Sampling Started
INFO:warmup_sampling:Configuration: 5 subjects × 25 trials each
INFO:warmup_sampling:Total budget: 125 trials
...
INFO:warmup_sampling:[OK] Subject 0: 25 trials -> ..\results\251112_004342\subject_0.csv
INFO:warmup_sampling:[OK] Subject 1: 25 trials -> ..\results\251112_004342\subject_1.csv
...
INFO:warmup_sampling:[OK] SCOUT Phase-1 Warmup Sampling Completed Successfully
```

---

## 常见问题

**Q: 为什么每个被试的试验类型比例不同？**  
A: StudyCoordinator 优化全局试验计划，然后按被试分配。每个被试获得该计划的一个子集。

**Q: 能改成不同数量的被试吗？**  
A: 可以！修改 `--n_subjects` 和 `--trials_per_subject` 参数：

```bash
python run_warmup_sampling.py --n_subjects 10 --trials_per_subject 20
```

**Q: 设计空间可以自定义吗？**  
A: 可以！创建你自己的 CSV 文件并使用 `--design_csv` 参数指定。

---

## 运行日期

- **最后验证**: 2025-11-12 00:43:42
- **时间戳目录**: 251112_004342

---

## 修复历程

本脚本经历以下问题修复：

1. ✓ Unicode 编码错误 → ASCII 替代
2. ✓ 字符串算术错误 → 整数编码
3. ✓ 混合类型检测失败 → pd.to_numeric() 改进
4. ✓ design_row_id 浮点型 → 整数转换
5. ✓ 缺失 f-值 → 从设计空间回填
6. ✓ 虚拟被试创建 → 改进生成器调用
7. ✓ 不均匀试验分布 → 每被试生成器重构

所有测试通过 ✓

---

**状态**: ✓ 生产就绪  
**质量**: ✓ 所有数据检查通过  
**用户反馈**: 已满足原始需求 (5被试 × 25试验)
