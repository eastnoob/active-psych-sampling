# SCOUT Phase-1 Warmup Sampling Scripts

这些脚本实现了完整的SCOUT Phase-1预热采样工作流程。

## 快速开始

### 默认配置 (5被试 × 25次/被试)

```bash
cd scripts
python quick_start.py
```

或直接使用Python运行：

```bash
python run_warmup_sampling.py
```

### 自定义配置

```bash
python run_warmup_sampling.py \
  --design_csv <设计空间CSV路径> \
  --n_subjects 10 \
  --trials_per_subject 50 \
  --output_dir ./custom_results
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--design_csv` | `../../data/only_independences/...` | 设计空间CSV文件路径 |
| `--n_subjects` | 5 | 被试数量 |
| `--trials_per_subject` | 25 | 每个被试的测试次数 |
| `--output_dir` | `../results` | 输出目录（会创建时间戳子目录） |
| `--seed` | 42 | 随机种子（确保可重现性） |

## 工作流程

1. **加载设计空间**: 从CSV读取因子组合，整数编码所有因子
2. **全局规划**:
   - Core-1选择（所有被试共享的固定框架）
   - 交互对选择（基于方差启发式）
   - 边界库构建
3. **预算分配**: 根据维数自适应分配各被试的Core-1/Core-2/Individual配额
4. **试验生成**（每个被试）:
   - Core-1: 确定性（所有被试相同）
   - Core-2: 交互筛选（标记 block_type="core2" + interaction_pair_id 非空）
   - Individual: 边界点 + LHS随机采样
   - 所有试次的f值通过整数编码还原回原始值
5. **结果保存**:
   - `trial_schedule.csv`: 全被试合并试验表
   - `subject_0.csv, subject_1.csv, ...`: **新** 每个被试的独立试验表（带trial_number编号）
   - `subject_summaries.json`: 每个被试的统计摘要
   - `execution_summary.json`: 全局执行统计
   - `warmup_sampling.log`: 详细日志

## 输出目录结构

```
results/
├── 251112_003508/                  # 时间戳目录 (YYMMDD_HHMMSS)
│   ├── trial_schedule.csv          # 全被试合并试验表
│   ├── subject_0.csv               # 被试0独立试验表（新增）
│   ├── subject_1.csv               # 被试1独立试验表（新增）
│   ├── ...
│   ├── subject_summaries.json      # 每个被试的统计
│   ├── execution_summary.json      # 全局摘要
│   ├── generate_subject_csvs.log   # per-subject生成日志
│   └── warmup_sampling.log         # 详细日志
└── 251112_160530/
    └── ...
```

## 自动per-Subject CSV生成

运行 `run_warmup_sampling.py` 后，脚本会自动：

1. 生成 combined `trial_schedule.csv`（全被试）
2. 分离每个被试的试次生成独立 CSV 文件
3. 添加 `trial_number` 列用于本地编号（从1开始）
4. 按 batch_id 和 block_type 排序

如需单独运行：

```bash
python generate_subject_csvs.py --trial_schedule_csv <path>/trial_schedule.csv
```

## 输出示例

### trial_schedule.csv

```csv
subject_id,batch_id,block_type,is_core1_repeat,interaction_pair_id,design_row_id,f1,f2,f3,f4,f5,f6,seed
0,0,core1,False,None,42,0,1,0.0,low,A,True,42
0,0,core1,False,None,58,0,3,0.2,high,B,False,42
0,0,core2,False,0,105,0,2,0.1,mid,C,True,42
...
```

### execution_summary.json

```json
{
  "timestamp": "2025-01-12T15:30:22.123456",
  "configuration": {
    "n_subjects": 5,
    "trials_per_subject": 25,
    "total_budget": 125,
    "design_space_size": 1200,
    "n_factors": 6
  },
  "results": {
    "total_trials_generated": 125,
    "trials_by_type": {
      "core1": 40,
      "core2": 50,
      "individual": 25,
      "boundary": 10
    },
    "avg_coverage": 0.9823,
    "avg_gini": 0.0245,
    "avg_core1_repeat_rate": 0.0
  }
}
```

## 关键指标说明

- **Coverage Rate**: 设计空间的边际覆盖率（目标 >0.10）
- **Gini Coefficient**: 因子水平分布的不均匀性（目标 <0.40）
- **Core-1 Repeat Rate**: Core-1点重复率（Phase-1中应为0，Phase-2会≥50%）

## 特性

✅ **自动参数自适应**:

- N_BINS_CONTINUOUS: 根据维数自动调整 (d≤4→2, 5-8→3, 9-12→4, >12→5)
- 预算分配: 根据维数调整Core-1/Core-2/Individual比例

✅ **完整验证**:

- 覆盖率检查
- Gini系数监测
- 试验类型统计

✅ **可重现性**:

- 确定性随机数生成 (基于seed参数)
- 完整的日志记录

✅ **灵活配置**:

- 支持任意被试数量和试验次数
- 支持自定义设计空间
- 支持自定义输出目录

## 故障排除

### 问题: "design_row_id not found"

**原因**: 某些试验未成功匹配到设计空间行
**解决**: 通常自动处理，检查日志中的匹配统计

### 问题: "覆盖率 <0.10"

**原因**: 试验次数过少或设计空间过大
**解决**: 增加 `--trials_per_subject` 或减少因子数

### 问题: "Gini系数 >0.40"

**原因**: 某些因子水平未充分覆盖
**解决**: 使用自适应binning，增加试验次数

## 系统需求

- Python 3.8+
- pandas, numpy, scipy, scikit-learn
- （在AEPsych环境中运行，已安装所有依赖）

## 可重现性

使用相同seed参数会生成完全相同的试验序列：

```bash
# 两次运行会生成相同的试验表
python run_warmup_sampling.py --seed 42 --n_subjects 5 --trials_per_subject 25
python run_warmup_sampling.py --seed 42 --n_subjects 5 --trials_per_subject 25
```

## Phase-1/Phase-2的区别

### Phase-1 (此脚本实现)

- ✅ 预热设计: 无GP反馈
- ✅ Core-1重复: 0% (所有被试采样相同Core-1)
- ✅ 目标: 快速bootstrap，多样性优先
- ✅ 交互: 盲目筛选 (方差启发式)

### Phase-2 (将来)

- Core-1重复: ≥50% (与前批相同)
- GP在环: 基于观测数据动态优化
- 交互: 数据驱动的选择
- 目标: 高效主效应+交互估计

---

**Last Updated**: 2025-11-12  
**Version**: 1.0  
**Status**: 生产就绪 ⭐⭐⭐⭐⭐
