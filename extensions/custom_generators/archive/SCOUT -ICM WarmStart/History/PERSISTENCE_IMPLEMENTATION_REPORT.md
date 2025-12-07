# 持久化层实现完成报告

## 实现概要

已成功实现 StudyCoordinator 和 WarmupAEPsychGenerator 的持久化/状态管理方法，支持多主体、多批次的跨批实验设计继承和可审计性。

## 实现的方法

### StudyCoordinator 类 (5个方法)

#### 1. `save_global_plan(path: str) -> None`

- **功能**: 序列化不可变全局计划（库、预算、策略）
- **输出**: `global_plan.json` 包含：
  - 元数据（时间戳、科目数、批次数、种子）
  - 因子信息（名称、类型、离散化）
  - 全局组件（Core-1候选、交互对、边界库大小）
  - 预算分割和桥接计划
  - 警告信息

#### 2. `save_run_state(path: str, batch_k: int = 0, results: Dict = None) -> None`

- **功能**: 保存可变运行状态（检查点、科目状态、覆盖率轨迹）
- **输出**: `run_state.pkl` 包含：
  - 检查点信息（当前批、时间戳）
  - 科目状态（完成试验、Core-1重复、剩余预算）
  - 覆盖率/Gini轨迹
  - 交互覆盖和桥接完成度
  - RNG种子状态

#### 3. `load_run_state(path: str) -> Dict`

- **功能**: 从文件加载已保存的运行状态
- **返回**: 状态字典，包含所有检查点和科目追踪信息

#### 4. `export_subject_plan(subject_id: int, path: str) -> None`

- **功能**: 导出单个科目的完整计划供生成器使用
- **输出**: `subject_{id}.plan.json` 包含：
  - 科目ID和批次ID
  - 预算配额（Core-1、Core-2、个体、边界、LHS）
  - 约束（边界库、交互对）
  - 每科目的RNG种子
  - 模式版本

#### 5. `validate_global_constraints(trials_all: pd.DataFrame) -> Dict`

- **功能**: 在批次执行后验证全局约束
- **返回**: 验证结果字典：
  - Core-1重复比率（目标 ≥50%）
  - 桥接科目覆盖范围
  - 全局覆盖率和Gini系数
  - 警告列表

### WarmupAEPsychGenerator 类 (3个方法)

#### 1. `apply_plan(plan: Dict) -> Self`

- **功能**: 将协调器的科目计划注入生成器
- **效果**: 覆盖配额、约束、RNG种子
- **返回**: self 用于方法链接

#### 2. `generate_trials(save_to: Optional[str] = None) -> pd.DataFrame`

- **增强**: 添加 `save_to` 参数支持直接文件保存
- **格式**: 支持 .csv 和 .parquet（自动回退到CSV如果缺少依赖）

#### 3. `export_metadata(path: str) -> None`

- **功能**: 导出审计追踪元数据
- **输出**: `metadata.json` 包含：
  - 科目/批次ID
  - 时间戳和种子
  - 因子信息
  - 预算明细
  - 覆盖率和Gini系数
  - 任何警告信息

## 测试结果

### 8个综合集成测试 - 全部通过 ✓

| 测试 | 描述 | 状态 |
|------|------|------|
| 1 | save_global_plan() 序列化 | [OK] |
| 2 | save/load_run_state() 往返 | [OK] |
| 3 | export_subject_plan() | [OK] |
| 4 | apply_plan() 注入 | [OK] |
| 5 | generate_trials(save_to=path) | [OK] |
| 6 | export_metadata() 审计追踪 | [OK] |
| 7 | validate_global_constraints() | [OK] |
| 8 | 端到端工作流 | [OK] |

**结果**: `[SUCCESS] 所有8个持久化测试通过！`

## 三段式工作流

```
t0: 全局规划
  ├─ fit_initial_plan() → 生成全局库
  ├─ save_global_plan() → 序列化不可变计划
  └─ save_run_state(0) → 初始检查点

t_k: 批次执行（每个科目）
  ├─ load_run_state() → 恢复之前的检查点
  ├─ export_subject_plan() → 导出科目配额
  ├─ Generator.apply_plan() → 应用配额
  ├─ fit_planning() → 本地规划
  ├─ generate_trials(save_to=path) → 生成并保存
  └─ export_metadata() → 导出审计信息

t_N: 聚合验证
  ├─ 加载所有批次的试验
  ├─ validate_global_constraints() → 全局检查
  └─ save_run_state() → 最终检查点
```

## 文件结构

```
experiment_dir/
├── global_plan.json           # 全局不可变计划
├── run_state.pkl              # 可变检查点（跨批次）
├── subject_0.plan.json        # 科目0配额和约束
├── subject_0_batch_0.csv      # 科目0第0批试验
├── subject_0_metadata.json    # 科目0审计信息
├── subject_1.plan.json
├── subject_1_batch_0.csv
├── subject_1_metadata.json
└── ... （更多科目）
```

## 关键特性

✓ **多科目跨批继承**: 通过run_state.pkl维护全局状态
✓ **可审计**: 每个阶段导出元数据和配置
✓ **可复现**: 保存RNG种子和所有参数
✓ **自适应**: 支持基于结果调整未来批次
✓ **模块化**: Coordinator和Generator完全解耦
✓ **鲁棒**: 智能回退处理（parquet → CSV）

## 代码位置

**文件**:

- `study_coordinator.py` (第 676-771 行): 5个新方法
- `scout_warmup_generator.py` (第 134-243, 384-435 行): 3个新方法

**测试**:

- `test_persistence_integration.py`: 8个综合集成测试

## 验证检查清单

- [x] 所有方法实现完成
- [x] JSON/Pickle 序列化正常工作
- [x] 往返加载保留数据完整性
- [x] 错误处理（缺失属性、依赖回退）
- [x] 8/8 集成测试通过
- [x] 端到端工作流验证
- [x] 文件系统操作正确
- [x] 数据结构合理性

## 下一步（可选）

1. **自适应规划** - 实现 `adapt_next_batch_plan(results)` 根据覆盖率/Gini调整
2. **性能监控** - 添加详细的指标追踪（执行时间、内存等）
3. **分布式执行** - 支持多工作进程并行执行科目
4. **更多持久化格式** - SQLite / PostgreSQL 支持用于大规模研究

## 总结

持久化层已完全实现，支持复杂的多科目、多批次AEPsych研究的完整工作流。所有方法都经过严格测试，可以直接用于生产环境。
