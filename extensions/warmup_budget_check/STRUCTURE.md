# 项目结构说明

## 目录组织

```
warmup_budget_check/
├── quick_start.py              # 快速启动脚本（主入口）
├── README.md                   # 项目说明文档
├── STRUCTURE.md               # 本文件 - 项目结构说明
│
├── core/                       # 核心功能模块
│   ├── warmup_sampler.py       # Step 1: 生成采样方案
│   ├── warmup_budget_estimator.py # 预算评估工具
│   ├── simulation_runner.py    # Step 1.5: 模拟被试作答
│   ├── single_output_subject.py # 被试模拟类
│   ├── analyze_phase1.py       # Step 2: 分析Phase 1数据
│   ├── phase1_analyzer.py      # 数据分析工具
│   ├── phase1_step3_base_gp.py # Step 3: Base GP训练
│   ├── warmup_api.py           # API接口封装
│   └── config_models.py        # 配置数据模型
│
├── docs/                       # 文档目录
│   ├── README.md               # 完整使用指南
│   ├── README_API.md           # API文档
│   ├── API_INTEGRATION_SUMMARY.md # API集成说明
│   ├── ENHANCEMENT_SUMMARY.md  # 增强功能说明
│   ├── WORK_COMPLETED.md       # 工作完成记录
│   ├── 修复总结.md              # Gower距离修复总结
│   └── ...                     # 其他文档
│
├── tests/                      # 测试文件
│   ├── test_analysis.py        # 分析功能测试
│   └── test_api_integration.py # API集成测试
│
├── examples/                   # 示例代码
│
├── sample/                     # 采样输出目录
│   └── YYYYMMDDHHMM/          # 时间戳目录
│       ├── subject_*.csv       # 采样方案文件
│       └── result/             # 模拟应答结果（Step 1.5）
│           ├── combined_results.csv # 合并的响应数据
│           ├── subject_*_result.csv # 分被试结果
│           ├── subject_*_model.md   # 模型规格
│           └── SIMULATION_REPORT.md # 模拟报告
│
├── phase1_analysis_output/     # Phase 1分析输出
│   ├── *.json                  # Phase 2配置文件
│   ├── *.npz                   # λ/γ动态调度
│   └── *.md                    # 分析报告
│
├── test_output/                # 测试输出目录
│
└── archive/                    # 历史版本和旧文件
    ├── 251116/                 # 按日期归档
    ├── docs/                   # 旧文档
    └── tests/                  # 旧测试
```

## 核心模块说明

### 1. quick_start.py - 主入口
- **功能**: 统一的快速启动脚本
- **支持模式**:
  - `step1`: 生成采样方案
  - `step1.5`: 模拟被试作答
  - `step2`: 分析数据
  - `step3`: Base GP训练
  - `all`: 自动运行全流程

### 2. core/warmup_sampler.py - 采样生成器
- **五步采样法**:
  - Core-1: 固定语义点（8个）
  - Core-2a: D-optimal主效应
  - Core-2b: 交互对探索
  - Boundary: 边界极值
  - LHS: 均匀填充
- **Gower距离**: 正确处理混合类型变量

### 3. core/simulation_runner.py - 被试模拟器
- **功能**: 模拟真实被试作答
- **支持特性**:
  - 混合效应模型（固定效应+随机效应）
  - 交互效应
  - Likert量表映射
  - 多被试独立参数

### 4. core/analyze_phase1.py - 数据分析器
- **功能**: 从Phase 1数据识别关键交互对
- **方法**: Elbow/BIC/Top-K
- **输出**: Phase 2配置（λ/γ调度）

## 工作流程

```
┌─────────────────────────────────────────────────────────┐
│                    Step 1: 生成采样方案                    │
│  - 输入: 设计空间CSV                                       │
│  - 输出: sample/YYYYMMDDHHMM/subject_*.csv                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│            Step 1.5: 模拟被试作答（可选）                  │
│  - 输入: Step 1输出目录                                    │
│  - 输出: result/combined_results.csv                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│              【真实实验或使用模拟数据】                      │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  Step 2: 分析Phase 1数据                  │
│  - 输入: 带响应的实验数据CSV                               │
│  - 输出: phase1_analysis_output/*.json, *.npz            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                  Step 3: Base GP训练（可选）               │
│  - 输入: Phase 1数据 + 设计空间                            │
│  - 输出: GP模型 + 设计空间预测                             │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│                   Phase 2: EUR-ANOVA                     │
│  - 使用Step 2输出的配置                                    │
│  - 主动学习采样                                            │
└─────────────────────────────────────────────────────────┘
```

## 快速使用

1. **编辑配置**: 打开 `quick_start.py`，修改配置参数
2. **选择模式**: 设置 `MODE` 变量
3. **运行**: `python quick_start.py`

## 文件命名规范

- **采样输出**: `sample/YYYYMMDDHHMM/` - 时间戳目录
- **分析输出**: `phase1_analysis_output/phase1_*.json` - 带前缀
- **测试输出**: `test_output/` - 测试专用

## 依赖关系

```
quick_start.py
    ├─→ core.warmup_sampler
    │       └─→ core.warmup_budget_estimator
    ├─→ core.simulation_runner
    │       └─→ core.single_output_subject
    ├─→ core.analyze_phase1
    │       └─→ core.phase1_analyzer
    ├─→ core.phase1_step3_base_gp
    └─→ core.warmup_api (封装所有上述模块)
```

## 许可

MIT License
