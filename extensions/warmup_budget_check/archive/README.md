# Archive - 历史版本和测试文件

本文件夹包含旧版本、测试文件和已被替代的脚本。

## 已归档的文件

### 旧版脚本

- **quick_start_for_budget_check.py** - 旧的快速启动脚本（已被 quick_start.py 替代）
- **two_phase_planner.py** - 旧的集成版本（已被独立的 warmup_sampler.py 和 analyze_phase1.py 替代）
- **run_full_pipeline.py** - 旧的一键式流程脚本
- **example_two_phase_workflow.py** - 示例工作流（已整合到文档中）
- **warmup_budget_estimator copy.py** - 备份文件

### 旧版文档

- **README_TWO_PHASE.md** - 旧的两阶段规划文档
- **QUICKSTART.md** - 旧的快速开始指南
- **USAGE_GUIDE.md** - 旧的使用指南
- **COMPLETION_SUMMARY.md** - 工作完成总结
- **测试成功.txt** - 测试结果

### 配置和测试

- **config_template.py** - 旧的配置模板
- **test_standalone_workflow.py** - 独立工作流测试脚本

### 其他

- **plan/** - 规划文件夹
- **__pycache__/** - Python缓存文件

## 当前使用的核心文件

请使用父目录中的以下文件：

- **quick_start.py** - 快速启动脚本（推荐）
- **warmup_sampler.py** - 步骤1：生成预热采样方案
- **analyze_phase1.py** - 步骤2：分析Phase 1数据
- **warmup_budget_estimator.py** - 预算评估工具（依赖）
- **phase1_analyzer.py** - 数据分析工具（依赖）
- **README.md** - 主文档
- **README_STANDALONE.md** - 详细使用指南

## 说明

本archive文件夹中的文件仅供参考，不建议在实际项目中使用。如需使用这些功能，请参考父目录中的最新版本文件。
