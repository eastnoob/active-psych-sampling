# Handoff Documentation Index

**生成时间**：2025-12-14
**项目**：active-psych-sampling 预热采样模块修复
**目标读者**：后续接手的大模型/开发者

---

## 📂 文档结构

### 问题分析文档
1. **20251214_CORE2A_2B_SAMPLING_ALGORITHM_MISMATCH.md**
   - **用途**：问题发现和根因分析
   - **内容**：证据、影响范围、修复建议
   - **适合**：了解问题背景
   - **阅读时间**：10-15分钟

### 深度技术分析
2. **WARMUP_ARCHITECTURE_ANALYSIS.md** (项目根目录)
   - **用途**：完整的架构设计vs实现对比分析
   - **内容**：五步采样法每个模块的符合度评估
   - **适合**：理解整体设计缺陷和影响
   - **阅读时间**：20-30分钟

### 实施方案（本文档）
3. **20251214_CORE2A_DOPTIMAL_IMPLEMENTATION_PLAN.md**
   - **用途**：可执行的、极致精简的实现计划
   - **内容**：代码框架、步骤、检查清单
   - **适合**：直接开始编码
   - **阅读时间**：5-10分钟

---

## 🎯 快速导航

### 如果你想...

**了解发生了什么**
→ 阅读 `20251214_CORE2A_2B_SAMPLING_ALGORITHM_MISMATCH.md`

**理解根本设计缺陷**
→ 阅读 `WARMUP_ARCHITECTURE_ANALYSIS.md`

**开始实施修复**
→ 阅读 `20251214_CORE2A_DOPTIMAL_IMPLEMENTATION_PLAN.md`

**查看环境配置**
→ 查看 `pixi.toml` (pydoe3已安装)

---

## 📋 关键信息速记

### 问题本质
- Core-2a 和 Core-2b 采样被合并
- Core-2a 承诺用D-optimal，实际是随机
- 违反科学严谨性

### 解决方案
- 分离Step 3的采样逻辑
- 使用pyDOE3实现D-optimal
- 保留Core-2b现有功能

### 技术栈
- pyDOE3 1.6.1 (已装)
- numpy/scipy (已装)
- 不需要新增依赖

### 预计工作量
- **编码**：1-2小时
- **测试**：1.5-2小时
- **文档**：1小时
- **总计**：4-5小时

### 关键文件修改
| 文件 | 改动 |
|------|------|
| `core/warmup_sampler.py` | 分离采样逻辑 + 新增D-optimal方法 |
| `tests/test_core2a_doptimal.py` | 新建单元测试 |
| `STRUCTURE.md` | 更新说明 |

---

## 🔗 相关代码位置

### warmup_sampler.py
- 行 796-811：需要修改的Step 3 (Core-2a/2b合并处)
- 行 360-423：Core-1采样 (参考，无需改)
- 行 460-497：Boundary采样 (参考，无需改)
- 行 615-734：Core-2b采样 (参考，保留)
- 行 499-586：LHS采样 (参考，无需改)

### warmup_budget_estimator.py
- 行 98-105：五步采样法设计定义
- 行 143-199：预算分离逻辑
- 行 563-571：文档更新处

---

## 🧪 验证步骤速查

```bash
# 修改完成后执行

# 1. 单元测试
pixi run pytest tests/test_core2a_doptimal.py -v

# 2. 集成测试
pixi run pytest tests/ -v

# 3. 功能测试
pixi run python test_three_modes.py

# 4. 完整流程测试
pixi run python quick_start.py  # MODE='step1'
```

---

## 💡 LLM接手建议

### 开始前
1. ✓ 阅读本索引文件 (2分钟)
2. ✓ 快速读一遍 `20251214_CORE2A_DOPTIMAL_IMPLEMENTATION_PLAN.md` (5分钟)
3. ✓ 查看 `warmup_sampler.py` 第796-811行 (3分钟)

### 执行顺序
1. 修改 `_generate_five_step_samples()` 分离采样
2. 新增 `_select_doptimal_configs()` 方法
3. 新增 `_normalize_design_to_candidates()` 方法
4. 新建单元测试
5. 运行集成测试
6. 更新文档

### 可能的问题和解决
- **Q：pyDOE3导入失败**
  A：使用大写 `from pyDOE3.doe_optimal import optimal_design`

- **Q：标准化后维度错误**
  A：检查 one-hot编码是否过度，考虑限制类别数

- **Q：性能太慢**
  A：pyDOE3.optimal_design()是瓶颈，可增加采样上限参数

- **Q：D-efficiency很低**
  A：可能候选集太小，考虑先做全因子设计的candidate set

---

## 📞 信息来源

### 问题报告
- 原始问题发现：Claude Haiku 4.5 (2025-12-14)
- 根因分析：claude-code-guide 分析

### 环境验证
- pyDOE3：1.6.1 (conda-forge)
- 文档参考：https://pydoe3.readthedocs.io/en/stable/doe_optimal.html

---

## ✅ 完成标志

实施完成后，应该看到：

```
Step 3 输出示例：
  [Core-2a] D-optimal主效应采样
    配置数: 16
    D-efficiency: 35.2%

  [Core-2b] 交互对感知采样 (hybrid mode)
    配置数: 21
    指定对: [(3,4), (0,1)]
```

---

## 📝 文档维护

| 文档 | 最后更新 | 维护者 | 状态 |
|------|---------|--------|------|
| 本索引 | 2025-12-14 | Claude | ✅ 当前 |
| CORE2A实现计划 | 2025-12-14 | Claude | ✅ 待执行 |
| 架构分析 | 2025-12-14 | Claude | ✅ 完成 |
| 问题报告 | 2025-12-14 | Claude | ✅ 已验证 |

---

**下一步**：选择任何一个大模型，给它这三个文档，告诉它"按照20251214_CORE2A_DOPTIMAL_IMPLEMENTATION_PLAN.md的步骤1-5执行"
