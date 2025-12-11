# Handoff文档说明

本目录包含两份关于**有序参数(Ordinal)类型扩展**的实现计划文档。

## 📄 文件清单

### 1. `20251211_ordinal_monotonic_parameter_extension.md` (详细版)

**用途**: 供后续大模型(LLM)完整理解与实施  
**特点**:

- 🔍 **完整深度分析** (800+ 行)
  - AEPsych源码设计精读
  - dynamic_eur_acquisition集成分析
  - 详细的实现代码框架与伪代码
  - 完整的配置示例与API文档
  - 测试策略与关键决策说明
  
- 📋 **自包含** - 无需外部参考，包含所有必要背景信息
- 🎯 **精准指导** - 具体文件路径、行号、修改点明确
- ✅ **可直接实施** - 伪代码可直接转化为实际代码

**何时使用**: 指派给LLM进行完整实现时

---

### 2. `ORDINAL_QUICK_REF.md` (快速参考版)

**用途**: 快速理解任务本质与检查清单  
**特点**:

- ⚡ **精简** (200 行)
  - 核心概念一览
  - 修改范围表格(文件、行数、内容)
  - 关键API与配置格式
  - 测试清单
  - 常见问题
  
- 📊 **表格化** - 便于快速查阅关键信息
- 🚀 **实施步骤** - 按天分解的实现路线图
- 💡 **决策参考** - 核心设计决策的简要说明

**何时使用**: 快速理解任务范围、评估工作量时

---

## 🎯 两份文档的对应关系

```
快速参考 (ORDINAL_QUICK_REF.md)
    ↓
    详细版参考需要了解详细内容时
    ↓
详细版 (20251211_ordinal_monotonic_parameter_extension.md)
    ↓
    指派给LLM实施时
```

---

## 📊 任务概览表

| 维度 | 内容 |
|------|------|
| **目标** | 添加Ordinal参数类型以保留参数顺序关系 |
| **类型** | 两种: 等差有序 + 非等差单调 |
| **影响范围** | AEPsych (核心库) + dynamic_eur_acquisition (扩展库) |
| **总代码量** | ~340 LOC (aepsych 212 + extension 130) |
| **预计工作量** | 2-3 天 (Day 1-2核心, Day 3测试) |
| **复杂度** | 中等 - 需理解Transform + LocalSampler + Config体系 |
| **向后兼容** | ✅ 完全兼容, 无breaking changes |
| **关键依赖** | 理解aepsych参数变换机制, dynamic_eur的扰动逻辑 |

---

## 🔍 核心修改点速查

### AEPsych侧 (主库)

```
transforms/ops/ordinal.py          [新建] 150 LOC → Ordinal Transform类
transforms/ops/__init__.py         [修改] 2 LOC   → 导入
transforms/parameters.py           [修改] 50 LOC  → par_type处理
config.py                          [修改] 10 LOC  → 验证
```

### dynamic_eur_acquisition侧 (扩展库)

```
modules/local_sampler.py           [修改] 50 LOC  → _perturb_ordinal()
modules/config_parser.py           [修改] 30 LOC  → 类型识别
eur_anova_pair.py                  [修改] 20 LOC  → 类型推断
(modules/diagnostics.py)           [修改] 30 LOC  → 可选诊断报告
```

---

## 🚀 快速开始 (五分钟上手)

1. **了解任务** → 阅读本文件 + ORDINAL_QUICK_REF.md (10 min)
2. **深入理解** → 如需实施，阅读 20251211_ordinal_monotonic_parameter_extension.md (30 min)
3. **实施** → 按详细版中的三个Phase依次执行
4. **验证** → 运行测试清单中的单元 & 集成测试

---

## 📚 设计亮点

### 1. Transform空间设计

- **Why Rank Space**: 统一Categorical, bounds处理简单
- **原理**: Ordinal值 → rank (0,1,2,...) → GP学习相对顺序

### 2. 扰动策略

- **LocalSampler**: rank空间内高斯扰动+舍入 (vs值空间)
- **好处**: 自然的序列扰动, 避免跳跃

### 3. 混合策略支持

- **穷举采样**: 水平数≤阈值时覆盖所有rank
- **随机采样**: 高维ordinal时使用高斯扰动

---

## 💡 设计决策速查

**Q: 为什么不用Integer?**  
A: Integer仅整数, Ordinal支持任意数值 (0.1, 0.5, 2.0, ...)

**Q: Ordinal vs Categorical?**  
A: Categorical无序, Ordinal有序(1<2<3有意义)

**Q: 为什么在rank空间扰动?**  
A: 高斯扰动作用于整数序号更自然, 避免值空间的非线性

---

## ⚠️ 实施前注意

- [ ] 已阅读详细计划文档
- [ ] 已理解Transform、LocalSampler、Config体系
- [ ] 已准备好aepsych源代码查阅环境
- [ ] 已规划测试策略

---

## 📞 后续协调

当指派给LLM实施时，请提供:

1. ✅ 本文件 (overview)
2. ✅ 详细计划文件 (20251211_ordinal_monotonic_parameter_extension.md)
3. ✅ aepsych源代码路径或Git仓库链接
4. ✅ dynamic_eur_acquisition源代码
5. (可选) 已有的测试配置或示例数据

---

**创建时间**: 2025-12-11  
**版本**: v1.0  
**状态**: 待实施  
**所有者**: AI Assistant  
