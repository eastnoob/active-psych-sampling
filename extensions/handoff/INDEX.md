# 文档索引 | AEPsych Extensions 项目

**更新时间**: 2025-12-07
**状态**: 持续维护

---

## 🎯 给AI的核心文件 (必读)

### BaseGPResidualMixedFactory 项目 (已完成)

| 优先级 | 文件 | 用途 | 行数 | token预估 |
|--------|------|------|------|----------|
| ⭐⭐⭐ | `extensions/handoff/20251202_complete_handoff_for_ai.md` | **完整交接文档** (包含一切必需信息) | 321 | 1800 |
| ⭐⭐ | `DETAILED_EXECUTION_PLAN.txt` | 6阶段详细任务分解 | 450 | 2200 |
| ⭐⭐ | `IMPLEMENTATION_QUICK_OVERVIEW.md` | 快速总览 (5分钟速览) | 180 | 900 |

### Ordinal Parameter Support (可选增强)

| 优先级 | 文件 | 用途 | 状态 |
|--------|------|------|------|
| ⭐⭐ | `extensions/handoff/20251207_ordinal_parameter_support.md` | 有序类别参数支持实现计划 | 待实现 |

---

## 📚 参考和验证 (可选深入)

| 文件 | 内容 | 关键数据 |
|------|------|----------|
| `CATEGORICAL_ARD_DECISION_RECORD.md` | 为什么不实现自定义ARDKernel | 核值差异: 0.37 vs 0.61 (39%) |
| `DISCRETE_ARD_PER_DIMENSION.md` | 每维ARD独立支持的验证 | 实测: K[0,1]=0.5134 K[0,2]=0.7165 K[0,3]=0.8465 |
| `FINAL_ANALYSIS_SUMMARY.md` | 全面的分析总结 | FAQ + 时间表 + 风险评估 |
| `verify_discrete_ard_per_dim.py` | 可运行的验证脚本 | 证明botorch CategoricalKernel支持每维ARD |
| `analyze_categorical_ard_clean.py` | 加法vs乘法核对比 | 数值演示39%差异 |

---

## 📁 项目文件结构

```
f:\Github\aepsych-source\
├── extensions/
│   ├── custom_mean.py ✏️ [要修改]
│   ├── custom_factory.py ✏️ [要修改]
│   │
│   ├── custom_factory_mixed.py ✨ [新建，阶段2]
│   ├── test_custom_mixed.py ✨ [新建，阶段3]
│   ├── README_MIXED_RESIDUAL.md ✨ [新建，阶段5]
│   │
│   ├── config_residual_pure_continuous.ini ✨ [新建，阶段4]
│   ├── config_residual_learned_continuous.ini ✨ [新建，阶段4]
│   ├── config_residual_pure_mixed.ini ✨ [新建，阶段4]
│   ├── config_residual_learned_mixed.ini ✨ [新建，阶段4]
│   │
│   └── handoff/
│       ├── 20251201_basegp_residual_mixed_factory_handoff.md [已有]
│       └── 20251202_complete_handoff_for_ai.md [新建，此文档]
│
├── IMPLEMENTATION_QUICK_OVERVIEW.md [总览]
├── DETAILED_EXECUTION_PLAN.txt [详细计划]
├── CATEGORICAL_ARD_DECISION_RECORD.md [拒却理由]
├── DISCRETE_ARD_PER_DIMENSION.md [ARD验证]
├── FINAL_ANALYSIS_SUMMARY.md [全面分析]
├── verify_discrete_ard_per_dim.py [验证脚本]
└── analyze_categorical_ard_clean.py [对比脚本]
```

---

## 🚀 AI接手实现流程

### Step 1: 理解任务 (10分钟)

读: `extensions/handoff/20251202_complete_handoff_for_ai.md`

### Step 2: 了解计划 (10分钟)  

读: `IMPLEMENTATION_QUICK_OVERVIEW.md`

### Step 3: 开始实现 (按顺序)

**阶段1 (6小时)**: Mean模块扩展

- 读: `DETAILED_EXECUTION_PLAN.txt` 第1部分
- 修改: `extensions/custom_mean.py` + `extensions/custom_factory.py`
- 验证: 参数计数, 向后兼容

**阶段2 (8小时)**: 混合工厂实现

- 创建: `extensions/custom_factory_mixed.py`
- 核心: ProductKernel组合 + 维度映射
- 验证: 前向传播形状

**阶段3 (6小时)**: 单元测试

- 创建: `extensions/test_custom_mixed.py`
- 目标: 15+测试, >85%覆盖率
- 命令: `pytest extensions/test_custom_mixed.py`

**阶段4 (4小时)**: 配置系统

- 创建: 4个`.ini`配置文件
- 验证: 都能解析+初始化

**阶段5 (3小时)**: 文档

- 创建: `extensions/README_MIXED_RESIDUAL.md`
- 内容: API参考 + 示例 + 迁移指南

**阶段6 (1小时)**: 最终验证

- 向后兼容性 (旧代码还跑吗?)
- 性能检查 (训练速度可接受?)
- 覆盖检查 (所有测试通过?)

---

## ✅ 成功标准 (逐项验证)

```python
# 代码质量
assert all_tests_pass()                        # 15+单元测试
assert code_coverage() > 0.85                  # 覆盖率>85%
assert no_warnings()                           # 无编译警告

# 功能完整
assert mean_mode_works("pure_residual")        # pure_residual工作
assert mean_mode_works("learned_offset")       # learned_offset工作
assert product_kernel_combines_correctly()     # ProductKernel乘法正确
assert dimension_mapping_correct()             # 维度映射无误

# 向后兼容
assert BaseGPResidualFactory_unchanged()       # 旧工厂不受影响
assert old_configs_still_work()                # 旧配置无需修改
assert acquisition_compatible()                # acquisition函数可用

# 文档完整
assert README_comprehensive()                  # README>800字
assert API_documented()                        # API文档清晰
assert migration_guide_clear()                 # 迁移指南明确
```

---

## 🔑 关键决策一览 (不要推翻)

| 决策 | 方案 | 原因 | 验证 |
|------|------|------|------|
| Mean默认 | pure_residual | 参数效率最优 | ✅ |
| 离散核 | 单CategoricalKernel | 参数省+ARD独立 | ✅ |
| 核组合 | ProductKernel | 标准乘法 | ✅ Acquisition兼容 |
| 自定义核 | ❌不要 | 加法组合无理论 | ✅ |
| 每维ARD | ✅支持 | botorch原生 | ✅ 实测验证 |

**如果想改这些，先回顾决策记录** (`CATEGORICAL_ARD_DECISION_RECORD.md`)

---

## 📊 参数速查表

### Mean相关

```
mean_type = "pure_residual" (默认) | "learned_offset"
offset_prior_std = 0.10  [N(0, 0.1²)先验]
```

### Kernel相关

```
continuous_params = ['x1', 'x2', ...]  # 连续参数名
discrete_params = {'color': ['r','g','b'], ...}  # 离散参数名
discrete_kernel = "categorical" (推荐) | "index" (备选)
```

### Prior相关

```
lengthscale_prior: LogNormal(μ=log(basegp_ls)-log(d)/2+0.01, σ=0.1)
noise_prior: GammaPrior(2.0, 1.228)  [mode≈0.814]
outputscale_prior: LogNormal
```

### 维度映射约定

```
train_X: (n_batch, n_dims)
  前 len(continuous_params) 维 → 连续值
  后 len(discrete_params) 维 → 整数0-indexed (0到n_cat-1)
```

---

## 🆘 困境解决 (常见问题)

| 问题 | 症状 | 修复 |
|------|------|------|
| **参数形状错误** | 前向传播报形状不匹配 | 检查active_dims连续无重合 |
| **梯度为None** | backward不更新参数 | 检查参数是否register_parameter() |
| **核值异常** | K值接近0或1 | 检查lengthscale初始化/prior |
| **训练不收敛** | 损失函数振荡 | 检查noise_prior是否太宽 |
| **向后兼容破** | 旧代码报错 | 确保mean_type有默认值 |

---

## 💾 文档维护

创建新文件时遵循:

```
extensions/custom_factory_mixed.py
  - docstring清晰 (class + 主要方法)
  - 类型注解 (Type hints)
  - 参数说明 (Args/Returns)
  
extensions/test_custom_mixed.py
  - 每个测试函数名清晰 (test_xxx_yyy)
  - assertion错误消息有意义
  - 固定种子保证可重现
  
extensions/README_MIXED_RESIDUAL.md
  - 目录清晰 (## 标题)
  - 有工作代码示例
  - 常见问题/故障排查章节
```

---

## 🎓 深入阅读 (可选)

如果AI想深入理解背景:

1. **残差学习原理** → `FINAL_ANALYSIS_SUMMARY.md` 第2节
2. **核的数学** → `CATEGORICAL_ARD_DECISION_RECORD.md` 数值对比
3. **兼容性验证** → 相关verification脚本源代码
4. **AEPsych生态** → `temp_aepsych/aepsych/factory/` 源代码

---

## 📈 预期时间表

```
实际开发进度预测:

阶段1: 6h   ├─ 预计快速完成 (改2个文件)
           │
阶段2: 8h   ├─ 预计可能卡壳 (ProductKernel逻辑)
           │
阶段3: 6h   ├─ 预计中等复杂 (test覆盖)
阶段4: 4h   │  (可平行)
           │
阶段5: 3h   ├─ 预计快速完成 (写文档)
           │
阶段6: 1h   └─ 验证检查

如遇到问题，可能需要回头修改阶段2或3。
建议: 阶段1+2完成后先做完整单元测试 (阶段3)，
     再做配置系统 (阶段4)。
```

---

## 📞 如有问题

如果AI在实现过程中卡住，建议:

1. **参数问题** → 查看 `IMPLEMENTATION_QUICK_OVERVIEW.md` 参数表
2. **核函数问题** → 查看 `DISCRETE_ARD_PER_DIMENSION.md` 验证
3. **设计问题** → 查看 `CATEGORICAL_ARD_DECISION_RECORD.md` 拒却理由
4. **测试问题** → 参考现有 `test_*.py` 代码风格
5. **兼容问题** → 检查 `FINAL_ANALYSIS_SUMMARY.md` 风险章节

---

**此索引文档为导航用，具体细节见各分文档。**

**关键: 先读交接文档 (20251202_complete_handoff_for_ai.md)，然后开始编码。**
