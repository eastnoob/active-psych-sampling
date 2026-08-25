# 🚀 Lean Warmup 重构计划：极致精简与精准规格化

## 1. 核心目标

在“极致少预算”的约束下，通过“锚点定位 + 全局筛选 + 先验注入”的组合策略，为 Phase 2 的主动学习提供一个既稳健又具有前瞻性的 BaseGP 先验模型。

---

## 2. 面向的问题与解决方案

### 问题 A：预热预算冗余与效率低下

* **现状**：Core-1 占用 8 个点，且交互探索“走马观花”。
* **方案**：
  * **锚点精简**：Core-1 压减至 4 个关键点（Min, Max, Median, Strategic Random）。
  * **深度优先**：对用户指定的先验交互对执行深度采样（每对 4+ 点），其余预算执行广度扫描。

### 问题 B：小样本下的“复杂度诅咒”

* **现状**：GP 模型试图拟合所有可能的交互，导致在小样本下参数震荡或过拟合。
* **方案**：
  * **Lasso 全局筛选**：引入 Lasso 正则化进行联合效应分析，剔除伪交互。
  * **模型规格化**：在训练前确定精简的核函数结构。

### 问题 C：先验知识与数据发现的冲突

* **现状**：要么纯数据驱动，要么纯人工指定。
* **方案**：
  * **融合逻辑**：`[Lasso 发现项] + [用户怀疑项]`。
  * **柔性熔断**：设定推荐阈值（如 5 个），但允许在数据证据极强（R² 增益显著）时突破限制，并给出风险预警。

---

## 3. 调整后的三步走架构

### Step 1: Warmup Sampling (采样生成)

* **职责**：生成极致精简的采样方案。
* **核心逻辑**：4 锚点 + 先验深度优先。

### Step 2: Analysis & Specification (分析与规格化) —— [重点重构]

* **职责**：合并原 Step 2 与 Step 3 的分析部分。
* **内容**：
    1. **ICC 诊断**：评估个体差异。
    2. **Lasso 筛选**：全局敏感性分析。
    3. **效应融合**：执行“发现 + 怀疑”逻辑，计算各维度敏感度。
    4. **生成处方**：导出 `model_spec.json`，包含 Lengthscales 初值、核结构、噪声底噪。

### Step 3: BaseGP Training & Export (训练与导出)

* **职责**：按照 Step 2 的“处方”进行 GP 优化。
* **内容**：执行训练、导出模型、生成最终可视化报告。

---

## 4. 必要文件路径

| 功能模块 | 文件路径 |
| :--- | :--- |
| **预算估算逻辑** | `eur-warmup/core/internal/warmup_budget_estimator.py` |
| **采样核心逻辑** | `eur-warmup/core/internal/warmup_sampler.py` |
| **分析与筛选核心** | `eur-warmup/core/internal/phase1_analyzer.py` |
| **分析模块 (Step 2)** | `eur-warmup/modules/step2.py` |
| **训练模块 (Step 3)** | `eur-warmup/modules/step3.py` |
| **全流程配置模板** | `eur-warmup/config/full_workflow_template.toml` |

---

## 5. 交互融合与熔断的平衡策略 (Balanced Circuit Breaker)

系统将采取 **“推荐阈值 + 证据强度”** 的平衡算法：

1. **硬性保留**：用户指定的 `suspected_pairs` 永远保留。
2. **自动发现**：Lasso 筛选出的显著项。
3. **柔性熔断**：
    * 如果总数 $\le 5$：全部纳入。
    * 如果总数 $> 5$：
        * 计算额外交互项带来的 **BIC 改进** 或 **R² 增益**。
        * 如果增益超过显著性阈值（如 $\Delta R^2 > 0.05$），则允许纳入，但输出 **[黄色预警]** 提示过拟合风险。
        * 否则，按显著性截断至 5 个，并输出 **[建议]**。

---

## 6. 测试方案

### 单元测试 (Unit Tests)

* `test_lean_sampling`: 验证 4 锚点分布及先验深度采样。
* `test_lasso_screening`: 验证 Lasso 是否能正确剔除无关交互项。
* `test_fusion_logic`: 验证“发现 + 怀疑”在不同证据强度下的融合结果。

### 集成测试 (Integration Tests)

* `test_full_lean_flow`: 模拟从 Step 1 到 Step 3 的完整流程，检查 `model_spec.json` 的传递正确性及 BaseGP 的收敛速度。

---
**计划拟定人**：GitHub Copilot (Gemini 3 Flash)
**日期**：2025-12-27
