# Phase 1 完整分析报告

> **整合了 Step 2 (交互对分析) 和 Step 3 (Base GP 敏感度分析) 的结果**

**生成时间**: 2025-11-30 23:52:51

---

## 核心发现

### 1. 关键交互对（来自 Step 2）

筛选出 **5** 个重要交互对：

| 排名 | 交互对 | 评分 | 系数 |
|------|--------|------|------|
| 1 | **x3_OuterFurniture** × **x6_InnerFurniture** | 0.171 | 0.078 |
| 2 | **x1_CeilingHeight** × **x6_InnerFurniture** | 0.163 | -0.047 |
| 3 | **x2_GridModule** × **x4_VisualBoundary** | 0.141 | -0.273 |
| 4 | **x2_GridModule** × **x6_InnerFurniture** | 0.135 | -0.281 |
| 5 | **x1_CeilingHeight** × **x4_VisualBoundary** | 0.135 | 0.096 |

### 2. 因子敏感度排序（来自 Step 3 ARD）

基于 Base GP 的自动相关性判断（ARD）：

| 排名 | 因子 | 长度尺度 | 敏感度 | 参与交互数 |
|------|------|----------|--------|------------|
| 1 | x4_VisualBoundary | 1.37 | *** 高 | 2 个 |
| 2 | x5_PhysicalBoundary | 1.74 | *** 高 | 0 个 |
| 3 | x3_OuterFurniture | 1.80 | ** 中 | 1 个 |
| 4 | x6_InnerFurniture | 4.30 | ** 中 | 3 个 |
| 5 | x2_GridModule | 4.30 | * 低 | 2 个 |
| 6 | x1_CeilingHeight | 7.28 | * 低 | 2 个 |

## 核心洞察：交互模式分析

### 交互核心因子：x6_InnerFurniture

这些因子参与了 **3/5** 个交互对，表明其效果**高度依赖情境**。

### 因子特性对比

| 因子 | 主效应敏感度 | 交互参与度 | 特性 |
|------|--------------|------------|------|
| x1_CeilingHeight | 低 (LS=7.28) | 2 个 | 调节因子 |
| x2_GridModule | 低 (LS=4.30) | 2 个 | 调节因子 |
| x3_OuterFurniture | 中 (LS=1.80) | 1 个 | 调节因子 |
| x4_VisualBoundary | 高 (LS=1.37) | 2 个 | 主效应 + 交互 |
| x5_PhysicalBoundary | 高 (LS=1.74) | 0 个 | 独立主效应 |
| x6_InnerFurniture | 中 (LS=4.30) | 3 个 | **情境依赖型** |

## 推荐的初始采样点（来自 Step 3）

这些点可作为 Phase 2 的 warmup 初始化：

### Sample 1: Best Prior（预测最佳）

- **预测得分**: 1.529 (std=0.780)
- **参数配置**:
  - x1_CeilingHeight: 8.5
  - x2_GridModule: 6.5
  - x3_OuterFurniture: 2.0
  - x4_VisualBoundary: 2.0
  - x5_PhysicalBoundary: 1.0
  - x6_InnerFurniture: 0.0

### Sample 2: Worst Prior（预测最差）

- **预测得分**: -1.759 (std=0.782)
- **参数配置**:
  - x1_CeilingHeight: 2.8
  - x2_GridModule: 6.5
  - x3_OuterFurniture: 0.0
  - x4_VisualBoundary: 0.0
  - x5_PhysicalBoundary: 1.0
  - x6_InnerFurniture: 2.0

### Sample 3: Max Uncertainty（最不确定）

- **不确定性**: std=0.807 (mean=-0.101)
- **参数配置**:
  - x1_CeilingHeight: 8.5
  - x2_GridModule: 8.0
  - x3_OuterFurniture: 0.0
  - x4_VisualBoundary: 2.0
  - x5_PhysicalBoundary: 1.0
  - x6_InnerFurniture: 2.0

## Phase 2 实验建议

### 推荐策略

基于整合分析，建议 Phase 2 采用以下策略：

1. **EUR-ANOVA 配置**（来自 Step 2）:
   - 交互对: 5 个
   - λ (交互权重): 0.10 → 0.60
   - γ (覆盖权重): 0.30 → 0.06
   - 总预算: 500 次

2. **初始 Warmup 点**（来自 Step 3）:
   - 使用 3 个关键点作为初始采样
   - 覆盖设计空间的关键区域（最佳/最差/最不确定）

3. **探索优先级**:
   - **优先探索主效应**: x4_VisualBoundary, x5_PhysicalBoundary
   - **重点探索交互**: 涉及 x6_InnerFurniture 的组合

## 📦 输出文件

### Step 2 输出
- JSON配置: `extensions\warmup_budget_check\step2\202511302352\phase1_phase2_config.json`
- NumPy调度: `extensions\warmup_budget_check\step2\202511302352\phase1_phase2_schedules.npz`
- 详细报告: `extensions\warmup_budget_check\step2\202511302352\phase1_analysis_report.md`

### Step 3 输出
- GP模型: `extensions\warmup_budget_check\phase1_analysis_output\202511302352\base_gp\base_gp_state.pth`
- 关键点: `extensions\warmup_budget_check\phase1_analysis_output\202511302352\base_gp\base_gp_key_points.json`
- 长度尺度: `extensions\warmup_budget_check\phase1_analysis_output\202511302352\base_gp\base_gp_lengthscales.json`
- 设计空间扫描: `extensions\warmup_budget_check\phase1_analysis_output\202511302352\base_gp\design_space_scan.csv`
- 详细报告: `extensions\warmup_budget_check\phase1_analysis_output\202511302352\base_gp\base_gp_report.md`

### 整合报告
- **本报告**: `extensions\warmup_budget_check\step2\integrated_202511302352\INTEGRATED_ANALYSIS_REPORT.md`

---

*自动生成于 Phase 1 完整分析流程*
