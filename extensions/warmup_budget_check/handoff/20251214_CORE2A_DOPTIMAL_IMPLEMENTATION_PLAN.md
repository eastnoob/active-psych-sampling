# Core-2a D-optimal 实现计划

**状态**：待实施
**优先级**：P0 (科学严谨性)
**预计工作量**：4-5小时
**依赖**：pydoe3 1.6.1 (已安装在pixi.toml)

---

## 问题描述

**当前状况**：Core-2a 和 Core-2b 采样合并，两者都用随机选择，违反原设计承诺
**目标**：分离采样逻辑，为Core-2a实现D-optimal设计
**科学性**：绝对严谨，可发表论文

---

## 技术方案

### 核心修改 (warmup_sampler.py)

#### 改动1：分离Step 3的采样逻辑

**位置**：`_generate_five_step_samples()` 第 796-811 行

**原代码**：
```python
n_core2_total = budget["core2a_configs"] + budget["core2b_configs"]
core2_indices = self._select_interaction_aware_configs(n_configs=n_core2_total, ...)
```

**新代码**：
```python
# Step 3a: Core-2a - D-optimal主效应
n_core2a = budget["core2a_configs"]
core2a_indices = self._select_doptimal_configs(n_configs=n_core2a, used_indices=used_indices)
used_indices.update(core2a_indices)

# Step 3b: Core-2b - 交互对感知采样
n_core2b = budget["core2b_configs"]
core2b_indices = self._select_interaction_aware_configs(
    n_configs=n_core2b,
    interaction_mode=interaction_mode,
    interaction_pairs_to_explore=interaction_pairs_to_explore,
    min_config_per_pair=min_config_per_pair,
    used_indices=used_indices,
)
used_indices.update(core2b_indices)

core2_indices = core2a_indices + core2b_indices
```

#### 改动2：新增D-optimal采样方法

**位置**：新方法 `_select_doptimal_configs()`
**代码框架**：

```python
def _select_doptimal_configs(self, n_configs: int, used_indices: set) -> List[int]:
    """
    D-optimal设计：最大化Fisher信息矩阵行列式

    Args:
        n_configs: 选择的配置数
        used_indices: 已使用的索引集

    Returns:
        选中的配置索引列表
    """
    from pyDOE3.doe_optimal import optimal_design
    import numpy as np

    available = list(set(self.design_df.index) - used_indices)

    if len(available) <= n_configs:
        return available

    # 1. 准备候选集：将混合类型设计矩阵标准化到[0,1]
    candidates = self._normalize_design_to_candidates(available)

    # 2. 应用D-optimal设计
    design, info = optimal_design(
        candidates,
        n_points=n_configs,
        degree=1,           # 线性模型
        criterion="D",      # D-optimality
        method="detmax"     # Detmax算法
    )

    # 3. 映射回配置索引
    selected_indices = [available[i] for i in np.where(design.sum(axis=1) >= 0)[0]]
    selected_indices = selected_indices[:n_configs]

    return selected_indices


def _normalize_design_to_candidates(self, available_indices: List[int]) -> np.ndarray:
    """将设计矩阵标准化为[0,1]数值矩阵"""
    import numpy as np

    df_sub = self.design_df.iloc[available_indices]
    X_list = []

    for col in df_sub.columns:
        col_data = df_sub[col]

        if col_data.dtype in ['float64', 'int64', 'int32']:
            # 数值变量：min-max标准化
            col_min, col_max = col_data.min(), col_data.max()
            if col_max > col_min:
                X_list.append((col_data.values - col_min) / (col_max - col_min))
            else:
                X_list.append(np.zeros(len(col_data)))

        elif col_data.dtype == 'bool':
            # 布尔变量
            X_list.append(col_data.astype(int).values)

        else:
            # 分类变量：one-hot编码
            for cat in col_data.unique():
                X_list.append((col_data == cat).astype(int).values)

    return np.column_stack(X_list) if X_list else np.array([]).reshape(-1, 1)
```

---

## 实施步骤

### Step 1: 代码修改 (1小时)
- [ ] 修改 `_generate_five_step_samples()` 分离Core-2a/2b
- [ ] 新增 `_select_doptimal_configs()` 方法
- [ ] 新增 `_normalize_design_to_candidates()` 辅助方法

### Step 2: 单元测试 (1小时)
- [ ] 新建 `tests/test_core2a_doptimal.py`
  - 验证D-optimal选择的形状
  - 验证不重复选择
  - 验证效率评估 (D-efficiency > 0)

### Step 3: 集成测试 (1.5小时)
- [ ] 运行 `test_three_modes.py` 验证Core-2b不受影响
- [ ] 验证六因子设计空间的完整流程
- [ ] 检查输出日志中Core-2a/2b的分离

### Step 4: 文档更新 (1小时)
- [ ] 更新 STRUCTURE.md: Core-2a改为"D-optimal主效应"
- [ ] 更新 warmup_budget_estimator.py 的print说明
- [ ] 新增 Core-2a采样说明

### Step 5: 提交 (0.5小时)
- [ ] 生成新的handoff文档
- [ ] Git commit

**总计**：4-5小时

---

## 验证清单

### 科学性检查
- [ ] D-optimality值 > 0 (Fisher信息矩阵行列式)
- [ ] D-efficiency >= 20% (实际值应该30-50%)
- [ ] 选择的配置无重复
- [ ] 与已用配置无重叠

### 功能性检查
- [ ] Core-2a输出20个配置（示例值）
- [ ] Core-2b输出21个配置（示例值）
- [ ] 分配给5个被试时无遗漏
- [ ] 日志清晰显示分离

### 兼容性检查
- [ ] 现有三种交互模式正常工作
- [ ] LHS采样不受影响
- [ ] Boundary采样不受影响
- [ ] Core-1采样不受影响

### 回归测试
- [ ] `pixi run pytest tests/` 全部通过
- [ ] `test_three_modes.py` 三种模式都工作
- [ ] 模拟被试流程正常

---

## 文件修改清单

| 文件 | 改动 | 行数 |
|------|------|------|
| `core/warmup_sampler.py` | 分离+新增方法 | +80 -5 |
| `tests/test_core2a_doptimal.py` | 新建 | 100 |
| `STRUCTURE.md` | 更新说明 | +5 -2 |
| `core/warmup_budget_estimator.py` | 更新print | +2 -2 |

**总增删**：~180行

---

## 依赖检查

| 包 | 版本 | 状态 |
|---|---|---|
| pyDOE3 | 1.6.1 | ✅ pixi.toml已添加 |
| numpy | 2.3.5 | ✅ 已有 |
| scipy | 1.16.3 | ✅ 已有 |
| pandas | 2.3.3 | ✅ 已有 |

**无需额外安装**

---

## 已知风险 & 缓解方案

| 风险 | 概率 | 缓解 |
|------|------|------|
| pyDOE3与Windows编码冲突 | 低 | 使用英文注释，避免unicode |
| 候选集过大导致计算慢 | 低 | 当available > 1000时采样子集 |
| Categorical变量one-hot爆炸 | 中 | 限制类别数 <= 10，或采用目标编码 |
| 与现有API不兼容 | 极低 | 新增方法，保留原有接口 |

---

## 参考文献

```bibtex
Box, G.E.P., Hunter, J.S., & Hunter, W.G. (2005).
"Statistics for Experimenters: Design, Innovation, and Discovery" (2nd ed.).
Wiley. Chapter 7: D-optimal Designs.

pyDOE3 Documentation:
https://pydoe3.readthedocs.io/en/stable/doe_optimal.html
```

---

## 接手说明

**对后续维护者**：
1. 上述Step 1-5按顺序执行
2. 遇到问题先查看 `WARMUP_ARCHITECTURE_ANALYSIS.md` 了解背景
3. 所有D-optimal调用都应该检查返回的efficiency值
4. 如果某个因子的水平太多，应该考虑采用其他编码方式
5. 性能瓶颈通常在pyDOE3.optimal_design()，可增加采样次数上限

---

**更新于**：2025-12-14
**分析报告**：`WARMUP_ARCHITECTURE_ANALYSIS.md`
**背景问题**：`20251214_CORE2A_2B_SAMPLING_ALGORITHM_MISMATCH.md`
