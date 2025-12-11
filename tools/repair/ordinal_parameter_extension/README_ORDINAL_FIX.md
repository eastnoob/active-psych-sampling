# Ordinal Parameter Extension for AEPsych

**有序参数类型支持** - 扩展 AEPsych 支持稀疏采样的连续物理值参数

---

## 概述

本扩展为 AEPsych 添加了 **ordinal 参数类型** 支持，用于处理具有有序关系的稀疏离散值参数，如：

- 天花板高度：`[2.0, 2.5, 3.5]` 米
- 温度设置：`[16, 18, 20, 22, 24]` 摄氏度
- Likert 量表：`["非常不同意", "不同意", "中立", "同意", "非常同意"]`

### 核心设计理念

**规范化值空间 (Normalized Value Space)**

与传统的 rank 编码不同，ordinal 参数使用 **Min-Max 归一化** 保留间距信息：

```python
# 物理值
heights = [2.0, 2.5, 3.5]  # 米

# ❌ 错误：Rank 编码（丢失间距信息）
ranks = [0, 1, 2]  # 等间距，ANOVA 误判

# ✅ 正确：规范化值（保留间距比例）
normalized = [0.0, 0.333, 1.0]  # (2.5-2.0)/(3.5-2.0) = 0.333
```

**为什么保留间距很重要？**

- **ANOVA 分解**：主效应分析需要正确的相对间距关系
- **GP 建模**：高斯过程核函数依赖于距离度量
- **局部探索**：LocalSampler 在规范化空间扰动 + 最近邻约束

---

## 支持的参数类型

| 类型 | 描述 | 配置方式 | 示例 |
|------|------|---------|------|
| `custom_ordinal` | 等差有序参数 | `min_value` + `max_value` + `step` | 温度：16-24℃，步长2℃ |
| `custom_ordinal_mono` | 非等差单调参数 | `values` (直接指定列表) | 天花板：[2.0, 2.5, 3.5] |
| `ordinal` | 通用有序类型 | 两种方式均可 | Likert 量表 |

### 配置示例

#### 1. 等差有序参数 (custom_ordinal)

```ini
[temperature]
par_type = custom_ordinal
min_value = 16
max_value = 24
step = 2
# 自动生成: [16, 18, 20, 22, 24]
```

或使用 `num_levels`：

```ini
[brightness]
par_type = custom_ordinal
min_value = 0.0
max_value = 1.0
num_levels = 5
# 自动生成: [0.0, 0.25, 0.5, 0.75, 1.0]
```

#### 2. 非等差单调参数 (custom_ordinal_mono)

```ini
[ceiling_height]
par_type = custom_ordinal_mono
values = [2.0, 2.5, 3.5]
# 间距：[0.5, 1.0] 不等
```

#### 3. Likert 量表

```ini
[agreement]
par_type = custom_ordinal
levels = strongly_disagree, disagree, neutral, agree, strongly_agree
# 自动映射到整数序列: [0, 1, 2, 3, 4]
```

---

## 文件结构

```
tools/repair/ordinal_parameter_extension/
├── README_ORDINAL_FIX.md          # 本文件
├── apply_fix.py                   # 自动应用 patch 脚本
├── verify_fix.py                  # 验证 patch 应用脚本
├── files/
│   └── ordinal.py                 # AEPsych Ordinal Transform 完整实现
├── aepsych_ordinal_transforms.patch         # 新增 ordinal.py
├── aepsych_transforms_ops_init.patch        # 修改 transforms/ops/__init__.py
├── aepsych_transforms_parameters.patch      # 修改 parameters.py
└── aepsych_config.patch                     # 修改 config.py
```

### Extensions 自定义组件

```
extensions/dynamic_eur_acquisition/
├── transforms/
│   └── ops/
│       ├── __init__.py
│       └── custom_ordinal.py      # CustomOrdinal 类（不继承 BoTorch）
├── modules/
│   ├── local_sampler.py           # 添加 _perturb_ordinal 方法
│   └── config_parser.py           # 变量类型解析支持 ordinal
└── eur_anova_pair.py             # Transform 推断支持 ordinal
```

---

## 安装指南

### 前置条件

1. **Git 工作目录干净**
   ```bash
   cd aepsych
   git status  # 确保无未提交修改
   ```

2. **安装 patch 工具**
   - **Windows**: `choco install patch` 或使用 Git Bash
   - **Linux**: `sudo apt install patch`
   - **macOS**: `brew install gpatch`

### 自动应用 Patch

```bash
# 1. 进入 repair 目录
cd tools/repair/ordinal_parameter_extension

# 2. 检查 patch 可用性（dry-run）
python apply_fix.py --dry-run

# 3. 实际应用 patch
python apply_fix.py

# 4. 验证应用结果
python verify_fix.py
```

### 手动应用 Patch（备选）

如果自动脚本失败，可手动应用：

```bash
cd /path/to/aepsych

# 应用四个 patch
patch -p1 < /path/to/aepsych_ordinal_transforms.patch
patch -p1 < /path/to/aepsych_transforms_ops_init.patch
patch -p1 < /path/to/aepsych_transforms_parameters.patch
patch -p1 < /path/to/aepsych_config.patch
```

---

## 技术实现细节

### 1. Transform 层：规范化映射

**文件**: `aepsych/transforms/ops/ordinal.py`

```python
class Ordinal(InputTransform):
    def _build_normalized_mappings(self):
        """Min-Max 归一化到 [0, 1]"""
        for idx in self.indices:
            phys_vals = self.values[idx]  # e.g., [2.0, 2.5, 3.5]
            min_val = phys_vals.min()
            max_val = phys_vals.max()

            # 规范化
            norm_vals = (phys_vals - min_val) / (max_val - min_val)
            # Result: [0.0, 0.333, 1.0]

            self.normalized_values[idx] = norm_vals
```

### 2. LocalSampler：规范化空间扰动

**文件**: `extensions/dynamic_eur_acquisition/modules/local_sampler.py`

```python
def _perturb_ordinal(self, base, k, B):
    """在规范化值空间扰动 + 最近邻约束"""
    unique_normalized_vals = self._unique_vals_dict[k]  # [0.0, 0.333, 1.0]

    # 规范化空间的 span 总是 1.0
    sigma = self.local_jitter_frac * 1.0  # e.g., 0.1

    # 高斯扰动
    center_vals = base[:, :, k]  # 当前值（已规范化）
    noise = np.random.normal(0, sigma, size=(B, self.local_num))
    perturbed = center_vals + noise

    # 最近邻约束到有效值
    for i in range(B):
        for j in range(self.local_num):
            closest_idx = np.argmin(np.abs(unique_normalized_vals - perturbed[i, j]))
            samples[i, j] = unique_normalized_vals[closest_idx]
```

**关键设计**：
- ✅ 在规范化值空间扰动（保留间距信息）
- ✅ 最近邻约束确保合法值
- ✅ ANOVA 看到正确的相对间距

### 3. 混合扰动策略（可选）

对于低水平离散变量（≤3 水平），支持 **穷举模式**：

```python
# local_sampler.py 配置
local_sampler = LocalSampler(
    use_hybrid_perturbation=True,
    exhaustive_level_threshold=3,  # 2-3 水平用穷举
)

# 效果：
# 2水平变量 → [val0, val1, val0, val1, ...]（循环）
# 4水平变量 → 高斯扰动（传统方法）
```

---

## 配置集成示例

### AEPsych 配置文件

```ini
[common]
stimuli_per_trial = 1
outcome_types = binary

# ========== 参数定义 ==========
[ceiling_height]
par_type = custom_ordinal_mono
values = [2.0, 2.5, 3.5]  # 非等差

[temperature]
par_type = custom_ordinal
min_value = 16
max_value = 24
step = 2  # 等差序列

[lighting]
par_type = continuous
lower_bound = 0.0
upper_bound = 1.0

# ========== 模型配置 ==========
[GPClassificationModel]
stimuli_per_trial = 1
outcome_types = binary

# 变量类型自动推断（从 par_type）
# ceiling_height → custom_ordinal_mono
# temperature → custom_ordinal
# lighting → continuous
```

### Extensions EUR-ANOVA 配置

```python
# config_parser.py 自动识别
variable_types = {
    0: "custom_ordinal_mono",  # ceiling_height
    1: "custom_ordinal",       # temperature
    2: "continuous",           # lighting
}

# LocalSampler 自动应用正确的扰动策略
sampler = LocalSampler(
    variable_types=variable_types,
    local_jitter_frac=0.1,
    local_num=4,
)
```

---

## 测试指南

### 单元测试

```bash
python -m pytest tests/test_custom_ordinal_transform.py -v
```

测试覆盖：
- ✅ 规范化映射正确性
- ✅ Transform/untransform 往返一致性
- ✅ 边界变换
- ✅ 配置解析

### 集成测试

```bash
python -m pytest extensions/dynamic_eur_acquisition/test/test_ordinal_integration.py -v
```

测试覆盖：
- ✅ LocalSampler 扰动逻辑
- ✅ 变量类型推断
- ✅ EUR-ANOVA 采集函数兼容性

### 端到端测试

创建测试配置文件 `test_ordinal_e2e.ini`：

```ini
[common]
stimuli_per_trial = 1
outcome_types = binary

[height]
par_type = custom_ordinal
min_value = 2.0
max_value = 3.5
num_levels = 4

[GPClassificationModel]
stimuli_per_trial = 1
outcome_types = binary
```

运行完整实验流程：

```python
from aepsych.config import Config
from aepsych.server import AEPsychServer

config = Config.from_file("test_ordinal_e2e.ini")
server = AEPsychServer(config=config)

# 模拟实验
for trial in range(20):
    x = server.ask()
    y = simulate_response(x)  # 用户模拟函数
    server.tell(x, y)

# 验证结果
assert server.strat.model is not None
assert len(server.strat.x) == 20
```

---

## 故障排除

### 问题 1: Patch 应用失败

**症状**:
```
patching file aepsych/transforms/parameters.py
Hunk #1 FAILED at 42.
```

**原因**: AEPsych 版本不匹配或已有修改

**解决**:
1. 检查 AEPsych commit hash：`git log -1 --oneline`
2. 如果版本不匹配，手动合并代码
3. 参考 `files/ordinal.py` 完整实现

### 问题 2: 导入失败

**症状**:
```python
ImportError: cannot import name 'Ordinal' from 'aepsych.transforms.ops'
```

**检查**:
1. 文件是否存在：`ls aepsych/transforms/ops/ordinal.py`
2. `__init__.py` 是否导出：`grep Ordinal aepsych/transforms/ops/__init__.py`
3. 运行验证脚本：`python verify_fix.py`

### 问题 3: 规范化值异常

**症状**: ANOVA 效应分解结果不合理

**检查**:
1. 打印规范化值：
   ```python
   print(ordinal.normalized_values)
   # 应该是 [0.0, ..., 1.0]，保留间距比例
   ```
2. 验证 LocalSampler 是否使用规范化值：
   ```python
   print(sampler._unique_vals_dict[dim_idx])
   # 应该是规范化值，不是物理值
   ```

---

## 设计文档参考

详细设计理念和实现决策，请参考：

- **handoff/ordinal_normalized_design.md** - 规范化值 vs Rank 设计决策
- **handoff/20251211_ordinal_monotonic_parameter_extension.md** - 完整实现计划
- **handoff/AEPSYCH_MODIFICATIONS_PATCH_GUIDE.md** - Patch 创建指南

---

## 版本兼容性

| 组件 | 最低版本 | 推荐版本 | 备注 |
|------|---------|---------|------|
| AEPsych | 0.3.x | latest | 需要包含 `transforms` 模块 |
| BoTorch | 0.8.x | 0.9.x | InputTransform 基类 |
| Python | 3.8+ | 3.10+ | Type hints 支持 |
| NumPy | 1.20+ | 1.24+ | |
| PyTorch | 1.12+ | 2.0+ | |

---

## 常见问题 (FAQ)

### Q1: Ordinal 和 Categorical 有什么区别？

**A**:
- **Categorical**: 无序离散（如颜色：红/蓝/绿）
  - 扰动方式：随机采样
  - 编码：One-hot 或 Label encoding
- **Ordinal**: 有序离散（如温度：16/18/20℃）
  - 扰动方式：规范化空间高斯扰动 + 最近邻
  - 编码：规范化值（保留间距）

### Q2: 为什么不用 Rank 编码？

**A**: Rank 编码 `[0, 1, 2]` 假设等间距，会误导 ANOVA 分解。

例如：`[2.0, 2.5, 3.5]` → Rank `[0, 1, 2]`
- ❌ ANOVA 认为 2.5→3.5 的效应 = 2.0→2.5 的效应（错误）
- ✅ 规范化 `[0.0, 0.333, 1.0]` 保留真实间距比例

### Q3: 如何选择 custom_ordinal vs custom_ordinal_mono？

**A**:
- 等间距参数 → `custom_ordinal` + `step`/`num_levels`（自动生成）
- 不等间距参数 → `custom_ordinal_mono` + `values`（手动指定）

### Q4: LocalSampler 混合策略什么时候有用？

**A**:
- **2-3 水平变量**：用穷举（`exhaustive_level_threshold=3`）
  - 完全覆盖所有水平，无随机性
- **≥4 水平变量**：用高斯扰动
  - 探索性更强，适合高维空间

---

## 贡献者

- 设计与实现：[Your Team]
- 技术审查：[Reviewers]
- 文档编写：[Documentation Team]

---

## 许可证

本扩展遵循 AEPsych 主项目的许可证（Apache 2.0）。

---

## 更新日志

### v1.0.0 (2025-12-11)
- ✅ 初始版本发布
- ✅ 支持 custom_ordinal 和 custom_ordinal_mono
- ✅ LocalSampler 规范化空间扰动
- ✅ EUR-ANOVA 集成
- ✅ 自动化 patch 应用和验证脚本

---

**祝使用愉快！如有问题请提交 Issue。**
