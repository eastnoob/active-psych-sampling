# BaseGP 先验集成指南 (完整版)

**日期**: 2025-11-28 | **工作文件夹**: `F:\Github\aepsych-source\test\is_EUR_work\plans\20251128_2238\`

---

## 📍 0. 数据和文件获取路径汇总

### 0.1 BaseGP 训练输出文件

```
F:\Github\aepsych-source\extensions\warmup_budget_check\
└── phase1_analysis_output\
    └── 202511271557\                          # ← 训练时间戳
        ├── base_gp\
        │   ├── base_gp_report.md              # ← 包含所有参数统计
        │   ├── base_gp_lengthscales.json      # ← Lengthscales 数值
        │   ├── base_gp_state.pth              # ← PyTorch 模型状态
        │   ├── base_gp_subject_stats.json     # ← 被试数据统计
        │   └── design_space_scan.csv          # ← 设计空间扫描
        └── base_gp_key_points.json            # ← Phase 2 推荐点
```

**快速获取**:

```bash
# 打开报告查看汇总
cat "F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202511271557\base_gp\base_gp_report.md"

# 提取 lengthscales (JSON格式)
cat "F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202511271557\base_gp\base_gp_lengthscales.json"
```

### 0.2 BaseGP 核函数代码

**文件**: `F:\Github\aepsych-source\extensions\warmup_budget_check\core\phase1_step3_base_gp.py`

**关键位置**:

| 内容 | 行号 |
|------|------|
| 核函数定义 | 108-110 |
| 训练函数 | 123-170 |
| State 保存 | ~160 |

**核函数代码**:

```python
# line 108-110
self.covar_module = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.MaternKernel(
        nu=2.5,
        ard_num_dims=train_x.shape[-1],
    )
)
```

### 0.3 BaseGP 模型加载方式

```python
import torch
import gpytorch
from aepsych.factory import DefaultMeanCovarFactory

# 加载已训练的模型
state_path = r"F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202511271557\base_gp\base_gp_state.pth"

state = torch.load(state_path, map_location='cpu')

# 提取参数
lengthscales = state['model']['covar_module.base_kernel.raw_lengthscale']
# → 应用 softplus 变换: ls = softplus(raw_ls)

output_scale = state['model']['covar_module.raw_outputscale']
# → 应用 softplus 变换

mean_constant = state['model']['mean_module.raw_constant']
# → 不需要变换

noise_variance = state['likelihood']['noise_covar.raw_noise']
# → 应用 softplus 变换
```

---

## 1. BaseGP 学习到的参数完整清单

### 1.1 参数来源与数值

| 参数 | 原始值 | 经过 softplus 变换 | 来源文件 |
|------|--------|-----------------|----------|
| **Lengthscales (ARD)** | 见 raw_lengthscale | `[1.9395, 1.4659, 3.8931, 0.4367, 4.2384, 3.6145]` | `base_gp_lengthscales.json` |
| **Output Scale** | `-1.7046` | `0.167083` | state.pth |
| **Mean Constant** | `0.04302` | `0.04302` | state.pth |
| **Noise Variance** | `0.22842` | `0.813864` | state.pth |

### 1.2 核函数详细配置

**核函数类型**: Matern(ν=2.5) + ARD + Scale

**代码实现**:

```python
# File: phase1_step3_base_gp.py (line 108-110)
class _MaternARDGP(gpytorch.models.ExactGP):
    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(
                nu=2.5,                           # ← Matern 平滑度
                ard_num_dims=train_x.shape[-1],  # ← 6 维 ARD
            )
        )
```

**为什么选择 Matern(2.5)?**

- 光滑度 ν=2.5 = 2 倍可微，适合中等复杂度函数
- 比 RBF (无限光滑) 更能捕捉真实数据的不规则性
- ARD 允许每维度独立学习 lengthscale

### 1.3 各参数的含义

| 参数 | 含义 | 用途 |
|------|------|------|
| **Lengthscale** | 该维度上函数的变化尺度（越小越敏感） | 初始化 Phase 2 GP 的维度权重 |
| **Output Scale** | 整体方差幅度 | Regression 模型的 kernel variance |
| **Mean Constant** | 潜在函数的平均值 | 初始化模型的 mean 函数 |
| **Noise Variance** | 观测噪声大小 | Regression 模型的噪声项 |

### 1.4 维度敏感度排序 (从 Lengthscale 推导)

```
ARD Weight = 1/lengthscale (归一化)

Rank | 维度 | Lengthscale | ARD Weight | 敏感度
-----|------|-------------|-----------|-------
  1  | x3   | 0.4367      | 53.8%     | 🔴 最敏感
  2  | x1   | 1.4659      | 16.0%     | 🟡 高
  3  | x0   | 1.9395      | 12.1%     | 🟡 中等
  4  | x5   | 3.6145      |  6.5%     | 🟢 低
  5  | x2   | 3.8931      |  6.0%     | 🟢 低
  6  | x4   | 4.2384      |  5.5%     | 🟢 低
```

---

## 2. 如何提取 BaseGP 参数并转换为 INI 配置

### 2.1 自动提取脚本

```python
import json
import numpy as np
import torch
import math

# 路径定义
base_path = r"F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202511271557\base_gp"

# 方法1: 从 JSON 文件提取 (最简单)
with open(f"{base_path}/base_gp_lengthscales.json", "r") as f:
    ls_data = json.load(f)
    lengthscales = np.array(ls_data['lengthscales'])

# 方法2: 从 state.pth 提取 (完整参数)
state = torch.load(f"{base_path}/base_gp_state.pth", map_location='cpu')

def softplus(x):
    return np.log(1 + np.exp(x))

# 提取所有参数
raw_ls = state['model']['covar_module.base_kernel.raw_lengthscale'].numpy().flatten()
lengthscales = softplus(raw_ls)

output_scale = softplus(state['model']['covar_module.raw_outputscale'].item())

mean = state['model']['mean_module.raw_constant'].item()

noise = softplus(state['likelihood']['noise_covar.raw_noise'].item())

# 计算 INI 配置参数
sigma = 0.5  # 选择的不确定性
ls_loc = np.log(lengthscales) + sigma**2

print(f"For INI [DefaultMeanCovarFactory]:")
print(f'ls_loc = [{", ".join([f"{v:.4f}" for v in ls_loc])}]')
print(f'ls_scale = [{"5, " * 5}0.5]')
print(f"\nFor INI [ConfigurableGaussianLikelihood]:")
print(f"noise_prior_rate = {(2.0 - 1) / noise:.4f}  # mode = {noise:.4f}")
print(f"noise_init = {noise:.4f}")
```

### 2.2 转换结果

```ini
# 直接复制到 INI 文件

[DefaultMeanCovarFactory]
lengthscale_prior = lognormal
ls_loc = [0.9124, 0.6325, 1.6092, -0.5785, 1.6942, 1.5350]
ls_scale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
fixed_kernel_amplitude = True    ; OrdinalGP 用

[ConfigurableGaussianLikelihood]
noise_prior_concentration = 2.0
noise_prior_rate = 1.228         ; 对应 mode = 0.814
noise_init = 0.814               ; Regression 用
```

---

## 3. OrdinalGPModel 配置 (含详细说明)

### 3.1 完整 INI 配置

```ini
[OrdinalGPModel]
n_levels = 5
mean_covar_factory = DefaultMeanCovarFactory

[DefaultMeanCovarFactory]
# Lengthscale 先验类型
lengthscale_prior = lognormal

# 向量形式: 来自 BaseGP 学习结果
# 计算方式: log(basegp_ls) + 0.5^2
ls_loc = [0.9124, 0.6325, 1.6092, -0.5785, 1.6942, 1.5350]
ls_scale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# OrdinalGP 不使用 ScaleKernel (序数模型的识别性要求)
fixed_kernel_amplitude = True
```

### 3.2 配置在哪里

**文件位置**: `F:\Github\aepsych-source\test\is_EUR_work\eur_config_basegp_prior.ini`

**相关代码**:

- 配置解析: `temp_aepsych/aepsych/factory/default.py` (line 380-390)
- 模型初始化: `temp_aepsych/aepsych/models/ordinal_gp.py` (line 18-125)

---

## 4. GPRegressionModel 配置 (含自定义 Likelihood)

### 4.1 完整 INI 配置

```ini
[GPRegressionModel]
likelihood = ConfigurableGaussianLikelihood
mean_covar_factory = DefaultMeanCovarFactory

[ConfigurableGaussianLikelihood]
# Noise 先验参数
noise_prior_concentration = 2.0
noise_prior_rate = 1.228
noise_init = 0.814

[DefaultMeanCovarFactory]
lengthscale_prior = lognormal
ls_loc = [0.9124, 0.6325, 1.6092, -0.5785, 1.6942, 1.5350]
ls_scale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

# Regression 使用 ScaleKernel
fixed_kernel_amplitude = False
outputscale_prior = gamma
```

### 4.2 自定义 Likelihood 代码位置

**应该放在哪里**: 你的项目的主模块或 utils 文件中

**建议位置**:

```
F:\Github\aepsych-source\test\is_EUR_work\
├── custom_likelihoods.py  ← 创建此文件
└── run_*.py               ← 在这里 import 并注册
```

**代码内容**:

```python
# File: custom_likelihoods.py
import gpytorch
from aepsych.config import Config, ConfigurableMixin

class ConfigurableGaussianLikelihood(
    gpytorch.likelihoods.GaussianLikelihood, 
    ConfigurableMixin
):
    """GaussianLikelihood with configurable noise prior.
    
    Source: BaseGP learned noise_variance = 0.814
    """
    
    def __init__(
        self,
        noise_prior_concentration: float = 2.0,
        noise_prior_rate: float = 1.228,
        noise_init: float = 0.814,
        **kwargs
    ):
        noise_prior = gpytorch.priors.GammaPrior(
            concentration=noise_prior_concentration,
            rate=noise_prior_rate
        )
        noise_constraint = gpytorch.constraints.GreaterThan(
            1e-4, 
            initial_value=noise_init
        )
        super().__init__(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            **kwargs
        )

# 在模块初始化时注册
Config.register_object(ConfigurableGaussianLikelihood)
```

**在脚本中使用**:

```python
# File: run_regression_with_basegp.py
from custom_likelihoods import ConfigurableGaussianLikelihood
from aepsych.config import Config

# ConfigurableGaussianLikelihood 已自动注册
config = Config()
config.read('your_config.ini')
```

---

## 5. 核函数对比: BaseGP vs AEPsych 默认

### 5.1 核函数差异

| 特性 | BaseGP (Phase 1) | AEPsych 默认 | 影响 |
|------|------------------|-----------|------|
| **核函数** | Matern(ν=2.5) | RBF | 光滑度不同 |
| **ARD** | ✅ 6 维独立 | ✅ 支持 | 可配置 |
| **Scale** | ✅ ScaleKernel | ✅ 可选 | 取决于模型 |
| **参数转移** | ✅ 直接转移 ls | ✅ via INI | Lengthscale 兼容 |

### 5.2 为什么 BaseGP 用 Matern?

BaseGP 使用 Matern(2.5) 而非 RBF 的原因:

```
RBF:       函数无限光滑 (℃^∞)
           → 过度假设，可能过度正则化
           → 对噪声数据不稳健

Matern(2.5): 函数二阶可微 (℃²)
            → 现实中大多数物理过程的光滑度
            → 在噪声数据下表现更好

参考: Rasmussen & Williams, GPML, Section 4.2
```

### 5.3 如何在 AEPsych 中使用 Matern?

**问题**: AEPsych 默认 RBF，不支持 INI 配置切换核函数

**解决方案**: 修改 `DefaultMeanCovarFactory` 源代码

```python
# File: temp_aepsych/aepsych/factory/default.py
# Line ~360, 修改默认核函数

# 原代码:
kernel = gpytorch.kernels.RBFKernel

# 改为:
kernel = gpytorch.kernels.MaternKernel  # ← 改这里
# 需要在 __init__ 中添加 matern_nu 参数
```

---

## 6. 关键文件路径快速参考

### 数据文件

```
F:\Github\aepsych-source\extensions\warmup_budget_check\phase1_analysis_output\202511271557\base_gp\
├── base_gp_report.md              ← 人工可读的汇总报告
├── base_gp_lengthscales.json      ← JSON 格式的 lengthscales
├── base_gp_state.pth              ← PyTorch 完整模型状态
└── base_gp_key_points.json        ← Phase 2 推荐的 warmup 点
```

### 代码文件

```
F:\Github\aepsych-source\extensions\warmup_budget_check\core\phase1_step3_base_gp.py
  → 核函数定义 (line 108-110)
  → 训练逻辑 (line 123-170)

F:\Github\aepsych-source\temp_aepsych\aepsych\factory\default.py
  → DefaultMeanCovarFactory (line 89-233)
  → 配置解析逻辑 (line 380-410)

F:\Github\aepsych-source\temp_aepsych\aepsych\models\ordinal_gp.py
  → OrdinalGPModel 初始化 (line 18-88)

F:\Github\aepsych-source\temp_aepsych\aepsych\models\gp_regression.py
  → GPRegressionModel 初始化 (line 21-88)
```

### 配置文件

```
F:\Github\aepsych-source\test\is_EUR_work\eur_config_basegp_prior.ini
  → 当前 Phase 2 配置示例

F:\Github\aepsych-source\test\is_EUR_work\plans\20251128_2238\
  ├── BaseGP_Prior_Integration.md   ← 本文档
  └── README.md                      ← 快速索引
```

---

## 7. 执行步骤 (完整流程)

```
Step 1: 获取 BaseGP 参数
├── 位置: extensions/warmup_budget_check/phase1_analysis_output/202511271557/base_gp/
├── 文件: base_gp_lengthscales.json
└── 提取: lengthscales = [1.9395, 1.4659, ...]

Step 2: 计算 INI 配置值
├── 计算: ls_loc = log(lengthscales) + 0.5^2
└── 结果: [0.9124, 0.6325, 1.6092, -0.5785, 1.6942, 1.5350]

Step 3: 创建或修改 INI 配置
├── 文件: eur_config_basegp_prior.ini
├── [DefaultMeanCovarFactory] 部分
└── [ConfigurableGaussianLikelihood] 部分 (仅 Regression)

Step 4: 运行测试
├── OrdinalGPModel: 直接运行，无需额外代码
├── GPRegressionModel: 先定义 ConfigurableGaussianLikelihood
└── 验证: 检查 model.covar_module.lengthscale 是否 ≈ [1.94, 1.47, ...]
```

---

## 8. 故障排查

### 问题: Lengthscale 初始值不对

```
预期: [1.9395, 1.4659, 3.8931, 0.4367, 4.2384, 3.6145]
实际: [其他值]

原因: ls_loc 计算错误
解决:
  1. 检查 base_gp_lengthscales.json 值是否正确
  2. 验证计算: ls_loc = log(basegp_ls) + 0.25
  3. 确保 INI 中的 ls_loc 有 6 个值，与维度匹配
```

### 问题: Noise variance 无法设置 (Ordinal)

```
原因: OrdinalLikelihood 不包含噪声参数
解决: 
  - OrdinalGPModel 无法设置 noise_variance
  - 若需要，使用 GPRegressionModel + ConfigurableGaussianLikelihood
```

### 问题: ConfigurableGaussianLikelihood 未被识别

```
原因: 未注册到 Config
解决:
  1. 确保代码中有: Config.register_object(ConfigurableGaussianLikelihood)
  2. 确保在 Config.read() 前调用
  3. 检查 INI 中 [GPRegressionModel] likelihood = ConfigurableGaussianLikelihood 写法
```

---

**最后更新**: 2025-11-28 22:50  
**涵盖版本**: AEPsych (temp_aepsych/)  
**验证状态**: ✅ 已通过 API 测试验证
