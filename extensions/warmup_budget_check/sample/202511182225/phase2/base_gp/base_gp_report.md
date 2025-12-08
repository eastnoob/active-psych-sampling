# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.423 | 6.686e-01 | 0.718 |
| 25 | 1.350 | 7.235e-01 | 1.442 |
| 50 | 1.337 | 7.450e-01 | 2.025 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x2_5level_discrete | 1.6440 | 高敏感 (变化小即影响大) |
| 2 | x1_binary | 1.9636 | 高敏感 (变化小即影响大) |
| 3 | x6_binary | 2.0084 | 中等 |
| 4 | x5_3level_categorical | 2.4065 | 中等 |
| 5 | x3_5level_decimal | 2.4519 | 低敏感 |
| 6 | x4_4level_categorical | 2.6081 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| 1 | 3.640 | 1.196 | 1.196 | 25 |
| 2 | 3.000 | 1.356 | 1.356 | 25 |
| 3 | 1.840 | 0.967 | 0.967 | 25 |
| 4 | 4.680 | 0.786 | 0.786 | 25 |
| 5 | 4.640 | 0.625 | 0.625 | 25 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 0.660 (Std = 0.900)
- **Coordinates**: [0.0, 5.0, 0.0, 1.0, 1.0, 0.0]
- **Detailed**: x1_binary=0.0, x2_5level_discrete=5.0, x3_5level_decimal=0.0, x4_4level_categorical=1.0, x5_3level_categorical=1.0, x6_binary=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -0.891 (Std = 0.895)
- **Coordinates**: [0.0, 1.0, 0.0, 2.0, 2.0, 1.0]
- **Detailed**: x1_binary=0.0, x2_5level_discrete=1.0, x3_5level_decimal=0.0, x4_4level_categorical=2.0, x5_3level_categorical=2.0, x6_binary=1.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.924 (Mean = -0.365)
- **Coordinates**: [1.0, 1.0, 0.0, 3.0, 0.0, 1.0]
- **Detailed**: x1_binary=1.0, x2_5level_discrete=1.0, x3_5level_decimal=0.0, x4_4level_categorical=3.0, x5_3level_categorical=0.0, x6_binary=1.0


## 🧪 使用示例
```python
import torch, json, gpytorch
from phase1_step3_base_gp import _MaternARDGP
# 加载 state_dict
state = torch.load('base_gp_state.pth', map_location='cpu')
# 重建模型 (需知道输入维度)
D = 6
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = _MaternARDGP(torch.zeros(1, D), torch.zeros(1), likelihood)
model.load_state_dict(state['model'])
likelihood.load_state_dict(state['likelihood'])
model.eval(); likelihood.eval()
# 预测
with torch.no_grad():
    x = torch.randn(5, D)
    pred = likelihood(model(x))
    print(pred.mean, pred.stddev)
```

*自动生成*
