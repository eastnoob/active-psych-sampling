# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.471 | 7.186e-01 | 0.710 |
| 25 | 1.399 | 8.066e-01 | 1.240 |
| 50 | 1.383 | 8.208e-01 | 1.670 |
| 75 | 1.380 | 8.164e-01 | 1.981 |
| 100 | 1.378 | 8.155e-01 | 2.234 |
| 125 | 1.377 | 8.145e-01 | 2.431 |
| 150 | 1.377 | 8.140e-01 | 2.598 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x4_4level_categorical | 0.4367 | 高敏感 (变化小即影响大) |
| 2 | x2_5level_discrete | 1.4659 | 高敏感 (变化小即影响大) |
| 3 | x1_binary | 1.9395 | 中等 |
| 4 | x6_binary | 3.6145 | 中等 |
| 5 | x3_5level_decimal | 3.8931 | 低敏感 |
| 6 | x5_3level_categorical | 4.2384 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 2.040 | 0.824 | 0.824 | 25 |
| subject_2 | 3.000 | 1.265 | 1.265 | 25 |
| subject_3 | 2.120 | 0.863 | 0.863 | 25 |
| subject_4 | 4.160 | 0.674 | 0.674 | 25 |
| subject_5 | 4.440 | 0.637 | 0.637 | 25 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 0.766 (Std = 0.942)
- **Coordinates**: [0.0, 5.0, 0.5, 1.0, 0.0, 0.0]
- **Detailed**: x1_binary=0.0, x2_5level_discrete=5.0, x3_5level_decimal=0.5, x4_4level_categorical=1.0, x5_3level_categorical=0.0, x6_binary=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -0.721 (Std = 0.935)
- **Coordinates**: [0.0, 1.0, 0.5, 2.0, 1.0, 0.0]
- **Detailed**: x1_binary=0.0, x2_5level_discrete=1.0, x3_5level_decimal=0.5, x4_4level_categorical=2.0, x5_3level_categorical=1.0, x6_binary=0.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.966 (Mean = 0.236)
- **Coordinates**: [1.0, 5.0, 1.0, 3.0, 0.0, 0.0]
- **Detailed**: x1_binary=1.0, x2_5level_discrete=5.0, x3_5level_decimal=1.0, x4_4level_categorical=3.0, x5_3level_categorical=0.0, x6_binary=0.0


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
