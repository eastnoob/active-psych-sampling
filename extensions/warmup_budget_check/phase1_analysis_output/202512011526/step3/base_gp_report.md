# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.329 | 6.686e-01 | 0.718 |
| 25 | 1.208 | 4.497e-01 | 1.461 |
| 50 | 1.184 | 4.800e-01 | 2.069 |
| 75 | 1.172 | 4.871e-01 | 2.534 |
| 100 | 1.165 | 4.890e-01 | 2.923 |
| 125 | 1.161 | 4.896e-01 | 3.243 |
| 150 | 1.158 | 4.898e-01 | 3.522 |
| 175 | 1.156 | 4.899e-01 | 3.773 |
| 200 | 1.155 | 4.899e-01 | 4.001 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x4_VisualBoundary | 1.8586 | 高敏感 (变化小即影响大) |
| 2 | x3_OuterFurniture | 2.4018 | 高敏感 (变化小即影响大) |
| 3 | x5_PhysicalBoundary | 3.8510 | 中等 |
| 4 | x6_InnerFurniture | 4.3672 | 中等 |
| 5 | x2_GridModule | 4.4160 | 低敏感 |
| 6 | x1_CeilingHeight | 7.1114 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 1.500 | 1.025 | 1.025 | 30 |
| subject_2 | 3.767 | 1.627 | 1.627 | 30 |
| subject_3 | 2.767 | 1.521 | 1.521 | 30 |
| subject_4 | 5.000 | 0.000 | 1.723 | 30 |
| subject_5 | 4.667 | 0.789 | 0.789 | 30 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 1.378 (Std = 0.755)
- **Coordinates**: [8.5, 6.5, 2.0, 2.0, 1.0, 0.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=6.5, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -1.175 (Std = 0.749)
- **Coordinates**: [8.5, 8.0, 0.0, 0.0, 1.0, 2.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=0.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=2.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.767 (Mean = 0.114)
- **Coordinates**: [8.5, 8.0, 0.0, 2.0, 0.0, 0.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=0.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=0.0, x6_InnerFurniture=0.0


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
