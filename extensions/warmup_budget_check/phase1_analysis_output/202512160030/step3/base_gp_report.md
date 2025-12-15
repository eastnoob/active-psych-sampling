# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.301 | 6.686e-01 | 0.718 |
| 25 | 1.171 | 3.859e-01 | 1.421 |
| 50 | 1.145 | 4.312e-01 | 2.102 |
| 75 | 1.133 | 4.326e-01 | 2.651 |
| 100 | 1.125 | 4.316e-01 | 3.094 |
| 125 | 1.120 | 4.307e-01 | 3.447 |
| 150 | 1.117 | 4.299e-01 | 3.741 |
| 175 | 1.115 | 4.295e-01 | 3.998 |
| 200 | 1.114 | 4.292e-01 | 4.229 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x4_VisualBoundary | 1.6573 | 高敏感 (变化小即影响大) |
| 2 | x3_OuterFurniture | 3.3388 | 高敏感 (变化小即影响大) |
| 3 | x5_PhysicalBoundary | 3.4688 | 中等 |
| 4 | x6_InnerFurniture | 3.7269 | 中等 |
| 5 | x2_GridModule | 4.7738 | 低敏感 |
| 6 | x1_CeilingHeight | 8.4054 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 1.350 | 0.792 | 0.792 | 20 |
| subject_2 | 3.250 | 1.699 | 1.699 | 20 |
| subject_3 | 2.200 | 1.249 | 1.249 | 20 |
| subject_4 | 5.000 | 0.000 | 1.775 | 20 |
| subject_5 | 4.700 | 0.900 | 0.900 | 20 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 1.367 (Std = 0.726)
- **Coordinates**: [2.8, 6.5, 2.0, 2.0, 1.0, 0.0]
- **Detailed**: x1_CeilingHeight=2.8, x2_GridModule=6.5, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -1.057 (Std = 0.693)
- **Coordinates**: [8.5, 8.0, 0.0, 0.0, 1.0, 1.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=0.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=1.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.753 (Mean = 0.830)
- **Coordinates**: [8.5, 8.0, 2.0, 2.0, 0.0, 0.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=0.0, x6_InnerFurniture=0.0


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
