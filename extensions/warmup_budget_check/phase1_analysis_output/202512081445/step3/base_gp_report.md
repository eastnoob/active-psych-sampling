# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.357 | 6.686e-01 | 0.718 |
| 25 | 1.252 | 4.893e-01 | 1.261 |
| 50 | 1.230 | 5.455e-01 | 1.781 |
| 75 | 1.222 | 5.626e-01 | 2.196 |
| 100 | 1.217 | 5.648e-01 | 2.524 |
| 125 | 1.213 | 5.657e-01 | 2.794 |
| 150 | 1.211 | 5.665e-01 | 3.027 |
| 175 | 1.209 | 5.672e-01 | 3.232 |
| 200 | 1.208 | 5.677e-01 | 3.417 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x4_VisualBoundary | 1.2976 | 高敏感 (变化小即影响大) |
| 2 | x3_OuterFurniture | 2.3667 | 高敏感 (变化小即影响大) |
| 3 | x6_InnerFurniture | 3.3651 | 中等 |
| 4 | x5_PhysicalBoundary | 3.6479 | 中等 |
| 5 | x2_GridModule | 4.3414 | 低敏感 |
| 6 | x1_CeilingHeight | 5.4818 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 2.592 | 0.492 | 0.492 | 71 |
| subject_2 | 3.225 | 0.481 | 0.481 | 71 |
| subject_3 | 2.901 | 0.449 | 0.449 | 71 |
| subject_4 | 3.930 | 0.256 | 0.256 | 71 |
| subject_5 | 3.583 | 0.493 | 0.493 | 72 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 1.366 (Std = 0.788)
- **Coordinates**: [2.8, 6.5, 2.0, 2.0, 0.0, 0.0]
- **Detailed**: x1_CeilingHeight=2.8, x2_GridModule=6.5, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=0.0, x6_InnerFurniture=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -1.588 (Std = 0.785)
- **Coordinates**: [4.0, 6.5, 0.0, 0.0, 1.0, 2.0]
- **Detailed**: x1_CeilingHeight=4.0, x2_GridModule=6.5, x3_OuterFurniture=0.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=2.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.796 (Mean = 0.753)
- **Coordinates**: [8.5, 8.0, 2.0, 2.0, 1.0, 0.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=0.0


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
