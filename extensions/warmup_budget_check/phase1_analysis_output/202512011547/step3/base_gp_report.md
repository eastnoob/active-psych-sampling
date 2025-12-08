# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.300 | 6.686e-01 | 0.718 |
| 25 | 1.162 | 3.499e-01 | 1.437 |
| 50 | 1.136 | 4.384e-01 | 1.998 |
| 75 | 1.123 | 4.292e-01 | 2.470 |
| 100 | 1.117 | 4.274e-01 | 2.846 |
| 125 | 1.112 | 4.288e-01 | 3.162 |
| 150 | 1.109 | 4.295e-01 | 3.439 |
| 175 | 1.107 | 4.299e-01 | 3.687 |
| 200 | 1.105 | 4.301e-01 | 3.914 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x4_VisualBoundary | 2.0050 | 高敏感 (变化小即影响大) |
| 2 | x3_OuterFurniture | 2.6468 | 高敏感 (变化小即影响大) |
| 3 | x2_GridModule | 3.6532 | 中等 |
| 4 | x6_InnerFurniture | 4.1907 | 中等 |
| 5 | x5_PhysicalBoundary | 4.2154 | 低敏感 |
| 6 | x1_CeilingHeight | 6.7699 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 1.433 | 0.844 | 0.844 | 30 |
| subject_2 | 3.733 | 1.459 | 1.459 | 30 |
| subject_3 | 2.733 | 1.263 | 1.263 | 30 |
| subject_4 | 5.000 | 0.000 | 1.636 | 30 |
| subject_5 | 4.400 | 0.987 | 0.987 | 30 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 1.374 (Std = 0.705)
- **Coordinates**: [4.0, 6.5, 2.0, 2.0, 1.0, 0.0]
- **Detailed**: x1_CeilingHeight=4.0, x2_GridModule=6.5, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -1.306 (Std = 0.710)
- **Coordinates**: [8.5, 8.0, 0.0, 0.0, 1.0, 2.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=0.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=2.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.731 (Mean = -0.070)
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
