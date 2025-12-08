# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.269 | 6.686e-01 | 0.718 |
| 25 | 1.104 | 3.681e-01 | 1.494 |
| 50 | 1.067 | 4.249e-01 | 2.210 |
| 75 | 1.060 | 4.266e-01 | 2.599 |
| 100 | 1.057 | 4.228e-01 | 2.870 |
| 125 | 1.055 | 4.212e-01 | 3.116 |
| 150 | 1.053 | 4.205e-01 | 3.348 |
| 175 | 1.052 | 4.201e-01 | 3.567 |
| 200 | 1.051 | 4.198e-01 | 3.773 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x6_InnerFurniture | 2.2573 | 高敏感 (变化小即影响大) |
| 2 | x2_GridModule | 2.5979 | 高敏感 (变化小即影响大) |
| 3 | x1_CeilingHeight | 3.6032 | 中等 |
| 4 | x5_PhysicalBoundary | 4.2363 | 中等 |
| 5 | x4_VisualBoundary | 4.8224 | 低敏感 |
| 6 | x3_OuterFurniture | 5.1187 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 1.000 | 0.000 | 1.617 | 25 |
| subject_2 | 3.720 | 0.960 | 0.960 | 25 |
| subject_3 | 1.960 | 0.958 | 0.958 | 25 |
| subject_4 | 5.000 | 0.000 | 1.617 | 25 |
| subject_5 | 4.000 | 0.849 | 0.849 | 25 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 0.607 (Std = 0.674)
- **Coordinates**: [2.8, 8.0, 1.0, 0.0, 0.0, 2.0]
- **Detailed**: x1_CeilingHeight=2.8, x2_GridModule=8.0, x3_OuterFurniture=1.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=0.0, x6_InnerFurniture=2.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -0.832 (Std = 0.681)
- **Coordinates**: [8.5, 6.5, 1.0, 2.0, 1.0, 0.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=6.5, x3_OuterFurniture=1.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=0.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.689 (Mean = -0.235)
- **Coordinates**: [8.5, 6.5, 2.0, 0.0, 0.0, 2.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=6.5, x3_OuterFurniture=2.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=0.0, x6_InnerFurniture=2.0


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
