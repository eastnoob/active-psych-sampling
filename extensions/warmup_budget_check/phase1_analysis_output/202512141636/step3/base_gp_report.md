# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.281 | 6.686e-01 | 0.718 |
| 25 | 1.116 | 3.423e-01 | 1.413 |
| 50 | 1.087 | 4.382e-01 | 1.960 |
| 75 | 1.076 | 4.215e-01 | 2.376 |
| 100 | 1.070 | 4.198e-01 | 2.702 |
| 125 | 1.067 | 4.212e-01 | 2.966 |
| 150 | 1.065 | 4.218e-01 | 3.194 |
| 175 | 1.063 | 4.221e-01 | 3.397 |
| 200 | 1.062 | 4.224e-01 | 3.582 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x4_VisualBoundary | 2.3220 | 高敏感 (变化小即影响大) |
| 2 | x3_OuterFurniture | 2.4791 | 高敏感 (变化小即影响大) |
| 3 | x5_PhysicalBoundary | 3.3502 | 中等 |
| 4 | x2_GridModule | 3.3677 | 中等 |
| 5 | x6_InnerFurniture | 3.6464 | 低敏感 |
| 6 | x1_CeilingHeight | 6.3285 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 1.366 | 0.717 | 0.717 | 71 |
| subject_2 | 3.577 | 1.401 | 1.401 | 71 |
| subject_3 | 2.423 | 1.350 | 1.350 | 71 |
| subject_4 | 5.000 | 0.000 | 1.677 | 71 |
| subject_5 | 4.597 | 0.811 | 0.811 | 72 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 1.634 (Std = 0.678)
- **Coordinates**: [4.0, 6.5, 2.0, 2.0, 0.0, 0.0]
- **Detailed**: x1_CeilingHeight=4.0, x2_GridModule=6.5, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=0.0, x6_InnerFurniture=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -1.361 (Std = 0.675)
- **Coordinates**: [8.5, 8.0, 0.0, 0.0, 1.0, 1.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=0.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=1.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.688 (Mean = 0.658)
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
