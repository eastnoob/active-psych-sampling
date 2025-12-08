# Base GP (Matern 2.5 + ARD) 报告

## 📐 模型结构
- Kernel: Matern(ν=2.5) + ARD + Scale
- 输入维度: 6
- 设备: cpu

## 🔧 训练摘要
| Iter | Loss | Noise | Mean Lengthscale |
|------|------|-------|------------------|
| 1 | 1.374 | 6.686e-01 | 0.718 |
| 25 | 1.272 | 4.694e-01 | 1.337 |
| 50 | 1.240 | 4.802e-01 | 1.868 |
| 75 | 1.223 | 4.931e-01 | 2.259 |
| 100 | 1.213 | 4.960e-01 | 2.571 |
| 125 | 1.207 | 4.970e-01 | 2.835 |
| 150 | 1.202 | 4.975e-01 | 3.067 |
| 175 | 1.199 | 4.978e-01 | 3.275 |
| 200 | 1.196 | 4.981e-01 | 3.465 |

## 🎛️ 长度尺度 (Sensitivity)
| Rank | Factor | Lengthscale | Interpretation |
|------|--------|------------:|---------------|
| 1 | x4_VisualBoundary | 1.3694 | 高敏感 (变化小即影响大) |
| 2 | x5_PhysicalBoundary | 1.7425 | 高敏感 (变化小即影响大) |
| 3 | x3_OuterFurniture | 1.7970 | 中等 |
| 4 | x6_InnerFurniture | 4.3008 | 中等 |
| 5 | x2_GridModule | 4.3018 | 低敏感 |
| 6 | x1_CeilingHeight | 7.2792 | 低敏感 |

## 👥 被试标准化统计
| Subject | Mean | Std | Adjusted_Std_Used | N |
|---------|------|-----|-------------------|---|
| subject_1 | 1.667 | 1.135 | 1.135 | 30 |
| subject_2 | 3.500 | 1.628 | 1.628 | 30 |
| subject_3 | 2.833 | 1.572 | 1.572 | 30 |
| subject_4 | 4.900 | 0.396 | 0.396 | 30 |
| subject_5 | 4.400 | 0.879 | 0.879 | 30 |

## 📍 关键点 (设计空间) - 三个采样点
*供 Phase 2 直接使用的三个关键参数配方*

### 1️⃣ Sample 1 (Best Prior)
- **Score**: Mean = 1.529 (Std = 0.780)
- **Coordinates**: [8.5, 6.5, 2.0, 2.0, 1.0, 0.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=6.5, x3_OuterFurniture=2.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=0.0

### 2️⃣ Sample 2 (Worst Prior)
- **Score**: Mean = -1.759 (Std = 0.782)
- **Coordinates**: [2.8, 6.5, 0.0, 0.0, 1.0, 2.0]
- **Detailed**: x1_CeilingHeight=2.8, x2_GridModule=6.5, x3_OuterFurniture=0.0, x4_VisualBoundary=0.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=2.0

### 3️⃣ Sample 3 (Max Uncertainty / Center)
- **Score**: Std = 0.807 (Mean = -0.101)
- **Coordinates**: [8.5, 8.0, 0.0, 2.0, 1.0, 2.0]
- **Detailed**: x1_CeilingHeight=8.5, x2_GridModule=8.0, x3_OuterFurniture=0.0, x4_VisualBoundary=2.0, x5_PhysicalBoundary=1.0, x6_InnerFurniture=2.0


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
