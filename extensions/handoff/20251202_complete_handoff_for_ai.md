# BaseGPResidualMixedFactory 交接文档 | 20251202

## 🎯 任务本质 (60字)

为AEPsych残差学习框架添加**混合参数支持**：

- 现状：只支持连续参数 (Matern核)
- 需求：支持连续+离散参数 (ProductKernel: Matern × Categorical)
- 工作量：28小时，6阶段
- 验收标准：向后兼容 + 15+单元测试通过 + >85%覆盖率

---

## 📊 决策矩阵 (已验证)

| 决策项 | 方案 | 理由 | 验证 |
|--------|------|------|------|
| Mean模式 | pure_residual(默认) | 参数效率最优 | ✅ 分析+测试 |
| Mean模式 | learned_offset(可选) | 约束可学习 | ✅ 分析+测试 |
| 离散核 | 单CategoricalKernel | 参数省 + ARD独立 | ✅ 运行验证 |
| 离散核 | 否ProductKernel(IndexKernel×n) | 核值差37% | ❌ 不用 |
| 核组合 | ProductKernel(Matern,Cat) | 标准乘法 | ✅ Acquisition兼容 |
| 自定义核 | 否CategoricalARDKernel | 加法组合缺理论 | ❌ 不实现 |
| 每维ARD | ✅支持 | botorch原生 | ✅ 实测验证 |

---

## 📁 核心改动 (极简)

### 修改 (2个文件)

```
extensions/custom_mean.py
  + class MeanWithOffsetPrior(nn.Module)  [新类，1参数]
  + 保留BaseGPPriorMean不变 [兼容]

extensions/custom_factory.py
  + 参数: mean_type="pure_residual"|"learned_offset"
  + 参数: offset_prior_std=0.10
  + 修改_make_mean_module()分发逻辑
```

### 新建 (8个文件)

```
extensions/custom_factory_mixed.py [280行]
  class BaseGPResidualMixedFactory(MeanCovarFactory)
    __init__: continuous_params, discrete_params, mean_type, ...
    _make_mean_module(): 复用阶段1逻辑
    _make_covar_module(): 
      ├─ MaternKernel(ard_num_dims=len(continuous))
      ├─ CategoricalKernel(ard_num_dims=len(discrete))
      └─ ProductKernel组合 + ScaleKernel包装
    _get_active_dims_continuous()
    _get_active_dims_discrete()

extensions/test_custom_mixed.py [350行，15+个测试]
  Mean测试: 4个 (初始化×2 + forward + 梯度)
  工厂测试: 4个 (纯连续 + 纯离散 + 混合 + 参数计数)
  前向测试: 5个 (连续/离散/混合数据 + 形状 + 梯度)
  集成测试: 3个 (训练步骤 + Likelihood + Acquisition)

extensions/config_residual_pure_continuous.ini
extensions/config_residual_learned_continuous.ini
extensions/config_residual_pure_mixed.ini
extensions/config_residual_learned_mixed.ini
  [模板] mean_type选择 + 参数指定 + 先验配置

extensions/README_MIXED_RESIDUAL.md [800字]
  概览 + API参考 + 示例代码 + 迁移指南 + FAQ + 故障排查
```

---

## 🔧 实现细节 (必读)

### 参数流向

```
INI配置 
  ↓
Factory.__init__(continuous_params, discrete_params, mean_type, ...)
  ├─ _make_mean_module()
  │  └─ pure_residual → BaseGPPriorMean(0参)
  │  └─ learned_offset → MeanWithOffsetPrior(1参)
  │
  └─ _make_covar_module()
     ├─ continuous_kernel = MaternKernel(ard_num_dims=len(cont))
     ├─ discrete_kernel = CategoricalKernel(ard_num_dims=len(disc))
     └─ return ProductKernel(continuous_kernel, discrete_kernel)

总参数数 = continuous_ard + discrete_ard + offset(可选) + outputscale
```

### 维度映射约定 (必须遵守)

```
train_X形状: (n_batch, n_dims)
  前 len(continuous_params) 维: 连续值
  后 len(discrete_params) 维: 整数0-indexed (0到n_cat-1)

示例: continuous=['dur','freq'], discrete=['intensity','color']
  dim0: dur (连续) → active_dims=[0]
  dim1: freq (连续) → active_dims=[1]
  dim2: intensity (0/1/2) → active_dims=[2]
  dim3: color (0/1) → active_dims=[3]
```

### 关键参数值

```
mean_type = "pure_residual" (默认) | "learned_offset"
offset_prior_std = 0.10  [N(0, 0.10²)先验]
discrete_kernel = "categorical" (推荐) | "index" (备选)
lengthscale_prior: LogNormal(μ=log(basegp_ls)-log(d)/2+σ², σ²=0.1²)
noise_prior: GammaPrior(2.0, 1.228) [mode≈0.814，可用户调]
```

---

## 📈 实现路线 (6阶段)

| 阶段 | 任务 | 时间 | 依赖 | 输出 |
|------|------|------|------|------|
| 1 | Mean扩展 (add MeanWithOffsetPrior) | 6h | 无 | custom_mean/factory修改 |
| 2 | 混合工厂 (ProductKernel逻辑) | 8h | 1 | custom_factory_mixed.py |
| 3 | 单元测试 (15+覆盖>85%) | 6h | 2 | test_custom_mixed.py |
| 4 | 配置系统 (4个INI示例) | 4h | 3 | 4个config文件 |
| 5 | 文档编写 (README + 决策记录) | 3h | 4 | README_MIXED_RESIDUAL.md |
| 6 | 最终验证 (回归+性能检查) | 1h | 5 | 通过检查清单 |

**总计**: 28小时 / **关键路径**: 1→2→3→(4平行)→5→6

---

## ✅ 验收清单

### 代码

- [ ] 所有15+单元测试通过 (`pytest extensions/test_custom_mixed.py`)
- [ ] 测试覆盖率>85% (`coverage`)
- [ ] 无编译/运行时警告
- [ ] 参数计数正确: cont_ard + disc_ard + offset(opt) + 1scale

### 功能

- [ ] Mean两种模式正常工作
- [ ] ProductKernel正确组合 (乘法，非加法)
- [ ] 4个配置示例都能初始化+前向传播
- [ ] 梯度反向正常 (检查backward())

### 兼容性

- [ ] BaseGPResidualFactory旧行为不变 (mean_type默认="pure_residual")
- [ ] 旧配置文件无需修改即可运行
- [ ] acquisition函数可调用 (已验证ProductKernel兼容)

### 文档

- [ ] README_MIXED_RESIDUAL.md完整 (>800字含示例)
- [ ] API文档清晰 (**init**, _make_mean_module,_make_covar_module)
- [ ] 迁移指南明确 (如何从BaseGPResidual升级)

---

## 🔴 已拒绝的方案 (不要重复)

| 方案 | 为什么拒绝 | 证据 |
|------|----------|------|
| **自定义CategoricalARDKernel** (加法组合) | 核值差39% + 无理论依据 | `analyze_categorical_ard_clean.py` 实测 |
| **ProductKernel(IndexKernel×n)** | 核值差37% + 参数管理复杂 | `test_categorical_modes.py` 验证 |
| **为每个离散参数单独一个kernel** | 与CategoricalKernel的ARD功能重复 | botorch实测 |

---

## 📞 关键文档位置

| 文档 | 路径 | 用途 |
|------|------|------|
| **执行计划详细版** | DETAILED_EXECUTION_PLAN.txt | 任务分解参考 |
| **快速总览** | IMPLEMENTATION_QUICK_OVERVIEW.md | 5分钟速览 |
| **离散ARD验证** | DISCRETE_ARD_PER_DIMENSION.md | 理论支撑 |
| **CategoricalARD拒却** | CATEGORICAL_ARD_DECISION_RECORD.md | 为什么不用 |
| **全面分析** | FINAL_ANALYSIS_SUMMARY.md | 完整背景 |
| **验证脚本** | verify_discrete_ard_per_dim.py, analyze_categorical_ard_clean.py | 跑一遍看结果 |

---

## 🚀 快速开始

### 0. 理解基础

```bash
# 运行验证脚本，理解核值差异
pixi run python verify_discrete_ard_per_dim.py      # 每维ARD验证
pixi run python analyze_categorical_ard_clean.py    # 加法vs乘法对比
```

### 1. 启动阶段1 (6小时)

```python
# extensions/custom_mean.py 添加类
class MeanWithOffsetPrior(nn.Module):
    def __init__(self, basemodel, csv, offset_prior_std=0.10):
        self.base_mean = BaseGPPriorMean(...)
        self.register_parameter("offset", Parameter(torch.zeros(1)))
        # 设置N(0, offset_prior_std²)先验

# extensions/custom_factory.py 修改
class BaseGPResidualFactory:
    def __init__(self, ..., mean_type="pure_residual", offset_prior_std=0.10):
        ...
    def _make_mean_module(self, train_X):
        if self.mean_type == "pure_residual":
            return BaseGPPriorMean(...)
        elif self.mean_type == "learned_offset":
            return MeanWithOffsetPrior(...)
```

### 2. 启动阶段2 (8小时)

```python
# extensions/custom_factory_mixed.py
class BaseGPResidualMixedFactory(MeanCovarFactory):
    def __init__(self, continuous_params=None, discrete_params=None, ...):
        self.continuous_params = continuous_params or []
        self.discrete_params = discrete_params or {}
    
    def _make_covar_module(self, train_X):
        kernels = []
        if self.continuous_params:
            k_cont = MaternKernel(nu=2.5, ard_num_dims=len(self.continuous_params))
            kernels.append(k_cont)
        if self.discrete_params:
            k_disc = CategoricalKernel(ard_num_dims=len(self.discrete_params))
            kernels.append(k_disc)
        return ScaleKernel(ProductKernel(*kernels))
```

### 3. 启动阶段3 (6小时)

```python
# extensions/test_custom_mixed.py
def test_mean_learned_offset():
    mean = MeanWithOffsetPrior(basemodel, csv, offset_prior_std=0.10)
    params = list(mean.parameters())
    assert len(params) == 1
    assert params[0].shape == torch.Size([1])

def test_factory_mixed():
    factory = BaseGPResidualMixedFactory(
        continuous_params=['x1', 'x2'],
        discrete_params={'color': ['r','g','b']},
        mean_type="learned_offset"
    )
    model = factory.build_model(train_X, train_Y)
    # 验证: 参数数 = 2(cont_ard) + 1(disc_ard) + 1(offset) + 1(scale) = 5
```

---

## 🎓 技术备注

### Why ProductKernel (乘法)?

- 标准做法：Matern各维独立，Categorical各维独立，最终乘积
- 数学基础：独立性假设 (GP多输出)
- 已验证：与所有acquisition函数兼容 (lookahead, MI等)

### Why 单CategoricalKernel而非多个IndexKernel?

- 参数数相同 (n个离散维 → n个lengthscale)
- 但管理更简单 (1个kernel vs n个kernel)
- 核值更符合直觉 (K[0,1]=0.513 for ls=0.5单索引)

### Why learned_offset是可选的?

- 30样本预算下，固定mean(0参) + 大GP 优于 可学习mean(1参) + 小GP
- 但某些场景可能需要，所以提供选项
- 默认仍是pure_residual保证性能

---

## 🔗 相关代码参考

### AEPsych源码

- `temp_aepsych/aepsych/factory/mixed.py` - MixedMeanCovarFactory (参考继承)
- `temp_aepsych/aepsych/factory/default.py` - DefaultMeanCovarFactory (参考实现)
- `temp_aepsych/aepsych/acquisition/lookahead.py` - 验证ProductKernel兼容

### 我们的实现

- `extensions/custom_factory.py` - BaseGPResidualFactory (要修改)
- `extensions/custom_mean.py` - BaseGPPriorMean (要扩展)

---

## 📝 问题排查 (常见坑)

| 问题 | 症状 | 解决 |
|------|------|------|
| 维度映射错误 | 前向传播报错/结果错误 | 检查active_dims是否连续无重合 |
| 参数初始化崩溃 | NaN/Inf | 检查lengthscale先验是否过宽 |
| 向后兼容破坏 | 旧代码报参数不认识 | 确保mean_type有默认值"pure_residual" |
| Acquisition失效 | qEI/qKG无法优化 | 验证Posterior支持ProductKernel (应该的) |
| 离散参数编码错误 | 核值异常高/低 | 确保离散值是0-indexed整数 |

---

## 📊 对标指标 (性能参考)

当前 BaseGPResidualFactory (纯连续):

- 参数数: n_cont_ard + 1_outputscale
- 训练速度: ~50ms/batch (30samples, 2D)
- 收敛稳定: ✅

新 BaseGPResidualMixedFactory (混合):

- 参数数: n_cont_ard + n_disc_ard + 1_outputscale (+ 1_offset可选)
- 期望速度: ~55ms/batch (参数多1-2个) ← 可接受
- 期望稳定: ✅ (ProductKernel已验证)

---

## ⚡ 时间优化建议

如果赶时间:

1. **跳过**阶段4 (配置系统) - 代码工作，配置只是便利
2. **精简**阶段5 (文档) - 只写API doc，README可延后
3. **合并**阶段3和6 - 单元测试本身就是回归测试

**最小可行** = 阶段1 + 2 + 3 = 20小时

---

**交接时间**: 2025-12-02  
**优化程度**: ★★★★★ (极致精简但完整)  
**适合目标**: AI模型接手实现  
**预期消耗token**: <3000 (此文档)
