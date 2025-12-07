# 交接文档：BaseGPResidualMixedFactory 实现方案

**日期**：2025-12-01  
**任务**：为 BaseGPResidualFactory 添加混合核支持（连续+离散）  
**目标完成**：实现通用混合核工厂，支持任意参数类型组合

---

## 📌 核心需求

### 功能目标

用户定义多个维度，每个维度指定其核类型，最终通过**乘法**组合：

$$K_{\text{final}}(x, x') = K_1(x_1, x'_1) \times K_2(x_2, x'_2) \times \cdots \times K_n(x_n, x'_n)$$

### 示例场景

```
维度0：连续 → Matern-2.5 核
维度1：离散(3类) → CategoricalKernel  
维度2：连续 → Matern-2.5 核
维度3：离散(2类) → CategoricalKernel

最终：K_Matern(x0) × K_Cat(x1) × K_Matern(x2) × K_Cat(x3)
```

---

## 🏗️ 架构设计

### 继承关系

```
BaseGPResidualFactory (基础残差工厂)
    ↓ 继承
BaseGPResidualMixedFactory (新增：支持混合核)
    │
    ├─ Mean: BaseGPPriorMean（从 BaseGP 预计算）
    └─ Covar: 混合核（维度级乘积）
         ├─ 维0: Matern-2.5（如果连续）
         ├─ 维1: CategoricalKernel（如果离散）
         ├─ 维2: ...
         └─ 最终组合方式: ProductKernel(K0, K1, K2, ...)
```

### 关键改动点

#### 1. 新文件创建

**路径**：`extensions/custom_factory/basegp_residual_mixed_factory.py`

**主要类**：`BaseGPResidualMixedFactory`

- 继承：`BaseGPResidualFactory`
- 新增参数：`kernel_dims` (dict[int, str])
  - key: 维度索引
  - value: 核类型 ('matern25', 'categorical', 'rbf', 'matern12', 'matern32')

**核心方法**：

```python
def _make_covar_module(self) -> gpytorch.kernels.Kernel:
    """
    根据 kernel_dims 为每个维度构造对应的核，
    最终通过 ProductKernel 乘积组合
    """
    kernels = []
    for dim_idx in sorted(self.kernel_dims.keys()):
        kernel_type = self.kernel_dims[dim_idx]
        k = self._make_kernel_for_dim(dim_idx, kernel_type)
        kernels.append(k)
    
    return gpytorch.kernels.ProductKernel(*kernels)
```

#### 2. 配置参数格式

**INI 配置**：

```ini
[BaseGPResidualMixedFactory]
basegp_scan_csv = extensions/.../design_space_scan.csv

# 【关键】维度-核类型映射（字典格式）
# 格式：kernel_dims = {维度索引: 核类型, ...}
kernel_dims = {0: "matern25", 1: "categorical", 2: "matern25", 3: "categorical"}

# 【可选】离散参数信息（从 Config 自动推断）
# discrete_params = {1: 3, 3: 2}  # 维1有3类，维3有2类（通常自动获取）

lengthscale_prior = lognormal
ls_loc = [0.0166, -0.2634, 0.7133, -1.4744, 0.7983, 0.6391]
ls_scale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
fixed_kernel_amplitude = False
outputscale_prior = gamma
```

#### 3. 自动参数推断

**从 Config 自动检测**：

```python
@classmethod
def get_config_args(cls, config, name=None):
    # 自动检测 categorical 维度
    par_names = config.getlist("common", "parnames", element_type=str)
    kernel_dims = {}
    discrete_params = {}
    
    for i, par_name in enumerate(par_names):
        par_type = config.get(par_name, "par_type")
        
        if par_type == "categorical":
            kernel_dims[i] = "categorical"
            choices = config.getlist(par_name, "choices", element_type=str)
            discrete_params[i] = len(choices)
        else:
            kernel_dims[i] = "matern25"  # 默认连续用 Matern-2.5
    
    return {
        "kernel_dims": kernel_dims,
        "discrete_params": discrete_params,
        ...其他参数...
    }
```

---

## 📂 关键文件位置

| 文件 | 作用 |
|------|------|
| `extensions/custom_factory/basegp_residual_factory.py` | 基础工厂（已存在） |
| `extensions/custom_factory/basegp_residual_mixed_factory.py` | **新增：混合核工厂** |
| `extensions/custom_factory/__init__.py` | 导出新工厂（需修改） |
| `extensions/custom_mean/basegp_prior_mean.py` | BaseGP mean（已存在） |
| `extensions/custom_likelihood/configurable_gaussian_likelihood.py` | 配置化 likelihood（已存在） |
| `temp_aepsych/aepsych/factory/mixed.py` | 原始 MixedMeanCovarFactory（参考） |
| `temp_aepsych/botorch/models/kernels/categorical.py` | CategoricalKernel 实现（参考） |

---

## 🔧 实现清单

### Phase 1：核心实现

- [ ] 创建 `basegp_residual_mixed_factory.py`
  - [ ] 类定义与初始化
  - [ ] `_make_covar_module()` 实现
  - [ ] 为每个维度构造对应的核（Matern、Categorical、RBF）
  - [ ] ProductKernel 组合

### Phase 2：配置集成

- [ ] 实现 `get_config_args()` 自动参数推断
- [ ] 支持从 INI 解析 `kernel_dims`
- [ ] 自动检测 `discrete_params`
- [ ] 更新 `extensions/custom_factory/__init__.py` 导出

### Phase 3：测试与文档

- [ ] 单元测试（直接实例化 + 核计算）
- [ ] INI 配置文件示例
- [ ] 集成测试（与 BaseGPPriorMean + ConfigurableGaussianLikelihood）
- [ ] 更新 `CUSTOM_COMPONENTS_README.md`

### Phase 4：清理

- [ ] 删除冗余文件（我创建的 CategoricalMixedFactory 系统）

---

## 💡 关键实现细节

### 1. 维度-核映射的构造

```python
def _make_kernel_for_dim(self, dim_idx: int, kernel_type: str) -> gpytorch.kernels.Kernel:
    """为单个维度构造对应的核"""
    
    if kernel_type == "categorical":
        # 需要从 discrete_params[dim_idx] 获取类别数
        num_categories = self.discrete_params[dim_idx]
        kernel = botorch.models.kernels.CategoricalKernel(
            active_dims=(dim_idx,),
            ard_num_dims=1,
            lengthscale_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
    
    elif kernel_type in ["matern25", "matern12", "matern32", "rbf"]:
        # 连续核
        nu_map = {"matern25": 2.5, "matern12": 1.2, "matern32": 3.2}
        nu = nu_map.get(kernel_type, 2.5)
        kernel = gpytorch.kernels.MaternKernel(
            nu=nu,
            active_dims=(dim_idx,),
            ard_num_dims=1,
            lengthscale_prior=...从 ls_loc/ls_scale 构造...
        )
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    # 可选：包裹 ScaleKernel
    if not self.fixed_kernel_amplitude:
        kernel = gpytorch.kernels.ScaleKernel(kernel)
    
    return kernel
```

### 2. 乘法组合

```python
# 所有核通过 ProductKernel 乘法组合
return gpytorch.kernels.ProductKernel(*kernels)

# 若需要整体 ScaleKernel，在外层包裹
if not self.fixed_kernel_amplitude:
    return gpytorch.kernels.ScaleKernel(final_product_kernel)
```

### 3. 参数推断逻辑

```
INI 中 [BaseGPResidualMixedFactory] 节点：
  ├─ 若有 kernel_dims 字段 → 直接使用（用户显式指定）
  └─ 若无 kernel_dims 字段 → 自动推断
        └─ 遍历 parnames，检查每个 par_type：
            ├─ categorical → kernel_dims[i] = "categorical"
            └─ continuous/integer → kernel_dims[i] = "matern25"（默认）
```

---

## 📋 INI 配置示例

### 完整示例

```ini
[common]
parnames = [learning_rate, optimizer_type, regularization, activation]
stimuli_per_trial = 1

[learning_rate]
par_type = continuous
lower_bound = 0.0001
upper_bound = 0.1

[optimizer_type]
par_type = categorical
choices = [adam, sgd, rmsprop]

[regularization]
par_type = continuous
lower_bound = 0.0
upper_bound = 1.0

[activation]
par_type = categorical
choices = [relu, tanh]

[GPRegressionModel]
mean_covar_factory = BaseGPResidualMixedFactory
likelihood = ConfigurableGaussianLikelihood
max_fit_time = 5.0

[BaseGPResidualMixedFactory]
basegp_scan_csv = extensions/warmup_budget_check/phase1_analysis_output/xxx/base_gp/design_space_scan.csv

# 自动推断模式（推荐）：不指定 kernel_dims，系统会自动检测
# 结果：{0: "matern25", 1: "categorical", 2: "matern25", 3: "categorical"}

# 或显式指定（仅当需要自定义时）
# kernel_dims = {0: "matern25", 1: "categorical", 2: "matern25", 3: "categorical"}

lengthscale_prior = lognormal
ls_loc = [0.0166, -0.2634, 0.7133, -1.4744]
ls_scale = [0.5, 0.5, 0.5, 0.5]
fixed_kernel_amplitude = False
outputscale_prior = gamma

[ConfigurableGaussianLikelihood]
noise_prior_concentration = 2.0
noise_prior_rate = 2.0
noise_init = 0.5
```

---

## 🔗 依赖关系与集成

### 已有组件（复用）

1. **BaseGPPriorMean**：从 BaseGP 查找表读取 mean
2. **ConfigurableGaussianLikelihood**：可配置 noise prior
3. **BaseGPResidualFactory**：基础框架（继承）

### 外部依赖

- `gpytorch.kernels`：MaternKernel、RBFKernel、ProductKernel、ScaleKernel
- `botorch.models.kernels`：CategoricalKernel
- `aepsych.config`：Config、ConfigurableMixin

### 注意事项

- CategoricalKernel **不支持** ARD（自动相关性判定），只在离散维上工作
- ProductKernel 中所有核通过**乘法**组合（无选项）
- 若需要加法或其他组合，另外扩展即可

---

## 🎯 交接要点

### 什么要做

1. ✅ 实现 `BaseGPResidualMixedFactory` 类
2. ✅ 维度-核类型映射与构造
3. ✅ 从 Config 自动推断参数
4. ✅ ProductKernel 乘法组合
5. ✅ 单元测试（至少覆盖：直接实例化、配置解析、核计算）
6. ✅ 更新导出与注册

### 什么不要做

- ❌ 不修改原有 MixedMeanCovarFactory（保持现状）
- ❌ 不创建 sum/sum_and_prod 等其他组合模式（乘法即可）
- ❌ 不改动 BaseGPPriorMean 或 ConfigurableGaussianLikelihood

### 代码参考

- **参考 MixedMeanCovarFactory**（`temp_aepsych/aepsych/factory/mixed.py`）了解参数推断模式
- **参考 BaseGPResidualFactory**（`extensions/custom_factory/basegp_residual_factory.py`）了解继承与配置化模式
- **参考 CategoricalKernel**（botorch 源码）了解离散核用法

---

## 📝 预期最终状态

### 文件结构

```
extensions/
  ├─ custom_factory/
  │   ├─ basegp_residual_factory.py        （基础，已存在）
  │   ├─ basegp_residual_mixed_factory.py  （新增）✨
  │   └─ __init__.py                        （修改：导出新工厂）
  ├─ custom_mean/
  │   └─ basegp_prior_mean.py              （已存在）
  ├─ custom_likelihood/
  │   └─ configurable_gaussian_likelihood.py （已存在）
  ├─ test_custom_components.py              （修改：添加测试）
  └─ CUSTOM_COMPONENTS_README.md            （修改：补充文档）
```

### 用户体验

```python
# 使用方式 1：INI 配置（推荐）
config = Config(config_fnames=["my_config.ini"])
factory = BaseGPResidualMixedFactory.from_config(config)

# 使用方式 2：直接 Python
factory = BaseGPResidualMixedFactory(
    dim=4,
    kernel_dims={0: "matern25", 1: "categorical", 2: "matern25", 3: "categorical"},
    discrete_params={1: 3, 3: 2},
    basegp_scan_csv="...",
    ...
)
kernel = factory._make_covar_module()
# → ProductKernel(Matern(x0), Categorical(x1), Matern(x2), Categorical(x3))
```

---

**交接完成。等待后续实现。**
