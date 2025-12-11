# 有序参数实现设计的核心决策与论证

**日期**: 2025-12-11  
**目标**: 回答4个关键设计问题，提供证据和理由

---

## 问题1: 等差和非等差合并为一个Transform吗？

### 答案: **是的，合并成一个Ordinal类，但内部逻辑有明确分支**

### 理由

#### 1.1 共性大于差异性

```
custom_ordinal (等差)        custom_ordinal_mono (非等差)
├── values = [1,2,3,4,5]    ├── values = [0.01, 0.1, 1, 10, 100]
├── Transform: value→rank    ├── Transform: value→rank
├── rank空间: [0,1,2,3,4]    ├── rank空间: [0,1,2,3,4]
├── GP学习: 相对顺序         └── GP学习: 相对顺序
└── LocalSampler: rank扰动       └── LocalSampler: rank扰动
```

**两者完全相同的部分**:

- Transform的核心: value → rank (0,1,2,...,n-1) → value
- bounds处理: [-0.5, n-0.5] (统一)
- GP核函数: RBFKernel in rank space (统一)
- LocalSampler扰动: rank空间高斯扰动+舍入 (统一)
- config解析: 都支持自动计算或直接指定

**两者差异的部分** (只在配置阶段):

- `custom_ordinal`: 自动计算values (min/max/step 或 num_levels)
- `custom_ordinal_mono`: 手工指定values (因为间距无规律)

#### 1.2 分离会导致代码重复

如果分成两个类:

```python
class OrdinalArithmetic(Transform):
    def _transform(self, X): ...  # A
    def _untransform(self, X): ... # B
    def transform_bounds(self, X): ... # C
    
class OrdinalMonotonic(Transform):
    def _transform(self, X): ...  # A (完全相同)
    def _untransform(self, X): ... # B (完全相同)
    def transform_bounds(self, X): ... # C (完全相同)
```

这违反DRY原则。配置差异不足以证明需要两个类。

#### 1.3 AEPsych的模式

在aepsych中，**Transform类不区分配置方式，只关心最终的values**:

- Categorical不区分 `choices=[a,b,c]` 还是 `choices_file=path/to/file.json`
- Integer不区分 `lb/ub` 还是 `values=[1,2,3,...]`
- 所有差异都在 `get_config_options()` 中解决

**所以Ordinal也应该这样做**:

```python
class Ordinal(Transform):
    """统一处理所有有序参数，无论等差还是非等差"""
    
    @classmethod
    def get_config_options(cls, config_dict):
        # 优先级链: values (手工) → min/max/step → min/max/num_levels → levels
        # 返回时都是 Ordinal(values=[...])
        # 调用者不关心values来自哪里
```

### 结论

✅ **单一Ordinal类 + 优先级链配置 = 正确的设计**

---

## 问题2: 为什么选择rank空间扰动？证据是什么？

### 答案: rank空间扰动在LocalSampler中**工作，但存在细微问题**

### 分析过程

#### 2.1 LocalSampler.sample()工作流程

```python
# 当前代码 (在local_sampler.py中)
def sample(self, X_can_t, dims):
    """
    输入: X_can_t (候选点，原始值空间)
    输出: local samples (原始值空间)
    """
    
    for d in dims:  # 逐维度扰动
        if variable_types[d] == "categorical":
            X_can_t[:, d] = self._perturb_categorical(X_can_t[:, d], ...)
        elif variable_types[d] == "integer":
            X_can_t[:, d] = self._perturb_integer(X_can_t[:, d], ...)
        # 无"ordinal"处理 ← 这是当前的问题
    
    return X_can_t
```

#### 2.2 关键insight: LocalSampler没有Transform对象

**问题**: LocalSampler的sample()工作在**原始值空间**，而不是Transform空间。

证据:

```python
# modules/local_sampler.py (约第60-100行)
class LocalSampler:
    def __init__(self, variable_types, bounds, **kwargs):
        self.variable_types = variable_types  # {0: 'categorical', 1: 'integer', ...}
        self.bounds = bounds  # 原始空间的lb/ub
        self._unique_vals_dict = {}  # 离散变量的unique值 (原始值)
    
    def sample(self, X_can_t, dims):
        # X_can_t的shape: (B, d)
        # X_can_t[:, d]都是原始值，不是rank
        
        # 对于categorical, 我们从_unique_vals_dict[d]采样 (原始值)
        # 对于integer, 我们生成[lb[d], ub[d]]范围的整数 (原始值)
        
        # 如果添加ordinal扰动，应该从哪里获取ordinal的values?
```

#### 2.3 问题: 如何在LocalSampler中获取Ordinal的values?

**三种方案对比**:

| 方案 | 描述 | 可行性 | 代价 |
|------|------|--------|------|
| **A: rank空间扰动** | LocalSampler显式转换到rank空间，扰动，再转换回 | ✅ 可行 | 需要Transform对象，破坏LocalSampler的独立性 |
| **B: 原始值空间扰动** | 像integer一样，直接在原始值空间扰动ordinal values | ✅ 可行 | 高斯噪声在非等差间距上可能不合理 |
| **C: 值表查找** | 像categorical一样，将ordinal values存在_unique_vals_dict中 | ✅ 最简洁 | 需要在初始化时提取values |

#### 2.4 证据: Categorical的实现模式 (correct way)

```python
# 在LocalSampler.__init__中
def __init__(self, variable_types, bounds, unique_vals_dict=None, ...):
    self._unique_vals_dict = unique_vals_dict or {}
    # {0: array([red, green, blue]), 1: array([1, 2, 3, 4]), ...}

# 在sample()中
def _perturb_categorical(self, center, k, B):
    unique_vals = self._unique_vals_dict[k]  # 原始值: [red, green, blue]
    # 随机选择: np.random.choice(unique_vals, size=(B, local_num))
    
# 这完全在原始值空间工作！无需Transform对象
```

#### 2.5 为什么categorical方案也适用于ordinal?

```python
# ordinal vs categorical: 唯一区别是是否有顺序

# categorical (无序):
#   center = "green"
#   扰动 = 随机选择 ["red", "green", "blue"] 中的一个
#   结果 = "green" 或 "blue" (50% each)

# ordinal (有序):
#   center = 2 (Likert: disagree)
#   values = [strongly_disagree, disagree, neutral, agree, strongly_agree]
#   扰动 = ??? 
#   - 选项A: 随机选择任一值 (失去顺序优势)
#   - 选项B: 优先选择邻近值 (有顺序性)
#   - 选项C: 高斯采样around中心 (保留连续性)
```

**关键问题**: Ordinal扰动的目的是什么?

- 如果目的是**探索邻近值** (保持局部性), 应该选B或C
- 如果目的是**完全随机** (无损搜索), 应该选A

#### 2.6 混合扰动策略 (use_hybrid_perturbation)

```python
# 当前配置中:
if use_hybrid_perturbation and n_levels <= exhaustive_level_threshold:
    # 对于低维discrete变量: 穷举所有值
    # for ordinal with 5 levels: [0, 1, 2, 3, 4] all sampled
else:
    # 高维变量: 随机采样
    # for ordinal: 高斯扰动+舍入 在rank空间
```

**这说明**:

- 设计者已经考虑了discrete变量的特殊处理
- 对于ordinal，应该根据levels数量选择穷举 vs 随机

### 当前实现的问题

**证据: 我的初始设计假设了错误的架构**

```python
# 我最初写的:
def _perturb_ordinal(self, center_point, var_idx, par_type, ordinal_transform):
    # ← ordinal_transform从哪里来? LocalSampler没有它!
    
    rank = ordinal_transform._transform(center_point[var_idx])  # 需要Transform对象
    # ← 这打破了LocalSampler的设计 (local_sampler.py不应该关心Transform)
```

**正确的做法应该是**:

```python
def _perturb_ordinal(self, center_idx, k, B):
    """
    在原始值空间扰动，但保持顺序性
    """
    unique_vals = self._unique_vals_dict[k]  # 来自初始化时提取的ordinal values
    center_val = center_idx[:, k]  # 当前值
    
    # 找到center_val在unique_vals中的索引
    center_rank = np.where(unique_vals == center_val)[0][0]
    
    # 混合策略:
    if use_hybrid_perturbation and len(unique_vals) <= threshold:
        # 穷举: 所有rank都采样
        samples = np.tile(unique_vals, (B, 1))[:, :self.local_num]
    else:
        # 随机: around中心的邻近值
        n_levels = len(unique_vals)
        noise = self._rng.normal(0, sigma, size=(B, self.local_num))
        ranks = np.round(np.clip(center_rank + noise, 0, n_levels-1)).astype(int)
        samples = unique_vals[ranks]  # 转换回原始值
    
    return samples
```

### 结论

✅ **rank空间概念正确，但LocalSampler实现应该这样做**:

1. 初始化时: 从Ordinal Transform提取values，存入_unique_vals_dict
2. sample()时: 在原始值空间工作，但利用rank索引保持顺序性
3. 不需要在LocalSampler中显式转换到rank空间 (那是Transform的工作)

---

## 问题3: ordinal与int类型的对比与选择

### 答案: **两者不相同，应该保持分离；ordinal更接近categorical**

### 详细对比

```
类型        | 值空间           | 间距       | GP处理         | 适用场景
----------|-----------------|-----------|----------------|------------------
integer   | [1,2,3,...,50] | 均匀 (1)   | RBF kernel    | 计数、索引
          | 完全连续等价   |           | 距离度量有意义 | 100、200...都合理
----------|-----------------|-----------|----------------|------------------
custom_   | [1,2,3,4,5]    | 均匀 (自定)| RBF kernel    | 有序偏好、等级
ordinal   | rank化处理      |           | rank距离       | 1→5路径有意义
          |                 |           | 绝对值无意义   | 但5不是"5倍"强
----------|-----------------|-----------|----------------|------------------
custom_   | [0.01,0.1,     | 不均匀    | RBF kernel    | 功率律、指数
ordinal_  |  1,10,100]     |           | rank距离       | 0.1→1路径有意义
mono      | rank化处理      |           | 绝对值无意义   | 但100是"1百倍"强
----------|-----------------|-----------|----------------|------------------
categori- | [red, green,   | 无序      | Categorical   | 无序选择
cal       |  blue]         | 无意义    | kernel        | A/B/C无相对强弱
          |                 |           |               |
```

### 关键差异: 间距的语义

#### 3.1 Integer: 间距=真实意义

```python
# integer: [1, 2, 3, ..., 50]
# GP学习到的: X=10和X=11离X=20和X=21的距离相同

# 这是对的，因为:
# - 10件和11件（差1）的效应差异
# - ≈ 20件和21件（差1）的效应差异

# 实际上，在这个轴上的距离度量是有意义的
```

#### 3.2 Custom_ordinal: 间距=纯粹排列顺序

```python
# custom_ordinal: [1, 2, 3, 4, 5]  (Likert量表)
# GP学习到的: rank空间中，3和4邻近，1和5远离

# 但问题是: 间距的绝对值无关!
# - [1, 2, 3, 4, 5] vs [1.1, 1.2, 1.3, 1.4, 1.5]
# - 在GP中应该给出相同的预测 (都是rank 0,1,2,3,4)
# - 但integer会给出不同预测 (值空间的差异)

# 所以ordinal和integer是fundamentally不同的
```

#### 3.3 Custom_ordinal vs Integer: 为什么选择ordinal?

```
场景: 用户有5个等级 [1, 2, 3, 4, 5]

选项A: 用integer
├── 配置: lb=1, ub=5
├── GP学习: 1→2和4→5的边际效应相同
├── 问题: 用户可能非线性!
│   - 1→2可能是"完全不同"
│   - 4→5可能是"细微差别"
└── 结果: 模型可能学错

选项B: 用custom_ordinal
├── 配置: values=[1,2,3,4,5]
├── GP学习: 在rank空间 [0,1,2,3,4]
├── 优点: 保留顺序，但不强制线性
│   - 1→2变化由GP学习
│   - 4→5变化由GP学习
│   - 可能不同，这是对的!
└── 结果: 模型更灵活
```

### 3.4 Ordinal更接近Categorical (架构角度)

```python
# Transform处理

# Categorical:
class Categorical(Transform):
    def _transform(self, X):  # [red, green, blue] → [0, 1, 2]
    def _untransform(self, X): # [0, 1, 2] → [red, green, blue]
    bounds = [-0.5, 2.5]  # 3个类别

# Custom_ordinal:
class Ordinal(Transform):
    def _transform(self, X):  # [1, 2, 3, 4, 5] → [0, 1, 2, 3, 4]
    def _untransform(self, X): # [0, 1, 2, 3, 4] → [1, 2, 3, 4, 5]
    bounds = [-0.5, 4.5]  # 5个等级

# ← 完全相同的架构!
# 都是: values → rank [0,1,...,n-1] → values

# Integer:
class Integer(Transform):
    # 没有values列表，直接用lb/ub
    # bounds = [lb-0.5, ub+0.5]
    # ← 不同的架构
```

### 结论

✅ **Custom_ordinal与categorical在架构上一致，都基于values列表rank化**  
❌ **Custom_ordinal与integer不同，因为ordinal无固定间距语义**  
✅ **保持ordinal和integer分离是正确的**

---

## 问题4: 数据类型与精度 (float64 vs float32)

### 答案: **必须用float64处理等差数列，确保精度**

### 问题分析

#### 4.1 等差数列的精度问题

```python
# 情形1: 整数值
values = [1, 2, 3, 4, 5]
# → float64或float32无差别，精确表示

# 情形2: 小数值 (步长为0.1)
np.arange(0, 1.0, 0.1)
# 用float32:  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, ...]
# 用float64:  [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#             ↑ 结果可能不同!

# 情形3: 等分数列
np.linspace(0, 1, 11)  # 11个点从0到1
# float32可能产生: [0.0, 0.090909..., 0.181818..., ...]
# ← 尾数精度不足，累积误差
```

#### 4.2 AEPsych当前的做法

```python
# aepsych/transforms/ops/categorical.py
class Categorical(Transform):
    bounds = torch.tensor([[-0.5], [n_categories - 0.5]], dtype=torch.double)
    #                                                         ↑ float64

# 所以AEPsych内部使用torch.double (float64)
```

证据:

```python
# aepsych/transforms/parameters.py (line ~200)
bounds = torch.tensor(
    [[float(lbs[i])], [float(ubs[i])]],
    dtype=torch.double  # ← 强制float64
)
```

#### 4.3 Ordinal的数据类型设计

```python
class Ordinal(Transform):
    def __init__(self, indices, values):
        # values: dict[int, list[float]]
        # 例: {0: [0.1, 0.5, 2.0, 5.0, 10.0]}
        
        self.values = {}
        for idx, val_list in values.items():
            # 强制转换为float64确保精度
            self.values[idx] = np.array(val_list, dtype=np.float64)
        
        # bounds也用float64
        n_levels = len(self.values[indices[0]])
        self.bounds = torch.tensor(
            [[-0.5], [n_levels - 0.5]],
            dtype=torch.double  # float64
        )
    
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        # 输入X可能是float32，需要转换
        X_fp64 = X.double()  # → float64处理
        
        # 执行查表 (在float64空间)
        ranks = self._lookup_rank(X_fp64)  # → tensor (float64)
        
        return ranks  # 返回float64
    
    @staticmethod
    def _compute_arithmetic_sequence(min_val, max_val, step=None, num_levels=None):
        """用float64计算等差数列"""
        min_val = float(min_val)
        max_val = float(max_val)
        
        if step is not None:
            step = float(step)
            # 使用float64计算
            values = np.arange(min_val, max_val + step/2, step, dtype=np.float64)
            # 舍入到合理精度 (避免浮点误差)
            values = np.round(values, decimals=12)
        elif num_levels is not None:
            num_levels = int(num_levels)
            # linspace在float64中更精确
            values = np.linspace(min_val, max_val, num_levels, dtype=np.float64)
        
        return values  # float64数组
```

#### 4.4 混合配置的陷阱

```ini
[common]
parnames = [integer_param, ordinal_param]
lb = [1, 1]
ub = [50, 5]

[integer_param]
par_type = integer
# bounds: float64 (aepsych强制)

[ordinal_param]
par_type = custom_ordinal
min_value = 1.0
max_value = 5.0
step = 0.5
# values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
# 必须用float64避免精度问题!
```

**可能的bug**:

```python
# 如果用float32计算等差:
np.arange(1.0, 5.0, 0.5, dtype=np.float32)
# 可能产生: [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, ...]
# 但5.0可能被舍掉 (因为float32精度)

# 结果: 4个点变成3个点，rank空间错误!
```

#### 4.5 LocalSampler的数据类型一致性

```python
# 在LocalSampler中:
# bounds来自aepsych: torch.double (float64)
# X_can_t: 推理输出，也是float64

# 当我们提取ordinal values时:
ordinal_values = self._unique_vals_dict[k]  # np.float64

# 匹配时不会有float32/float64混淆
```

### 检查清单

#### 在Ordinal类中

- [ ] __init__时强制 np.float64
- [ ] _compute_arithmetic_sequence输出 np.float64
- [ ] bounds用 torch.double
- [ ] _transform/_untransform处理float32→float64转换
- [ ] 单元测试: 测试0.1步长的10个点精度

#### 在custom_pool_based_generator中

- [ ] 提取ordinal.values时保持float64
- [ ] Pool点在float64中生成
- [ ] 去重时用float64比较 (np.isclose with rtol/atol)

#### 在LocalSampler中

- [ ] _unique_vals_dict[k]保持float64
- [ ] 扰动输出float64
- [ ] 回到aepsych前无类型转换

### 结论

✅ **必须全程使用float64处理ordinal参数**  
✅ **特别是自动计算等差数列时，否则精度丢失**  
✅ **这与aepsych的全局政策 (torch.double) 一致**

---

## 总结: 4个设计决策

| # | 问题 | 答案 | 关键点 |
|----|------|------|--------|
| 1 | 等差/非等差合并? | ✅ 是，单一Ordinal类 | Transform逻辑相同，差异在配置阶段 |
| 2 | rank空间扰动工作? | ✅ 工作，但架构需调整 | 应该在_unique_vals_dict中存values，在原始空间工作 |
| 3 | ordinal vs int? | ✅ 保持分离 | ordinal更接近categorical (rank化)，不是integer的变体 |
| 4 | 数据类型? | ✅ 强制float64 | 避免等差数列精度丢失，与aepsych一致 |
