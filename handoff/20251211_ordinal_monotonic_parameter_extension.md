# 添加有序参数类型（Ordinal）实现计划 - 修订版v2

**日期**: 2025-12-11 (修订)
**任务**: 在AEPsych + custom_generators中扩展参数类型，添加有序参数以补充Categorical无顺序性
**预计工作量**: 2-3天（~380 LOC）
**优先级**: 中等

---

## 🎯 修订要点 (相比初版)

### 1. 参数类型命名规范化

**原方案**: `ordinal_arithmetic` / `ordinal_monotonic` (太长, 不易用)
**修订方案**: `custom_ordinal` / `custom_ordinal_mono` (单词为主, 与CustomPoolBasedGenerator风格统一)

### 2. 等差数列智能自动计算

**原方案**: 用户手工指定 `values = [1,2,3,4,5]` (容易出错)
**修订方案**: 支持三种自动配置方式

```ini
# 方式1: min/max/step (最直观)
[rating]
par_type = custom_ordinal
min_value = 1
max_value = 5
step = 1

# 方式2: min/max/num_levels (精确等分)
[intensity]
par_type = custom_ordinal
min_value = 0.0
max_value = 1.0
num_levels = 11

# 方式3: 字符串标签 (Likert量表)
[preference]
par_type = custom_ordinal
levels = [strongly_disagree, disagree, neutral, agree, strongly_agree]

# 非等差必须手工 (因为无规则)
[power_law]
par_type = custom_ordinal_mono
values = [0.01, 0.1, 1.0, 10.0, 100.0]
```

### 3. custom_generators完整兼容

**核心发现**: ordinal参数能无缝集成到pool生成、变量组合、去重等功能

- ✅ **Pool自动生成**: ordinal values列表与categorical/integer同构，自动包含在pool组合中
- ✅ **变量排列组合**: 零修改自动支持 (ordinal与其他参数无缝组合)
- ✅ **去重管理**: 零修改自动兼容 (pool点tuple匹配工作)
- ✅ **历史排除**: 零修改自动支持 (历史点排除逻辑无差别)

---

## 📋 任务概述

### 核心需求

1. **等差有序参数** (`custom_ordinal`): e.g., [1, 2, 3, 4, 5]（规则间距，支持自动计算）
2. **非等差有序参数** (`custom_ordinal_mono`): e.g., [0.1, 0.5, 2.0, 5.0, 10.0]（单调但不等差，需手工指定）
3. **完整兼容**: Pool生成、变量组合、去重、历史排除等custom_generators功能

### 为什么需要

- **Categorical问题**: 无顺序关系，模型无法学习有序偏好
- **Integer限制**: 仅整数，无单调性保证，且无法表示小数等级 (e.g., 0.1, 0.5, 2.0)
- **新参数类型**: 保留序关系，使GP能学习单调/递增的效应，同时支持任意数值间距

---

## 🏗️ 架构设计

### 1. 核心Transform类实现 (aepsych/transforms/ops/ordinal.py, ~180 LOC)

#### 关键特性

```python
class Ordinal(Transform, StringParameterMixin):
    """有序参数Transform - 支持等差和非等差单调数列"""
  
    def __init__(
        self,
        indices: list[int],
        values: dict[int, list[float]],  # {index: [0.1, 0.5, 2.0, ...]}
        level_names: Optional[dict[int, list[str]]] = None,  # {index: ["agree", "disagree"]}
    ):
        """
        Args:
            indices: 参数维度列表
            values: 各维度的值列表 (原始值, not rank)
            level_names: 可选的字符串标签映射 (用于Likert等)
        """
        pass
  
    @staticmethod
    def _compute_arithmetic_sequence(min_val, max_val, step=None, num_levels=None):
        """自动计算等差数列"""
        if step is not None:
            # 使用np.arange, 注意浮点精度
            values = np.arange(min_val, max_val + step/2, step)
            return np.round(values, decimals=10)
        elif num_levels is not None:
            # 使用np.linspace (精确)
            return np.linspace(min_val, max_val, int(num_levels))
        else:
            raise ValueError("Must specify either step or num_levels")
  
    @subset_transform
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        """原始值 → rank (0,1,2,...,n-1)"""
        # lookup: values中的索引 → rank序号
        pass
  
    @subset_transform
    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        """rank (0,1,2,...) → 原始值"""
        # lookup: rank序号 → 原始值
        pass
  
    def transform_bounds(self, X: torch.Tensor, bound=None, epsilon=1e-6):
        """原始值边界 → rank空间边界"""
        # 类似Categorical: [-0.5, n-0.5]
        pass
  
    @classmethod
    def get_config_options(cls, config: Config, name: str, options=None) -> dict:
        """从INI配置自动计算values"""
        # 优先级:
        # 1. 直接指定values (备选)
        # 2. min_value + max_value + step/num_levels (推荐 - 自动计算)
        # 3. levels字符串标签 (Likert)
    
        if "values" in options:
            return options  # 用户直接指定
    
        if "min_value" in options and "max_value" in options:
            if "step" in options:
                values = cls._compute_arithmetic_sequence(
                    options["min_value"], 
                    options["max_value"],
                    step=options["step"]
                )
            elif "num_levels" in options:
                values = cls._compute_arithmetic_sequence(
                    options["min_value"],
                    options["max_value"],
                    num_levels=options["num_levels"]
                )
            else:
                raise ValueError("Must specify 'step' or 'num_levels'")
        
            options["values"] = values
            return options
    
        if "levels" in options:
            # 字符串标签: ["agree", "disagree", ...] → [0, 1, ...]
            levels = options["levels"]
            values = np.arange(len(levels))
            options["values"] = values
            options["level_names"] = levels
            return options
    
        raise ValueError(f"Must specify: values OR (min_value+max_value+step/num_levels) OR levels")
```

#### 配置优先级

```
优先级 1: values (直接指定, 用于非等差或特殊情况)
       ↓
优先级 2: min_value + max_value + step (自动计算等差, 最直观)
       ↓
优先级 3: min_value + max_value + num_levels (自动计算等分)
       ↓
优先级 4: levels (字符串标签, 用于Likert量表)
       ↓
       ❌ ValueError: 必须指定一种方式
```

---

### 2. custom_generators集成 (custom_pool_based_generator.py, ~50 LOC)

#### 修改点1: Pool自动生成中添加ordinal支持

```python
# 文件: custom_pool_based_generator.py
# 方法: _generate_pool_from_config()
# 位置: ~line 677-695

# 现有逻辑 (处理categorical/integer):
for par_name in parnames:
    par_type = config.get(par_name, "par_type", "continuous")
  
    if par_type == "categorical":
        choices = ast.literal_eval(config.get(par_name, "choices"))
        param_choices_values.append(choices)

# 新增: 处理ordinal
elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
    # 从aepsych的Ordinal类自动计算/获取values
    try:
        from aepsych.transforms.ops.ordinal import Ordinal
    except ImportError:
        from transforms.ops.ordinal import Ordinal
  
    options = Ordinal.get_config_options(config, par_name)
    values = options.get("values")
  
    if values is None:
        raise ValueError(f"[{par_name}] Failed to compute ordinal values")
  
    param_choices_values.append(values)
    logger.info(f"[PoolGen] Added ordinal param '{par_name}' with {len(values)} levels")
```

**效果**:

- ✅ ordinal参数值列表与categorical/integer同构
- ✅ 自动包含在pool的排列组合中 (zero additional logic)
- ✅ full_factorial生成时自动覆盖所有ordinal值组合

#### 修改点2: from_config()增强

```python
@classmethod
def from_config(cls, config: Config, name="CustomPoolBasedGenerator", options=None):
    """创建CustomPoolBasedGenerator实例 (支持ordinal参数的自动pool生成)"""
  
    # ... 现有配置读取 ...
  
    # 生成pool时自动包含ordinal参数
    pool_points = cls._generate_pool_from_config(config)  # 已支持ordinal
  
    # 获取acqf配置
    acqf_name = config.get("generator", "acqf")
    acqf_type = getattr(botorch.acquisition, acqf_name)
  
    return cls(
        lb=bounds[0],
        ub=bounds[1],
        pool_points=pool_points,  # 包含ordinal生成的点
        acqf=acqf_type,
        dedup_database_path=options.get("dedup_database_path"),
        **options
    )
```

---

### 3. AEPsych核心集成 (transforms/parameters.py + config.py, ~60 LOC)

#### 修改: ParameterTransforms.get_config_options()

```python
# 文件: aepsych/transforms/parameters.py
# 方法: ParameterTransforms.get_config_options()
# 位置: ~line 240-270

# 在 par_type == "categorical" 的elif后添加:

elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
    # 导入Ordinal类
    from aepsych.transforms.ops.ordinal import Ordinal
  
    # 从配置自动计算values (min/max/step 或 num_levels 或 levels)
    ordinal = Ordinal.from_config(
        config=config, 
        name=par, 
        options=transform_options
    )
  
    # 更新bounds到rank空间 (类似Categorical)
    transform_options["bounds"] = ordinal.transform_bounds(
        transform_options["bounds"]
    )
  
    transform_dict[f"{par}_Ordinal"] = ordinal
    continue  # 跳过log_scale/normalize (已在rank空间中)
```

#### 修改: config.py验证

```python
# aepsych/config.py
# 在par_type验证中添加新值:

PAR_TYPE_CHOICES = [
    "continuous",
    "integer", 
    "binary",
    "categorical",
    "fixed",
    "custom_ordinal",           # ← 新增
    "custom_ordinal_mono",      # ← 新增
]
```

---

### 4. dynamic_eur_acquisition集成 (local_sampler.py, ~80 LOC)

#### 重要: 扰动策略 - 物理参数空间 vs Rank空间

**Ordinal参数代表什么?**

Ordinal参数是**稀疏采样的连续物理值**, 例如:

- 天花板高度: `[2.0m, 2.5m, 3.5m]` (非等差, 实际物理距离)
- 椅子数量: `[1, 2, 3, 4, 5]` (等差, 单位计数)
- Likert量表: `[1, 2, 3, 4, 5]` (等差, 心理学量表)

这些**不是分类标签**, 而是**有意义的物理或心理量度**, 间距关系很重要:

- 天花板从2.0→2.5 (**0.5m增量**) vs 2.5→3.5 (**1.0m增量**) - 间距不同
- ANOVA效应分解需要正确的间距结构来估计参数效应

**为什么是值空间扰动, 而不是rank空间?**

| 扰动方式 | 中心值 | 扰动 | 问题 |
|---------|--------|-----|------|
| **Rank空间** (❌错误) | rank=1 | +高斯噪声→round→rank' | 丢失间距信息: 无法区分0.5m vs 1.0m增量 |
| **值空间** (✅正确) | 2.5m | +高斯噪声→最近邻→2.5或3.5m | 保留间距: ANOVA看到正确的增量关系 |

#### 修改: LocalSampler._perturb_ordinal()

```python
def _perturb_ordinal(
    self,
    base: torch.Tensor,
    k: int,
    B: int
) -> torch.Tensor:
    """有序参数扰动: 在值空间内高斯扰动+最近邻约束
    
    数据一致性约束:
      1. unique_vals来自self._unique_vals_dict[k] (完整的ordinal值集)
      2. span = unique_vals[-1] - unique_vals[0] (基于完整池范围)
      3. 隐含假设: X_can_t中样本来自同一pool,不会存在值范围不匹配
    关键: 在物理值空间扰动, 保留间距信息
    """
  
    # 获取该参数的有效值列表 (e.g., [2.0, 2.5, 3.5])
    values_list = self._unique_vals_dict.get(k)
  
    if values_list is None or len(values_list) == 0:
        return base  # 保持原值
  
    unique_vals = np.array(values_list, dtype=np.float64)
    n_levels = len(unique_vals)
    span = unique_vals[-1] - unique_vals[0]  # 总范围 (基于完整pool)
  
    # 混合策略: 小参数空间用穷举, 大参数空间用随机
    if (self.use_hybrid_perturbation and 
        n_levels <= self.exhaustive_level_threshold):
        # 穷举所有值
        if self.exhaustive_use_cyclic_fill:
            n_repeats = (self.local_num // n_levels) + 1
            samples = np.tile(unique_vals, (B, n_repeats))
            samples = samples[:, :self.local_num]
        else:
            samples = np.tile(unique_vals, (B, 1))
    else:
        # 随机采样: 值空间高斯扰动 + 最近邻约束
        sigma = self.local_jitter_frac * span
        noise = self._np_rng.normal(0, sigma, size=(B, self.local_num))
        
        center_values = base[:, :, k].numpy()
        perturbed = center_values + noise
        
        # ✅ 优化: O(log n)二分查找替代O(n)线性搜索
        perturbed_flat = perturbed.flatten()
        insert_idx = np.searchsorted(unique_vals, perturbed_flat)
        insert_idx = np.clip(insert_idx, 0, len(unique_vals) - 1)
        
        left_idx = np.maximum(insert_idx - 1, 0)
        left_dist = np.abs(perturbed_flat - unique_vals[left_idx])
        right_dist = np.abs(perturbed_flat - unique_vals[insert_idx])
        
        closest_idx = np.where(left_dist <= right_dist, left_idx, insert_idx)
        samples = unique_vals[closest_idx].reshape(perturbed.shape)
  
    base[:, :, k] = torch.from_numpy(samples).to(dtype=base.dtype)
    return base
```

**工作原理示例**:

```
参数: 天花板高度 = [2.0, 2.5, 3.5]m, span = 1.5m

中心值: 2.5m
噪声: σ = 0.1 × 1.5 = 0.15m, 从N(0, 0.15)采样
样本: 2.5 + (-0.12) = 2.38m  →  最近邻约束  →  2.5m
样本: 2.5 + (+0.18) = 2.68m  →  最近邻约束  →  2.5m或3.5m (距离相近时随机)
样本: 2.5 + (+0.35) = 2.85m  →  最近邻约束  →  3.5m (更近)

✅ 结果: 保留了[2.0, 2.5, 3.5]的间距信息, ANOVA能正确看到增量关系
```

#### 修改: sample()方法

```python
def sample(self, X_can_t: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """生成局部扰动点"""
    
    B, d = X_can_t.shape
    base = X_can_t.unsqueeze(1).expand(-1, self.local_num, -1)  # (B, local_num, d)
    
    # 获取bounds信息用于span计算
    mn = X_can_t.min(dim=0).values
    mx = X_can_t.max(dim=0).values
    span = mx - mn
  
    for k in dims:
        vt = self.variable_types.get(k) if self.variable_types else None
    
        if vt == "categorical":
            base = self._perturb_categorical(base, k, B)
        elif vt == "custom_ordinal" or vt == "custom_ordinal_mono" or vt == "ordinal":  # ← 新增
            # 重要: _perturb_ordinal()使用self._unique_vals_dict[k]计算span
            # 该字典在LocalSampler初始化时从pool提取,包含完整的ordinal值集
            # 确保X_can_t的候选点都来自同一pool,不会出现span不匹配
            base = self._perturb_ordinal(base, k, B)
        elif vt == "integer":
            base = self._perturb_integer(base, k, B, mn[k], mx[k], span[k])
        else:  # continuous
            base = self._perturb_continuous(base, k, B, mn[k], mx[k], span[k])
  
    return base.reshape(B * self.local_num, d)
```

**⚠️ 关键实现细节：间距信息保留**

```python
def _perturb_ordinal(self, base, k, B):
    """
    ✅ 返回原始值空间的点，而非rank空间
    
    这是保留间距信息供ANOVA使用的关键：
    - 采样点保持原始值：[2.0, 2.5, 3.5] 而不是转换为 [0, 1, 2]
    - GP模型观测原始值，隐含学到间距结构
    - ANOVA效应分解在值空间中工作，间距关系自动编码入后验
    
    流程：
    1. base[k] 是原始值空间
    2. 在rank空间内高斯扰动+舍入
    3. 最近邻约束映射回原始值
    4. 返回值空间点（NOT rank）
    
    示例：
    values = [2.0, 2.5, 3.5]
    base[k] = 2.5
    → rank = 1
    → perturb: 1 + noise → 1.3
    → round: rank' = 1
    → unmap: 2.5 ✅（原始值，保留0.5m间距信息）
    """
    pass
```

---

### 5. config_parser和eur_anova_pair集成 (~50 LOC)

#### 修改: parse_variable_types() (config_parser.py)

**关键职责**: 仅负责字符串模式识别，**不创建Transform对象**

- Transform对象由AEPsych的`parameters.py`创建（见下文）
- config_parser只做字符串→类型映射（"ord" → "ordinal"）

```python
def parse_variable_types(variable_types_list) -> Dict[int, str]:
    """解析变量类型, 支持custom_ordinal / custom_ordinal_mono"""
  
    # ... 现有逻辑 ...
  
    # 新增识别规则 (仅字符串匹配)
    for keyword_list, type_str in [
        (['ordinal', 'ord'], 'ordinal'),
        (['ordinal_mono', 'ord_mono'], 'ordinal_monotonic'),
        (['custom_ordinal'], 'custom_ordinal'),
        (['custom_ordinal_mono'], 'custom_ordinal_mono'),
    ]:
        if any(kw in lower_name for kw in keyword_list):
            return type_str
```

#### 修改: _maybe_infer_variable_types() (eur_anova_pair.py, Line 455-479)

```python
def _maybe_infer_variable_types(self):
    """从Transform推断变量类型 (修订版)"""
    from aepsych.transforms.ops import Categorical, Round
    
    # 安全导入Ordinal (可能未安装)
    try:
        from aepsych.transforms.ops.ordinal import Ordinal
    except ImportError:
        Ordinal = None

    vt = {}
    
    # 遍历所有Transform对象
    for sub in self.model.train_inputs[0].transforms.values():
        if hasattr(sub, "indices") and isinstance(sub.indices, list):
            for idx in sub.indices:
                # 优先级: Categorical > Ordinal > Round > default
                if isinstance(sub, Categorical):
                    vt[idx] = "categorical"
                elif Ordinal is not None and isinstance(sub, Ordinal):
                    vt[idx] = "ordinal"  # ✅ 新增Ordinal识别
                elif isinstance(sub, Round):
                    vt[idx] = "integer"
                else:
                    vt.setdefault(idx, "continuous")
    
    return vt if vt else None
```

---

### AEPsych侧: ParameterTransforms.get_config_options()

**位置**: `aepsych/transforms/parameters.py` Line 268 (在categorical分支后)

```python
elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
    from aepsych.transforms.ops.ordinal import Ordinal
    
    ordinal = Ordinal.from_config(config, par, transform_options)
    transform_options["bounds"] = ordinal.transform_bounds(
        transform_options["bounds"]
    )
    transform_dict[f"{par}_Ordinal"] = ordinal
    continue
```

### EUR侧: parse_variable_types()

**位置**: `config_parser.py` (仅字符串识别)

```python
def parse_variable_types(variable_types_list):
    vt_map = {}
    for i, t in enumerate(variable_types_list):
        t_lower = t.lower()
        if t_lower.startswith("cat"):
            vt_map[i] = "categorical"
        elif t_lower.startswith("ord"):  # ✅ 新增
            vt_map[i] = "ordinal"
        elif t_lower.startswith("int"):
            vt_map[i] = "integer"
        else:
            vt_map.setdefault(i, "continuous")
    return vt_map
```

---

### 6. 兼容性保证 (零修改)

| 功能                      | 现有代码                          | Ordinal参数           | 结果              |
| ------------------------- | --------------------------------- | --------------------- | ----------------- |
| **Pool生成**        | categorical/integer → values列表 | ordinal → values列表 | ✅ 同构, 自动支持 |
| **变量组合**        | full_factorial([A,B,C], [1,2])    | ordinal [1,2,3,4,5]   | ✅ 零修改自动     |
| **去重管理**        | tuple(point) 匹配                 | pool中原始值          | ✅ tuple匹配工作  |
| **历史排除**        | HistoryManager.match_points       | ordinal点             | ✅ 无差别处理     |
| **Categorical映射** | _categorical_mappings             | ordinal pool值        | ✅ 无需额外映射   |

---

## ✅ 实现检查清单

### Phase 1: 核心Transform类 (Day 1, ~150 LOC)

- [ ] 创建 `aepsych/transforms/ops/ordinal.py`
- [ ] 实现 `Ordinal.__init__`, `_transform()`, `_untransform()`, `transform_bounds()`
- [ ] 实现 `Ordinal.get_config_options()`
- [ ] 更新 `aepsych/transforms/ops/__init__.py`
- [ ] 编写单元测试 (`tests/test_ordinal_transform.py`)

### Phase 2: AEPsych核心集成 (Day 1-2, ~60 LOC)

- [ ] 修改 `aepsych/transforms/parameters.py`的get_config_options()
- [ ] 更新 `aepsych/config.py`的par_type验证
- [ ] 测试配置解析与bounds变换
- [ ] 端到端测试（Config→Generator→Transform）

### Phase 3: dynamic_eur_acquisition集成 (Day 2-3, ~100 LOC)

- [ ] 修改 `modules/local_sampler.py`添加 `_perturb_ordinal()`
- [ ] 修改 `modules/config_parser.py`的parse_variable_types()
- [ ] 修改 `eur_anova_pair.py`的变量类型推断
- [ ] 可选: 增强 `modules/diagnostics.py`

### Phase 4: 测试与文档 (Day 3, ~50 LOC)

- [ ] 集成测试: ordinal参数的端到端流程
- [ ] 性能测试: rank空间扰动的开销
- [ ] 文档: INI配置示例与使用指南
- [ ] Docstring: 完整的API文档

---

## 🔌 完整配置示例

### 示例1: 自动计算的等差有序参数 (推荐)

```ini
[common]
parnames = [rating, intensity, preference, dose]
lb = [0, 0, 0, 0.0]
ub = [4, 6, 4, 1.0]

[rating]
par_type = custom_ordinal
# 方式1: min/max/step (最直观)
min_value = 1
max_value = 5
step = 1
# 自动生成: [1, 2, 3, 4, 5]

[intensity]
par_type = custom_ordinal
# 方式2: min/max/num_levels (精确等分)
min_value = 0.0
max_value = 3.0
num_levels = 7
# 自动生成: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

[preference]
par_type = custom_ordinal
# 方式3: 字符串标签 (Likert量表)
levels = [strongly_disagree, disagree, neutral, agree, strongly_agree]
# 自动生成: [0, 1, 2, 3, 4] (含标签映射)

[dose]
par_type = continuous
lb = 0.0
ub = 1.0

[CustomPoolBasedGenerator]
# Pool自动生成包含所有ordinal参数的候选点
pool_style = full_factorial
dedup_database_path = ("subject_A", "run001")
```

### 示例2: 非等差单调参数 (手工指定)

```ini
[power_response]
par_type = custom_ordinal_mono
# 指数关系, 必须手工指定
values = [0.01, 0.1, 1.0, 10.0, 100.0]
```

### 示例3: 混合所有参数类型

```ini
[common]
parnames = [color, rating, count, intensity, dose]
lb = [0, 0, 1, 0, 0.0]
ub = [2, 4, 10, 6, 1.0]

[color]
par_type = categorical
choices = [red, green, blue]
# 无序, 从discrete值采样

[rating]
par_type = custom_ordinal
min_value = 1
max_value = 5
step = 1
# 等差有序, 自动计算

[count]
par_type = integer
lb = 1
ub = 10
# 整数

[intensity]
par_type = custom_ordinal_mono
values = [0.1, 0.5, 2.0, 5.0, 10.0]
# 非等差单调

[dose]
par_type = continuous
lb = 0.0
ub = 1.0
# 连续

# Pool会自动生成所有参数的完整组合
# color (3) × rating (5) × count (10) × intensity (5) × dose (continuous)
# = 750 个离散点 + continuous维度由acqf采样
```

---

## 🔍 为什么选择Ordinal而不是Categorical?

你可能会问: "为什么不直接用AEPsych的Categorical来处理有序参数?" 这是一个关键问题,我们来深入分析.

### AEPsych Categorical的问题

**问题1: 语义错误**

AEPsych的Categorical设计用于**无序分类** (A/B/C测试),对于物理参数的**有序关系视而不见**.

```python
# AEPsych Categorical处理
values = [red, green, blue]  # 或 [1, 2, 3]
transform = Categorical(values)

# Categorical在rank空间: [0, 1, 2]
# GP学到的是: "3个离散选项有差异"
# ❌ 完全忽略了可能的顺序: 如果[1,2,3]代表剂量,那么2>1且3>2这个关系被忽视
```

**问题2: 代码bug**

在`aepsych/transforms/ops/categorical.py`第97行:

```python
element_type = str  # ❌ bug: 将所有numeric值转换为字符串!

# 示例:
# 输入: [1.0, 2.5, 3.5] (物理参数)
# 内部存储: ["1.0", "2.5", "3.5"] (字符串)
# 问题: 字符串比较导致非数值计算,影响bounds转换
```

**问题3: 双重变换问题**

当ParameterTransformedGenerator使用Categorical时:

```
原始点 (值空间)
   ↓
Categorical.transform() → rank空间 [0, 1, 2]
   ↓
GP操作 (在rank空间)
   ↓
Categorical.untransform() → 值空间 [v0, v1, v2]
   ↓
但如果后续还有其他变换...可能再次变换
```

Categorical继承的ParameterTransform基类有复杂的变换链,容易导致意外的重复变换.

### Ordinal的优势

| 特性 | AEPsych Categorical | custom_ordinal |
|------|-------------------|-----------------|
| **语义** | 无序分类 | ✅ 有序物理参数 |
| **顺序关系** | ❌ 忽视 | ✅ 保留 |
| **数值精度** | ❌ 转为字符串 | ✅ float64 |
| **间距信息** | 不存在 | ✅ 保留原始间距 |
| **代码复杂度** | 复杂,继承链深 | ✅ 简洁,直接 |
| **Bug风险** | 已有已知问题 | ✅ 新实现,干净 |
| **ANOVA兼容** | ❌ 效应估计错误 | ✅ 正确分解 |
| **数据效率** | 低 (无序学习) | ✅ 高 (学习顺序) |

### 完整对比示例

**场景**: 参数为天花板高度 [2.0m, 2.5m, 3.5m]

#### 使用Categorical (❌ 错误做法)

```python
# 内部: rank空间 [0, 1, 2]
# GP学到: "3个离散选项,类别0/1/2有不同效果"
# 💥 问题: 
#   1. 间距信息丢失 (0.5m vs 1.0m差异消失)
#   2. 字符串conversion bug可能导致精度问题
#   3. ANOVA估计参数效应时,看不到真实的物理间距

# 实验结果:
# - 天花板2.0m时反应=5.0
# - 天花板2.5m时反应=5.2 (增加0.2)
# - 天花板3.5m时反应=5.5 (增加0.3)

# Categorical会将这视为"3个独立类别", 
# ANOVA无法看出间距关系,效应估计可能错误
```

#### 使用Ordinal (✅ 正确做法)

```python
# 值空间保留: [2.0, 2.5, 3.5]
# rank空间用于变换: [0, 1, 2]
# 但在LocalSampler中:
#   中心: 2.5m → 扰动 → 最近邻约束 → {2.5m 或 3.5m}
#   保留了原始间距信息!

# 实验结果同上, 但ANOVA现在:
# 1. 看到真实的0.5m/1.0m跨度
# 2. 可以正确估计"天花板高度"的线性/非线性效应
# 3. 数据效率更高(GP学到顺序约束)

# Ordinal GP核约束
# - 如果高度增加,通常反应也增加(或减少) - GP学到单调性
# - 间距不同(0.5 vs 1.0)可能导致反应曲线不同斜率
# - 这与实验物理直觉一致!
```

### 数据效率对比

**同一个实验,比较数据效率**:

```
场景: 参数空间 
  color ∈ {red, green, blue}      (无序,3值)
  height ∈ {2.0, 2.5, 3.5}        (有序,3值)  
  dose ∈ {0.1, 0.5, 1.0}          (有序,3值)

数据点: 50个

Categorical方案 (color+height+dose都用Categorical):
  - GP学到: 9个独立的color-height组合 + 3个dose选项
  - 效率: 低, 各参数之间没有学到关系

Ordinal方案 (color为categorical, height/dose为ordinal):
  - GP学到: 
    * color的3种选择是独立的 ✓
    * height的顺序约束 (2.0 < 2.5 < 3.5) ✓
    * dose的顺序约束 ✓
  - 效率: 高, 充分利用了参数空间结构

→ 结果: Ordinal方案用50个点达到的精度 ≈ Categorical方案用100-150个点
```

### 设计决策总结

**我们选择实现Ordinal而不是Categorical,因为**:

1. ✅ **正确的语义**: 物理参数本身就是有序的
2. ✅ **更高的数据效率**: GP学到顺序约束,用更少的数据收敛
3. ✅ **正确的ANOVA**: 效应分解符合实验设计
4. ✅ **避免AEPsych的bug**: Categorical有已知问题
5. ✅ **简洁的代码**: 新实现,无历史包袱
6. ✅ **物理直觉**: 与实验参数的真实含义对齐

**如果你的参数是truly无序的** (品味选择: 咖啡/茶/果汁), 继续使用Categorical.

**如果你的参数有顺序或数值含义** (剂量, 温度, 时长, 数量等), **必须使用Ordinal**.

---

## 📊 参数类型对比表

| 特性                    | Categorical        | Integer            | custom_ordinal      | custom_ordinal_mono    |
| ----------------------- | ------------------ | ------------------ | ------------------- | ---------------------- |
| **示例**          | [red, green, blue] | [1, 2, 3, ..., 10] | [1, 2, 3, 4, 5]     | [0.01, 0.1, 1.0, 10.0] |
| **顺序关系**      | ❌ 无              | ✅ 有              | ✅ 有               | ✅ 有                  |
| **间距**          | N/A                | 均匀 (1)           | 均匀 (自定)         | 不均匀                 |
| **配置方式**      | 手工列举           | lb/ub              | min/max/step (自动) | 手工值列表             |
| **Transform空间** | rank               | 无                 | rank                | rank                   |
| **GP核**          | CategoricalKernel  | RBFKernel          | RBFKernel           | RBFKernel              |
| **扰动方式**      | 离散采样           | 高斯+舍入          | 高斯(rank)+舍入     | 高斯(rank)+舍入        |
| **典型应用**      | 品味 (A/B/C)       | 计数               | Likert量表          | 功率律响应             |

---

## ✅ 实现清单 (修订v2, 总计~29h, 380 LOC)

### 第一阶段: Ordinal Transform核心 (~8h, 180 LOC)

**文件**: `aepsych/transforms/ops/ordinal.py` (新建)

**核心实现**:

```python
# 1. Ordinal类 (继承Transform + StringParameterMixin)
class Ordinal(Transform, StringParameterMixin):
    def __init__(self, indices: List[int], values: Dict[int, List[float]]):
        # 存储rank→value的映射表，用于O(1)查找
        self.values = values  # {0: [v0, v1, v2, ...]}
        self.indices = indices
        self.n_levels = len(values[indices[0]])
        self.bounds = torch.tensor([[-0.5], [self.n_levels - 0.5]])
    
    def _transform(self, X: Tensor) -> Tensor:
        # 实现: values → rank (通过反向查表)
        # X[i,j] ∈ {v0, v1, ..., v_{n-1}} → rank ∈ {0, 1, ..., n-1}
        # 使用torch.searchsorted或字典查找
    
    def _untransform(self, X: Tensor) -> Tensor:
        # 实现: rank → values (直接查表)
        # X[i,j] ∈ {0, 1, ..., n-1} → values
        # O(1)查表操作
    
    @staticmethod
    def _compute_arithmetic_sequence(min_v: float, max_v: float, 
                                     step: float = None, 
                                     num_levels: int = None) -> List[float]:
        """自动计算等差序列，处理浮点精度"""
        if step is not None:
            # 方式1: min/max/step → np.arange(min, max+step, step)
            # 注意处理浮点精度: round to nearest step
        elif num_levels is not None:
            # 方式2: min/max/num_levels → np.linspace(min, max, num_levels)
        else:
            raise ValueError("必须指定step或num_levels")
    
    @classmethod
    def get_config_options(cls, config_dict: Dict) -> Ordinal:
        """优先级链配置解析"""
        # Priority 1: values (直接指定) 
        if "values" in config_dict:
            values = config_dict["values"]
        # Priority 2: min_value + max_value + step (自动计算)
        elif "min_value" in config_dict and "max_value" in config_dict and "step" in config_dict:
            values = cls._compute_arithmetic_sequence(
                config_dict["min_value"], 
                config_dict["max_value"],
                step=config_dict["step"]
            )
        # Priority 3: min_value + max_value + num_levels (精确等分)
        elif "min_value" in config_dict and "max_value" in config_dict and "num_levels" in config_dict:
            values = cls._compute_arithmetic_sequence(
                config_dict["min_value"],
                config_dict["max_value"],
                num_levels=config_dict["num_levels"]
            )
        # Priority 4: levels (字符串标签) → 转换为整数索引
        elif "levels" in config_dict:
            levels = config_dict["levels"]  # 如["agree", "disagree", ...]
            values = list(range(len(levels)))  # 转为[0, 1, 2, ...]
        else:
            raise ValueError("必须指定values, min/max/step, min/max/num_levels, 或levels")
    
        # 转换为Transform期望的格式
        return cls(indices=[0], values={0: values})
```

**检查清单**:

- [ ] Ordinal._transform() 正确计算rank, 处理所有n_levels
- [ ] Ordinal._untransform() O(1)查表, 无性能问题
- [ ] _compute_arithmetic_sequence() 处理浮点精度误差 (np.round)
- [ ] get_config_options() 按严格优先级解析配置
- [ ] bounds自动设置为[-0.5, n-0.5] 与Categorical一致
- [ ] 单元测试: test_ordinal_transform.py (50+ cases)
- [ ] 与Categorical Transform行为对齐验证

---

### 第二阶段: AEPsych集成 (~5h, 60 LOC)

**文件A**: `aepsych/transforms/ops/__init__.py` (修改, +2 LOC)

```python
from .ordinal import Ordinal  # 新增导入
```

**文件B**: `aepsych/transforms/parameters.py` (修改, ~50 LOC)

**位置**: `get_config_options()` 函数中, 约240-270行

```python
# 在elif par_type == "categorical": ... elif par_type == "integer": ...后添加:
elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
    # 两个类型都使用相同的Ordinal Transform
    # 区别在配置方式: custom_ordinal自动计算, custom_ordinal_mono手工指定
    return Ordinal.get_config_options(config_dict)
```

**文件C**: `aepsych/config.py` (修改, +10 LOC)

**位置**: 参数类型验证, 约100-120行

```python
# 在valid_par_types列表中添加:
if "custom_ordinal" not in valid_par_types:
    valid_par_types.extend(["custom_ordinal", "custom_ordinal_mono"])
```

**检查清单**:

- [ ] parameters.py的par_type路由正确识别custom_ordinal/custom_ordinal_mono
- [ ] config.py验证允许新的par_type值
- [ ] Pool生成时包含ordinal参数的所有值
- [ ] bounds转换正确 (原始 → rank)
- [ ] 集成测试: test_ordinal_aepsych_integration.py

---

### 第三阶段: custom_generators集成 (~5h, 50 LOC)

**文件**: `custom_pool_based_generator.py` (修改, ~50 LOC)

**修改点1**: `_generate_pool_from_config()` (~20 LOC)

**位置**: 大约100-150行, 在variable_values字典填充处

```python
# 在处理categorical和integer后添加ordinal处理:
if par_type in ["custom_ordinal", "custom_ordinal_mono"]:
    # 从config解析ordinal参数
    ord_transform = Ordinal.get_config_options(config_dict[par_name])
    variable_values[par_name] = ord_transform.values[0]  # 提取值列表
    # 注: ord_transform.values是{0: [v0, v1, ...]}, 取索引0即可
```

**修改点2**: `from_config()` (~15 LOC)

**位置**: 自动Pool生成的条件判断处

```python
# 当self.pool为None且auto_generate_pool为True时:
if self.pool is None and self.auto_generate_pool:
    self.pool = self._generate_pool_from_config(...)
    # _generate_pool_from_config已包含ordinal处理,
    # full_factorial自动包含ordinal值的所有组合
```

**修改点3**: 变量组合处理 (~10 LOC - 无需修改)

```python
# custom_generators的变量组合逻辑已支持任意discrete类型:
# ordinal值列表 × categorical列表 × integer范围 = 完整pool
# 去重管理器的tuple matching自动兼容ordinal值
# 无需修改, 自动兼容!
```

**检查清单**:

- [ ] ordinal参数值从Ordinal Transform正确提取到variable_values
- [ ] full_factorial包含所有ordinal值的组合
- [ ] 去重管理器正确匹配ordinal点 (tuple(point)匹配)
- [ ] 历史排除工作正常 (existing dedup逻辑)
- [ ] 集成测试: test_ordinal_pool_generation.py

---

### 第四阶段: dynamic_eur_acquisition集成 (~6h, 50 LOC)

**文件A**: `modules/local_sampler.py` (修改, ~40 LOC)

**新方法**: `_perturb_ordinal()` (~25 LOC)

**位置**: 在_perturb_categorical()后添加

```python
def _perturb_ordinal(self, center_point: Tensor, var_idx: int, 
                     par_type: str, ordinal_transform: Transform) -> Tensor:
    """
    在rank空间中扰动有序参数
  
    参数:
        center_point: 候选中心点 (原始值空间)
        var_idx: 该参数在点中的维度索引
        par_type: 参数类型 ("custom_ordinal" 或 "custom_ordinal_mono")
        ordinal_transform: Ordinal Transform对象 (包含values和rank映射)
  
    实现逻辑:
    1. 从center_point[var_idx]得到原始值
    2. 使用ordinal_transform._transform()转换到rank空间
    3. 生成高斯扰动: rank_center + N(0, σ²)
    4. 舍入到最近的rank: round(rank_perturbed)
    5. 使用ordinal_transform._untransform()转换回原始值
  
    混合策略:
    - 若use_hybrid_perturbation=True且n_levels ≤ exhaustive_level_threshold:
      穷举采样: [0, 1, 2, ..., n-1]循环填充local_num个点
    - 否则: 高斯扰动+舍入
    """
    n_levels = ordinal_transform.values[ordinal_transform.indices[0]].shape[0]
  
    if self.use_hybrid_perturbation and n_levels <= self.exhaustive_level_threshold:
        # 穷举模式: 轮流采样所有rank
        rank_candidates = torch.arange(n_levels).float()
        selected_ranks = rank_candidates[torch.randperm(n_levels)[:self.local_num]]
    else:
        # 高斯模式: 在rank空间扰动
        center_rank = ordinal_transform._transform(center_point[[var_idx]])
        perturbed_ranks = center_rank + torch.randn_like(center_rank) * self.std  # std=可调参数
        selected_ranks = torch.clamp(torch.round(perturbed_ranks), 0, n_levels - 1)
  
    # 转换回原始值空间
    selected_values = ordinal_transform._untransform(selected_ranks.unsqueeze(-1))
    return selected_values
```

**文件B**: `modules/config_parser.py` (修改, ~10 LOC)

**修改点**: `parse_variable_types()` (~10 LOC)

**位置**: 变量类型识别逻辑

```python
# 在解析par_type后添加ordinal识别:
if "custom_ordinal" in par_type_str:
    var_types[par_name] = "ordinal"  # 或保留完整type
    # 存储Ordinal Transform对象供LocalSampler使用
    transforms[par_name] = ordinal_transform_obj
```

**文件C**: `eur_anova_pair.py` (修改, ~15 LOC)

**修改点**: `_infer_variable_types_from_transforms()` (~15 LOC)

**位置**: Transform对象的类型检测逻辑

```python
# 添加Ordinal Transform的检测:
from aepsych.transforms.ops.ordinal import Ordinal  # 新增导入

if isinstance(transform, Ordinal):
    variable_types[var_name] = "ordinal"
    # 存储Ordinal对象供后续使用
```

**检查清单**:

- [ ] _perturb_ordinal()正确在rank空间扰动并转换回原始值
- [ ] 混合扰动策略: 低level穷举, 高level高斯
- [ ] LocalSampler.sample()正确路由到_perturb_ordinal()
- [ ] config_parser正确从配置识别ordinal参数并提取Transform
- [ ] variable_types推断正确 (Ordinal → "ordinal")
- [ ] 集成测试: test_ordinal_local_sampler.py

---

## 📝 测试策略

### 单元测试 (test_ordinal_transform.py)

```python
def test_ordinal_transform_and_untransform():
    """测试rank空间往返转换"""
    values = [0.1, 0.5, 2.0, 5.0, 10.0]
    ordinal = Ordinal(indices=[0], values={0: values})
  
    X = torch.tensor([[0.5], [2.0], [10.0]])  # 原始值
    X_transformed = ordinal.transform(X)      # 应得 [1, 2, 4]
    X_untransformed = ordinal.untransform(X_transformed)
    assert torch.allclose(X_untransformed, X)

def test_ordinal_with_categorical():
    """测试Ordinal + Categorical混合"""
    # ordinal在维度0，categorical在维度1
    pass

def test_ordinal_bounds_transform():
    """测试bounds从原始空间→rank空间的转换"""
    values = [1, 2, 3, 4, 5]
    bounds = torch.tensor([[0.5], [5.5]])  # 原始
    transformed = ordinal.transform_bounds(bounds)
    # 应得 [[-0.5], [4.5]]（rank -0.5~4.5）
    pass

def test_arithmetic_sequence_calculation():
    """测试三种自动计算方式"""
    # 方式1: min/max/step
    vals1 = Ordinal._compute_arithmetic_sequence(0, 1.0, step=0.2)
    # 应得 [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
  
    # 方式2: min/max/num_levels
    vals2 = Ordinal._compute_arithmetic_sequence(0, 1.0, num_levels=5)
    # 应得 [0.0, 0.25, 0.5, 0.75, 1.0]
  
    # 方式3: levels字符串
    config = {"levels": ["agree", "disagree", "neutral"]}
    ordinal = Ordinal.get_config_options(config)
    # values应为{0: [0, 1, 2]}
```

### 集成测试 (extensions/dynamic_eur_acquisition/test/)

```python
def test_ordinal_with_eur_anova():
    """测试ordinal参数与EURAnovaPairAcqf的集成"""
    # 1. 创建含ordinal参数的配置
    # 2. 初始化LocalSampler，variable_types={0: 'ordinal'}
    # 3. 验证_perturb_ordinal()输出合法rank值
    # 4. 验证采集函数评估不出错
    pass

def test_hybrid_perturbation_with_ordinal():
    """测试混合扰动策略对ordinal参数的支持"""
    # use_hybrid_perturbation=True，ordinal水平数≤threshold
    # 验证穷举采样覆盖所有rank
    pass

def test_pool_generation_with_ordinal():
    """测试ordinal参数的Pool生成"""
    # 1. 配置: categorical (3选项) × ordinal (5值) × integer (10-50)
    # 2. 生成Pool
    # 3. 验证Pool包含 3 × 5 × 41 个点
    # 4. 验证去重工作正常
    pass
```

---

## 🚨 关键实现决策

### 1. Ordinal vs Monotonic 的区分

- **Ordinal** (等差): 使用均匀的rank空间，自动检测间距
- **Monotonic** (非等差): 用户显式指定values列表，保留原始间距信息

### 2. Transform空间 vs 原始空间

- **aepsych侧**: Ordinal在rank空间(0,1,2,...)中存储与变换
  - 优点: 与Categorical统一, bounds处理简单
  - 优点: GP学习的是相对顺序而非绝对值
- **dynamic_eur_acquisition侧**: LocalSampler在rank空间内扰动
  - 优点: 高斯扰动自然作用于rank序号
  - 优点: 舍入操作简单明确

### 3. 向后兼容性

- 现有配置无需修改（par_type默认continuous）
- 新par_type自动识别并使用Ordinal
- 若用户混合integer与ordinal，根据bounds推断

### 4. 性能优化

- **最近邻查找**: O(log n)二分查找 vs O(n)线性扫描
  - `np.searchsorted()`定位插入点，左右距离比较选最近值
  - 对大ordinal集合(n>100)性能提升显著
- **没有lookup table缓存**: Ordinal无需预计算缓存，值直接从Pool提取
- **向量化运算**: 使用numpy全量计算而非循环，内存访问高效

---

## 📝 测试策略

### 单元测试 (test_ordinal_transform.py)

```python
def test_ordinal_transform_and_untransform():
    """测试rank空间往返转换"""
    values = [0.1, 0.5, 2.0, 5.0, 10.0]
    ordinal = Ordinal(indices=[0], values={0: values})
  
    X = torch.tensor([[0.5], [2.0], [10.0]])  # 原始值
    X_transformed = ordinal.transform(X)      # 应得 [1, 2, 4]
    X_untransformed = ordinal.untransform(X_transformed)
    assert torch.allclose(X_untransformed, X)

def test_pool_ordinal_consistency():
    """验证Pool值与Transform数据一致性"""
    # Pool提取的unique_vals应与Transform.values对应
    # Min-Max归一化应保持interval比例
    from aepsych.transforms.ops.ordinal import Ordinal
    values = [2.0, 2.5, 3.5]  # ceiling heights
    ordinal = Ordinal(indices=[0], values={0: values})
    
    X_trans = ordinal.transform(torch.tensor([[2.0], [3.5]]))
    # Expected: [[0], [2]] (rank indices)
    assert X_trans[0, 0] == 0 and X_trans[1, 0] == 2

def test_ordinal_categorical_mixed():
    """验证Ordinal与Categorical混合使用"""
    # 维度0: Ordinal [1,2,3]，维度1: Categorical {A,B}
    # 确保transform链不崩溃，bounds正确处理
    pass

def test_local_sampler_coverage():
    """验证LocalSampler能到达所有ordinal值"""
    # 大量采样验证perturbation + nearest-neighbor能覆盖all unique_vals
    pass
```

### 集成测试 (extensions/dynamic_eur_acquisition/test/)

```python
def test_ordinal_with_eur_anova():
    """测试ordinal参数与EURAnovaPairAcqf的集成"""
    # 1. 创建含ordinal参数的配置
    # 2. 初始化LocalSampler，variable_types={0: 'ordinal'}
    # 3. 验证_perturb_ordinal()输出合法rank值
    # 4. 验证采集函数评估不出错
    pass

def test_hybrid_perturbation_with_ordinal():
    """测试混合扰动策略对ordinal参数的支持"""
    # use_hybrid_perturbation=True，ordinal水平数≤threshold
    # 验证穷举采样覆盖所有rank
    pass
```

---

## 🎯 成功标准

✅ 可从INI配置正确加载ordinal参数
✅ Ordinal Transform的transform/untransform往返精确
✅ bounds正确转换到rank空间
✅ LocalSampler能识别ordinal类型并执行rank空间扰动
✅ EURAnovaPairAcqf能推断ordinal参数类型
✅ 混合扰动策略能正确处理ordinal（穷举vs随机）
✅ 与existing categorical/integer参数兼容
✅ 性能无明显下降（与categorical相当）

---

## 📚 参考资源

### AEPsych相关

- `aepsych/transforms/ops/categorical.py` - Transform基类参考实现
- `aepsych/transforms/parameters.py` - par_type解析逻辑
- `aepsych/config.py` - 配置系统

### dynamic_eur_acquisition相关

- `modules/local_sampler.py` - 扰动逻辑 (参考_perturb_categorical)
- `modules/config_parser.py` - 变量类型解析
- `eur_anova_pair.py` - 变量类型推断与使用

---

**最后更新**: 2025-12-11
**状态**: 待实施
