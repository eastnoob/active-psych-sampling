# Ordinal参数实现：规范化值方案

**基于评审反馈的修订设计**
**日期**: 2025-12-11

---

## 核心设计原则

### 原则1: Transform产生保留间距的规范化值

```
物理值空间          规范化值空间 (模型输入)      内部Rank空间 (可选)
[2.0, 2.5, 3.5]  →  [0.0, 0.25, 1.0]      →  [0, 1, 2]
   ↑                      ↑                        ↑
 用户配置            GP/ANOVA看到的值        离散约束用
```

**关键**: 模型空间使用 `[0.0, 0.25, 1.0]` 而非 `[0, 1, 2]`

### 原则2: LocalSampler在规范化值空间扰动

- `base[:,:,k]` 包含规范化值 (e.g., 0.25)
- 高斯扰动: `0.25 + N(0, σ)` → 最近邻约束 → `{0.0, 0.25, 1.0}`
- 无需Transform对象，只需知道有效的规范化值列表

### 原则3: ANOVA看到正确间距

当分解主效应时:
```python
# 参数效应计算
effect_of_height = model.predict([0.0, ...]) vs model.predict([0.25, ...]) vs model.predict([1.0, ...])
# ANOVA看到：0→0.25 (小增量) vs 0.25→1.0 (大增量)
# 这正确反映了 2.0→2.5 (0.5m) vs 2.5→3.5 (1.0m) 的物理关系
```

---

## Ordinal Transform实现

### 核心类设计

```python
class Ordinal(Transform, StringParameterMixin):
    """有序参数Transform - 输出保留间距的规范化值"""

    def __init__(
        self,
        indices: list[int],
        values: dict[int, list[float]],  # {0: [2.0, 2.5, 3.5]}
        level_names: Optional[dict[int, list[str]]] = None,
    ):
        super().__init__()
        self.indices = indices
        self.values = values  # 原始物理值
        self.level_names = level_names

        # 计算规范化映射
        self._build_normalized_mappings()

    def _build_normalized_mappings(self):
        """构建物理值 ↔ 规范化值的双向映射"""
        self.normalized_values = {}  # {index: [norm_v0, norm_v1, ...]}
        self.physical_to_normalized = {}  # {index: {phys_val: norm_val}}
        self.normalized_to_physical = {}  # {index: {norm_val: phys_val}}

        for idx in self.indices:
            phys_vals = np.array(self.values[idx], dtype=np.float64)

            # Min-max归一化到[0, 1]
            min_val = phys_vals.min()
            max_val = phys_vals.max()

            if max_val - min_val < 1e-10:
                # 所有值相同，归一化为0
                norm_vals = np.zeros_like(phys_vals)
            else:
                norm_vals = (phys_vals - min_val) / (max_val - min_val)

            # 保存映射
            self.normalized_values[idx] = norm_vals

            # 构建双向字典 (处理浮点精度)
            self.physical_to_normalized[idx] = {
                round(p, 10): round(n, 10)
                for p, n in zip(phys_vals, norm_vals)
            }
            self.normalized_to_physical[idx] = {
                round(n, 10): round(p, 10)
                for n, p in zip(norm_vals, phys_vals)
            }

    @subset_transform
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        """物理值 → 规范化值

        输入: [[2.5], [3.5], [2.0]]
        输出: [[0.25], [1.0], [0.0]]
        """
        X_normalized = X.clone()

        for i, idx in enumerate(self.indices):
            phys_vals = X[..., i].cpu().numpy()
            norm_vals = np.zeros_like(phys_vals)

            # 查表转换
            phys_to_norm = self.physical_to_normalized[idx]
            for j, pv in enumerate(phys_vals.flat):
                pv_rounded = round(pv, 10)
                if pv_rounded not in phys_to_norm:
                    # 最近邻匹配 (容错)
                    closest = min(phys_to_norm.keys(), key=lambda x: abs(x - pv_rounded))
                    norm_vals.flat[j] = phys_to_norm[closest]
                else:
                    norm_vals.flat[j] = phys_to_norm[pv_rounded]

            X_normalized[..., i] = torch.from_numpy(norm_vals).to(dtype=X.dtype)

        return X_normalized

    @subset_transform
    def _untransform(self, X: torch.Tensor) -> torch.Tensor:
        """规范化值 → 物理值

        输入: [[0.25], [1.0], [0.0]]
        输出: [[2.5], [3.5], [2.0]]
        """
        X_physical = X.clone()

        for i, idx in enumerate(self.indices):
            norm_vals = X[..., i].cpu().numpy()
            phys_vals = np.zeros_like(norm_vals)

            # 查表转换
            norm_to_phys = self.normalized_to_physical[idx]
            for j, nv in enumerate(norm_vals.flat):
                nv_rounded = round(nv, 10)
                if nv_rounded not in norm_to_phys:
                    # 最近邻匹配
                    closest = min(norm_to_phys.keys(), key=lambda x: abs(x - nv_rounded))
                    phys_vals.flat[j] = norm_to_phys[closest]
                else:
                    phys_vals.flat[j] = norm_to_phys[nv_rounded]

            X_physical[..., i] = torch.from_numpy(phys_vals).to(dtype=X.dtype)

        return X_physical

    def transform_bounds(
        self,
        X: torch.Tensor,
        bound: Literal["lb", "ub"] | None = None,
        epsilon: float = 1e-6
    ) -> torch.Tensor:
        """物理边界 → 规范化边界

        输入: [[2.0], [3.5]] (物理值)
        输出: [[-ε], [1.0+ε]] (规范化值，加小偏移保证覆盖)
        """
        X_bounds = X.clone()

        for i, idx in enumerate(self.indices):
            # 规范化后的边界总是[0, 1]
            if bound == "lb":
                X_bounds[0, i] = -epsilon  # 下界稍微扩展
            elif bound == "ub":
                X_bounds[0, i] = 1.0 + epsilon  # 上界稍微扩展
            else:  # both bounds
                X_bounds[0, i] = -epsilon
                X_bounds[1, i] = 1.0 + epsilon

        return X_bounds

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str,
        options: dict = None
    ) -> dict:
        """从配置解析ordinal参数"""
        options = options or {}

        # 优先级1: 直接指定values
        if "values" in options:
            values = options["values"]
            return {"indices": [0], "values": {0: list(values)}}

        # 优先级2: min/max + step
        if "min_value" in options and "max_value" in options:
            min_val = float(options["min_value"])
            max_val = float(options["max_value"])

            if "step" in options:
                step = float(options["step"])
                # 使用linspace避免累积误差
                num_steps = int(round((max_val - min_val) / step)) + 1
                values = np.linspace(min_val, max_val, num_steps)
            elif "num_levels" in options:
                num_levels = int(options["num_levels"])
                values = np.linspace(min_val, max_val, num_levels)
            else:
                raise ValueError(
                    f"[{name}] Must specify 'step' or 'num_levels' with min/max_value"
                )

            return {"indices": [0], "values": {0: list(values)}}

        # 优先级3: levels (字符串标签)
        if "levels" in options:
            levels = options["levels"]
            if isinstance(levels, str):
                levels = [s.strip() for s in levels.split(',')]

            # 字符串标签 → 整数序列 (等差)
            values = list(range(len(levels)))
            return {
                "indices": [0],
                "values": {0: values},
                "level_names": {0: levels}
            }

        raise ValueError(
            f"[{name}] Must specify one of:\n"
            "  1. 'values' (direct list)\n"
            "  2. 'min_value' + 'max_value' + ('step' or 'num_levels')\n"
            "  3. 'levels' (string labels)"
        )
```

---

## 🚨 关键数据流：unique_vals_dict的初始化

### 问题分析

**核心问题**: LocalSampler的 `_unique_vals_dict` 需要包含规范化值 `[0.0, 0.333, 1.0]`，但这个数据从哪里来？

**数据流追踪**:

```python
# 1. Transform层生成规范化值
ordinal_transform = Ordinal(...)
ordinal_transform.normalized_values = {0: [0.0, 0.333, 1.0]}

# 2. Pool生成使用规范化值
pool = [[0.0], [0.333], [1.0]]  # ✓ 包含规范化值

# 3. LocalSampler初始化 ← 🚨 缺失环节！
local_sampler = LocalSampler(
    variable_types={0: 'ordinal'},
    unique_vals_dict={0: ???}  # ← 从哪里获取 [0.0, 0.333, 1.0]？
)
```

### 解决方案：从Pool直接提取（最简方案）

**核心洞察**: Pool已经包含了正确的规范化值，直接从pool提取即可！

#### 修改：LocalSampler初始化支持从pool自动提取

```python
class LocalSampler:
    def __init__(
        self,
        local_num: int,
        local_jitter_frac: float,
        variable_types: Dict[int, str],
        pool: torch.Tensor = None,  # ← 新增pool参数
        unique_vals_dict: Dict[int, np.ndarray] = None,
        use_hybrid_perturbation: bool = False,
        ...
    ):
        # 优先使用显式提供的unique_vals_dict
        if unique_vals_dict is not None:
            self._unique_vals_dict = unique_vals_dict
        elif pool is not None:
            # 🔑 关键：从pool自动提取unique值
            self._unique_vals_dict = self._extract_unique_vals_from_pool(
                pool, variable_types
            )
        else:
            self._unique_vals_dict = {}
            warnings.warn(
                "LocalSampler initialized without pool or unique_vals_dict. "
                "Ordinal/categorical perturbation may not work correctly."
            )

    @staticmethod
    def _extract_unique_vals_from_pool(
        pool: torch.Tensor,
        variable_types: Dict[int, str]
    ) -> Dict[int, np.ndarray]:
        """从pool提取ordinal/categorical的unique值

        Args:
            pool: 候选点pool (已经在规范化值空间)
            variable_types: 变量类型字典

        Returns:
            unique_vals_dict: {dim_idx: unique_values_array}
        """
        unique_vals_dict = {}

        for k in range(pool.shape[1]):
            vt = variable_types.get(k)

            if vt in ["ordinal", "custom_ordinal", "custom_ordinal_mono"]:
                # Ordinal: 需要排序以便最近邻查找正确工作
                # torch.unique()自动排序，但我们显式调用np.sort()确保意图清晰
                unique_vals = torch.unique(pool[:, k]).cpu().numpy()
                unique_vals = np.sort(unique_vals)  # 确保升序排列
                unique_vals_dict[k] = unique_vals

                logger.debug(
                    f"[LocalSampler] Extracted {len(unique_vals)} unique ordinal values "
                    f"for dimension {k}: {unique_vals} (sorted)"
                )

            elif vt == "categorical":
                # Categorical: 顺序不重要，但保持一致性也排序
                unique_vals = torch.unique(pool[:, k]).cpu().numpy()
                unique_vals_dict[k] = unique_vals  # 不需要额外排序

                logger.debug(
                    f"[LocalSampler] Extracted {len(unique_vals)} unique categorical values "
                    f"for dimension {k}: {unique_vals}"
                )

        return unique_vals_dict
```

#### 使用示例

```python
# 在EURAnovaPairAcqf初始化LocalSampler
class EURAnovaPairAcqf:
    def __init__(self, model, pool, ...):
        # 解析变量类型（保持现有逻辑）
        variable_types = self._infer_variable_types_from_transforms(model.transforms)

        # 初始化LocalSampler - 自动从pool提取unique值
        self.local_sampler = LocalSampler(
            local_num=self.local_num,
            local_jitter_frac=self.local_jitter_frac,
            variable_types=variable_types,
            pool=pool,  # ← 只需传入pool，自动提取！
            use_hybrid_perturbation=self.use_hybrid_perturbation,
            ...
        )
```

#### Advantages of Pool-Based Extraction

**Principle 1: Simplicity**
- Implementation: 3 core lines of logic in `_extract_unique_vals_from_pool()`
- No new config_parser functions required
- Single responsibility: LocalSampler extracts what it needs from pool

**Principle 2: Data Consistency**
- Pool serves as single source of truth for normalized values
- Avoids synchronization issues between Transform and Pool
- Direct extraction ensures what model sees matches what LocalSampler perturbs

**Principle 3: Zero Breaking Changes**
- Pool parameter is optional in LocalSampler.__init__()
- Existing code paths remain unchanged
- Backward compatible with manual unique_vals_dict provision

**Principle 4: Automatic Operation**
- Pool already contains normalized values from Transform
- No additional Transform object dependency in LocalSampler
- Extraction happens transparently during initialization

**Principle 5: Architectural Alignment**
- Follows EUR's existing pattern: Pool → LocalSampler
- Matches categorical/integer handling philosophy
- Maintains separation of concerns: Transform for conversion, Pool for candidates, LocalSampler for perturbation

### Complete Data Flow: Pool Extraction Approach

**Overview**: This design combines the best of both approaches - Transform handles normalization, Pool serves as single source of truth, and LocalSampler extracts directly from Pool.

```yaml
Stage_1_Configuration_Parsing:
  input:
    - config_file: "[height] par_type=custom_ordinal, values=[2.0, 2.5, 3.5]"
  process:
    - action: "Initialize Ordinal Transform"
    - computation: "min_max_normalization"
    - formula: "(value - min) / (max - min)"
    - example: "(2.0-2.0)/(3.5-2.0)=0.0, (2.5-2.0)/1.5=0.333, (3.5-2.0)/1.5=1.0"
  output:
    - transform_object: "Ordinal"
    - normalized_values: "[0.0, 0.333, 1.0]"
    - physical_to_normalized_map: "{2.0: 0.0, 2.5: 0.333, 3.5: 1.0}"

Stage_2_Pool_Generation:
  input:
    - ordinal_transform: "from Stage_1"
  process:
    - function: "CustomPoolBasedGenerator._generate_pool_from_config()"
    - action: "Extract normalized_values from Transform.normalized_values[0]"
    - note: "Pool stores normalized values, NOT physical values"
  output:
    - pool_tensor: "torch.tensor([[0.0], [0.333], [1.0]])"
    - data_type: "float (normalized)"
    - interpretation: "Each pool point is already in model input space"

Stage_3_LocalSampler_Initialization:
  design_choice: "Pool-based extraction (NEW)"
  rationale:
    - pool_already_contains: "normalized values from Stage_2"
    - no_transform_dependency: "LocalSampler doesn't need Transform object"
    - single_source_of_truth: "Pool is authoritative"
  input:
    - pool: "from Stage_2"
    - variable_types: "{0: 'ordinal'}"
  process:
    - function: "LocalSampler.__init__(pool=pool, variable_types={0: 'ordinal'})"
    - conditional_check:
        if_unique_vals_dict_provided:
          action: "Use provided dict directly"
        elif_pool_provided:
          action: "Call _extract_unique_vals_from_pool()"
          implementation: "torch.unique(pool[:, 0]).cpu().numpy()"
          result: "Extract [0.0, 0.333, 1.0] from pool"
        else:
          action: "Warn and use empty dict"
  output:
    - local_sampler_object: "LocalSampler"
    - internal_state: "_unique_vals_dict = {0: np.array([0.0, 0.333, 1.0])}"
    - advantage: "Direct extraction, zero Transform dependency"

Stage_4_Perturbation:
  input:
    - candidate_point: "torch.tensor([[0.333]])  # normalized value"
    - unique_vals_dict: "from Stage_3: {0: [0.0, 0.333, 1.0]}"
  process:
    - function: "_perturb_ordinal(base, k=0, B)"
    - step_1_retrieve: "unique_vals = self._unique_vals_dict[0]  # [0.0, 0.333, 1.0]"
    - step_2_perturb:
        method: "Gaussian noise in normalized space"
        formula: "perturbed = 0.333 + N(0, sigma)"
        sigma: "0.1 * 1.0 = 0.1  # 10% of normalized range"
        example_samples: "[0.283, 0.453, 0.253, ...]"
    - step_3_constrain:
        method: "Nearest neighbor to valid values"
        implementation: "np.argmin(np.abs(unique_vals - perturbed_value))"
        example_mapping:
          - "0.283 → 0.333 (closest)"
          - "0.453 → 0.333 (distance 0.12 to 0.333, distance 0.547 to 1.0)"
          - "0.253 → 0.333 (closest)"
  output:
    - perturbed_samples: "[0.333, 0.333, 0.333, 1.0, 0.0, ...]  # valid normalized values"
    - guarantee: "All outputs are in normalized space and match pool values"

Key_Design_Insight:
  data_flow: "Transform → Pool → LocalSampler"
  extraction_point: "Stage_3 extracts from Pool (Stage_2 output)"
  not_extraction_point: "Stage_3 does NOT go back to Transform (Stage_1)"
  reason: "Pool is single source of truth after Stage_2"
  benefit: "Simpler dependency graph, automatic synchronization"
```

**Critical Implementation Note**: The `_extract_unique_vals_from_pool()` method is called during LocalSampler initialization, making the extraction automatic and transparent. Users only need to pass the pool parameter.

---

## LocalSampler集成

### _perturb_ordinal实现

```python
def _perturb_ordinal(
    self,
    base: torch.Tensor,
    k: int,
    B: int
) -> torch.Tensor:
    """有序参数扰动：在规范化值空间扰动 + 最近邻约束

    假设: base[:,:,k] 包含规范化值 (e.g., [0.0, 0.25, 1.0])
    """
    # 获取有效的规范化值列表
    unique_normalized_vals = self._unique_vals_dict.get(k)

    if unique_normalized_vals is None or len(unique_normalized_vals) == 0:
        warnings.warn(f"Ordinal dimension {k}: no unique values found, keeping original")
        return base

    unique_vals = np.array(unique_normalized_vals, dtype=np.float64)
    n_levels = len(unique_vals)

    # 混合策略
    if (self.use_hybrid_perturbation and
        n_levels <= self.exhaustive_level_threshold):
        # ========== 穷举模式 ==========
        if self.exhaustive_use_cyclic_fill:
            n_repeats = (self.local_num // n_levels) + 1
            samples = np.tile(unique_vals, (B, n_repeats))
            samples = samples[:, :self.local_num]
        else:
            samples = np.tile(unique_vals, (B, 1))

        base[:, :samples.shape[1], k] = torch.from_numpy(samples).to(
            dtype=base.dtype, device=base.device
        )
    else:
        # ========== 高斯扰动模式 ==========
        # 规范化值空间的span总是1.0 (因为已归一化到[0,1])
        span = 1.0
        sigma = self.local_jitter_frac * span  # e.g., 0.1 * 1.0 = 0.1

        # 在规范化值空间扰动
        center_vals = base[:, :, k].cpu().numpy()  # (B, local_num)
        noise = self._np_rng.normal(0, sigma, size=(B, self.local_num))
        perturbed = center_vals + noise

        # 约束到最近的有效规范化值
        samples = np.zeros_like(perturbed)
        for i in range(B):
            for j in range(self.local_num):
                closest_idx = np.argmin(np.abs(unique_vals - perturbed[i, j]))
                samples[i, j] = unique_vals[closest_idx]

        base[:, :, k] = torch.from_numpy(samples).to(
            dtype=base.dtype, device=base.device
        )

    return base
```

### sample()方法集成

```python
def sample(self, X_can_t: torch.Tensor, dims: Sequence[int]) -> torch.Tensor:
    """生成局部扰动点"""
    B, d = X_can_t.shape
    base = X_can_t.unsqueeze(1).expand(-1, self.local_num, -1)

    for k in dims:
        vt = self.variable_types.get(k) if self.variable_types else None

        if vt == "categorical":
            base = self._perturb_categorical(base, k, B)
        elif vt in ["ordinal", "custom_ordinal", "custom_ordinal_mono"]:
            base = self._perturb_ordinal(base, k, B)  # ← 新增
        elif vt == "integer":
            base = self._perturb_integer(base, k, B, mn[k], mx[k], span[k])
        else:  # continuous
            base = self._perturb_continuous(base, k, B, mn[k], mx[k], span[k])

    return base.reshape(B * self.local_num, d)
```

---

## CustomPoolBasedGenerator集成

### Pool生成使用规范化值

```python
def _generate_pool_from_config(cls, config: Config) -> torch.Tensor:
    """生成pool，ordinal参数使用规范化值"""
    param_choices_values = []

    for par_name in parnames:
        par_type = config.get(par_name, "par_type", "continuous")

        if par_type == "categorical":
            choices = config.getlist(par_name, "choices")
            # Categorical用索引 [0, 1, 2, ...]
            param_choices_values.append(list(range(len(choices))))

        elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
            from aepsych.transforms.ops.ordinal import Ordinal

            # 创建Ordinal transform
            options = {}
            for key in ["values", "min_value", "max_value", "step", "num_levels", "levels"]:
                if config.has_option(par_name, key):
                    options[key] = config.get(par_name, key)

            ordinal_config = Ordinal.get_config_options(config, par_name, options)
            ordinal = Ordinal(**ordinal_config)

            # 使用规范化值
            normalized_vals = ordinal.normalized_values[ordinal.indices[0]]
            param_choices_values.append(list(normalized_vals))

            logger.info(
                f"[PoolGen] Added ordinal param '{par_name}' with {len(normalized_vals)} "
                f"levels (normalized values: {normalized_vals})"
            )

        elif par_type == "integer":
            lb = config.getint(par_name, "lb")
            ub = config.getint(par_name, "ub")
            param_choices_values.append(list(range(lb, ub + 1)))

    # 生成完整组合
    pool = generate_full_factorial(param_choices_values)
    return pool
```

---

## 工作流示例

### 示例1: 非等差物理参数

```ini
[height]
par_type = custom_ordinal
values = [2.0, 2.5, 3.5]  # 非等差
```

**处理流程**:

```
1. 用户配置物理值
   values = [2.0, 2.5, 3.5]

2. Ordinal Transform初始化
   min = 2.0, max = 3.5, span = 1.5
   normalized_values = [(2.0-2.0)/1.5, (2.5-2.0)/1.5, (3.5-2.0)/1.5]
                     = [0.0, 0.333, 1.0]

3. Pool生成
   param_choices_values = [[0.0, 0.333, 1.0]]
   pool = torch.tensor([[0.0], [0.333], [1.0]])

4. LocalSampler扰动
   base = [[0.333, 0.333, ...]]  # 中心点: 规范化值0.333
   noise = N(0, 0.1)  # σ = 0.1 * 1.0
   perturbed = 0.333 + [-0.05, 0.12, -0.08, ...]
             = [0.283, 0.453, 0.253, ...]

   最近邻约束:
   0.283 → 0.333 (最近)
   0.453 → 0.333 (与0.333距离0.12，与1.0距离0.547)
   0.253 → 0.333 (最近)
   ...

   samples = [0.333, 0.333, 0.333, 1.0, 0.0, ...]

5. GP训练
   X_train包含规范化值 [0.0, 0.333, 1.0, ...]
   GP学到: f(0.0) vs f(0.333) vs f(1.0)

6. ANOVA分解
   主效应: Δ(0.0→0.333) vs Δ(0.333→1.0)
   ANOVA看到: 0.333间距 vs 0.667间距
   这正确反映了物理上 0.5m vs 1.0m 的比例关系！

7. 用户查询结果
   untransform: [0.333] → [2.5m]
```

### 示例2: 等差Likert量表

```ini
[agreement]
par_type = custom_ordinal
min_value = 1
max_value = 5
step = 1
```

**处理流程**:

```
1. 配置解析
   values = [1, 2, 3, 4, 5]

2. 规范化
   normalized_values = [0.0, 0.25, 0.5, 0.75, 1.0]

3. Pool生成
   pool包含 [0.0, 0.25, 0.5, 0.75, 1.0]

4. ANOVA看到等间距
   Δ = 0.25 (所有相邻级别间距相同)
   这符合Likert量表的心理学假设
```

---

## 优势总结

### 1. 保留物理间距信息 ✅

- ANOVA分解看到正确的相对间距
- GP学习时利用间距结构
- 效应估计更准确

### 2. 架构一致性 ✅

- LocalSampler无需修改签名
- 与categorical/integer处理模式统一
- 无破坏性变更

### 3. 实现简洁 ✅

- Transform负责归一化逻辑
- LocalSampler只需最近邻约束
- Pool生成自然包含规范化值

### 4. 数学合理性 ✅

- 规范化到[0,1]是标准做法
- 保留间距比例信息
- 离散约束通过最近邻实现

### 5. 向后兼容 ✅

- 现有参数类型不受影响
- 配置格式保持简洁
- API无破坏性变更

---

## 与原计划对比

| 方面 | 原计划 (rank空间) | 修订方案 (规范化值) |
|------|------------------|--------------------|
| **模型输入** | [0, 1, 2] (纯rank) | [0.0, 0.25, 1.0] (保留间距) |
| **ANOVA间距** | ❌ 看到等间距1 | ✅ 看到真实比例 |
| **LocalSampler** | rank空间扰动 | 规范化值空间扰动 |
| **Transform复杂度** | 简单 (值↔rank) | 中等 (值↔规范化值) |
| **物理含义** | ❌ 丢失间距信息 | ✅ 保留相对间距 |
| **架构一致性** | ✅ 零破坏性变更 | ✅ 零破坏性变更 |

---

## 实现检查清单

### Phase 1: Ordinal Transform (Day 1)

- [ ] 实现 `_build_normalized_mappings()`
- [ ] 实现 `_transform()` (物理值→规范化值)
- [ ] 实现 `_untransform()` (规范化值→物理值)
- [ ] 实现 `transform_bounds()` (规范化边界)
- [ ] 实现 `get_config_options()` (配置解析)
- [ ] 单元测试: 间距保留验证
- [ ] 单元测试: 浮点精度边界case

### Phase 2: AEPsych集成 (Day 1-2)

- [ ] 修改 `aepsych/transforms/parameters.py`
- [ ] 更新 `aepsych/config.py` par_type验证
- [ ] 测试Transform往返一致性
- [ ] 测试bounds转换

### Phase 3: CustomPoolBasedGenerator集成 (Day 2)

- [ ] 修改 `_generate_pool_from_config()` 使用规范化值
- [ ] 验证pool包含正确的规范化值
- [ ] 测试pool去重逻辑

### Phase 4: LocalSampler集成 (Day 2-3)

- [ ] 实现 `_perturb_ordinal()` (规范化值空间扰动)
- [ ] 修改 `sample()` 路由到ordinal扰动
- [ ] 测试最近邻约束逻辑
- [ ] 测试混合扰动策略

### Phase 5: 测试与验证 (Day 3)

- [ ] 端到端测试: 非等差参数的间距保留
- [ ] ANOVA��证: 效应估计正确性
- [ ] 性能测试: vs categorical baseline
- [ ] 文档: 空间约定说明

---

**状态**: 设计完成，待实施
**推荐**: 立即采用此方案替代原rank空间方案
