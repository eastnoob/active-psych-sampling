# 🧑‍🎄 EUR ANOVA 算法实现细节

本文档详细介绍EUR ANOVA采集函数的各个组件实现，使用Python伪代码展示核心逻辑。每个部分包含：

- 功能概述
- 关键参数
- 实现伪代码
- 设计考虑

## 1. 核心采集函数 (EURAnovaMultiAcqf.forward)

### 功能概述

实现批量ANOVA分解和动态权重融合的主采集函数。支持多阶交互（主效应+二阶+三阶），通过一次性模型调用优化性能。

### 关键参数

- `X`: 候选点张量 (B, q, d) 或 (B, d)
- `effects`: 效应列表 (MainEffect, PairwiseEffect, ThreeWayEffect)
- `lambda_t`: 当前交互权重
- `gamma_t`: 当前覆盖权重

### 实现伪代码

```python
def forward(self, X: torch.Tensor) -> torch.Tensor:
    # 数据同步和预处理
    self._ensure_fresh_data()
    
    # 验证输入形状
    if X.dim() < 3:
        raise ValueError("输入必须为 (..., q, d) 形状")
    
    batch_shape = X.shape[:-2]
    q, d = X.shape[-2:]
    
    if q != 1:
        raise AssertionError("当前实现仅支持 q=1")
    
    X_flat = X.reshape(-1, d)
    X_canonical = self._canonicalize_torch(X_flat)
    
    # 初始化ANOVA引擎（延迟加载）
    if self.anova_engine is None:
        self.anova_engine = ANOVAEffectEngine(
            metric_fn=self._metric,
            local_sampler=self.local_sampler.sample
        )
    
    # 构造效应列表（基于配置）
    effects = create_effects_from_config(
        n_dims=d,
        enable_main=self.enable_main,
        interaction_pairs=self._pairs,
        interaction_triplets=self._triplets
    )
    
    # ========== 核心：批量效应计算 ==========
    results = self.anova_engine.compute_effects(X_canonical, effects)
    
    # 提取各阶效应贡献
    main_sum = results['aggregated'].get('order_1', torch.zeros_like(results['baseline']))
    pair_sum = results['aggregated'].get('order_2', torch.zeros_like(results['baseline']))
    triplet_sum = results['aggregated'].get('order_3', torch.zeros_like(results['baseline']))
    
    # 计算动态权重
    lambda_2_t = self.weight_engine.compute_lambda(model=self.model) if self.use_dynamic_lambda_2 else self.lambda_2
    gamma_t = self.weight_engine.compute_gamma(model=self.model)
    
    # 信息项融合
    info_raw = self.main_weight * main_sum
    if self.enable_pairwise and len(self._pairs) > 0:
        info_raw += lambda_2_t * pair_sum
    if self.enable_threeway and len(self._triplets) > 0:
        info_raw += self.lambda_3 * triplet_sum
    
    # 覆盖项计算
    cov_t = self.coverage_helper.compute_coverage(X_canonical)
    
    # 批内标准化
    def standardize(x):
        return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)
    
    info_normalized = standardize(info_raw)
    cov_normalized = standardize(cov_t)
    
    # 最终融合
    if self.fusion_method == "multiplicative":
        acquisition_values = info_normalized * (1.0 + gamma_t * cov_normalized)
    else:  # additive
        acquisition_values = info_normalized + gamma_t * cov_normalized
    
    # 平局抖动
    if acquisition_values.max() - acquisition_values.min() < 1e-9:
        acquisition_values += 1e-3 * torch.rand_like(acquisition_values)
    
    # 诊断信息更新
    self.diagnostics.update_effects(
        main_sum=main_sum, pair_sum=pair_sum, triplet_sum=triplet_sum,
        info_raw=info_raw, cov=cov_t
    )
    
    return acquisition_values.view(batch_shape)
```

## 2. ANOVA效应引擎 (ANOVAEffectEngine)

### 功能概述

层次化计算ANOVA效应，避免重复计算低阶效应。支持任意阶数扩展，通过依赖管理确保计算顺序。

### 实现伪代码

```python
class ANOVAEffectEngine:
    def __init__(self, metric_fn, local_sampler):
        self.metric_fn = metric_fn
        self.local_sampler = local_sampler
    
    def compute_effects(self, X_candidates, effects, enable_orders=None):
        B = X_candidates.shape[0]
        
        # 阶数过滤
        if enable_orders:
            effects = [e for e in effects if e.order in enable_orders]
        
        if not effects:
            baseline = self.metric_fn(X_candidates)
            return self._empty_results(baseline)
        
        # ========== 批量扰动点构造 ==========
        X_all_perturbed = []
        segment_info = []
        
        for effect in effects:
            X_pert = self.local_sampler(X_candidates, list(effect.indices))
            segment_info.append((effect.indices, len(X_all_perturbed)))
            X_all_perturbed.append(X_pert)
        
        # ========== 批量模型评估（核心优化）==========
        if X_all_perturbed:
            X_batch = torch.cat(X_all_perturbed, dim=0)
            I_batch = self.metric_fn(X_batch)  # 仅此一次模型调用！
            
            # 解析结果到各效应
            raw_results = {}
            current_idx = 0
            
            for indices, _ in segment_info:
                seg_size = B * self.local_sampler.local_num
                I_seg = I_batch[current_idx:current_idx + seg_size]
                I_averaged = I_seg.view(B, self.local_sampler.local_num).mean(dim=1)
                raw_results[indices] = I_averaged
                current_idx += seg_size
        
        # ========== 基线评估 ==========
        I_baseline = self.metric_fn(X_candidates)
        
        # ========== 层次化贡献计算 ==========
        effects_by_order = {}
        for e in effects:
            effects_by_order.setdefault(e.order, []).append(e)
        
        contributions = {}
        for order in sorted(effects_by_order.keys()):
            for effect in effects_by_order[order]:
                I_current = effect.compute_contribution(
                    I_current, I_baseline, raw_results
                )
                contributions[effect.indices] = delta
        
        # ========== 阶数聚合 ==========
        aggregated = {}
        for order, order_effects in effects_by_order.items():
            order_contribs = [
                contributions[e.indices] for e in order_effects
                if e.indices in contributions
            ]
            if order_contribs:
                aggregated[f'order_{order}'] = torch.stack(order_contribs, dim=1).mean(dim=1)
            else:
                aggregated[f'order_{order}'] = torch.zeros_like(I_baseline)
        
        return {
            'baseline': I_baseline,
            'raw_results': raw_results,
            'contributions': contributions,
            'aggregated': aggregated
        }
```

## 3. 效应类实现

### MainEffect (主效应)

```python
class MainEffect(ANOVAEffect):
    def __init__(self, dim_idx: int):
        super().__init__(order=1, indices=(dim_idx,))
    
    def get_dependencies(self):
        return []  # 无依赖
    
    def compute_contribution(self, I_current, I_baseline, lower_results):
        delta = I_current - I_baseline
        return torch.clamp(delta, min=0.0)  # 不确定性导向
```

### PairwiseEffect (二阶交互)

```python
class PairwiseEffect(ANOVAEffect):
    def __init__(self, i: int, j: int):
        super().__init__(order=2, indices=tuple(sorted([i, j])))
    
    def get_dependencies(self):
        i, j = self.indices
        return [(i,), (j,)]  # 依赖两个主效应
    
    def compute_contribution(self, I_current, I_baseline, lower_results):
        i, j = self.indices
        I_i = lower_results.get((i,))
        I_j = lower_results.get((j,))
        
        if I_i is None or I_j is None:
            return torch.zeros_like(I_baseline)
        
        delta = I_current - I_i - I_j + I_baseline
        return torch.clamp(delta, min=0.0)
```

### ThreeWayEffect (三阶交互)

```python
class ThreeWayEffect(ANOVAEffect):
    def __init__(self, i: int, j: int, k: int):
        super().__init__(order=3, indices=tuple(sorted([i, j, k])))
    
    def get_dependencies(self):
        i, j, k = self.indices
        return [
            (i,), (j,), (k,),  # 一阶
            (min(i,j), max(i,j)), (min(i,k), max(i,k)), (min(j,k), max(j,k))  # 二阶
        ]
    
    def compute_contribution(self, I_current, I_baseline, lower_results):
        i, j, k = self.indices
        
        I_i = lower_results.get((i,))
        I_j = lower_results.get((j,))
        I_k = lower_results.get((k,))
        I_ij = lower_results.get((min(i,j), max(i,j)))
        I_ik = lower_results.get((min(i,k), max(i,k)))
        I_jk = lower_results.get((min(j,k), max(j,k)))
        
        deps = [I_i, I_j, I_k, I_ij, I_ik, I_jk]
        if any(dep is None for dep in deps):
            return torch.zeros_like(I_baseline)
        
        delta = (I_current - I_ij - I_ik - I_jk + I_i + I_j + I_k - I_baseline)
        return torch.clamp(delta, min=0.0)
```

## 4. 动态权重引擎

### SPS收敛检测

```python
class SPS_Tracker:
    def __init__(self, bounds, sensitivity=8.0, ema_alpha=0.7):
        self.bounds = bounds
        self.sensitivity = sensitivity
        self.ema_alpha = ema_alpha
        self.skeleton_points = self._generate_skeleton_points()
        self.prev_predictions = None
        self.r_t_smoothed = None
    
    def compute_r_t(self, model):
        with torch.no_grad():
            posterior = model.posterior(self.skeleton_points)
            current_preds = posterior.mean.squeeze()
        
        if self.prev_predictions is None:
            self.prev_predictions = current_preds
            self.r_t_smoothed = 1.0
            return 1.0
        
        diff = current_preds - self.prev_predictions
        delta_t = torch.norm(diff) / (torch.norm(current_preds) + 1e-8)
        r_t_raw = torch.tanh(self.sensitivity * delta_t)
        
        if self.r_t_smoothed is None:
            r_t = r_t_raw
        else:
            r_t = self.ema_alpha * self.r_t_smoothed + (1 - self.ema_alpha) * r_t_raw
        
        self.prev_predictions = current_preds
        self.r_t_smoothed = r_t
        return r_t
```

### λ_t 计算

```python
def compute_lambda(self, model=None):
    if not self.use_dynamic_lambda:
        return self.lambda_max
    
    current_model = model or self.model
    
    if self.use_piecewise_lambda:
        n = self._n_train
        if n < self.piecewise_phase1_end:
            return self.piecewise_lambda_low
        elif n >= self.piecewise_phase2_end:
            return self.piecewise_lambda_high
        else:
            progress = (n - self.piecewise_phase1_end) / (self.piecewise_phase2_end - self.piecewise_phase1_end)
            return self.piecewise_lambda_low + progress * (self.piecewise_lambda_high - self.piecewise_lambda_low)
    else:
        if self.use_sps and self.sps_tracker:
            r_t = self.sps_tracker.compute_r_t(current_model)
        else:
            r_t = self.compute_relative_main_variance()
        
        if r_t > self.tau1:
            return self.lambda_min
        elif r_t < self.tau2:
            return self.lambda_max
        else:
            ratio = (self.tau1 - r_t) / (self.tau1 - self.tau2)
            return self.lambda_min + ratio * (self.lambda_max - self.lambda_min)
```

## 5. 局部采样器

### 混合扰动实现

```python
def sample(self, X_base, dims):
    B, d = X_base.shape
    base = X_base.unsqueeze(1).repeat(1, self.local_num, 1)
    
    for dim in dims:
        var_type = self.variable_types.get(dim)
        
        if var_type == "categorical":
            unique_vals = self._get_unique_vals(dim)
            if unique_vals is None:
                continue
            
            n_levels = len(unique_vals)
            
            if self.use_hybrid_perturbation and n_levels <= self.exhaustive_level_threshold:
                if self.exhaustive_use_cyclic_fill:
                    n_repeats = (self.local_num // n_levels) + 1
                    samples = np.tile(unique_vals, (B, n_repeats))[:, :self.local_num]
                else:
                    samples = np.tile(unique_vals, (B, 1))
                base[:, :len(samples[0]), dim] = torch.from_numpy(samples)
            else:
                samples = np.random.choice(unique_vals, size=(B, self.local_num))
                base[:, :, dim] = torch.from_numpy(samples)
        
        elif var_type == "integer":
            int_min, int_max = self._get_int_bounds(dim)
            all_ints = np.arange(int_min, int_max + 1)
            n_levels = len(all_ints)
            
            if self.use_hybrid_perturbation and n_levels <= self.exhaustive_level_threshold:
                if self.exhaustive_use_cyclic_fill:
                    n_repeats = (self.local_num // n_levels) + 1
                    samples = np.tile(all_ints, (B, n_repeats))[:, :self.local_num]
                else:
                    samples = np.tile(all_ints, (B, 1))
                base[:, :len(samples[0]), dim] = torch.from_numpy(samples)
            else:
                sigma = self.local_jitter_frac * self._get_range(dim)
                noise = torch.randn(B, self.local_num) * sigma
                perturbed = torch.round(base[:, :, dim] + noise)
                base[:, :, dim] = torch.clamp(perturbed, min=int_min, max=int_max)
        
        else:  # continuous
            sigma = self.local_jitter_frac * self._get_range(dim)
            noise = torch.randn(B, self.local_num) * sigma
            base[:, :, dim] = torch.clamp(base[:, :, dim] + noise,
                                         min=self.ranges[0][dim],
                                         max=self.ranges[1][dim])
    
    return base.reshape(B * self.local_num, d)
```

## 6. 信息度量

### 序数熵计算

基于序数似然的熵计算，使用累积分布函数(CDF)计算各分类区间的概率：

$$
\Phi(z) = \frac{1}{2} \left(1 + \erf\left(\frac{z}{\sqrt{2}}\right)\right)
$$

其中 $z = \frac{c_k - \mu}{\sigma}$，$c_k$ 为分类阈值。

```python
def compute_entropy(self, X):
    posterior = self.model.posterior(X)
    mean = posterior.mean.squeeze(-1)
    var = posterior.variance.squeeze(-1)
    
    std = torch.sqrt(torch.clamp(var, min=1e-8))
    cutpoints = self._get_cutpoints()
    
    z = (cutpoints.unsqueeze(0) - mean.unsqueeze(1)) / std.unsqueeze(1)
    cdfs = 0.5 * (1 + torch.erf(z / np.sqrt(2)))
    
    p0 = cdfs[:, :1]
    p_last = 1.0 - cdfs[:, -1:]
    mids = torch.clamp(cdfs[:, 1:] - cdfs[:, :-1], min=1e-8)
    probs = torch.cat([p0, mids, p_last], dim=1)
    
    probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
    probs = torch.clamp(probs, min=1e-8, max=1-1e-8)
    
    log_probs = torch.log(probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    entropy = torch.where(torch.isnan(entropy) | torch.isinf(entropy),
                         torch.full_like(entropy, 0.5), entropy)
    
    return entropy
```

## 7. 覆盖计算

### Gower距离

混合变量距离度量，加权平均不同类型变量的距离：

对于连续变量: $d_j = \frac{|x_{1j} - x_{2j}|}{r_j}$
对于分类变量: $d_j = 0$ (相同) 或 $1$ (不同)

总距离: $d = \frac{\sum w_j d_j}{\sum w_j}$

```python
def gower_distance(x1, x2, variable_types, ranges, weights=None):
    total_distance = 0.0
    total_weight = 0.0
    
    for j in range(len(x1)):
        w_j = weights[j] if weights else 1.0
        vt = variable_types.get(j, "continuous")
        
        if vt == "continuous":
            if np.isnan(x1[j]) or np.isnan(x2[j]):
                continue
            range_j = ranges.get(j, 1.0)
            dist_j = abs(x1[j] - x2[j]) / range_j
        
        elif vt == "categorical":
            dist_j = 0.0 if x1[j] == x2[j] else 1.0
        
        total_distance += w_j * dist_j
        total_weight += w_j
    
    return total_distance / total_weight if total_weight > 0 else 0.0
```

### 覆盖度量

```python
def compute_coverage(self, x, X_sampled, method="min_distance"):
    if len(X_sampled) == 0:
        return 1.0
    
    distances = [gower_distance(x, sampled, self.variable_types, self.ranges, self.ard_weights)
                 for sampled in X_sampled]
    
    if method == "min_distance":
        return min(distances)
    elif method == "mean_distance":
        return np.mean(distances)
    elif method == "median_distance":
        return np.median(distances)
    else:
        raise ValueError(f"未知方法: {method}")
```
