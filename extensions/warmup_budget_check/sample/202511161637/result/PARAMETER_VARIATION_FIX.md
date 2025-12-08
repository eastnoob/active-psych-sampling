# Parameter Variation Fix - Problem & Solution

**Date:** 2025-11-17  
**Status:** ✅ RESOLVED

---

## Problem Statement

用户报告问题：
> "5个subject的参数咋全他妈一样操你妈，不是说参数从窄分布采样，误差从较大的分布取样但均值是0，使用百分数控制吗？"

### Root Cause Analysis

问题在于 `run_simulation.py` 中 subject 的创建逻辑：

**之前的代码逻辑（错误）：**

```python
# 创建一个共享的群体权重
base = MixedEffectsLatentSubject(num_features=6, seed=42, ...)
pop_weights = base.get_population_weights()

# 然后所有被试都传入同一份 pop_weights
for idx, csv_path in enumerate(csvs, start=1):
    subject_seed = seed + idx
    subj = SingleOutputLatentSubject(
        seed=subject_seed,
        population_weights=pop_weights,  # ← 所有被试用同一份群体权重！
        individual_std=base.individual_std,
        ...
    )
```

**问题所在：**

1. 虽然设置了不同的 `subject_seed`（43, 44, 45, ...）
2. 但传入的 `population_weights` 完全相同
3. 在 deterministic 模式（`use_latent=False`）下，固定权重 `fixed_weights` 也自动生成但结果相同
4. 导致所有5个被试的参数完全一致

---

## Solution Implemented

### Key Changes in `run_simulation.py`

#### 1. **不再共享群体权重**

```python
# 移除了：
pop_weights = base.get_population_weights()

# 改成每个被试独立生成群体权重和个体偏差
for idx, csv_path in enumerate(csvs, start=1):
    subject_seed = seed + idx
    
    # 关键：不传入 population_weights，让每个被试独立采样
    subj = SingleOutputLatentSubject(
        seed=subject_seed,
        population_weights=None,  # ← 强制独立生成
        factor_loadings=None,      # ← 强制独立生成
        item_biases=None,          # ← 强制独立生成
        item_noises=None,          # ← 强制独立生成
        ...
    )
```

#### 2. **在 Deterministic 模式下为每个被试生成不同的固定权重**

当使用 `use_latent=False`（deterministic 模式）且 `individual_std_percent > 0` 时：

```python
# 每个被试使用独立的 RNG 生成偏差
rng_subj = np.random.RandomState(subject_seed)

# 从基础固定权重开始
base_fw = fixed_weights_map.get("global")

# 添加个体偏差：w_individual = w_base + deviation
base_fw_arr = np.array(base_fw, dtype=float)
indiv_std_val = args.individual_std_percent * args.population_std
deviation = rng_subj.normal(0, indiv_std_val, size=base_fw_arr.shape)
subject_fixed_weights = base_fw_arr + deviation

# 传入被试特定的权重
subj = SingleOutputLatentSubject(
    ...,
    fixed_weights=subject_fixed_weights,  # ← 每个被试有不同的权重
    ...
)
```

---

## Verification Results

### Before Fix ❌

所有5个被试的 x1 权重都相同：

```
Subject 1: -0.12546
Subject 2: -0.12546  ← 完全相同
Subject 3: -0.12546  ← 完全相同
Subject 4: -0.12546  ← 完全相同
Subject 5: -0.12546  ← 完全相同
```

### After Fix ✅

现在所有5个被试的 x1 权重都不同：

```
Subject 1: -0.11516
Subject 2: -0.15548  ← 不同
Subject 3: -0.12440  ← 不同
Subject 4: -0.10206  ← 不同
Subject 5: -0.15938  ← 不同
```

### 同一配置的多被试响应差异

输入配置：`(x1=0, x2=1, x3=0.0, x4=max, x5=C, x6=True)`

不同被试的响应（Likert 1-5）：

```
Subject 1: 4
Subject 2: 1  ← 不同
Subject 3: 1  ← 不同
Subject 4: 1  ← 不同
Subject 5: 5  ← 不同
```

---

## Model Parameters Distribution

### Configuration

- **Population Mean:** 0.0（所有被试围绕0均值）
- **Population Std:** 0.05（狭窄分布）
- **Individual Std Percent:** 0.8
  - 实际个体偏差标准差 = 0.8 × 0.05 = 0.04
  - 这是一个"较大的分布"相对于狭窄的群体分布
- **Likert Mapping:** Percentile binning（确保均匀分布）

### 参数采样流程（现在的正确实现）

```
步骤1：群体共性（群体固定效应）
    w_population ~ N(population_mean=0.0, population_std=0.05)
    
步骤2：个体差异（随机效应）
    ε_individual ~ N(0, individual_std_percent * population_std)
              = N(0, 0.8 * 0.05)
              = N(0, 0.04)
    
步骤3：个体权重
    w_individual = w_population + ε_individual
    
结果：
    - 所有被试围绕同一群体分布
    - 但每个被试有独立的采样变异
    - 导致被试间的参数差异
```

---

## Files Modified

1. **`run_simulation.py`**
   - 移除了 `pop_weights = base.get_population_weights()` 共享逻辑
   - 添加了被试特定的固定权重生成代码
   - 修复了 `subject_specs` dict 中的重复键（`"fixed_weights"`）
   - 添加了注释解释每个被试的独立参数生成

2. **`MixedEffectsLatentSubject.py`**
   - 改进了 `_init_rng()` 的文档说明
   - 确保每个被试初始化时使用其独立的种子

---

## Generated Output

所有生成文件都已更新，包括：

- ✅ `subject_1_model.md` 到 `subject_5_model.md`：显示不同的固定权重
- ✅ `combined_results.csv`：显示不同被试对同一输入的不同响应
- ✅ `subjects_parameters_summary.md`：参数对比和距离矩阵
- ✅ `SIMULATION_REPORT.md`：完整的实验报告

---

## Testing Command

```bash
cd "D:\WORKSPACE\python\aepsych-source"
pixi run python "D:\WORKSPACE\python\aepsych-source\extensions\warmup_budget_check\sample\202511161637\result\run_simulation.py" --seed 42
```

---

## Key Learnings

1. **共享 vs 独立：** 当想要群体共性但个体差异时，不应该共享已生成的参数矩阵，而应让每个被试独立采样
2. **RNG 隔离：** 使用 `np.random.RandomState(seed)` 为每个被试创建独立的 RNG 实例，确保参数间不相关
3. **参数层级：**
   - 固定效应（群体权重）：所有被试围绕同一分布
   - 随机效应（个体偏差）：每个被试独立采样
   - 结果：被试间参数有差异，但保持群体特征
