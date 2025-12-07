# Phase 1: 预热阶段

## 定位

**作用**: 筛选器 + 初始化器  
**输入**: 完整设计空间CSV（1200个配置）  
**输出**: 3个候选交互对 + 初始GP + 不确定性地图  
**预算**: 7人 × 25次 = 175次（23%总预算）

---

## 目标

| 维度 | 指标 | 目标 |
|------|------|------|
| **测量** | ICC | ≥0.35 |
| **统计** | 主效应SE | <0.18 |
| **统计** | 交互对筛选 | 15个→3个 |
| **预测** | GP RMSE | <0.95 |
| **预测** | 覆盖率 | >10% |

---

## 采样结构

```
175次 = Core-1(56) + Core-2(70) + 个体点(49)

Core-1 (32%): 8个固定点 × 7人 → ICC + 空间骨架
Core-2 (40%): 主效应45次 + 交互初筛25次
个体点(28%): 边界20次 + LHS 29次
```

---

## 采样策略

### 1. Core-1：固定重复点（8个）

**目的**: ICC估计 + 空间参考系

**策略**: 角点 + 中心点

```
FROM design_space:
  # 读取因子范围
  FOR each factor_i:
    min_i = design_space[factor_i].min()
    max_i = design_space[factor_i].max()
    center_i = median(design_space[factor_i])
  
  # 生成角点（2^d的子集）
  corners = [
    all_min,           # 全取最小值
    all_max,           # 全取最大值
    alternating_4,     # 4个交替模式（奇数维高，偶数维低等）
  ]
  
  # 生成中心点
  centers = [
    median_point,      # 所有维度取中位数
    perturbed_center   # 中位数 + 小扰动
  ]
  
  core1_points = corners + centers  # 共8个
  
  # 从design_space中找最近邻匹配
  FOR each target in core1_points:
    selected = argmin(distance(design_space, target))
```

**返回**: 8个design_space中的行索引

---

### 2. Core-2a：主效应覆盖（45次）

**目的**: 确保每个因子水平≥7次

**策略**: D-optimal设计

```
FROM design_space:
  # 计算每个因子的水平数
  factor_levels = {
    factor_i: unique_values(design_space[factor_i])
    for each factor_i
  }
  
  # 构建候选池（排除Core-1已选的8个）
  candidate_pool = design_space.drop(core1_indices)
  
  # D-optimal选点
  selected = []
  FOR i in 1 to 45:
    # 评估每个候选点的D-optimal分数
    FOR each candidate in candidate_pool:
      # 计算信息矩阵
      X_temp = [selected, candidate]
      info_matrix = X_temp.T @ X_temp
      score = log(det(info_matrix))  # D-optimal准则
      
      # 加权：优先填补采样不足的因子水平
      FOR each factor:
        level = candidate[factor]
        if count(selected, level) < 7:
          score += bonus
    
    best = argmax(score)
    selected.append(best)
    candidate_pool.remove(best)
  
  RETURN selected
```

**返回**: 45个design_space行索引

---

### 3. Core-2b：交互初筛（25次）

**目的**: 测试5个优先候选对，每对5次

**策略**: 象限均衡采样

```
# 输入：5个优先交互对（基于领域知识/文献）
priority_pairs = [(f1,f2), (f3,f4), (f5,f6), (f7,f8), (f9,f10)]

FOR each pair (fi, fj):
  # 定义象限
  median_i = median(design_space[fi])
  median_j = median(design_space[fj])
  
  quadrants = {
    "low_low":   (design_space[fi] <= median_i) & (design_space[fj] <= median_j),
    "low_high":  (design_space[fi] <= median_i) & (design_space[fj] > median_j),
    "high_low":  (design_space[fi] > median_i) & (design_space[fj] <= median_j),
    "high_high": (design_space[fi] > median_i) & (design_space[fj] > median_j)
  }
  
  # 从每个象限采样1次 + 中心/随机1次
  selected_for_pair = []
  FOR each quadrant:
    candidates = design_space[quadrant]
    # 排除已选点
    candidates = candidates.drop(all_selected_so_far)
    # 选择最接近象限中心的点
    quadrant_center = [median(candidates[fi]), median(candidates[fj])]
    best = argmin(distance(candidates, quadrant_center))
    selected_for_pair.append(best)
  
  # 第5次：随机或中心
  remaining = design_space.drop(selected_for_pair)
  selected_for_pair.append(random_choice(remaining))

RETURN all_selected  # 5对×5次=25个
```

**返回**: 25个design_space行索引

---

### 4. 个体点a：边界极端（20次）

**目的**: 外推能力 + 极端组合探索

**策略**: 分层极端点

```
FROM design_space:
  # 单维极端（d个）
  single_extremes = []
  FOR each factor_i:
    # 只有该因子取极值，其他取中位数
    config_min = {
      factor_i: min(design_space[factor_i]),
      others: median(design_space[others])
    }
    config_max = {
      factor_i: max(design_space[factor_i]),
      others: median(design_space[others])
    }
    single_extremes += [
      nearest_match(design_space, config_min),
      nearest_match(design_space, config_max)
    ]
  
  # 二维极端（高频组合）
  two_way_extremes = []
  high_freq_pairs = priority_pairs  # 复用交互候选对
  FOR each (fi, fj) in high_freq_pairs:
    # 4个组合：(min,min), (min,max), (max,min), (max,max)
    FOR (vi, vj) in [(min,min), (min,max), (max,min), (max,max)]:
      config = {
        fi: vi(design_space[fi]),
        fj: vj(design_space[fj]),
        others: median(design_space[others])
      }
      two_way_extremes.append(nearest_match(design_space, config))
  
  # 全局极端（2个）
  global_extremes = [
    nearest_match(design_space, all_min),
    nearest_match(design_space, all_max)
  ]
  
  # 组合并去重
  boundary_pool = unique(single_extremes + two_way_extremes + global_extremes)
  
  # 按到已选点的最小距离排序，优先选远离点
  FOR each candidate in boundary_pool:
    min_dist = min(distance(candidate, all_selected_so_far))
    candidate.score = min_dist
  
  boundary_selected = top_k(boundary_pool, k=20, by=score)
  
RETURN boundary_selected
```

**返回**: 20个design_space行索引

---

### 5. 个体点b：分层LHS（29次）

**目的**: 空间均匀填充

**策略**: 约束LHS + Gower距离

```
FROM design_space:
  # 统计当前覆盖
  current_selected = core1 + core2 + boundary
  coverage = count_by_level(current_selected)
  
  # 识别采样不足的因子水平
  undersampled_levels = []
  FOR each factor_i:
    FOR each level_v in unique(design_space[factor_i]):
      if coverage[factor_i][level_v] < 3:
        undersampled_levels.append((factor_i, level_v))
  
  # 生成LHS候选（在连续空间）
  lhs_samples = latin_hypercube_sample(n=29, d=n_factors)
  
  # 映射到design_space + 约束满足
  lhs_mapped = []
  FOR each lhs_point:
    # 找最近邻
    candidates = design_space
    
    # 优先匹配采样不足的水平
    FOR (factor_i, level_v) in undersampled_levels:
      if lhs_point[factor_i] close_to level_v:
        candidates = candidates[candidates[factor_i] == level_v]
    
    # 在候选中找距离最近的
    best = argmin(gower_distance(candidates, lhs_point))
    
    # 避免与已选点过近
    if min_distance(best, current_selected) > threshold:
      lhs_mapped.append(best)
      current_selected.append(best)
  
  # 如果不足29个，用maximin补充
  while len(lhs_mapped) < 29:
    remaining = design_space.drop(current_selected)
    # 选择到已选点最远的
    best = argmax(min_distance(remaining, current_selected))
    lhs_mapped.append(best)
    current_selected.append(best)
  
RETURN lhs_mapped
```

**返回**: 29个design_space行索引

---

## 数据分析

### 1. ICC估计

```
混合效应模型:
  rating ~ 1 + (1|subject) + (1|stimulus)

提取:
  ICC = σ²_between / (σ²_between + σ²_within)
  95% CI（bootstrap）

决策:
  IF ICC < 0.28 → 警告，检查测量流程
```

---

### 2. 主效应估计

```
固定效应回归:
  rating ~ β₀ + Σ(β_i × factor_i)

提取:
  β估计值, SE, 95% CI

质量检查:
  IF any SE > 0.22 → Phase 2前期加强该因子
```

---

### 3. 交互对筛选（核心）

```
FOR each tested_pair (fi, fj):
  # 拟合模型
  model: rating ~ main_effects + β_ij×(fi×fj) + (1|subject)
  
  # 评估指标
  p_value = test(β_ij ≠ 0)
  effect_se_ratio = |β_ij| / SE(β_ij)
  
  # 象限范围检验
  quadrant_means = [
    mean(rating | fi=low, fj=low),
    mean(rating | fi=low, fj=high),
    mean(rating | fi=high, fj=low),
    mean(rating | fi=high, fj=high)
  ]
  quadrant_range = max(quadrant_means) - min(quadrant_means)
  
  # BIC改善
  bic_additive = fit(rating ~ main_effects).BIC
  bic_interaction = fit(rating ~ main_effects + fi×fj).BIC
  bic_improvement = bic_additive - bic_interaction
  
  # 综合决策
  scores = {
    "p_value": int(p_value < 0.30),
    "effect_se": int(effect_se_ratio > 0.7),
    "quadrant": int(quadrant_range > 1.0),
    "bic": int(bic_improvement > 1.0)
  }
  
  IF sum(scores.values()) >= 2:
    priority = "high"
  ELIF sum(scores.values()) == 1:
    priority = "medium"
  ELSE:
    priority = "low"

# 排序并选择top-3
selected_pairs = top_k(tested_pairs, k=3, by=priority+effect_size)

RETURN selected_pairs
```

---

### 4. GP训练

```
数据准备:
  y_centered = y - mean(y_per_subject)

核函数: Matérn-5/2 + ARD

超参数:
  noise_level_init = sqrt(1 - ICC)
  length_scales_init = [1.0] × d
  
训练:
  maximize marginal_likelihood
  validate with 5-fold CV

评估:
  cv_rmse, cv_r2

质量检查:
  IF cv_rmse > 1.10 → 简化核或增加正则化
```

---

### 5. 不确定性地图

```
FOR each config in design_space (1200个):
  pred_mean, pred_std = gp.predict(config)
  uncertainty_map[config] = pred_std

# 识别高不确定性区域
high_uncertainty_configs = uncertainty_map[pred_std > 1.2]

# 生成优先采样列表（top-50）
priority_list = top_k(high_uncertainty_configs, k=50, by=pred_std)

# 空间聚类（避免过于集中）
priority_list_clustered = spatial_cluster(priority_list, n_clusters=10)

RETURN uncertainty_map, priority_list
```

---

## Phase1Output

```json
{
  "measurement_quality": {
    "icc": value, "ci_95": [low, high]
  },
  "factor_effects": {
    "main_effects": {factor_i: {beta, se, ci_95}},
    "lambda_main": value
  },
  "interaction_screening": {
    "tested_pairs": [5个，含优先级],
    "selected_for_phase2": [3个],
    "untested_pairs": [10个，Phase 2中期诊断用],
    "lambda_interaction": value
  },
  "gp_model": {
    "model_file": "phase1_gp.pkl",
    "cv_rmse": value,
    "kernel_params": {...}
  },
  "spatial_coverage": {
    "coverage_rate": value,
    "uncertainty_map_file": "uncertainty_map.csv",
    "priority_sampling_list": [50个配置]
  },
  "phase2_initialization": {
    "eur_params": {
      "interaction_pairs": [3个],
      "lambda_init": 0.8,
      "gamma_init": 0.35
    }
  },
  "decision": {
    "proceed_to_phase2": boolean
  }
}
```

---

## 给Phase 2的关键信息

```
1. 筛选后的3个交互对 
   → EUR-ANOVA专注目标

2. 初始GP + 不确定性地图
   → EUR采集函数的pred_std输入

3. lambda_main, lambda_interaction
   → EUR权重初始化

4. ICC估计
   → GP noise_level = sqrt(1-ICC)

5. 10个未测试候选对
   → Phase 2第300次中期诊断时补查
```

---

## 决策规则

```
进入Phase 2的必要条件（全部满足）:
✓ ICC ≥ 0.28
✓ 所有主效应SE < 0.22
✓ 筛选到1-4个交互对
✓ GP CV-RMSE < 1.10
✓ 覆盖率 > 8%
```

---

## 总结

**用7人×25次（175次）**:
1. 测试5个优先交互对 → 筛选到3个
2. 保留10个候选对 → Phase 2中期诊断补查
3. 建立ICC基线 + 主效应粗估 + 初始GP
4. 生成不确定性地图 → 指导Phase 2采样

**效率**: 用23%预算为Phase 2提供精确导航，节省60%交互估计预算
