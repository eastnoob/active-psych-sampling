================================================================================
改进总结: 你的4个问题暴露的关键设计缺陷
================================================================================

【核心发现】
你的4个问题指出了我初始设计中的3个关键问题:

1. ✓ 等差/非等差合并 - 正确 (单一Ordinal类)
2. ✗ LocalSampler架构 - 有缺陷 (显式Transform调用)
3. ✗ ordinal与int的关系 - 理解有偏差 (更接近categorical)
4. ✓ float64精度 - 重要但已考虑 (需显式强调)

================================================================================
【改进1】LocalSampler的架构改进
================================================================================

初始错误设计:
──────────────────────────────────────────────────────────────────────────────

def _perturb_ordinal(self, center_point, var_idx, par_type, ordinal_transform):
    """❌ 问题: LocalSampler不应该关心Transform对象"""

    rank = ordinal_transform._transform(center_point[var_idx])  # ← 违反架构
    perturbed_rank = center_rank + noise
    original_value = ordinal_transform._untransform(perturbed_rank)
    
    # ❌ 后果:
    # - LocalSampler现在依赖aepsych的Transform类
    # - 需要在初始化时传入ordinal_transform对象
    # - 增加模块耦合，难以测试

正确的架构 (改进后):
──────────────────────────────────────────────────────────────────────────────

class LocalSampler:
    def __init__(self, variable_types, bounds, unique_vals_dict=None, ...):
        """✓ unique_vals_dict来自外部，包含所有离散变量的values"""
        self.variable_types = variable_types  # {0: 'categorical', 1: 'ordinal', ...}
        self._unique_vals_dict = unique_vals_dict
        # {
        #   0: array([red, green, blue]),
        #   1: array([1.0, 2.0, 3.0, 4.0, 5.0]),  # ordinal values (float64)
        #   2: array([0.01, 0.1, 1.0, 10.0, 100.0])  # ordinal_mono values
        # }

def _perturb_ordinal(self, center_idx, k, B):
    """✓ 完全在原始值空间工作，不需要Transform对象"""

    unique_vals = self._unique_vals_dict[k]  # 原始值列表
    center_val = center_idx[:, k]  # 当前点的k维值
    
    # 找到中心值的rank (通过索引查找)
    center_rank = np.where(unique_vals == center_val)[0][0]
    n_levels = len(unique_vals)
    
    # 混合策略
    if self.use_hybrid_perturbation and n_levels <= self.exhaustive_level_threshold:
        # 低维discrete: 穷举所有rank
        if self.exhaustive_use_cyclic_fill:
            samples = np.repeat(unique_vals, (B * self.local_num) // n_levels)
            samples = samples[:B * self.local_num]
        else:
            samples = np.tile(unique_vals, (B, 1))[:, :self.local_num]
    else:
        # 高维discrete: 在rank空间高斯扰动 + 舍入 + 转换回值
        sigma = self.local_jitter_frac * n_levels
        noise = self._np_rng.normal(0, sigma, size=(B, self.local_num))
        
        # 在rank空间扰动
        perturbed_ranks = np.round(np.clip(
            center_rank + noise,
            a_min=0,
            a_max=n_levels - 1
        )).astype(int)
        
        # 从rank转换回原始值
        samples = unique_vals[perturbed_ranks]  # ← 简单查表，无Transform
    
    return samples  # 返回原始值空间的点

def sample(self, X_can_t, dims):
    """采样程序"""
    B = X_can_t.shape[0]
    d = X_can_t.shape[1]
    base = X_can_t.repeat(self.local_num, axis=0)  # (B*local_num, d)

    for k in dims:
        var_type = self.variable_types.get(k) if self.variable_types else None
        
        if var_type == "categorical":
            base[:, k] = self._perturb_categorical(base, k, B)
        elif var_type in ["ordinal", "custom_ordinal", "custom_ordinal_mono"]:
            # ✓ 与categorical完全相同的模式
            base[:, k] = self._perturb_ordinal(base, k, B)
        elif var_type == "integer":
            base[:, k] = self._perturb_integer(base, k, B, ...)
        else:  # continuous
            base[:, k] = self._perturb_continuous(base, k, B, ...)
    
    return base

好处:
  ✓ LocalSampler完全独立，无需关心Transform
  ✓ unique_vals_dict由上层(config_parser/eur_anova_pair)提供
  ✓ 架构与Categorical完全一致
  ✓ 易于测试，无外部依赖
  ✓ 集成自然，无额外复杂度

================================================================================
【改进2】ordinal与int的关系明确化
================================================================================

错误认知:
──────────────────────────────────────────────────────────────────────────────

"ordinal是integer的特例，处理非等差值" ← ❌ 错

正确认知:
──────────────────────────────────────────────────────────────────────────────

"ordinal与categorical在架构上一致，都使用rank化；与integer完全不同"

证据表:

┌────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ 维度           │ integer          │ custom_ordinal   │ categorical      │
├────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ 值表示         │ [1,2,3,...,100]  │ [1,2,3,4,5]      │ [red, green, ..] │
│ 间距含义       │ 数学上有意义     │ 无固定含义       │ 无序             │
│ Transform      │ 无 (直接用bounds)│ value→rank化     │ value→rank化     │
│ GP核函数       │ RBF (原始空间)   │ RBF (rank空间)   │ Categorical      │
│ bounds         │ [lb-0.5, ub+0.5] │ [-0.5, n-0.5]    │ [-0.5, n-0.5]    │
│ LocalSampler   │ 高斯+舍整        │ 高斯+舍整(rank)  │ 随机选择         │
│ 实现地点       │ parameters.py    │ ordinal.py       │ categorical.py   │
└────────────────┴──────────────────┴──────────────────┴──────────────────┘

为什么ordinal≠integer:

  场景: 药剂学中的剂量等级 [1mg, 2mg, 3mg, 4mg, 5mg]
  
  用integer (错的):
    bounds = [1, 5]
    GP: value=2和value=3离value=4和value=5距离相同
    假设: 1mg→2mg的效应 ≈ 4mg→5mg的效应
    问题: ❌ 非线性! 每倍增可能翻倍!

  用custom_ordinal (对的):
    values = [1.0, 2.0, 3.0, 4.0, 5.0]  (存float64)
    rank = [0, 1, 2, 3, 4]
    GP: value=2→3和value=4→5的距离相同 (都是rank +1)
    假设: 保留顺序，但边际效应可非线性
    优点: ✓ 灵活拟合，无线性假设

为什么ordinal≈categorical:

  架构视角:

    Categorical:
      values = ["red", "green", "blue"]
      rank = [0, 1, 2]
      bounds = [-0.5, 2.5]
      _transform: "red"(0)→0, "green"(1)→1, "blue"(2)→2
    
    Ordinal:
      values = [1.0, 2.0, 3.0, 4.0, 5.0]
      rank = [0, 1, 2, 3, 4]
      bounds = [-0.5, 4.5]
      _transform: 1.0→0, 2.0→1, 3.0→2, 4.0→3, 5.0→4
    
    LocalSampler处理:
      
      Categorical._perturb_categorical:
          unique_vals = [red, green, blue]
          sample: np.random.choice(unique_vals, ...)
      
      Ordinal._perturb_ordinal:
          unique_vals = [1.0, 2.0, 3.0, 4.0, 5.0]
          # 混合策略: 穷举 或 高斯扰动(rank)
          sample: unique_vals[rank_samples]
    
    ← 架构完全相同!

================================================================================
【改进3】float64精度的显式强调
================================================================================

需要在3个地方强调float64的必要性:

1. Ordinal.__init__:
   ──────────────────────────────────────────────────────────────────────
   def __init__(self, indices, values):
       # ✓ 强制float64
       self.values = {}
       for idx, val_list in values.items():
           self.values[idx] = np.array(val_list, dtype=np.float64)

       n_levels = len(self.values[indices[0]])
       self.bounds = torch.tensor(
           [[-0.5], [n_levels - 0.5]],
           dtype=torch.double  # float64
       )

2. _compute_arithmetic_sequence:
   ──────────────────────────────────────────────────────────────────────
   @staticmethod
   def_compute_arithmetic_sequence(min_val, max_val, step=None, num_levels=None):
       """✓ 全程float64避免精度丢失"""
       min_val = float(min_val)
       max_val = float(max_val)

       if step is not None:
           step = float(step)
           # np.arange可能有精度问题，使用float64 + 舍入
           values = np.arange(min_val, max_val + step/2, step, dtype=np.float64)
           # 舍入到合理小数位，避免累积误差
           values = np.round(values, decimals=12)
       elif num_levels is not None:
           num_levels = int(num_levels)
           # np.linspace在float64中最精确
           values = np.linspace(min_val, max_val, num_levels, dtype=np.float64)
       
       return values  # float64数组

3. custom_pool_based_generator:
   ──────────────────────────────────────────────────────────────────────
   def_generate_pool_from_config(self, config):
       # ...
       if par_type in ["custom_ordinal", "custom_ordinal_mono"]:
           ordinal = Ordinal.from_config(...)  # ordinal.values已是float64
           values = ordinal.values[indices[0]]  # 提取，保持float64
           # ✓ 不做任何类型转换
           param_choices_values.append(values)

精度问题的具体例子:

  问题: min=0, max=1.0, step=0.1 的等差数列
  
  ❌ 用float32:
     np.arange(0, 1.0, 0.1, dtype=np.float32)
     → [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
     → 缺少1.0!
     → values只有10个，不是11个!
     → rank空间错误
  
  ✓ 用float64:
     np.arange(0, 1.0 + 0.1/2, 0.1, dtype=np.float64)
     → [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
     → 正确11个值
     → rank空间正确: [0,1,2,3,4,5,6,7,8,9,10]

================================================================================
【改进4】(已正确) 等差/非等差单一Transform
================================================================================

这个设计已经正确，无需改进。

要点:
  ✓ 单一Ordinal类处理两种类型
  ✓ 优先级链配置在get_config_options()中
  ✓ Transform逻辑完全相同，差异在配置解析

================================================================================
【改进建议清单】
================================================================================

需要在实现文档中更新:

1. 【LocalSampler架构】
   ✓ 增加unique_vals_dict参数说明
   ✓ 说明ordinal/categorical使用相同的模式
   ✓ 强调不需要Transform对象

2. 【ordinal与int的关系】
   ✓ 增加明确的架构对比表
   ✓ 说明为什么ordinal≈categorical≠integer
   ✓ 提供选择指南 (何时用哪个)

3. 【float64精度】
   ✓ 在Ordinal类描述中显式强调float64
   ✓ 提供精度丢失的具体例子
   ✓ 在_compute_arithmetic_sequence中注明float64

4. 【初始化流程】
   ✓ config_parser需要提取ordinal.values到unique_vals_dict
   ✓ LocalSampler初始化时接收unique_vals_dict
   ✓ 清楚说明数据流: config → Ordinal → unique_vals_dict → LocalSampler

================================================================================
