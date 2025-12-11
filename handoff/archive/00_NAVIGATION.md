================================================================================
handoff/ 文件导航和使用指南
================================================================================

生成日期: 2025-12-11
状态: 修订版v2 (已整合用户反馈)

================================================================================
📁 文件清单
================================================================================

1. 20251211_ordinal_monotonic_parameter_extension.md (主文档)
   ──────────────────────────────────────────────────────────────────────────
   目的: 完整的实现计划和架构设计

   包含内容:
   ✓ 修订要点 (与初版的差异)
   ✓ 架构设计 (Ordinal核心类, custom_generators集成, local_sampler改进)
   ✓ 配置示例 (3种自动计算方式 + 混合参数示例)
   ✓ 参数类型对比表
   ✓ 实现清单 (五个阶段的详细工作量)
   ✓ 测试策略 (单元测试 + 集成测试)
   ✓ 关键实现决策 (为什么这样设计)

   何时使用: 实现时的主要参考文档
   行数: 1016

2. 20251211_DESIGN_RATIONALE.md (设计论证)
   ──────────────────────────────────────────────────────────────────────────
   目的: 回答四个核心设计问题，提供详细论证

   包含内容:
   ✓ 【问题1】等差/非等差为什么合并？
     - 共性大于差异性分析
     - 代码重复问题
     - AEPsych的设计模式

   ✓ 【问题2】rank空间扰动的可行性与证据
     - LocalSampler的架构分析
     - 与Categorical的对比
     - 初始设计的缺陷与改进

   ✓ 【问题3】ordinal与integer的区别
     - 架构对比表
     - 为什么ordinal≠integer
     - 为什么ordinal≈categorical

   ✓ 【问题4】数据类型与精度
     - float64的必要性
     - 精度丢失的具体例子
     - 实现检查清单

   何时使用: 需要理解设计决策的理由时
   行数: ~550

3. 20251211_IMPROVEMENTS_SUMMARY.md (改进总结)
   ──────────────────────────────────────────────────────────────────────────
   目的: 总结用户反馈所暴露的缺陷和改进方案

   包含内容:
   ✓ 【改进1】LocalSampler架构 (从Transform显式调用到unique_vals_dict)
   ✓ 【改进2】ordinal与int关系的澄清
   ✓ 【改进3】float64精度的强调
   ✓ 【改进4】确认等差/非等差单一Transform正确

   每个改进包含:
   - 初始错误设计 (代码示例)
   - 正确的做法 (改进后的代码)
   - 好处说明

   何时使用: 理解初始设计的问题和改进方案时
   行数: ~350

4. 20251211_QUICK_ANSWERS.txt (快速答案)
   ──────────────────────────────────────────────────────────────────────────
   目的: 用ASCII表格和要点形式快速回答四个问题

   包含内容:
   ✓ 问题1: 是否合并? (快速答案 + 理由简述)
   ✓ 问题2: rank空间工作吗? (证据 + 修正说明)
   ✓ 问题3: ordinal vs int? (对比表 + 选择指南)
   ✓ 问题4: float64精度? (必要性 + 检查清单)

   何时使用: 需要快速查询答案时
   行数: ~250

5. README_ORDINAL_HANDOFF.md (README)
   ──────────────────────────────────────────────────────────────────────────
   目的: 手交前的导航和快速入门

   包含内容:
   ✓ 项目概述
   ✓ 文件使用顺序
   ✓ 关键概念速览
   ✓ 常见问题

   何时使用: 开始实现前的入门
   行数: ~200

6. ORDINAL_QUICK_REF.md (快速参考)
   ──────────────────────────────────────────────────────────────────────────
   目的: 配置示例和API参考

   包含内容:
   ✓ 配置示例 (3种自动计算方式)
   ✓ 参数类型对比
   ✓ API签名速览

   何时使用: 配置实验或调试时
   行数: ~150

================================================================================
📖 推荐阅读顺序
================================================================================

【新手入门】

1. README_ORDINAL_HANDOFF.md (5分钟)
   └─ 快速了解项目结构和文件用途

2. ORDINAL_QUICK_REF.md (10分钟)
   └─ 看配置例子，理解参数类型

3. 20251211_QUICK_ANSWERS.txt (15分钟)
   └─ 四个核心问题的快速答案

【深度理解】
4. 20251211_DESIGN_RATIONALE.md (30分钟)
   └─ 理解设计的论证和背景

5. 20251211_IMPROVEMENTS_SUMMARY.md (20分钟)
   └─ 理解初始设计的缺陷和改进

【实现】
6. 20251211_ordinal_monotonic_parameter_extension.md (全天)
   └─ 主要实现参考，分五个阶段

   分阶段阅读:

- 第一阶段: Ordinal Transform核心 (1-2h)
- 第二阶段: AEPsych集成 (0.5-1h)
- 第三阶段: custom_generators集成 (1h)
- 第四阶段: local_sampler集成 (1.5h)
- 第五阶段: 测试和文档 (1h)

================================================================================
🔑 关键概念速查
================================================================================

【ordinal参数的三个关键点】

1. Ordinal Transform (aepsych/transforms/ops/ordinal.py)
   - 单一类，处理等差和非等差
   - Transform: value → rank [0,1,2,...,n-1] → value
   - bounds: [-0.5, n-0.5] (与Categorical相同)
   - 优先级链配置: values < min/max/step < min/max/num_levels < levels

2. Pool生成集成 (custom_pool_based_generator.py)
   - 从Ordinal.get_config_options()提取values
   - 自动包含在full_factorial的排列组合中
   - 零修改兼容: 去重、变量组合、历史排除都自动工作

3. LocalSampler扰动 (modules/local_sampler.py)
   - unique_vals_dict[k] 包含ordinal的values列表 (float64)
   - _perturb_ordinal() 在原始值空间工作 (like Categorical)
   - 混合策略: 低维穷举, 高维高斯扰动(rank)+舍入
   - 不需要显式Transform调用

【参数类型选择指南】

continuous (连续)
├─ 用途: 光滑变化 (e.g., 温度, 浓度)
└─ 特点: 无bounds离散化

integer (整数)
├─ 用途: 计数, 索引 (e.g., 件数, 倍数)
├─ 特点: 间距=1的严格含义
└─ 注意: 避免用于有序偏好

categorical (分类)
├─ 用途: 无序选择 (e.g., A/B/C, red/green/blue)
├─ 特点: 无顺序关系
└─ Transform: 值→rank化

custom_ordinal (等差有序)
├─ 用途: Likert量表, 等级, 有序偏好
├─ 特点: 间距均匀, 自动计算
├─ 配置: min/max/step 或 min/max/num_levels 或 levels字符串
└─ Transform: 值→rank化

custom_ordinal_mono (非等差有序)
├─ 用途: 功率律, 指数关系 (e.g., [0.01, 0.1, 1, 10, 100])
├─ 特点: 间距不均匀, 需手工指定
├─ 配置: values = [...]
└─ Transform: 值→rank化

【架构对比】

integer vs custom_ordinal:

  integer[1,2,3,4,5]:
    GP学到: value=2→3 ≈ value=4→5 (距离相同)
    问题: ❌ 可能非线性!
  
  custom_ordinal[1,2,3,4,5]:
    rank = [0,1,2,3,4]
    GP学到: rank=1→2 ≈ rank=3→4 (rank距离相同)
    优点: ✓ 保留顺序，灵活拟合非线性

categorical vs custom_ordinal:

  都是rank化:
  
  categorical["red", "green", "blue"]:
    rank = [0, 1, 2]
    LocalSampler: 随机选择 (无顺序)

  custom_ordinal[1, 2, 3, 4, 5]:
    rank = [0, 1, 2, 3, 4]
    LocalSampler: 高斯扰动(rank) (保留顺序)

  ← 架构相同，只有扰动策略不同

================================================================================
⚠️ 常见陷阱
================================================================================

【陷阱1】混淆ordinal与integer

错误:
  [rating]
  par_type = integer  # ❌ 会强制线性假设
  lb = 1
  ub = 5
  
正确:
  [rating]
  par_type = custom_ordinal  # ✓ 灵活拟合
  min_value = 1
  max_value = 5
  step = 1

【陷阱2】忘记float64处理

错误:
  values = [0.1, 0.2, 0.3, 0.4, 0.5]  # 可能float32精度丢失
  
正确:
  values = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)

# 或: np.arange(0.1, 0.6, 0.1, dtype=np.float64)

【陷阱3】在LocalSampler中显式调用Transform

错误:
  def _perturb_ordinal(self, center_point, ordinal_transform):
      rank = ordinal_transform._transform(center_point)  # ❌ 增加耦合
      ...
  
正确:
  def _perturb_ordinal(self, center_idx, k):
      unique_vals = self._unique_vals_dict[k]  # ✓ 从初始化时提供
      center_rank = np.where[unique_vals == center_idx[:, k]](0)[0]
      ...

【陷阱4】等差数列的端点

错误:
  np.arange(0, 1.0, 0.1)  # 可能缺少1.0
  
正确:
  np.arange(0, 1.0 + 0.1/2, 0.1)  # 确保包含max_value

# 或: np.linspace(0, 1.0, 11)  # 精确11个点

【陷阱5】混合参数类型的配置

错误:
  [common]
  parnames = [cat, ord, cont]
  lb = [0, 0, 0.0]     # ❌ lb/ub不适用所有参数类型
  ub = [2, 4, 1.0]
  
  [cat]
  par_type = categorical
  choices = [A, B, C]
  
  [ord]
  par_type = custom_ordinal

# ❌ min_value/max_value怎么指定?
  
正确:
  [common]
  parnames = [cat, ord, cont]
  
  [cat]
  par_type = categorical
  choices = [A, B, C]

# lb/ub无关 (rank空间: [-0.5, 2.5])
  
  [ord]
  par_type = custom_ordinal
  min_value = 1.0
  max_value = 5.0
  step = 0.5

# 用min/max/step，不用lb/ub
  
  [cont]
  par_type = continuous
  lb = 0.0
  ub = 1.0

# 用lb/ub

================================================================================
📋 实现检查清单
================================================================================

【第一阶段: Ordinal Transform】

- [ ] 创建 aepsych/transforms/ops/ordinal.py (~180 LOC)
- [ ] 实现 __init__, _transform,_untransform
- [ ] 实现 _compute_arithmetic_sequence (float64精度)
- [ ] 实现 get_config_options (优先级链)
- [ ] 单元测试覆盖: rank往返, 浮点精度, 配置解析
- [ ] 验证bounds处理: [-0.5, n-0.5]

【第二阶段: AEPsych集成】

- [ ] 更新 aepsych/transforms/ops/__init__.py (+2 LOC)
- [ ] 更新 aepsych/transforms/parameters.py (~50 LOC)
- [ ] 更新 aepsych/config.py (+10 LOC)
- [ ] 集成测试: pool生成包含ordinal参数
- [ ] 验证bounds转换: 原始→rank空间

【第三阶段: custom_generators】

- [ ] 更新 custom_pool_based_generator.py (~50 LOC)
- [ ] 修改 _generate_pool_from_config() 提取ordinal values
- [ ] 验证 full_factorial 包含所有ordinal组合
- [ ] 验证去重工作正常

【第四阶段: local_sampler】

- [ ] 更新 LocalSampler.__init__ 接收 unique_vals_dict
- [ ] 实现 _perturb_ordinal() (~25 LOC, 原始值空间)
- [ ] 更新 sample() 路由到 _perturb_ordinal
- [ ] 更新 config_parser.py (~10 LOC)
- [ ] 更新 eur_anova_pair.py (~15 LOC)
- [ ] 集成测试: 扰动策略, 混合参数

【第五阶段: 测试与文档】

- [ ] 单元测试: test_ordinal_transform.py (50+ cases)
- [ ] 集成测试: test_ordinal_aepsych_integration.py
- [ ] 集成测试: test_ordinal_pool_generation.py
- [ ] 集成测试: test_ordinal_local_sampler.py
- [ ] 端到端测试: config→pool→sample
- [ ] 更新用户文档

================================================================================
联系与问题
================================================================================

如果在实现中遇到以下问题，请参考:

Q: values应该是什么数据类型?
A: 见DESIGN_RATIONALE.md 【问题4】, float64确保精度

Q: Ordinal和Categorical有什么区别?
A: 见QUICK_ANSWERS.txt 【问题3】, 架构相同但扰动不同

Q: LocalSampler中如何扰动ordinal?
A: 见IMPROVEMENTS_SUMMARY.md 【改进1】, unique_vals_dict模式

Q: 什么时候用ordinal vs integer?
A: 见ORDINAL_QUICK_REF.md 【参数类型选择指南】

Q: min/max/step的精度问题?
A: 见QUICK_ANSWERS.txt 【问题4】和DESIGN_RATIONALE.md详细分析

================================================================================
