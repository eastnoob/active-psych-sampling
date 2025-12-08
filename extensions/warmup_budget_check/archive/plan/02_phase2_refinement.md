# 阶段2：精炼阶段（Refinement Phase）

## 核心定位

**角色**：精密仪器（Refiner & Discoverer）\
**性质**：估计 ≈ 预测（动态平衡）\
**输出**：精确的效应估计 + 中期诊断报告 + 精化的GP模型

***

## 要做什么

### 双重目标（动态权重调整）

```
目标1: 统计推断（前期70% → 后期30%）
├─ 精确估计3-5个二阶交互对（SE < 0.12）
├─ 精化主效应估计（SE < 0.10）
└─ 发现三阶交互线索（第300次中期诊断）

目标2: 预测建模（前期30% → 后期70%）
├─ 降低全局预测不确定性（mean_pred_std < 0.7）
├─ 填补阶段1识别的高不确定性区域
└─ 为最终贝叶斯模型提供均衡数据
```

***

## 目标与成功标准

| 目标维度     | 具体指标           | 目标值     | 用途     |
| -------- | -------------- | ------- | ------ |
| **统计推断** | 主效应SE          | <0.10   | 精确估计   |
| <br />   | 二阶交互SE         | <0.12   | 精确估计   |
| <br />   | 象限采样均衡         | 每象限≥20次 | 可识别性   |
| <br />   | 三阶交互发现         | 0-2个候选  | 触发阶段3  |
| **预测建模** | GP CV-RMSE     | <0.75   | 精化模型   |
| <br />   | 覆盖率            | >25%    | 空间探索   |
| <br />   | 90%空间pred\_std | <1.0    | 不确定性降低 |
| <br />   | 边界pred\_std    | <0.8    | 外推准备   |

***

## 资源分配（450次观测）

```
总计: 18人 × 25次 = 450次

动态分配（由EUR-ANOVA自动调整）：
├─ 交互对关键组合: ~315次（70%，前期权重高）
│   └─ EUR驱动，针对3-5个交互对的4个象限
│       根据当前不确定性动态分配
│
└─ 空间填充: ~135次（30%，后期权重高）
    └─ GP驱动，填补高不确定性区域
        覆盖权重从0.3→0.05逐渐降低
```

***

## 批次设计（18人，分批实施）

```
每批2-3人（灵活调整）
每人25次

批次间隔：1-2周
├─ 优势1: 及时发现数据质量问题
├─ 优势2: 可以根据中期结果调整策略
└─ 优势3: 减少被试疲劳（分散测试）

关键检查点：
├─ 第150次（第6人）: 检查交互对SE是否下降
├─ 第300次（第12人）: 中期诊断（三阶交互线索）
└─ 第450次（第18人）: 最终评估
```

***

## 方法论

### 1. 主动学习框架（EUR-ANOVA + GP混合）

**核心流程**：

```伪代码
初始化:
  gp_model = Phase1Output.gp_model  # 从阶段1继承
  interaction_pairs = Phase1Output.selected_pairs  # 3-5个
  lambda_main = Phase1Output.lambda_main
  lambda_interaction = Phase1Output.lambda_interaction

FOR trial = 1 to 450:
  # 1. 生成候选点（从未测试的配置中）
  X_candidates = sample_untested_configs(n=128)
  
  # 2. 动态权重计算
  lambda_t = compute_dynamic_lambda(trial, total=450)
    # 前200次: 0.8-1.0（高交互权重）
    # 中150次: 0.5-0.7（中等）
    # 后100次: 0.2-0.4（低交互权重）
  
  gamma_t = compute_dynamic_gamma(trial, total=450)
    # 前200次: 0.3（低覆盖权重）
    # 中150次: 0.2
    # 后100次: 0.05（高覆盖权重，但仍以信息为主）
  
  # 3. EUR-ANOVA评分
  scores = EUR_ANOVA(
    X_candidates,
    gp_model,
    interaction_pairs,
    lambda_t,
    gamma_t
  )
  
  # 4. 选择最高分配置
  X_next = X_candidates[argmax(scores)]
  
  # 5. 实验测试
  Y_next = conduct_experiment(subject_id, X_next)
  
  # 6. 更新GP模型
  gp_model.add_data(X_next, Y_next)
  gp_model.refit()
  
  # 7. 定期诊断
  IF trial % 50 == 0:
    print_diagnostics(gp_model, lambda_t, gamma_t)
    check_interaction_SE(current_data, interaction_pairs)
  
  # 8. 中期决策点
  IF trial == 300:
    run_mid_phase_diagnostic()  # 详见下节
```

***

### 2. EUR-ANOVA采集函数详解

**信息项（主效应 + 交互效应）**：

```伪代码
FOR 每个候选点 x:
  # 主效应贡献（Δ_main）
  FOR 每个因子 i:
    扰动点 x_i = local_perturb(x, dim=i)
    Δ_i = I(x_i) - I(x)  # 不确定性变化
    # I(x) = GP的后验方差
  
  Δ_main = mean(所有Δ_i)
  
  # 交互效应贡献（Δ_interaction）
  FOR 每个筛选出的交互对 (i,j):
    扰动点 x_ij = local_perturb(x, dims=[i,j])
    单独扰动 x_i, x_j
    Δ_ij = I(x_ij) - I(x_i) - I(x_j) + I(x)  # 交互项不确定性
  
  Δ_interaction = mean(所有Δ_ij)
  
  # 融合（动态权重）
  α_info(x) = (1.0 × Δ_main) + (λ_t × Δ_interaction)
```

**覆盖项（空间填充）**：

```伪代码
FOR 每个候选点 x:
  # 计算到已观测点的最小距离
  distances = []
  FOR 每个已观测点 x_obs:
    d = gower_distance(x, x_obs)  # 混合变量距离
    distances.append(d)
  
  COV(x) = min(distances)  # 最小距离越大，覆盖贡献越大
```

**最终评分**：

```伪代码
α(x) = α_info(x) + γ_t × COV(x)

# 标准化（确保可比性）
α_info_normalized = standardize(α_info)
COV_normalized = standardize(COV)
α_final = α_info_normalized + γ_t × COV_normalized
```

***

### 3. 中期诊断（第300次，关键决策点）

**目的**：发现三阶交互线索，决定是否启动阶段3

```伪代码
中期诊断流程:

1. 拟合当前最优模型
   data_so_far = 阶段1(350) + 阶段2前300次
   
   bayesian_model = fit(
     formula = "Y ~ main_effects + 2nd_order_interactions + (1|subject)",
     interactions = Phase1Output.selected_pairs
   )

2. 计算残差
   residuals = observed_Y - predicted_Y

3. 对每个候选三阶交互进行模式检测
   候选三阶 = C(6, 3) = 20个可能组合
   
   FOR 每个三阶交互 (i,j,k):
     # 将数据分成2×2×2的8个象限
     octants = group_by_levels(data, factors=[i,j,k])
     
     # 检查残差是否呈现三阶交互的"签名"
     pattern_signature = {
       (低i,低j,低k): mean_residual_1,
       (低i,低j,高k): mean_residual_2,
       (低i,高j,低k): mean_residual_3,
       ...
       (高i,高j,高k): mean_residual_8
     }
     
     # 三阶交互的特征：8个象限的残差呈现特定模式
     # 不是简单的加性关系
     pattern_score = compute_3rd_order_score(pattern_signature)
     
     IF pattern_score > 0.3:  # 阈值
       候选三阶交互.append((i,j,k))

4. 统计检验（Bootstrap）
   FOR 每个候选三阶交互:
     # Bootstrap检验：如果只是噪声，pattern_score应该很小
     bootstrap_scores = []
     FOR 1000次重采样:
       shuffle_residuals()
       recompute_pattern_score()
       bootstrap_scores.append(score)
     
     p_value = P(bootstrap_score > observed_score)
     
     IF p_value < 0.05:
       confirmed_candidates.append(候选)

5. 决策
   IF len(confirmed_candidates) > 0:
     decision = "proceed_to_phase3"
     # 调整阶段2剩余150次的策略：
     # - 70%继续EUR（完成二阶交互精化）
     # - 30%探索三阶交互的关键象限（为阶段3做准备）
   ELSE:
     decision = "continue_phase2_as_planned"
     # 剩余150次全部用于二阶交互精化+预测优化
```

**输出报告**：

```markdown
## 中期诊断报告（Trial 300）

### 当前模型性能
- 主效应SE: 0.105（目标<0.10，接近达标）
- 二阶交互SE: 平均0.115（目标<0.12，接近达标）
- GP CV-RMSE: 0.78（目标<0.75，需进一步降低）
- 覆盖率: 22%（目标>25%，需加强）

### 交互对精度评估
| 交互对 | 当前SE | 目标SE | 象限采样均衡 | 状态 |
|-------|--------|--------|-------------|------|
| (0,3) | 0.11 | <0.12 | [22,25,28,24] | ✓ 达标 |
| (1,2) | 0.13 | <0.12 | [18,20,32,25] | ⚠️ 需加强 |
| (0,1) | 0.12 | <0.12 | [21,23,22,21] | ✓ 达标 |

### 三阶交互候选发现
**候选1**: (density × greenery × style)
- Pattern score: 0.42
- Bootstrap p-value: 0.03
- 残差模式：(高密度, 高绿化, 现代风格)系统性偏高+0.48
- **建议**: 启动阶段3验证

**候选2**: (height × street_width × landmark)
- Pattern score: 0.28
- Bootstrap p-value: 0.12
- 残差模式：不稳定，可能是噪声
- **建议**: 不验证

### 决策
✓ **进入阶段3**
- 理由: 发现1个高置信度三阶交互候选
- 阶段2剩余150次调整策略（详见Phase2Output）
```

***

### 4. 后150次策略调整（如果触发阶段3）

```伪代码
IF 中期诊断发现三阶交互候选:
  # 修改EUR权重配置
  FOR trial = 301 to 450:
    lambda_t = 0.5  # 中等交互权重（不再递减）
    gamma_t = 0.1   # 中等覆盖权重
    
    # 新增：三阶探索权重
    alpha_3rd = 0.3  # 30%预算用于三阶象限探索
    
    IF random() < alpha_3rd:
      # 探索三阶交互的关键象限
      X_next = sample_from_3rd_order_octants(候选三阶)
    ELSE:
      # 继续EUR-ANOVA
      X_next = EUR_ANOVA(...)
ELSE:
  # 按原计划继续
  FOR trial = 301 to 450:
    lambda_t = 递减到0.2
    gamma_t = 递减到0.05
    X_next = EUR_ANOVA(...)
```

***

## 数据分析（阶段2结束）

### 最终统计模型拟合

```伪代码
数据整合:
  all_data = Phase1(350) + Phase2(450) = 800次
  
  # 如果有批次效应，校准
  IF Phase1Output.batch_effects > 0.3:
    apply_calibration(all_data, Phase1Output.calibration_factors)

拟合贝叶斯层次模型:
  formula = "Y ~ β0 + 
             Σ(β_i × factor_i) +                    # 主效应
             Σ(β_ij × factor_i × factor_j) +        # 二阶交互
             (1 | subject) + 
             ε"
  
  # 使用Phase1的主效应估计作为先验
  priors = {
    "main_effects": Phase1Output.main_effects,
    "interactions": Phase1Output.lambda_interaction
  }
  
  model = fit_bayesian_model(all_data, formula, priors)

后验诊断:
  ├─ 检查MCMC收敛（R-hat < 1.01）
  ├─ 后验预测检验（PPC）
  └─ 残差分析

提取结果:
  FOR 每个参数:
    posterior_mean = mean(MCMC samples)
    posterior_std = std(MCMC samples)
    ci_95 = quantile([2.5%, 97.5%])
    p_value = P(参数 > 0 | data)  # 贝叶斯p值
```

### GP模型最终评估

```伪代码
性能评估:
  # 5-fold交叉验证
  cv_rmse = cross_validate(gp_model, folds=5)
  
  # 留一被试验证（测试泛化）
  loso_rmse = []
  FOR 每个被试 s:
    train_without_s = all_data.exclude(s)
    test_on_s = all_data.only(s)
    gp_temp = train_gp(train_without_s)
    rmse_s = evaluate(gp_temp, test_on_s)
    loso_rmse.append(rmse_s)
  
  mean_loso_rmse = mean(loso_rmse)

不确定性地图更新:
  FOR 每个配置 in 1200种:
    pred_mean, pred_std = gp_model.predict(config)
    uncertainty_map[config] = {
      "mean": pred_mean,
      "std": pred_std,
      "ci_95": [pred_mean - 1.96*pred_std, pred_mean + 1.96*pred_std]
    }

覆盖分析:
  coverage_rate = n_unique_configs_tested / 1200
  
  高不确定性区域（pred_std > 1.0）:
    n_high_uncertainty = count(pred_std > 1.0)
    percent = n_high_uncertainty / 1200
  
  边界外推能力:
    boundary_configs = 筛选极端配置
    boundary_mean_std = mean(pred_std for boundary_configs)
```

***

## Phase2Output（交付清单）

```json
{
  "metadata": {
    "phase": 2,
    "n_subjects": 18,
    "n_observations": 450,
    "total_observations": 800,  // 含阶段1
    "completion_date": "2025-XX-XX"
  },
  
  "statistical_inference": {
    "main_effects": {
      // 精化后的主效应估计
      "density": {
        "posterior_mean": [β0, β1, β2, β3, β4],
        "posterior_std": [0.08, 0.09, 0.08, 0.09, 0.10],  // < 0.10
        "ci_95": [[low, high], ...],
        "p_values": [...]
      },
      // ... 其他因子
    },
    
    "2nd_order_interactions": {
      "(0,3)": {
        "beta": 0.22,
        "SE": 0.10,  // < 0.12 达标
        "ci_95": [0.02, 0.42],
        "p_value": 0.03,
        "BIC_gain": 12.5,
        "quadrant_sampling": {
          "(低,低)": 22,
          "(低,高)": 25,
          "(高,低)": 28,
          "(高,高)": 24
        }
      },
      // ... 其他交互对
    },
    
    "model_comparison": {
      "additive_model": {"BIC": 2450},
      "with_2nd_order": {"BIC": 2380},
      "BIC_improvement": 70
    }
  },
  
  "3rd_order_discovery": {
    "mid_phase_diagnostic_trial": 300,
    "candidates_found": [
      {
        "factors": "(0,3,4)",
        "names": "(density, greenery, style)",
        "pattern_score": 0.42,
        "bootstrap_p": 0.03,
        "evidence": "Residual pattern in (high,high,modern) octant: +0.48",
        "recommendation": "validate_in_phase3"
      }
    ],
    "decision": "proceed_to_phase3"
  },
  
  "prediction_model": {
    "gp_cv_rmse": 0.72,
    "gp_loso_rmse": 0.79,
    "coverage_rate": 0.27,
    "mean_pred_std": 0.68,
    "n_high_uncertainty_configs": 85,  // pred_std > 1.0
    "boundary_mean_std": 0.76
  },
  
  "phase3_initialization": {
    "trigger": true,
    "target_3rd_order": ["(0,3,4)"],
    "octant_design": {
      "(0,0,0)": "density_low, greenery_low, style_traditional",
      "(0,0,1)": "density_low, greenery_low, style_modern",
      // ... 8个象限
    },
    "recommended_samples_per_octant": 25,
    "total_budget": 200
  },
  
  "quality_metrics": {
    "all_2nd_order_SE_met": true,
    "all_main_SE_met": true,
    "gp_performance_met": true,
    "coverage_target_met": true
  }
}
```

***

## 潜在风险与应对

### 风险1：某个交互对SE无法降低（卡在0.14）

```
原因诊断:
├─ 该象限被试间分歧大（某些被试认为协调，某些认为冲突）
├─ 该象限真实效应弱（接近0），噪声占主导
└─ 采样不均衡（某个象限<15次）

应对策略:
IF 采样不均衡:
  → 手动干预，强制下50次都采该象限
ELSE IF 被试间分歧大（ICC_pair < 0.2）:
  → 接受SE=0.14，在论文中说明"该交互对存在个体差异"
ELSE IF 真实效应弱:
  → 考虑从模型中移除该交互对
```

### 风险2：GP模型过拟合（训练RMSE低，CV-RMSE高）

```
现象: 训练RMSE=0.5, CV-RMSE=0.9

应对:
1. 简化核函数（Matérn-5/2 → Matérn-3/2）
2. 增加噪声水平（noise_level从0.6→0.8）
3. 减少核长度尺度的自由度（ARD → isotropic）
4. 增加正则化（调整先验）
```

### 风险3：中期诊断时发现多个三阶交互候选（>3个）

```
决策困境: 预算不足以验证所有候选

应对策略:
1. 按pattern_score排序，只验证top-2
2. 其他候选在论文中报告为"exploratory finding"
3. 建议未来研究验证
```

***

## 关键里程碑

```
Week 12-13: 前150次（6人）→ 初步评估交互对SE
Week 14:    第300次 → 中期诊断（三阶交互检测）
Week 16:    阶段2完成 → Phase2Output生成
Week 17:    决策：是否启动阶段3
```

***

## 与阶段1的区别

| 维度       | 阶段1        | 阶段2        |
| -------- | ---------- | ---------- |
| **目标**   | 探索+筛选      | 精化+发现      |
| **采样策略** | 固定设计       | 自适应（主动学习）  |
| **精度要求** | SE<0.15（粗） | SE<0.10（精） |
| **交互对数** | 15个候选      | 3-5个确认     |
| **三阶交互** | 不考虑        | 主动发现       |
| **GP作用** | 初始训练       | 在线更新+驱动采样  |
| **决策权**  | 研究者设计      | 算法自适应      |

