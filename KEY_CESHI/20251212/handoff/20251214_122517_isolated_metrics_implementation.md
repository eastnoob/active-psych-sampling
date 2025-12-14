# 孤立阶段评估指标实施计划

**日期**: 2025-12-14
**目标**: 扩展现有评估模块以支持孤立阶段4核心指标
**状态**: 待实施

---

## 一、现状分析

### 1.1 现有模块功能梳理

| 模块 | 当前名称 | 主要功能 | 覆盖指标 |
|------|---------|---------|----------|
| **效应捕捉评估** | evaluation_v3.py | 主效应/交互效应识别精度评估 | Detection rate, sign accuracy, Top-k ranking |
| **自动模型对比** | evaluation_v4.py | 闭卷模式下的模型结构发现 | Structure discovery (precision/recall/F1), Top-k表 (AIC/BIC/R²) |
| **数据保存** | data_saver.py | 保存采样历史、EUR诊断、实验摘要 | iterations.csv (phase, y_value, EUR诊断) |

### 1.2 孤立阶段4核心指标缺口

| 指标 | 当前状态 | 缺口描述 |
|------|---------|----------|
| **1. Learning curves** | ❌ 缺失 | iterations.csv无 cumulative_rmse, cumulative_r2, cumulative_mae |
| **2. Structure discovery** | ✅ 完整 | v4已提供 precision/recall/F1 |
| **3. Effect discovery timeline** | ❌ 缺失 | 无函数追踪各效应首次达显著性的时间点 |
| **4. Top-k model evaluation** | ⚠️ 部分 | model_comparison表缺 delta_bic, rmse, mae, significant_coefs_ratio |

**覆盖率**: v4已完成70%，需扩展30%

---

## 二、模块重命名建议

### 2.1 命名原则

- **功能导向**: 名称应直接反映模块的核心功能，而非版本号
- **可读性**: 避免v2/v3/v4等版本号混淆，使用描述性名称
- **一致性**: 与项目其他模块命名风格保持一致

### 2.2 重命名方案

| 当前名称 | 建议新名称 | 理由 |
|---------|-----------|------|
| evaluation_v2.py | evaluation_effect_recovery.py | 主要功能：线性回归系数与Oracle权重对标 |
| evaluation_v3.py | evaluation_effect_capture.py | 主要功能：效应识别精度评估（detection/sign/ranking） |
| evaluation_v4.py | evaluation_model_discovery.py | 主要功能：闭卷模式下的自动模型结构发现 |

**实施建议**:
1. 创建新文件名副本
2. 更新所有导入引用（run_eur_residual.py, data_saver.py等）
3. 保留旧文件7天后删除（防止依赖遗漏）

---

## 三、实施方案

### 3.1 扩展点 1: Learning Curves in iterations.csv

**目标**: 在每次迭代中记录累积性能指标

**修改文件**: [data_saver.py:236-301](KEY_CESHI/20251212/scripts/modules/data_saver.py#L236-L301)

**修改位置**: `save_iterations_csv()` 函数

**新增列**:
```python
- cumulative_rmse: float  # 截至当前迭代，模型在完整设计空间上的RMSE
- cumulative_r2: float    # 截至当前迭代，模型在完整设计空间上的R²
- cumulative_mae: float   # 截至当前迭代，模型在完整设计空间上的MAE
```

**计算逻辑**:
1. 对于每次迭代 t (warmup后):
   - 从 model 提取训练数据 (train_inputs, train_targets)
   - 在设计空间上预测: y_pred = model.predict(design_space)
   - 计算与Oracle的误差:
     - RMSE = sqrt(mean((y_oracle - y_pred)^2))
     - R² = 1 - SS_res / SS_tot
     - MAE = mean(|y_oracle - y_pred|)
2. warmup阶段: 这3列填 `None`

**输入依赖**:
- `model`: 当前迭代的模型实例
- `design_space`: 完整设计空间 (1200×6)
- `oracle`: Oracle实例，用于获取真实y值

**输出格式** (iterations.csv新增列示例):
```
iteration,phase,y_value,...,cumulative_rmse,cumulative_r2,cumulative_mae
1,warmup,2.0,...,None,None,None
2,warmup,3.0,...,None,None,None
3,warmup,1.0,...,None,None,None
4,eur,2.5,...,0.523,0.612,0.421
5,eur,3.2,...,0.498,0.641,0.398
```

---

### 3.2 扩展点 2: Effect Discovery Timeline

**目标**: 追踪每个效应（主效应+交互）首次达到统计显著性的迭代次数

**修改文件**: [evaluation_v4.py](KEY_CESHI/20251212/scripts/modules/evaluation_v4.py) (建议重命名为 evaluation_model_discovery.py)

**新增函数**:
```python
def track_effect_discovery_timeline(
    model_history: List[object],  # 从warmup到当前的所有模型快照
    oracle_params: Dict,          # Oracle规格
    alpha: float = 0.05           # 显著性阈值
) -> Dict[str, int]:
    """
    追踪各效应首次达到统计显著性的迭代次数。

    Returns:
        {
            "x0": 5,           # 主效应x0在第5次迭代达到p<0.05
            "x1": 3,
            "x0*x5": 12,       # 交互效应x0*x5在第12次迭代达到显著
            "x1*x2": None,     # 从未达到显著
            ...
        }
    """
```

**算法逻辑**:
1. 提取Oracle的真实显著效应列表 (|coef| >= 0.05)
2. 对于每个迭代 t:
   - 拟合线性模型: y ~ intercept + x0...x5 + interactions
   - 提取每个系数的 p-value
   - 记录首次满足 p < alpha 的效应
3. 返回字典: {effect_name: first_significant_iteration}

**输出格式** (保存到 summary.json):
```json
"effect_discovery_timeline": {
    "x0": 4,
    "x1": 3,
    "x2": 7,
    "x3": null,  // 从未达到显著
    "x4": 6,
    "x5": 5,
    "x0*x5": 15,
    "x1*x2": 12,
    "x3*x4": 18
}
```

**调用位置**: 在 [run_eur_residual.py](KEY_CESHI/20251212/scripts/run_eur_residual.py) 的评估阶段，传入模型历史列表

---

### 3.3 扩展点 3: Top-k Model Detailed Evaluation

**目标**: 为 Top-k 模型添加详细评估指标

**修改文件**: [evaluation_v4.py:445-454](KEY_CESHI/20251212/scripts/modules/evaluation_v4.py#L445-L454)

**当前输出** (model_comparison表):
```python
{
    'name': 'M2_main+1int_((0,5),)',
    'aic': 245.3,
    'bic': 251.2,
    'rsquared_adj': 0.612,
    'interactions': ['x0*x5']
}
```

**扩展输出**:
```python
{
    'name': 'M2_main+1int_((0,5),)',
    'aic': 245.3,
    'bic': 251.2,
    'rsquared_adj': 0.612,
    'interactions': ['x0*x5'],
    # === 新增指标 ===
    'delta_bic': 0.0,          # 相对best model的BIC差 (best model = 0.0)
    'delta_aic': 0.0,
    'rmse': 0.421,             # 模型在训练集上的拟合RMSE
    'mae': 0.312,              # 模型在训练集上的拟合MAE
    'significant_coefs_ratio': 0.875,  # 显著系数占比 (p<0.05的系数数 / 总系数数)
    'n_params': 8              # 参数数量 (用于理解模型复杂度)
}
```

**计算逻辑**:
1. **delta_bic / delta_aic**:
   ```python
   best_bic = min(m['bic'] for m in models)
   delta_bic = model['bic'] - best_bic
   ```

2. **rmse / mae**:
   ```python
   y_pred = model.predict(X_train)
   rmse = np.sqrt(np.mean((y_train - y_pred)**2))
   mae = np.mean(np.abs(y_train - y_pred))
   ```

3. **significant_coefs_ratio**:
   ```python
   pvalues = model.pvalues[1:]  # 排除intercept
   n_significant = np.sum(pvalues < 0.05)
   ratio = n_significant / len(pvalues)
   ```

4. **n_params**:
   ```python
   n_params = len(model.params)  # intercept + main + interactions
   ```

**修改位置**: 在 `_enumerate_candidate_models()` 返回前，或在构建 model_comparison 时统一计算

---

## 四、实施步骤

### Phase 1: 模块重命名 (可选，建议优先级低)

1. 创建重命名副本:
   ```bash
   cp evaluation_v2.py evaluation_effect_recovery.py
   cp evaluation_v3.py evaluation_effect_capture.py
   cp evaluation_v4.py evaluation_model_discovery.py
   ```

2. 更新导入引用:
   - run_eur_residual.py
   - data_saver.py (如果有引用)
   - 任何测试文件

3. 添加过渡期兼容性 (在旧文件顶部):
   ```python
   # evaluation_v4.py
   import warnings
   warnings.warn(
       "evaluation_v4 is deprecated. Use evaluation_model_discovery instead.",
       DeprecationWarning
   )
   from evaluation_model_discovery import *
   ```

### Phase 2: 实施扩展 (核心，高优先级)

#### Step 1: Learning Curves

**文件**: data_saver.py

**修改**:
1. 修改 `save_iterations_csv()` 函数签名:
   ```python
   def save_iterations_csv(
       result_dir: Path,
       interaction_logs: List[Dict],
       eur_diagnostics: Optional[Dict[str, List]] = None,
       warmup_budget: int = 3,
       model: Optional[object] = None,        # 新增
       design_space: Optional[np.ndarray] = None,  # 新增
       oracle: Optional[object] = None         # 新增
   ) -> None:
   ```

2. 在循环内添加计算逻辑:
   ```python
   for log in interaction_logs:
       trial = log['trial']
       phase = 'warmup' if trial < warmup_budget else 'eur'

       # 基础行
       row = {'iteration': trial + 1, 'phase': phase, 'y_value': log['y']}

       # EUR诊断数据
       if eur_diagnostics and trial >= warmup_budget:
           # ... 现有代码 ...

       # === 新增：Learning curves ===
       if phase == 'eur' and model is not None and design_space is not None and oracle is not None:
           # 计算累积性能
           with torch.no_grad():
               y_oracle = np.array([oracle(x) - 1 for x in design_space])
               y_pred = model.predict(torch.from_numpy(design_space))

               rmse = np.sqrt(np.mean((y_oracle - y_pred)**2))
               ss_res = np.sum((y_oracle - y_pred)**2)
               ss_tot = np.sum((y_oracle - y_oracle.mean())**2)
               r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
               mae = np.mean(np.abs(y_oracle - y_pred))

               row['cumulative_rmse'] = rmse
               row['cumulative_r2'] = r2
               row['cumulative_mae'] = mae
       else:
           row['cumulative_rmse'] = None
           row['cumulative_r2'] = None
           row['cumulative_mae'] = None

       rows.append(row)
   ```

3. 更新列顺序:
   ```python
   base_cols = ['iteration', 'phase', 'y_value', 'lambda_t', 'gamma_t', 'r_t', 'n_train',
                'acqf_mean', 'acqf_std', 'min_distance',
                'cumulative_rmse', 'cumulative_r2', 'cumulative_mae']  # 新增
   ```

4. 更新调用代码 (run_eur_residual.py):
   ```python
   save_iterations_csv(
       result_dir=result_dir,
       interaction_logs=interaction_logs,
       eur_diagnostics=eur_diagnostics,
       warmup_budget=warmup_budget,
       model=model,              # 传递模型
       design_space=design_space,  # 传递设计空间
       oracle=oracle              # 传递Oracle
   )
   ```

#### Step 2: Effect Discovery Timeline

**文件**: evaluation_v4.py (或重命名后的 evaluation_model_discovery.py)

**新增函数**:
```python
def track_effect_discovery_timeline(
    model_history: List[object],
    oracle_params: Dict,
    alpha: float = 0.05
) -> Dict[str, Optional[int]]:
    """
    追踪各效应首次达到统计显著性的迭代次数。

    Args:
        model_history: 模型快照列表 [model_t3, model_t4, ..., model_tN]
        oracle_params: Oracle规格 {'main_weights': [...], 'interaction_weights': {...}}
        alpha: 显著性阈值

    Returns:
        {effect_name: first_iteration or None}
    """
    # 提取真实显著效应
    true_main_weights = np.array(oracle_params.get('main_weights', []))
    true_interaction_dict = oracle_params.get('interaction_weights', {})

    # 构建效应列表
    effect_names = [f"x{i}" for i in range(len(true_main_weights))]
    for key in true_interaction_dict.keys():
        effect_names.append(key)  # e.g., "x0*x5"

    # 初始化发现时间
    discovery_times = {name: None for name in effect_names}

    # 遍历模型历史
    for iteration, model in enumerate(model_history, start=1):
        # 提取训练数据
        train_X = model.train_inputs[0].detach().cpu().numpy()
        train_y_raw = model.train_targets.detach().cpu().numpy()

        # 处理序数模型
        if hasattr(model.likelihood, "cutpoints"):
            with torch.no_grad():
                probs = model.predict_probs(torch.from_numpy(train_X).to(torch.float64))
                probs_np = probs.cpu().numpy()
                train_y = np.array([
                    np.sum(np.arange(len(probs_np[i])) * probs_np[i])
                    for i in range(len(probs_np))
                ])
        else:
            train_y = train_y_raw.flatten()

        # 拟合线性模型
        # 提取交互对
        interaction_pairs = [
            tuple(map(int, key.replace('x', '').split('*')))
            for key in true_interaction_dict.keys()
        ]

        X_design = np.column_stack([
            np.ones(len(train_X)),
            train_X,
            *(train_X[:, i] * train_X[:, j] for i, j in interaction_pairs)
        ])

        lr_model = sm.OLS(train_y, X_design).fit()

        # 提取p-values (排除intercept)
        pvalues = lr_model.pvalues[1:]

        # 检查各效应的显著性
        n_main = len(true_main_weights)
        for i, effect_name in enumerate(effect_names):
            if discovery_times[effect_name] is None:  # 尚未发现
                if pvalues[i] < alpha:
                    discovery_times[effect_name] = iteration

    logger.info(f"Effect discovery timeline: {discovery_times}")
    return discovery_times
```

**调用位置** (run_eur_residual.py):
```python
# 在评估阶段添加
if model_history:  # 如果保存了模型历史
    from modules.evaluation_model_discovery import track_effect_discovery_timeline

    effect_timeline = track_effect_discovery_timeline(
        model_history=model_history,
        oracle_params={
            'main_weights': oracle.get_model_spec()['weights'],
            'interaction_weights': oracle.get_model_spec()['interaction_terms']
        },
        alpha=0.05
    )

    # 保存到summary
    eval_results_v4['effect_discovery_timeline'] = effect_timeline
```

**前置条件**: 需要在采样循环中保存模型历史:
```python
# run_eur_residual.py
model_history = []  # 初始化

for trial_idx in range(total_budget):
    # ... tell ...

    # 保存模型快照 (仅warmup后)
    if trial_idx >= warmup_budget:
        import copy
        model_history.append(copy.deepcopy(strat.model))
```

#### Step 3: Top-k Model Detailed Evaluation

**文件**: evaluation_v4.py:445-454

**修改**: 在构建 model_comparison 时添加新指标

```python
# 当前代码 (第445-454行)
model_comparison = [
    {
        'name': m['name'],
        'aic': float(m['aic']),
        'bic': float(m['bic']),
        'rsquared_adj': float(m['rsquared_adj']),
        'interactions': [f"x{i}*x{j}" for i, j in m['interactions']]
    }
    for m in sorted(candidate_models, key=lambda m: m[criterion])[:10]
]

# 扩展后代码
best_bic = min(m['bic'] for m in candidate_models)
best_aic = min(m['aic'] for m in candidate_models)

model_comparison = []
for m in sorted(candidate_models, key=lambda m: m[criterion])[:10]:
    # 计算拟合误差
    y_pred = m['model'].fittedvalues
    rmse = float(np.sqrt(np.mean((train_y - y_pred)**2)))
    mae = float(np.mean(np.abs(train_y - y_pred)))

    # 计算显著系数占比
    pvalues = m['model'].pvalues[1:]  # 排除intercept
    n_significant = int(np.sum(pvalues < 0.05))
    significant_ratio = float(n_significant / len(pvalues)) if len(pvalues) > 0 else 0.0

    model_comparison.append({
        'name': m['name'],
        'aic': float(m['aic']),
        'bic': float(m['bic']),
        'delta_aic': float(m['aic'] - best_aic),
        'delta_bic': float(m['bic'] - best_bic),
        'rsquared_adj': float(m['rsquared_adj']),
        'rmse': rmse,
        'mae': mae,
        'significant_coefs_ratio': significant_ratio,
        'n_params': int(len(m['model'].params)),
        'interactions': [f"x{i}*x{j}" for i, j in m['interactions']]
    })
```

### Phase 3: 测试验证

1. **单元测试** (可选):
   - 测试 learning curves 计算逻辑
   - 测试 effect discovery timeline 函数
   - 测试 Top-k 扩展指标计算

2. **集成测试**:
   ```bash
   cd KEY_CESHI/20251212/scripts
   pixi run python run_eur_residual.py --budget 10
   ```

3. **验证检查点**:
   - ✅ iterations.csv 包含 cumulative_rmse, cumulative_r2, cumulative_mae 列
   - ✅ summary.json 包含 effect_discovery_timeline 字段
   - ✅ model_comparison_table 包含 delta_bic, rmse, mae, significant_coefs_ratio
   - ✅ 无报错，数值合理

---

## 五、文件修改清单

| 文件 | 修改类型 | 修改内容 |
|------|---------|----------|
| [data_saver.py](KEY_CESHI/20251212/scripts/modules/data_saver.py) | ✏️ 扩展 | save_iterations_csv() 添加 learning curves |
| [evaluation_v4.py](KEY_CESHI/20251212/scripts/modules/evaluation_v4.py) | ✏️ 扩展 | 1. 新增 track_effect_discovery_timeline() <br> 2. 扩展 model_comparison 表 |
| [run_eur_residual.py](KEY_CESHI/20251212/scripts/run_eur_residual.py) | ✏️ 修改 | 1. 保存 model_history <br> 2. 调用 effect timeline 函数 <br> 3. 传递额外参数给 save_iterations_csv() |
| evaluation_model_discovery.py | 🆕 可选 | 重命名自 evaluation_v4.py |
| evaluation_effect_capture.py | 🆕 可选 | 重命名自 evaluation_v3.py |
| evaluation_effect_recovery.py | 🆕 可选 | 重命名自 evaluation_v2.py |

---

## 六、输出格式变化

### 6.1 iterations.csv (新增3列)

```csv
iteration,phase,y_value,lambda_t,gamma_t,r_t,n_train,acqf_mean,acqf_std,min_distance,cumulative_rmse,cumulative_r2,cumulative_mae,x0,x1,x2,x3,x4,x5
1,warmup,2.0,None,None,None,None,None,None,None,None,None,None,0.12,0.45,0.78,...
2,warmup,3.0,None,None,None,None,None,None,None,None,None,None,0.34,0.67,0.23,...
3,warmup,1.0,None,None,None,None,None,None,None,None,None,None,0.56,0.89,0.01,...
4,eur,2.5,0.312,0.543,0.123,3,0.234,0.112,0.045,0.523,0.612,0.421,0.78,0.12,...
5,eur,3.2,0.298,0.567,0.098,4,0.221,0.098,0.052,0.498,0.641,0.398,0.23,0.45,...
...
```

### 6.2 summary.json (新增字段)

```json
{
  "effect_capture_v4": {
    "evaluation_mode": "auto_discovery",
    "structure_discovery": { ... },
    "best_model": { ... },
    "model_comparison_table": [
      {
        "name": "M2_main+1int_((0,5),)",
        "aic": 245.3,
        "bic": 251.2,
        "delta_aic": 0.0,
        "delta_bic": 0.0,
        "rsquared_adj": 0.612,
        "rmse": 0.421,
        "mae": 0.312,
        "significant_coefs_ratio": 0.875,
        "n_params": 8,
        "interactions": ["x0*x5"]
      },
      { ... }  // Top 10 models
    ],
    "effect_discovery_timeline": {
      "x0": 4,
      "x1": 3,
      "x2": 7,
      "x3": null,
      "x4": 6,
      "x5": 5,
      "x0*x5": 15,
      "x1*x2": 12,
      "x3*x4": 18
    },
    "statistical_power": { ... },
    "effect_size_comparison": { ... }
  }
}
```

---

## 七、风险与注意事项

### 7.1 性能影响

- **Learning curves**: 每次迭代需在完整设计空间 (1200点) 上预测，增加约0.1-0.5秒
  - **缓解**: 仅在 EUR 阶段计算，warmup跳过

- **Effect discovery timeline**: 需保存模型历史，内存开销约 50MB/模型 × N次迭代
  - **缓解**: 使用 `copy.deepcopy()` 保存模型状态，或只保存关键checkpoint

### 7.2 兼容性

- **旧数据**: 修改后的 iterations.csv 格式不向后兼容
  - **缓解**: 旧实验结果保持不变，仅影响新运行

- **函数签名变更**: save_iterations_csv() 新增3个可选参数
  - **缓解**: 使用默认值 None，保持向后兼容

### 7.3 数值稳定性

- **R² 可能为负**: 当模型极差时，SS_res > SS_tot
  - **处理**: 添加检查逻辑，负值clip为0或报warning

- **RMSE/MAE 在序数模型上可能不准确**: 序数模型预测概率，转换为期望评分有误差
  - **处理**: 文档说明该指标仅供参考

---

## 八、后续工作 (对比阶段)

孤立阶段完成后，对比阶段需新增以下指标：

1. **Sample efficiency ratio**: EUR样本数 / Random样本数 (达到相同效应识别精度)
2. **Top-k stability across runs**: 多次运行的Top-k模型一致性
3. **Cross-run stability**: 跨run的效应发现时间点稳定性
4. **Win rate**: EUR vs Random 的胜率统计

**实施时机**: 孤立阶段测试通过后，再规划对比阶段模块

---

## 九、Checklist

- [ ] Phase 1: 模块重命名 (可选)
  - [ ] 创建重命名副本
  - [ ] 更新导入引用
  - [ ] 添加过渡期兼容性

- [ ] Phase 2: 实施扩展
  - [ ] Step 1: Learning curves in data_saver.py
  - [ ] Step 2: Effect discovery timeline in evaluation_v4.py
  - [ ] Step 3: Top-k detailed evaluation in evaluation_v4.py
  - [ ] 更新 run_eur_residual.py 调用代码

- [ ] Phase 3: 测试验证
  - [ ] 运行 budget=10 测试
  - [ ] 检查 iterations.csv 格式
  - [ ] 检查 summary.json 新字段
  - [ ] 检查数值合理性

- [ ] 文档更新
  - [ ] 更新 README.md (如需)
  - [ ] 记录 CHANGELOG (如需)

---

**实施优先级**: Phase 2 (扩展) > Phase 3 (测试) > Phase 1 (重命名，可后移)

**预计工作量**: 2-3小时（不含重命名）
