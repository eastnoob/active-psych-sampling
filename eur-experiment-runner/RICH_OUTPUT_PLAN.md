# Rich Output Implementation Plan
# 丰富输出实现计划

## 目标
复刻 KEY_CESHI 原始实现的丰富分析输出，包括采样轨迹、效应恢复、模型比较等。

## 当前输出 vs 目标输出

### 当前输出（简化版）
```
output/keyceshi_eur/eur_TIMESTAMP/
├── config.ini                    # 使用的配置
├── experiment_summary.json       # 简单摘要
└── sampling_history.csv          # 采样历史
```

### 目标输出（完整版）
```
output/keyceshi_eur/eur_TIMESTAMP/
├── config.ini                    # 使用的配置
├── data_files/
│   ├── oracle_spec.json          # Oracle 规格（含交互项）
│   ├── summary.json              # 详细分析报告
│   ├── interaction_log.json      # 交互项日志
│   ├── iterations.csv            # 迭代数据
│   ├── sampling_history.npy      # NumPy 格式采样历史
│   └── warmup_logs.json          # Warmup 日志
└── temp_config.ini               # 临时配置
```

## 实现步骤

### Phase 1: 基础数据收集（必需）
**目标**：在采样过程中收集必要的数据

#### 1.1 扩展 Context 类
**文件**：`core/context.py`

添加字段：
```python
class Context:
    # 现有字段...

    # 新增字段
    sampling_trajectory: Dict = {}  # 采样轨迹数据
    warmup_data: List = []          # Warmup 数据
    eur_dynamics: Dict = {}         # EUR 动态参数
    iteration_details: List = []    # 每次迭代的详细信息
```

#### 1.2 修改 SingleRun Behavior
**文件**：`behaviors/single_run.py`

在采样循环中收集数据：
```python
def run(self, roles, config, context):
    # ... 现有代码 ...

    for trial in range(budget):
        # Ask
        next_x = ask(...)

        # Tell
        response = oracle.query(next_x)
        tell(...)

        # 【新增】收集迭代数据
        iteration_data = {
            'trial': trial,
            'phase': 'warmup' if trial < warmup_points else 'main',
            'x': next_x.tolist(),
            'y': response,
            'timestamp': datetime.now().isoformat()
        }

        # 【新增】如果是 EUR 阶段，收集动态参数
        if trial >= warmup_points:
            # 从 acquisition function 提取 lambda_t, gamma_t, r_t
            eur_params = self._extract_eur_params(server)
            iteration_data['eur_params'] = eur_params

        context.iteration_details.append(iteration_data)
```

#### 1.3 提取 EUR 动态参数
**新增方法**：`behaviors/single_run.py`

```python
def _extract_eur_params(self, server) -> Dict:
    """从 AEPsych server 提取 EUR 动态参数"""
    try:
        # 获取当前 acquisition function
        acqf = server.strat.generator.acqf

        # 提取动态参数
        params = {}
        if hasattr(acqf, 'lambda_t'):
            params['lambda_t'] = float(acqf.lambda_t)
        if hasattr(acqf, 'gamma_t'):
            params['gamma_t'] = float(acqf.gamma_t)
        if hasattr(acqf, 'r_t'):
            params['r_t'] = float(acqf.r_t)

        return params
    except Exception as e:
        logger.warning(f"Failed to extract EUR params: {e}")
        return {}
```

### Phase 2: 后处理分析（核心）
**目标**：实验结束后进行深度分析

#### 2.1 创建 Analyzer 模块
**新文件**：`utils/analyzer.py`

```python
class ExperimentAnalyzer:
    """实验结果分析器"""

    def __init__(self, context: Context):
        self.context = context
        self.oracle = context.oracle
        self.iteration_details = context.iteration_details

    def analyze(self) -> Dict:
        """执行完整分析"""
        return {
            'oracle_spec': self._analyze_oracle_spec(),
            'sampling_trajectory': self._analyze_sampling_trajectory(),
            'effect_recovery': self._analyze_effect_recovery(),
            'effect_capture_v3': self._analyze_effect_capture_v3(),
            'effect_capture_v4': self._analyze_effect_capture_v4(),
            'prediction_quality': self._analyze_prediction_quality()
        }

    def _analyze_oracle_spec(self) -> Dict:
        """分析 Oracle 规格"""
        return self.oracle.get_model_spec()

    def _analyze_sampling_trajectory(self) -> Dict:
        """分析采样轨迹（lambda_t, gamma_t, r_t 动态）"""
        eur_trials = [d for d in self.iteration_details if d['phase'] == 'main']

        lambda_values = [d['eur_params']['lambda_t'] for d in eur_trials if 'eur_params' in d]
        gamma_values = [d['eur_params']['gamma_t'] for d in eur_trials if 'eur_params' in d]
        r_values = [d['eur_params']['r_t'] for d in eur_trials if 'eur_params' in d]

        return {
            'lambda_t': {
                'initial': lambda_values[0] if lambda_values else None,
                'final': lambda_values[-1] if lambda_values else None,
                'mean': np.mean(lambda_values) if lambda_values else None,
                'std': np.std(lambda_values) if lambda_values else None
            },
            'gamma_t': {
                'initial': gamma_values[0] if gamma_values else None,
                'final': gamma_values[-1] if gamma_values else None,
                'mean': np.mean(gamma_values) if gamma_values else None,
                'std': np.std(gamma_values) if gamma_values else None
            },
            'r_t': {
                'initial': r_values[0] if r_values else None,
                'final': r_values[-1] if r_values else None,
                'mean': np.mean(r_values) if r_values else None,
                'std': np.std(r_values) if r_values else None
            },
            'diversity': self._calculate_diversity()
        }

    def _analyze_effect_recovery(self) -> Dict:
        """效应恢复分析（需要拟合模型）"""
        # 1. 收集所有采样点和响应
        X = np.array([d['x'] for d in self.iteration_details])
        y = np.array([d['y'] for d in self.iteration_details])

        # 2. 拟合线性模型（含交互项）
        from sklearn.linear_model import LinearRegression

        # 构建特征矩阵（主效应 + 交互项）
        X_features = self._build_feature_matrix(X)

        # 拟合
        model = LinearRegression()
        model.fit(X_features, y)

        # 3. 比较估计权重 vs 真实权重
        true_weights = self.oracle.weights
        estimated_weights = model.coef_[:len(true_weights)]

        # 计算相关性和误差
        correlation = np.corrcoef(true_weights, estimated_weights)[0, 1]
        rmse = np.sqrt(np.mean((true_weights - estimated_weights) ** 2))
        mae = np.mean(np.abs(true_weights - estimated_weights))

        return {
            'main_effects': {
                'correlation': float(correlation),
                'rmse': float(rmse),
                'mae': float(mae)
            },
            'interaction_effects': self._analyze_interaction_recovery(X, y),
            'model_fit': {
                'r_squared': float(model.score(X_features, y))
            }
        }

    def _analyze_effect_capture_v3(self) -> Dict:
        """效应捕获 v3（检测率、符号准确性、Top-k 排名）"""
        # 实现检测率分析
        # 实现符号准确性分析
        # 实现 Top-k 排名分析
        pass

    def _analyze_effect_capture_v4(self) -> Dict:
        """效应捕获 v4（结构发现、模型比较、统计功效）"""
        # 实现结构发现（发现交互项）
        # 实现模型比较（AIC/BIC）
        # 实现统计功效分析
        pass
```

#### 2.2 集成 Analyzer 到 SingleRun
**文件**：`behaviors/single_run.py`

```python
def run(self, roles, config, context):
    # ... 采样循环 ...

    # 【新增】实验结束后进行分析
    logger.info("Running post-experiment analysis...")
    analyzer = ExperimentAnalyzer(context)
    analysis_results = analyzer.analyze()

    # 保存分析结果
    self._save_rich_output(context, analysis_results)
```

### Phase 3: 输出格式化（完善）
**目标**：生成与原始实现一致的输出文件

#### 3.1 创建 OutputWriter 模块
**新文件**：`utils/output_writer.py`

```python
class RichOutputWriter:
    """丰富输出写入器"""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.data_dir = output_dir / 'data_files'
        self.data_dir.mkdir(exist_ok=True)

    def write_all(self, context: Context, analysis: Dict):
        """写入所有输出文件"""
        self.write_oracle_spec(analysis['oracle_spec'])
        self.write_summary(context, analysis)
        self.write_interaction_log(analysis)
        self.write_iterations_csv(context)
        self.write_sampling_history_npy(context)
        self.write_warmup_logs(context)

    def write_oracle_spec(self, oracle_spec: Dict):
        """写入 oracle_spec.json"""
        path = self.data_dir / 'oracle_spec.json'
        with open(path, 'w') as f:
            json.dump(oracle_spec, f, indent=2)

    def write_summary(self, context: Context, analysis: Dict):
        """写入 summary.json（完整分析报告）"""
        summary = {
            'experiment': {
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'config': 'keyceshi_eur_complete.ini',
                'result_dir': str(self.output_dir)
            },
            'parameters': {
                'total_budget': len(context.iteration_details),
                'warmup_budget': len([d for d in context.iteration_details if d['phase'] == 'warmup']),
                'eur_budget': len([d for d in context.iteration_details if d['phase'] == 'main']),
                'seed': context.oracle.seed
            },
            'oracle': analysis['oracle_spec'],
            'sampling_trajectory': analysis['sampling_trajectory'],
            'effect_recovery': analysis['effect_recovery'],
            'effect_capture_v3': analysis['effect_capture_v3'],
            'effect_capture_v4': analysis['effect_capture_v4'],
            'prediction_quality': analysis['prediction_quality']
        }

        path = self.data_dir / 'summary.json'
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)

    # ... 其他写入方法 ...
```

## 实现优先级

### P0 - 必需（立即实现）
1. ✅ Oracle 交互项支持（已完成）
2. 📝 基础数据收集（Context 扩展 + 迭代数据收集）
3. 📝 Oracle spec 输出

### P1 - 核心（第二阶段）
4. 📝 采样轨迹分析（lambda_t, gamma_t, r_t）
5. 📝 效应恢复分析（相关性、RMSE、MAE）
6. 📝 输出文件格式化（data_files 目录结构）

### P2 - 高级（第三阶段）
7. 📝 效应捕获 v3（检测率、符号准确性、Top-k）
8. 📝 效应捕获 v4（结构发现、模型比较、统计功效）
9. 📝 交互项日志
10. 📝 Warmup 日志

## 技术挑战

### 挑战 1: 提取 EUR 动态参数
**问题**：需要从 AEPsych server 内部提取 lambda_t, gamma_t, r_t

**解决方案**：
- 通过 `server.strat.generator.acqf` 访问 acquisition function
- 检查 `EURAnovaMultiAcqf` 是否暴露这些参数
- 如果没有，需要修改 `EURAnovaMultiAcqf` 添加参数跟踪

### 挑战 2: 效应恢复需要模型拟合
**问题**：需要拟合线性模型并比较估计权重 vs 真实权重

**解决方案**：
- 使用 sklearn.LinearRegression
- 构建特征矩阵（主效应 + 交互项）
- 计算相关性、RMSE、MAE

### 挑战 3: 模型比较需要 AIC/BIC
**问题**：需要比较多个模型（不同交互项组合）

**解决方案**：
- 使用 statsmodels.OLS
- 枚举交互项组合
- 计算 AIC/BIC 并排序

## 估计工作量

- **P0（必需）**：2-3 小时
- **P1（核心）**：4-6 小时
- **P2（高级）**：6-8 小时

**总计**：12-17 小时

## 下一步行动

1. ✅ 修复 Oracle 交互项解析（已完成）
2. 测试 Oracle 交互项是否正确工作
3. 实现 P0 功能（基础数据收集）
4. 逐步实现 P1、P2 功能
