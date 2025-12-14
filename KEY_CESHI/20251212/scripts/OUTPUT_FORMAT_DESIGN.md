# EUR Residual Test - 输出格式设计方案

## 1. 当前问题

### 1.1 缺失的文件
- ❌ `oracle_model_spec.json` - Oracle模型规格(权重、交互项系数)
- ❌ `test_data.csv` - 详细的迭代记录(含动态权重轨迹)
- ❌ 完整的 `summary.json` - 缺少评估指标

### 1.2 Summary内容对比

**参考格式包含** (251206版本):
```json
{
  // 基础配置
  "config", "budget", "strategy", "tag", "timestamp",
  "interaction_pairs", "warmup_budget", "eur_budget",
  "golden_points", "config_path", "db_path",

  // 动态权重统计 ⭐
  "r_t_statistics": {min, max, mean, initial, final, variance},
  "lambda_statistics": {...},
  "gamma_statistics": {...},

  // 采样质量 ⭐
  "sampling_diversity": {min_distance_mean, min_distance_min, unique_samples},

  // 效应识别 ⭐
  "effect_identification": {main_accuracy, pair_accuracy, identified_main, ...},

  // 预测质量 ⭐
  "prediction_quality": {R2, RMSE, MAE, Accuracy, y_true, y_pred},

  // 效应恢复v2 ⭐⭐ (最重要)
  "effect_recovery_v2": {
    "main_correlation": {spearman, spearman_pval, pearson},
    "interaction_correlation": {...},
    "main_effects": {oracle, estimated, rmse, mae},
    "interaction_effects": {pairs, oracle, estimated, rmse, mae},
    "prediction_quality": {R2, RMSE, MAE, n_test},
    "model_fit": {lr_r2, n_train, model_type}
  }
}
```

**当前格式只有**:
```json
{
  "timestamp", "budget", "seed", "n_warmup", "n_eur_samples",
  "parnames", "oracle_weights", "oracle_bias",
  "config_path", "design_space_path", "result_dir"
}
```

### 1.3 问题诊断

参考格式的问题:
1. **信息冗余**: `prediction_quality` 出现两次(根级和effect_recovery_v2内)
2. **数组过大**: y_true和y_pred包含完整的1000个测试样本
3. **调试文件过多**: debug/目录下有5个文件(52K+36K+8K)

---

## 2. 改进设计方案

### 2.1 文件结构

```
results/{timestamp}/
├── data_files/
│   ├── interaction_log.json      # 每个trial的详细记录
│   ├── sampling_history.npy      # 采样历史数组
│   ├── oracle_spec.json          # NEW: Oracle模型规格
│   ├── iterations.csv            # NEW: 迭代详情(phase,lambda_t,gamma_t,r_t等)
│   └── summary.json              # ENHANCED: 完整评估报告
├── diagnostics/                   # NEW: 诊断数据(可选)
│   ├── lambda_t.npy
│   ├── gamma_t.npy
│   ├── r_t.npy
│   └── min_distance.npy
└── temp_config.ini               # 临时配置(用于复现)
```

**关键改进**:
- ✅ 合并 `test_data.csv` → `iterations.csv`
- ✅ 新增 `oracle_spec.json`
- ✅ 增强 `summary.json`
- ✅ 移除 debug/ 目录(experiment.db/log过大且价值低)
- ✅ EUR诊断数据单独放在 diagnostics/ (可选生成)

---

### 2.2 oracle_spec.json

```json
{
  "model_type": "linear",
  "num_features": 6,
  "seed": 42,
  "bias": 3.0,
  "noise_std": 0.1,
  "weights": [0.248, -0.069, ...],  // 主效应权重
  "weight_std": 0.157,
  "interaction_pairs": [[0,1], [3,4]],
  "interaction_terms": {           // 交互项系数
    "x0*x1": -0.068,
    "x3*x4": 0.040
  },
  "source": "SimpleOracle with seed=42"
}
```

---

### 2.3 iterations.csv

```csv
iteration,phase,y_value,lambda_t,gamma_t,r_t,n_train,acqf_mean,acqf_std,min_distance,x0,x1,x2,x3,x4,x5
1,warmup,4.24,,,,,,,, 0.0,0.0,1.0,1.0,0.0,0.0
2,warmup,2.97,,,,,,,3.61, 0.5,0.0,0.0,0.0,1.0,1.0
3,warmup,4.10,,,,,,,1.0, 1.0,1.0,1.0,1.0,1.0,0.0
4,eur,5.68,0.40,0.30,0.50,3,-0.000,1.24,1.80, 8.5,6.5,1.0,1.0,0.0,1.0
5,eur,4.93,0.36,0.28,0.50,4,-0.000,0.99,1.41, 8.5,8.0,2.0,0.0,0.0,2.0
```

**列说明**:
- `iteration`: 试次编号(1-based)
- `phase`: `warmup` (前3个golden points) 或 `eur` (EUR采集)
- `y_value`: 观测响应值
- `lambda_t`, `gamma_t`, `r_t`: EUR动态权重(warmup阶段为空)
- `n_train`: 当前训练样本数
- `acqf_mean`, `acqf_std`: 采集函数统计(可选)
- `min_distance`: 到最近历史点的距离
- `x0-x5`: 参数值(实际值,非归一化)

---

### 2.4 summary.json (精简版)

```json
{
  // ========== 基础配置 ==========
  "experiment": {
    "timestamp": "20251213_100821",
    "config": "eur_residual_test.ini",
    "tag": null,
    "result_dir": "...",
    "config_path": "..."
  },

  "parameters": {
    "total_budget": 5,
    "warmup_budget": 3,
    "eur_budget": 2,
    "seed": 42,
    "interaction_pairs": [[0,1], [3,4]],
    "parnames": ["x1_CeilingHeight", ...]
  },

  // ========== Oracle规格 ==========
  "oracle": {
    "model_type": "linear",
    "bias": 3.0,
    "noise_std": 0.1,
    "main_weights": [0.248, -0.069, 0.324, 0.762, -0.117, -0.117],
    "interaction_weights": {"x0*x1": -0.068, "x3*x4": 0.040}
  },

  // ========== 采样轨迹统计 ==========
  "sampling_trajectory": {
    "lambda_t": {"initial": 0.40, "final": 0.36, "mean": 0.38, "std": 0.02},
    "gamma_t": {"initial": 0.30, "final": 0.28, "mean": 0.29, "std": 0.01},
    "r_t": {"initial": 0.50, "final": 0.50, "mean": 0.50, "std": 0.00},
    "diversity": {"min_distance_mean": 1.96, "unique_samples": 5}
  },

  // ========== 效应恢复评估 ⭐⭐ (最核心) ==========
  "effect_recovery": {
    // 主效应恢复
    "main_effects": {
      "correlation": {
        "spearman": -0.20,
        "spearman_pval": 0.704,
        "pearson": -0.54
      },
      "rmse": 0.319,
      "mae": 0.294
    },

    // 交互效应恢复
    "interaction_effects": {
      "correlation": {
        "spearman": 1.0,
        "spearman_pval": null,
        "pearson": 1.0
      },
      "rmse": 0.057,
      "mae": 0.056
    },

    // 模型拟合质量
    "model_fit": {
      "lr_r2": 1.0,        // 线性回归R²
      "n_train": 5,
      "model_type": "regression"
    }
  },

  // ========== 预测质量 (在测试集上) ==========
  "prediction_quality": {
    "R2": -0.629,
    "RMSE": 0.493,
    "MAE": 0.242,
    "n_test": 1000,
    "test_accuracy": 0.791  // 对于分类任务
  },

  // ========== 效应对比 (简化,不存完整数组) ==========
  "effect_comparison": {
    "main_effects": {
      "oracle": [0.279, 0.297, 0.355, -0.092, 0.062, 0.095],
      "estimated": [0.000, 0.000, 0.000, 0.417, -0.167, 0.000],
      "error": [0.279, 0.297, 0.355, 0.508, 0.229, 0.095]  // abs diff
    },
    "interaction_effects": {
      "pairs": [[0,1], [3,4]],
      "oracle": [-0.068, 0.040],
      "estimated": [0.000, 0.083],
      "error": [0.068, 0.043]
    }
  }
}
```

**关键改进**:
1. ✅ **结构化分组**: experiment, parameters, oracle, sampling_trajectory, effect_recovery, prediction_quality
2. ✅ **移除冗余**: 不再重复存储 prediction_quality
3. ✅ **精简数组**: effect_comparison只存统计量,不存y_true/y_pred的1000个值
4. ✅ **突出核心指标**: effect_recovery放在显著位置,包含Spearman相关性
5. ✅ **保留关键轨迹**: lambda_t, gamma_t, r_t的初始/最终/均值/标准差

---

## 3. 实现计划

### 3.1 data_saver.py 增强

新增函数:
```python
def save_oracle_spec(result_dir, oracle) -> None:
    """保存Oracle规格到oracle_spec.json"""

def save_iterations_csv(result_dir, interaction_logs, eur_diagnostics) -> None:
    """生成iterations.csv,合并trial记录和EUR诊断数据"""

def save_enhanced_summary(result_dir, summary_dict) -> None:
    """保存增强的summary.json,包含完整评估结果"""
```

### 3.2 run_eur_residual.py 集成

在采样完成后:
```python
# 1. 保存Oracle规格
save_oracle_spec(result_dir, oracle)

# 2. 调用evaluation_v2评估效应恢复
from modules.evaluation_v2 import evaluate_effect_recovery_v2
eval_results = evaluate_effect_recovery_v2(
    server=server,
    oracle=oracle,
    design_space=design_space_array,
    configured_pairs=[(0,1), (3,4)]
)

# 3. 生成增强的summary
summary = build_enhanced_summary(
    config=config,
    oracle=oracle,
    interaction_logs=interaction_logs,
    eur_diagnostics=eur_diagnostics,
    eval_results=eval_results
)
save_enhanced_summary(result_dir, summary)

# 4. 生成iterations.csv
save_iterations_csv(result_dir, interaction_logs, eur_diagnostics)
```

---

## 4. 优先级

### 高优先级 (必须实现)
- ✅ `oracle_spec.json`
- ✅ 增强 `summary.json` (包含effect_recovery)
- ✅ `iterations.csv`

### 中优先级 (可选)
- `diagnostics/` 目录 (EUR诊断数据单独存储)
- 更详细的采样质量指标

### 低优先级 (不实现)
- ❌ debug/ 目录 (experiment.db/log过大)
- ❌ 完整的y_true/y_pred数组 (1000个值太冗余)
- ❌ aepsych_response_t0.json (价值低)

---

## 5. 测试验证

运行命令:
```bash
pixi run python run_eur_residual.py --budget 10
```

检查输出:
```
results/{timestamp}/
├── data_files/
│   ├── interaction_log.json  ✓
│   ├── sampling_history.npy  ✓
│   ├── oracle_spec.json      ✓ NEW
│   ├── iterations.csv        ✓ NEW
│   └── summary.json          ✓ ENHANCED
└── temp_config.ini           ✓
```

验证summary.json包含:
- ✓ oracle规格
- ✓ sampling_trajectory (lambda_t, gamma_t, r_t统计)
- ✓ effect_recovery (Spearman相关性)
- ✓ prediction_quality
- ✓ effect_comparison (不含完整数组)
