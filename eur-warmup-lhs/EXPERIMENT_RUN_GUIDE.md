# EUR-Warmup-LHS 实验运行说明

## 1. 目标与范围

本目录用于在小预算场景下执行模型无关的预热采样，并保持与原流程一致的输出结构：

- `step1`: 预热采样（已替换为 LHS）
- `step1_5`: 响应模拟
- `step2`: 交互分析与规格生成
- `step3`: Base GP 训练与扫描

说明：

- 预热采样阶段（Step1）不依赖后验模型形状，只依赖设计空间与预算。
- Step2/Step3 是预热后的分析与建模链路，保留原项目兼容性。

## 2. 输出结构（固定）

无论参数怎么调，步骤目录名固定如下：

```text
output_root/
  YYYYMMDD_HHMMSS/
    step1/
    step1_5/
    step2/
    step3/
```

可配置项只有 `step1.output_root`，用于切换总输出根目录。

## 3. Step1 采样策略

Step1 采用全局 LHS 映射到离散设计空间，并支持两种分配模式：

1. `allocation_mode = "disjoint"`
   - 默认模式。
   - 每位被试拿不同点，优先扩大覆盖。

2. `allocation_mode = "shared"`
   - 所有人共享同一组点。
   - 适合强调重复一致性，不适合最大化覆盖。

附加参数：

- `shared_points`: 在 `disjoint` 模式下保留的公共点数量。
- `lhs_oversample_factor`: LHS 候选放大倍数，越高越容易映射到分散点。
- `repeat_one_point`: 每位被试内可选重复点，用于纯误差估计。

## 4. 推荐配方（仅保留两个最终方案）

当前只保留两个最终可直接运行的配方：

1. `config/lhs_budget_5.toml`
   - 每人 5 次，极小预算冷启动。
   - 默认 `disjoint + shared_points=0`。

2. `config/lhs_budget_10.toml`
   - 每人 10 次，默认冷启动。
   - 默认 `disjoint + shared_points=1`。

这两个文件是当前推荐的最终结果入口。

## 5. 运行方式

### 5.1 环境准备

```powershell
Set-Location d:/ENVS/active-psych-sampling/eur-warmup-lhs
pixi install
```

### 5.2 交互式运行

```powershell
pixi run cli
```

在 CLI 中可选择：

- `1` 仅采样
- `all` 跑完整链路（1 -> 1.5 -> 2 -> 3）

### 5.3 使用指定配方

把目标配方复制/改名为 `config/temp_config.toml` 后，在 CLI 中选择该配置执行。

## 6. 各步骤产物说明

### Step1

- `subject_*.csv`：每位被试的试次配置
- `README_sampling.txt`：本次采样摘要

### Step1_5

- `subject_*.csv`：带响应列 `y`
- `MODEL_SUMMARY.txt`：模拟器参数摘要

### Step2

- `analysis_results.json`：分析详情
- `model_spec.json`：供 Step3 使用的规格处方
- `model_spec_summary.md`：人工可读摘要

### Step3

- `base_gp_state.pth`：模型参数
- 相关扫描/关键点输出（取决于配置）

## 7. 与原版兼容性

- 目录结构兼容：保持 `step1/step1_5/step2/step3`。
- 上下文兼容：`Context` 与模块串联方式不变。
- 唯一策略替换：Step1 从五段混合采样改为 LHS 采样分配。

## 8. 验证建议

最低验证：

1. 执行 Step1，确认输出路径是 `output_root/timestamp/step1`。
2. 执行 `pixi run test`，确认全流程联通。

进阶验证：

1. 对比 `lhs_budget_5.toml` 与 `lhs_budget_10.toml` 的覆盖率与 Step2 稳定性。
2. 若有先验交互，填写 `interaction_pairs` 与 `suspected_pairs` 做定向对比。
