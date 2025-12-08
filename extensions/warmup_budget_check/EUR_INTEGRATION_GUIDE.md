# EUR 验证 - 使用复制被试群体指南

> **目标：将 reproduce_subject_cluster.py 创建的被试用于 EUR 采集函数验证**

---

## 🎯 使用场景

你已经：
1. ✅ 完成 Phase 1 实验（5个被试，150次采样）
2. ✅ 运行 `reproduce_subject_cluster.py` 创建了同群体的被试（例如20个）
3. 🔜 现在需要用这些被试验证 EUR 采集函数的效果

---

## 🚀 快速开始

### 场景1：基础验证（单个被试）

```bash
cd extensions/warmup_budget_check

# 使用被试1进行50次EUR采样
python run_eur_with_reproduced_subjects.py \
    --subject_spec phase1_analysis_output/202512011547/step1_5/result/reproduced_subjects/subject_cluster_specs.json \
    --subject_id 1 \
    --budget 50
```

**输出：**
- `eur_results/subject_1_时间戳/subject_info.json` - 被试信息
- （待添加）EUR 采样历史、效应识别结果等

---

### 场景2：批量验证（多个被试）

如果你想对所有20个被试分别运行验证：

```bash
cd extensions/warmup_budget_check

for subject_id in {1..20}; do
    echo "Running subject $subject_id..."
    python run_eur_with_reproduced_subjects.py \
        --subject_spec phase1_analysis_output/202512011547/step1_5/result/reproduced_subjects/subject_cluster_specs.json \
        --subject_id $subject_id \
        --budget 50
done
```

---

## 📋 参数说明

| 参数 | 默认值 | 说明 |
|-----|--------|------|
| `--subject_spec` | **必填** | 被试规格JSON路径（来自 reproduce_subject_cluster.py） |
| `--subject_id` | 1 | 要使用的被试ID（1-N） |
| `--budget` | 50 | EUR采样次数 |
| `--config` | eur_config_sps.ini | EUR配置文件 |
| `--output_dir` | `eur_results/subject_ID_时间戳` | 输出目录 |

---

## 🔧 工作原理

### 1. 被试对象对比

#### 原始 EUR 脚本（test/is_EUR_work/run_eur_verification_sps.py）

```python
# Line 249-257
oracle = SingleSubject(
    seed=123,
    likert_levels=5,
    weight_std=0.7,
    noise_std=0.35,
    interaction_pairs=[(1, 2), (3, 4), (0, 5)],
    interaction_scale=0.45,
    likert_sensitivity=2.2,
)
```

- 使用 `SingleSubject` 类（来自 `subject_simulator`）
- 每次运行都用随机权重初始化
- 适合测试EUR采集函数的**通用行为**

#### 新脚本（run_eur_with_reproduced_subjects.py）

```python
oracle = LinearSubject(
    weights=np.array(subject_spec['subject_weights']),
    interaction_weights=interaction_weights,
    bias=subject_spec['bias'],
    noise_std=0.0,
    likert_levels=subject_spec['likert_levels'],
    likert_sensitivity=subject_spec['likert_sensitivity'],
    seed=subject_spec['seed']
)
```

- 使用 `LinearSubject` 类（来自 `subject_simulator_v2.linear`）
- 加载**预定义的权重**（来自 Phase 1 分析）
- 适合测试EUR在**特定群体**上的表现

### 2. 数据流

```
Phase 1 实验 (5个被试)
    ↓
Step 1.5: 估计群体参数
    ↓ (产生 fixed_weights_auto.json)
reproduce_subject_cluster.py
    ↓ (生成 subject_cluster_specs.json)
run_eur_with_reproduced_subjects.py
    ↓
加载被试规格 → 创建 LinearSubject 对象
    ↓
EUR 采样循环 (调用 oracle(x) 获取响应)
    ↓
效应识别 + 预测质量评估
```

### 3. 关键接口：`oracle(x)` 调用

EUR 验证脚本在每次迭代中会调用：

```python
# Line 457 in run_eur_verification_sps.py
y_raw = oracle(x_array)  # x_array shape: (6,)
y_likert = int(np.clip(y_raw, 1, 5))
y = y_likert - 1  # 转换为 0-4
```

`LinearSubject` 完全兼容这个接口：

```python
def __call__(self, x: np.ndarray) -> Union[float, int]:
    # 1. 计算主效应
    y = self.bias + np.dot(self.weights, x)

    # 2. 添加交互效应
    for (i, j), weight in self.interaction_weights.items():
        y += weight * x[i] * x[j]

    # 3. 转换为 Likert (1-5)
    if self.likert_levels is not None:
        return self._to_likert(y)
    return y
```

✅ **完全兼容！**

---

## 📊 预期结果

### Phase 1 群体的特征（来自 202512011547）

从 `subject_cluster_summary.txt` 我们知道：

```
Population weights: [ 0.12417854 -0.03456608  0.16192213  0.38075746 -0.05853834 -0.05853424]
Bias: -0.2181
Individual std: 0.1
```

这意味着：
- **x3 (OuterFurniture)** 和 **x4 (VisualBoundary)** 对响应影响最大
- **x1 (CeilingHeight)** 和 **x2 (GridModule)** 影响较小
- 个体差异标准差 = 0.1 （较低，群体一致性高）

### 预期 EUR 验证结果

1. **效应识别准确率：**
   - 主效应：应该能正确识别 x3 和 x4 为关键因子
   - 交互效应：应该能识别 (3,4) 和 (0,1) 交互对

2. **预测质量：**
   - R² > 0.75（因为群体一致性高）
   - RMSE < 0.30（被试间差异小）

3. **被试间差异：**
   - 20个被试的 EUR 结果应该**较为一致**
   - Between-subject SD ~ 0.6-0.7（来自 individual_std=0.1）

---

## 🔍 调试与验证

### 检查被试是否正确加载

在 `run_eur_with_reproduced_subjects.py` 运行后，你应该看到：

```
【被试模型规格 - Subject 1】
============================================================
  特征数量: 6
  Likert级别: 5
  噪声标准差: 0.0 (确定性)

  主效应权重:
    x0: -0.050561
    x1: 0.000811
    x2: +0.277138
    x3: +0.355514
    x4: +0.039684
    x5: -0.007352

  交互项权重:
    x3×x4: +0.120000
    x0×x1: -0.020000

  Bias: -0.218100
============================================================
```

### 验证响应范围

手动测试一个输入：

```python
import numpy as np
x_test = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
y_test = oracle(x_test)
print(f"测试响应: {y_test} (应该在 1-5 之间)")
```

### 检查被试间差异

如果你对多个被试运行验证，比较：

```bash
# 提取每个被试的 y_mean
for d in eur_results/subject_*/; do
    echo "$d: $(python -c "import json; print(json.load(open('$d/summary.json'))['y_statistics']['mean'])")"
done
```

预期：所有被试的 y_mean 应该在 ±0.6 范围内（因为 individual_std=0.1）

---

## 💡 进阶用法

### 1. 添加测量噪声

如果你想模拟真实实验的**试次内变异**：

```python
# 在 run_eur_with_reproduced_subjects.py 中修改
oracle = LinearSubject(
    weights=np.array(subject_spec['subject_weights']),
    interaction_weights=interaction_weights,
    bias=subject_spec['bias'],
    noise_std=0.35,  # 添加噪声（与原始 EUR 脚本一致）
    likert_levels=subject_spec['likert_levels'],
    likert_sensitivity=subject_spec['likert_sensitivity'],
    seed=subject_spec['seed']
)
```

### 2. 使用不同的 EUR 配置

```bash
python run_eur_with_reproduced_subjects.py \
    --subject_spec phase1_analysis_output/202512011547/step1_5/result/reproduced_subjects/subject_cluster_specs.json \
    --subject_id 1 \
    --budget 100 \
    --config custom_eur_config.ini
```

### 3. 对比不同群体

如果你创建了多个被试群体（例如不同的 `individual_std`）：

```bash
# 群体A：低个体差异 (individual_std=0.08)
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 10 \
    --individual_std 0.08 \
    --output_dir cluster_A

# 群体B：高个体差异 (individual_std=0.15)
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 10 \
    --individual_std 0.15 \
    --output_dir cluster_B

# 分别验证
python run_eur_with_reproduced_subjects.py \
    --subject_spec cluster_A/subject_cluster_specs.json \
    --subject_id 1 \
    --budget 50 \
    --output_dir eur_cluster_A

python run_eur_with_reproduced_subjects.py \
    --subject_spec cluster_B/subject_cluster_specs.json \
    --subject_id 1 \
    --budget 50 \
    --output_dir eur_cluster_B
```

**对比结果：**
- 群体A（低差异）：EUR 预测 R² 应该更高
- 群体B（高差异）：EUR 需要更多预算才能达到相同精度

---

## ⚠️ 注意事项

### 1. 特征映射一致性

确保 `reproduce_subject_cluster.py` 使用的设计空间与 EUR 验证使用的**完全一致**：

```python
# 在 ALL_CONFIG 中
"design_csv": "data/only_independences/.../6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"
```

与 EUR 脚本中的设计空间（Line 138-145）应该是同一个文件。

### 2. 交互对索引

`reproduce_subject_cluster.py` 使用 **0-based 索引**：
- 交互对 `(3, 4)` 指的是 x3 × x4

EUR 脚本也使用 0-based 索引，因此**完全兼容**。

### 3. Likert 映射

- `LinearSubject` 输出：1-5 (Likert)
- EUR 脚本期望：1-5 (Likert)
- EUR 内部编码：0-4 (`y = y_likert - 1`)

✅ **无需额外转换**

### 4. 确定性 vs 随机性

当前脚本使用 `noise_std=0.0`（确定性输出），这对于**调试**非常有用。

但真实实验有**测量噪声**，建议在最终验证时添加噪声（见进阶用法1）。

---

## 🆘 故障排查

### 问题1：找不到被试规格文件

**错误：**
```
[错误] 被试规格文件不存在: phase1_analysis_output/.../subject_cluster_specs.json
```

**解决：**
```bash
# 确认文件存在
ls phase1_analysis_output/202512011547/step1_5/result/reproduced_subjects/

# 如果不存在，重新运行
python reproduce_subject_cluster.py \
    --base_dir phase1_analysis_output/202512011547/step1_5/result \
    --n_subjects 20
```

### 问题2：被试ID超出范围

**错误：**
```
[错误] 找不到被试ID=25
可用被试ID: [1, 2, 3, ..., 20]
```

**解决：**
使用 1-20 范围内的ID，或者创建更多被试。

### 问题3：响应值超出范围

**错误：**
```
RuntimeError: Likert response out of bounds: y=6
```

**解决：**
检查 `likert_sensitivity` 参数（应该在 1.5-2.5 范围内）。

如果权重过大，降低 `population_std` 或 `interaction_scale`。

---

## 🔗 相关文档

- **复制被试工具**: `REPRODUCE_SUBJECTS_GUIDE.md`
- **ALL_CONFIG 配置**: `ALL_CONFIG_GUIDE.md`
- **Phase 1 分析报告**: `phase1_analysis_output/202512011547/step2/phase1_analysis_report.md`
- **原始 EUR 脚本**: `test/is_EUR_work/run_eur_verification_sps.py`

---

## 📦 当前进度

✅ **已完成：**
1. 加载被试规格
2. 创建 LinearSubject 对象
3. 打印被试模型规格
4. 保存被试信息到结果目录

🔜 **待完成：**
1. 加载设计空间（从 CSV）
2. 初始化 AEPsych Server（从 .ini 配置）
3. 运行 EUR 采样循环
4. 效应识别验证
5. 预测质量评估
6. 生成可视化报告

**提示：** 你可以复制 `test/is_EUR_work/run_eur_verification_sps.py` 的步骤1-6 代码到 `run_eur_with_reproduced_subjects.py` 的步骤3中，只需确保使用我们创建的 `oracle` 对象即可。

---

**最后更新**: 2025-12-01
**版本**: 1.0
**作者**: Claude Code
