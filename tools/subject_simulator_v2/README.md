# Subject Simulator V2

**简洁、可靠、易用的被试模拟器**

## 核心理念

- **简单**：线性模型 `y = weights·x + interactions + noise`
- **可靠**：参数完整保存，100%可复现
- **正态性**：自动检查响应分布，避免单调被试
- **灵活**：支持任意长度输入，自动适配交互效应

## 快速开始

### 场景1：生成被试集群

```python
from subject_simulator_v2 import ClusterGenerator
import numpy as np

# 设计空间 (324个条件, 6个特征)
design_space = np.load("design_space.npy")

# 生成5个被试的集群
gen = ClusterGenerator(
    design_space=design_space,
    n_subjects=5,
    population_std=0.3,     # 群体权重标准差（窄）
    individual_std=0.1,     # 个体偏差标准差（更窄）
    interaction_pairs=[(3,4), (0,1)],
    ensure_normality=True   # 自动检查正态性
)

cluster = gen.generate_cluster(output_dir="output/cluster_001")
```

**输出文件**：
```
output/cluster_001/
├── subject_1.csv              # 被试1的响应数据
├── subject_1_spec.json        # 被试1的完整参数
├── subject_2.csv
├── subject_2_spec.json
...
├── combined_results.csv       # 合并数据
└── cluster_summary.json       # 集群摘要
```

### 场景2：加载已有被试

```python
from subject_simulator_v2 import load_subject

# 精确复现被试
subject = load_subject("output/cluster_001/subject_1_spec.json")

# 逐行作答
x = np.array([0, 0, 0, 2, 1, 2])
y = subject(x)  # 返回Likert评分
```

### 场景3：创建单个被试

```python
from subject_simulator_v2 import LinearSubject
import numpy as np

subject = LinearSubject(
    weights=np.array([0.2, -0.3, 0.5, 0.1, -0.4, 0.3]),
    interaction_weights={(3, 4): -0.15, (0, 1): 0.12},
    bias=0.0,
    noise_std=0.1,
    likert_levels=5,
    likert_sensitivity=2.0
)

# 保存
subject.save("my_subject.json")
```

## 核心特性

### ✓ 参数完整保存

所有参数都保存在JSON中，无遗漏：
- 主效应权重 `weights`
- 交互效应权重 `interaction_weights`
- 截距 `bias`、噪声 `noise_std`
- Likert参数 `likert_levels`, `likert_sensitivity`

### ✓ 正态性保障

自动检查被试响应分布：
- 至少覆盖3个Likert等级
- 单个等级占比不超过60%
- 均值在2-4之间

不符合的被试会自动重采样（最多20次）。

### ✓ 100%可复现

```python
# 保存
subject.save("subject.json")

# 加载后完全一致
subject2 = load_subject("subject.json")
assert subject(x) == subject2(x)  # ✓
```

## 架构

```
subject_simulator_v2/
├── __init__.py          # API导出
├── base.py              # BaseSubject抽象类
├── linear.py            # LinearSubject实现
├── cluster.py           # ClusterGenerator
├── validators.py        # 正态性检查
├── io.py                # 保存/加载
└── examples/            # 示例脚本
```

## 迁移指南

从旧版本 `MixedEffectsLatentSubject` 迁移：

**旧版本（复杂）**：
```python
subject = MixedEffectsLatentSubject(
    num_latent_vars=2,
    num_observed_vars=1,
    factor_loadings=...,
    item_biases=...,
    item_noises=...,
    population_weights=...,
    # 10+个参数
)
```

**新版本（简洁）**：
```python
subject = LinearSubject(
    weights=...,
    interaction_weights=...,
    # 4个核心参数
)
```

## 版本

- **v1.0** (2025-11-30): 初始版本，核心功能完整
- 基于3个关键bug修复的经验重构

## 许可

MIT
