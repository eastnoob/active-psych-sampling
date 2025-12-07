# WarmupMixedPoolGenerator

面向混合类型自变量的池式（pool-based）高信息密度 warmup 选择器：
从候选池中挑选一小撮点，使主效应信息量（D-Optimal）最大化，允许二阶交互混淆；可选加入“中心点”。

## 支持的变量类型

- 连续（continuous）、整数（integer）、序数（ordinal）
- 类别（categorical，多水平）、布尔（boolean）

## 选择策略（要点）

- 将混合型变量按“主效应编码”构造成模型矩阵（含截距）。
  - 连续/整数/序数：缩放到 [-1,+1] 主效应列
  - 布尔：{0,1} -> {-1,+1}
  - 类别：Treatment 编码（L-1 列）
- 在池上做贪心 D-Optimal 子集选择：每次加入使 log|X'X| 增幅最大的候选
- 可选从池中预留“中心点”（按归一化空间欧氏距离最近）
- 目标规模 n_runs 默认由规则给出（与维度 k 近似一致），并确保 ≥ 列数（满秩）

## 使用

```python
import torch
from extensions.custom_generators.warmup_mixed_pool import WarmupMixedPoolGenerator

# 构造池 P: [N x d]
P = ...  # 每列为一个变量的原始取值（类别/序数可以是整数标签）

schema = [
    {"type": "continuous", "lb": -5.0, "ub": 5.0},
    {"type": "integer", "lb": 0.0, "ub": 5.0},
    {"type": "ordinal", "levels": [1, 2, 3]},
    {"type": "categorical", "levels": [0, 1, 2, 3]},
    {"type": "boolean"},
]

gen = WarmupMixedPoolGenerator(pool_points=P, schema=schema, n_runs=None, add_center=True)

pts = []
while not gen.finished:
    pts.append(gen.gen(5))
X = torch.cat(pts, dim=0)
```

## 测试（Pixi）

```powershell
pixi run -- python extensions/custom_generators/warmup_mixed_pool/tests/run_tests.py
```

## 说明与边界

- 若类别水平较多，则主效应编码列数会增大，最小可行运行数也会随之升高；生成器会自动提升到满秩所需规模（并遵循规则上限，如 12/16/20）。
- 该选择器专注主效应的可估性与信息量；二阶交互建议由后续 EUR 在池中继续逐步学习。
- 输出点均来自原始池；生成器本身不依赖里克特因变量的分布或模型。
