# WarmupMinimalGenerator

高信息密度、极简 warmup 设计生成器（只关注主效应，允许二阶交互混淆），用于以尽量少的预算快速建立主效应方向，为后续的 EUR 优化留出预算。

## 核心原则

- 用最少点数获得主效应信息
- 牺牲部分交互效应换取效率（允许二阶交互混淆）
- 为后续 EUR 节省预算
- 确保主效应设计满秩（无主效应混淆）

## 设计规则（已实现）

- 3 因子以下：4 点（L4 2 水平正交阵）
- 4–7 因子：8 点（2^3 完全因子设计 + 派生列）
- 8 因子：为了保持满秩，自动提升为 12 点 PB 设计（会记录 warning）
- 9–12 因子：k+1 点（12-run Plackett–Burman，取前 k 列）
- 13+ 因子：选择最近的 4 倍数点，目前支持到 16（Hadamard）与 20（PB）

备注：中心点可选，默认添加；添加后可辅助方差估计与模型初始稳健性。

## 使用方法

```python
import torch
from extensions.custom_generators.warmup_minimal import WarmupMinimalGenerator

# 以 5 因子为例，范围 [0, 1]
lb = torch.zeros(5)
ub = torch.ones(5)

# 默认添加中心点
gen = WarmupMinimalGenerator(lb=lb, ub=ub, add_center=True)

# 逐点获取
pts = []
while not gen.finished:
    pts.append(gen.gen(1))
X = torch.cat(pts, dim=0)  # [n_runs x 5]
```

生成的点均在 [lb, ub] 内；当 add_center=True 时，最后一行是中心点（各维度中点）。

## 输出格式

- 与普通 AEPsych generator 一致，`gen(num_points)` 返回 `[num_points x dim]` 的张量，可直接接入现有流程。

## 设计实现细节

- L4（2 水平 4 试验 3 列）用于 k≤3 的场景。
- 8-run（2^3 完全因子 + 派生列）用于 4≤k≤7，列依次为 A,B,C,AB,AC,BC,ABC，再裁剪到前 k 列。
- PB(12) 与 PB(20) 使用常见标准表（硬编码一个等价变体）；16-run 用 Sylvester 构造的 Hadamard。
- 将编码空间 {-1, +1} 线性映射到 [lb, ub]；中心点编码为 0 并映射到 (lb+ub)/2。

## 测试

在本目录的 `tests/` 下提供了一个轻量脚本：

```bash
python extensions/custom_generators/warmup_minimal/tests/run_tests.py
```

该脚本会检查：

- 设计规模是否满足规则（含中心点时会多 1 行）
- 点是否落在边界内
- 去除中心点后，编码矩阵的秩 ≥ min(试验数, 因子数)

通过后会打印 `ALL TESTS PASSED`。

## 注意事项

- 当前版本针对连续型/可线性编码的维度设计；若含有纯分类变量，应在上层将其固定或拆分场景后再使用。
- 当因子数较大（>19）且需要 PB 更大规模时，可拓展 PB 表或使用更通用的 OA 生成器；目前实现到 20-run。
