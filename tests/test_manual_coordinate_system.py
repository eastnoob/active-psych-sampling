#!/usr/bin/env python3
"""测试: ManualGenerator 在不同参数类型下的坐标系统"""

import torch
from aepsych.config import Config
from aepsych.generators import ManualGenerator

print("="*80)
print("测试: ManualGenerator 的坐标系统行为")
print("="*80)

# 测试1: 全部 numeric categorical
print("\n测试1: 全部 Numeric Categorical (choices用数值)")
print("-"*80)

config1_str = """
[common]
parnames = [x1, x2]
lb = [2.8, 6.5]
ub = [8.5, 8.0]

[x1]
par_type = categorical
choices = [2.8, 4.0, 8.5]

[x2]
par_type = categorical
choices = [6.5, 8.0]

[test_strat]
min_asks = 3
generator = ManualGenerator

[ManualGenerator]
points = [[2.8, 6.5], [4.0, 6.5], [8.5, 8.0]]
"""

try:
    config1 = Config(config_str=config1_str)
    gen1 = ManualGenerator.from_config(config1, "ManualGenerator")

    print(f"[OK] 配置加载成功")
    print(f"  lb: {gen1.lb}")
    print(f"  ub: {gen1.ub}")
    print(f"  points[0]: {gen1.points[0]}")
    print(f"  _skip_untransform: {gen1._skip_untransform}")

    # 检查points是否在lb/ub范围内
    point = gen1.points[0]
    in_bounds = torch.all(point >= gen1.lb) and torch.all(point <= gen1.ub)
    print(f"  points[0]在lb/ub范围内: {in_bounds}")

except Exception as e:
    print(f"[FAIL] 测试1失败: {e}")

# 测试2: 混合 numeric + string categorical
print("\n测试2: 混合 Numeric + String Categorical")
print("-"*80)

config2_str = """
[common]
parnames = [x1, x2]
lb = [0, 0]
ub = [2, 1]

[x1]
par_type = categorical
choices = [2.8, 4.0, 8.5]

[x2]
par_type = categorical
choices = [Chaos, Strict]

[test_strat]
min_asks = 3
generator = ManualGenerator

[ManualGenerator]
points = [[2.8, Chaos], [4.0, Chaos], [8.5, Strict]]
"""

try:
    config2 = Config(config_str=config2_str)
    gen2 = ManualGenerator.from_config(config2, "ManualGenerator")

    print(f"[OK] 配置加载成功")
    print(f"  lb: {gen2.lb}")
    print(f"  ub: {gen2.ub}")
    print(f"  points[0]: {gen2.points[0]}")
    print(f"  _skip_untransform: {gen2._skip_untransform}")

    # 检查points是否在lb/ub范围内
    point = gen2.points[0]
    in_bounds = torch.all(point >= gen2.lb) and torch.all(point <= gen2.ub)
    print(f"  points[0]在lb/ub范围内: {in_bounds}")

    if not in_bounds:
        print(f"\n[WARN] 警告: points[0]={point} 不在lb={gen2.lb}/ub={gen2.ub}范围内!")
        print(f"  这就是产生invalid values的原因!")

except Exception as e:
    print(f"[FAIL] 测试2失败: {e}")

# 测试3: 混合但使用索引
print("\n测试3: 混合 Numeric + String Categorical (使用索引)")
print("-"*80)

config3_str = """
[common]
parnames = [x1, x2]
lb = [0, 0]
ub = [2, 1]

[x1]
par_type = categorical
choices = [2.8, 4.0, 8.5]

[x2]
par_type = categorical
choices = [Chaos, Strict]

[test_strat]
min_asks = 3
generator = ManualGenerator

[ManualGenerator]
points = [[0, 0], [1, 0], [2, 1]]
"""

try:
    config3 = Config(config_str=config3_str)
    gen3 = ManualGenerator.from_config(config3, "ManualGenerator")

    print(f"[OK] 配置加载成功")
    print(f"  lb: {gen3.lb}")
    print(f"  ub: {gen3.ub}")
    print(f"  points[0]: {gen3.points[0]}")
    print(f"  _skip_untransform: {gen3._skip_untransform}")

    # 检查points是否在lb/ub范围内
    point = gen3.points[0]
    in_bounds = torch.all(point >= gen3.lb) and torch.all(point <= gen3.ub)
    print(f"  points[0]在lb/ub范围内: {in_bounds}")

except Exception as e:
    print(f"��� 测试3失败: {e}")

print("\n" + "="*80)
print("结论")
print("="*80)
print("如果测试1用实际值成功，测试2用实际值失败但测试3用索引成功,")
print("则证明: 混合string categorical时必须用索引，纯numeric时可以用实际值")
