#!/usr/bin/env python3
"""测试: continuous vs categorical 的坐标系统差异"""

import torch
from aepsych.config import Config
from aepsych.generators import ManualGenerator

print("="*80)
print("测试: par_type=continuous vs par_type=categorical")
print("="*80)

# 测试1: par_type=continuous with numeric values
print("\n测试1: par_type=continuous (discrete values)")
print("-"*80)

config1_str = """
[common]
parnames = [x1, x2]
lb = [2.8, 6.5]
ub = [8.5, 8.0]

[x1]
par_type = continuous
values = [2.8, 4.0, 8.5]

[x2]
par_type = continuous
values = [6.5, 8.0]

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
    print(f"\n  结论: par_type=continuous 时, lb/ub使用实际值空间!")

except Exception as e:
    print(f"[FAIL] 测试1失败: {e}")

# 测试2: par_type=categorical with numeric choices
print("\n测试2: par_type=categorical (numeric choices)")
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
choices = [6.5, 8.0]

[test_strat]
min_asks = 3
generator = ManualGenerator

[ManualGenerator]
points = [[0, 0], [1, 0], [2, 1]]
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
    print(f"\n  结论: par_type=categorical 时, lb/ub使用索引空间 [0, n_choices-1]!")

except Exception as e:
    print(f"[FAIL] 测试2失败: {e}")

print("\n" + "="*80)
print("最终结论")
print("="*80)
print("par_type=continuous: lb/ub是实际值空间, ManualGenerator points用实际值")
print("par_type=categorical: lb/ub是索引空间, ManualGenerator points必须用索引")
print("\n这与是否有string choices无关, 只取决于par_type!")
