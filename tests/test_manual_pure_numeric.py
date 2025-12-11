#!/usr/bin/env python3
"""测试: 纯 numeric categorical 时 ManualGenerator 的行为"""

import torch
from aepsych.server import AEPsychServer
from pathlib import Path

# 测试1: 纯 numeric categorical (无 string)
config_pure_numeric = """
[common]
stimuli_per_trial = 1
outcome_types = [binary]
strategy_names = [warmup]
parnames = [x1, x2]
use_ax = False

[x1]
par_type = continuous
values = [2.8, 4.0, 8.5]

[x2]
par_type = continuous
values = [6.5, 8.0]

[warmup]
generator = ManualGenerator
points = [[2.8, 6.5], [4.0, 6.5], [8.5, 8.0]]
min_asks = 3
max_asks = 3
"""

# 测试2: 混合 numeric + string categorical
config_mixed = """
[common]
stimuli_per_trial = 1
outcome_types = [binary]
strategy_names = [warmup]
parnames = [x1, x2]
use_ax = False

[x1]
par_type = continuous
values = [2.8, 4.0, 8.5]

[x2]
par_type = categorical
choices = [Chaos, Strict]

[warmup]
generator = ManualGenerator
points = [[2.8, Chaos], [4.0, Chaos], [8.5, Strict]]
min_asks = 3
max_asks = 3
"""

print("="*80)
print("测试1: 纯 Numeric Categorical (无 string)")
print("="*80)

try:
    server1 = AEPsychServer()
    setup_msg1 = {"type": "setup", "message": {"config_str": config_pure_numeric}}
    server1.handle_request(setup_msg1)

    strat1 = server1.strats[0]
    gen1 = strat1.generator

    print(f"\nGenerator类型: {type(gen1).__name__}")
    print(f"lb: {gen1.lb}")
    print(f"ub: {gen1.ub}")
    print(f"points (前3个): {gen1.points[:3]}")
    print(f"_skip_untransform: {gen1._skip_untransform}")

    # 获取第一个点
    ask_msg1 = {"type": "ask"}
    response1 = server1.handle_request(ask_msg1)
    print(f"\nAsk返回的第一个点: {response1}")

    # 检查是否在有效范围内
    x1_val = response1["config"]['x1'][0]
    valid_x1 = [2.8, 4.0, 8.5]
    print(f"\nx1值: {x1_val}")
    print(f"是否在有效值中: {x1_val in valid_x1 or any(abs(x1_val - v) < 0.01 for v in valid_x1)}")

except Exception as e:
    print(f"测试1失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("测试2: 混合 Numeric + String Categorical")
print("="*80)

try:
    server2 = AEPsychServer()
    setup_msg2 = {"type": "setup", "message": {"config_str": config_mixed}}
    server2.handle_request(setup_msg2)

    strat2 = server2.strats[0]
    gen2 = strat2.generator

    print(f"\nGenerator类型: {type(gen2).__name__}")
    print(f"lb: {gen2.lb}")
    print(f"ub: {gen2.ub}")
    print(f"points (前3个): {gen2.points[:3]}")
    print(f"_skip_untransform: {gen2._skip_untransform}")

    # 获取第一个点
    ask_msg2 = {"type": "ask"}
    response2 = server2.handle_request(ask_msg2)
    print(f"\nAsk返回的第一个点: {response2}")

    # 检查是否在有效范围内
    x1_val = response2["config"]['x1'][0]
    valid_x1 = [2.8, 4.0, 8.5]
    print(f"\nx1值: {x1_val}")
    print(f"是否在有效值中: {x1_val in valid_x1 or any(abs(x1_val - v) < 0.01 for v in valid_x1)}")

except Exception as e:
    print(f"测试2失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("结论")
print("="*80)
print("如果测试1成功、测试2失败，则证明问题确实是由 string categorical 导致的")
