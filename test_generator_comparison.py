#!/usr/bin/env python3
"""
直接对比ManualGenerator和CustomPoolBasedGenerator的返回值
在经过ParameterTransformedGenerator wrap后的行为
"""
import sys
sys.path.insert(0, 'd:/ENVS/active-psych-sampling/extensions/custom_generators')

import torch
from aepsych.server import AEPsychServer
import json

# 使用实际的config
config_path = 'd:/ENVS/active-psych-sampling/tests/is_EUR_work/00_plans/251206/scripts/eur_config_residual.ini'

# 创建server
server = AEPsychServer(database_path=':memory:')

# Setup
with open(config_path, 'r', encoding='utf-8') as f:
    config_str = f.read()

setup_msg = {
    'type': 'setup',
    'version': '0.01',
    'message': {'config_str': config_str}
}

try:
    server.handle_request(setup_msg)
except Exception as e:
    print(f"Setup error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 80)
print("Phase 1: ManualGenerator (warmup)")
print("=" * 80)

# Ask 3 warmup points from ManualGenerator
for i in range(3):
    ask_msg = {'type': 'ask', 'message': ''}
    response = server.handle_request(ask_msg)
    
    print(f"\n--- Warmup {i+1} ---")
    print(f"Response keys: {list(response.keys())}")
    
    # 提取参数值
    x_dict = {}
    for key in response.keys():
        if key.startswith("'x") or key.startswith("x"):
            clean_key = key.strip("'")
            val = response[key]
            if isinstance(val, list):
                val = val[0]
            x_dict[clean_key] = val
    
    print(f"x1_CeilingHeight: {x_dict.get('x1_CeilingHeight', 'N/A')}")
    print(f"x2_GridModule: {x_dict.get('x2_GridModule', 'N/A')}")
    
    # Tell
    tell_msg = {
        'type': 'tell',
        'message': {
            'config': x_dict,
            'outcome': 4.0
        }
    }
    server.handle_request(tell_msg)

print("\n" + "=" * 80)
print("Phase 2: CustomPoolBasedGenerator (EUR)")
print("=" * 80)

# Check generator type
strat = server._strats[1]  # opt_strat
gen = strat.generator

print(f"\nGenerator type: {type(gen).__name__}")
print(f"Has _base_obj: {hasattr(gen, '_base_obj')}")

if hasattr(gen, '_base_obj'):
    base_gen = gen._base_obj
    print(f"Base generator type: {type(base_gen).__name__}")
    
    # Patch to see what base generator returns
    original_gen = base_gen.gen
    
    captured_output = []
    
    def logged_gen(num_points, model=None, **kwargs):
        print(f"\n[BaseGen.gen] Called with num_points={num_points}")
        result = original_gen(num_points, model, **kwargs)
        print(f"[BaseGen.gen] Returning shape: {result.shape}")
        print(f"[BaseGen.gen] Returning first point: {result[0].tolist()}")
        captured_output.append(result.clone())
        return result
    
    base_gen.gen = logged_gen

# Also patch wrapper's gen
if hasattr(gen, 'transforms'):
    print(f"\nTransforms present: {list(gen.transforms._modules.keys())}")
    
    original_wrapper_gen = gen.gen
    
    def logged_wrapper_gen(num_points, model=None, **kwargs):
        print(f"\n[Wrapper.gen] Called")
        print(f"[Wrapper.gen] About to call base generator...")
        result = original_wrapper_gen(num_points, model, **kwargs)
        print(f"[Wrapper.gen] After untransform, returning: {result[0].tolist()}")
        return result
    
    gen.gen = logged_wrapper_gen

# Ask for next point (EUR)
print("\n--- EUR Ask 1 ---")
ask_msg = {'type': 'ask', 'message': ''}
response = server.handle_request(ask_msg)

print(f"\n=== Final Response ===")
x_dict = {}
for key in response.keys():
    if key.startswith("'x") or key.startswith("x"):
        clean_key = key.strip("'")
        val = response[key]
        if isinstance(val, list):
            val = val[0]
        x_dict[clean_key] = val

print(f"x1_CeilingHeight: {x_dict.get('x1_CeilingHeight', 'N/A')}")
print(f"x2_GridModule: {x_dict.get('x2_GridModule', 'N/A')}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
if captured_output:
    base_output = captured_output[0][0]
    print(f"\nBase generator returned: {base_output.tolist()}")
    print(f"Final response x1: {x_dict.get('x1_CeilingHeight', 'N/A')}")
    
    if base_output[0].item() == 2.8:
        print("\n⚠️  Base generator returns actual values (2.8)")
        print(f"   But final response is {x_dict.get('x1_CeilingHeight', 'N/A')}")
        print(f"   Ratio: {x_dict.get('x1_CeilingHeight', 0) / 2.8:.2f}")
