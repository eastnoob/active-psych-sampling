#!/usr/bin/env python3
"""
Test what transforms are actually created for categorical
"""
from aepsych.config import Config

config_str = """
[common]
parnames = ['x1']
stimuli_per_trial = 1
outcome_types = [continuous]
strategy_names = [test_strat]

[x1]
par_type = categorical
choices = [2.8, 4.0, 8.5]

[test_strat]
min_asks = 1
generator = ManualGenerator
model = GPRegressionModel

[ManualGenerator]
points = [[2.8]]

[GPRegressionModel]
inducing_size = 10
"""

config = Config()
config.update(config_str=config_str)

# Get transforms
from aepsych.transforms.parameters import ParameterTransforms
transforms = ParameterTransforms.from_config(config)

print("=== Transforms Created ===")
for name, transform in transforms._modules.items():
    print(f"{name}: {type(transform).__name__}")
    if hasattr(transform, 'indices'):
        print(f"  indices: {transform.indices}")
    if hasattr(transform, 'categories'):
        print(f"  categories: {transform.categories}")

print(f"\n=== Test Transform ===")
import torch
test_input = torch.tensor([[2.8]])
print(f"Input: {test_input}")
transformed = transforms.transform(test_input)
print(f"After transform: {transformed}")
untransformed = transforms.untransform(transformed)
print(f"After untransform: {untransformed}")
