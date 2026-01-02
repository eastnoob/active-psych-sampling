#!/usr/bin/env python3
"""Test to reproduce train_targets data loss with ordinal likelihood."""

import torch
import numpy as np
from aepsych.models import OrdinalGPModel
from aepsych.config import Config
import tempfile

print("Testing ordinal model data storage...")
print("=" * 80)

# Create a simple config for ordinal model
config_str = """
[common]
parnames = [x1]
outcome_type = single_ordinal
stimuli_per_trial = 1

[x1]
par_type = continuous
lower_bound = 0
upper_bound = 1

[OrdinalGPModel]
n_levels = 5
"""

with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
    f.write(config_str)
    config_path = f.name

config = Config()
config.update(config_str=config_str)

print("\n1. Creating ordinal model...")
model = OrdinalGPModel.from_config(config)

print(f"   train_inputs: {model.train_inputs}")
print(f"   train_targets: {model.train_targets}")

# Add some data
X = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

print(f"\n2. Fitting with {len(X)} data points:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   y values: {y}")

try:
    model.fit(X, y)
    print("\n3. After fit:")
    print(f"   train_inputs: {model.train_inputs}")
    if model.train_inputs is not None and len(model.train_inputs) > 0:
        print(f"   train_inputs[0] shape: {model.train_inputs[0].shape}")
    print(f"   train_targets: {model.train_targets}")
    if model.train_targets is not None:
        print(f"   train_targets shape: {model.train_targets.shape}")
        print(f"   train_targets values: {model.train_targets}")

    # Check if data loss occurred
    if model.train_inputs is not None and len(model.train_inputs) > 0:
        n_inputs = len(model.train_inputs[0])
        n_targets = len(model.train_targets) if model.train_targets is not None else 0

        print(f"\n4. Data consistency check:")
        print(f"   Number of input samples: {n_inputs}")
        print(f"   Number of target samples: {n_targets}")

        if n_inputs != n_targets:
            print(f"   ❌ DATA LOSS DETECTED: {n_inputs - n_targets} target values missing!")
        else:
            print(f"   ✓ Data consistent")

except Exception as e:
    print(f"\nError during fit: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
