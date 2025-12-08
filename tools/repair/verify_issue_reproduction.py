#!/usr/bin/env python3
"""
Verify if the reproduction script in the Issue document can reproduce the problem.
This script behaves differently before and after change.

Usage:
    cd d:\ENVS\active-psych-sampling
    pixi run python tools/repair/verify_issue_reproduction.py
"""
import torch
import sys

# No need to modify sys.path - aepsych is installed via pixi
from aepsych.models import GPRegressionModel
from aepsych.transforms.parameters import ParameterTransformedModel
from botorch.models.transforms.input import ChainedInputTransform

print("=" * 70)
print("ParameterTransformedModel.train_inputs Issue reproduction and verification")
print("=" * 70)

# Setup
print("\n[Setup] Creating GPRegressionModel and its wrapper...")
base_model = GPRegressionModel(dim=2)
wrapped = ParameterTransformedModel(
    model=base_model, transforms=ChainedInputTransform(**{})
)

# Initial fit
print("\n[Fit #1] Fitting with 3 data points...")
x1 = torch.rand(3, 2)
y1 = torch.rand(3)
wrapped.fit(x1, y1)

result1_wrapped = wrapped.train_inputs[0].shape
result1_base = wrapped._base_obj.train_inputs[0].shape
print(f"  wrapped.train_inputs[0].shape     = {result1_wrapped}")
print(f"  wrapped._base_obj.train_inputs[0].shape = {result1_base}")
print(f"  [ :) ] Fit #1 passed (both are torch.Size([3, 2]))")

# Second fit with more data
print("\n[Fit #2] Fitting with 5 data points...")
x2 = torch.rand(5, 2)
y2 = torch.rand(5)
wrapped.fit(x2, y2)

result2_wrapped = wrapped.train_inputs[0].shape
result2_base = wrapped._base_obj.train_inputs[0].shape
print(f"  wrapped.train_inputs[0].shape     = {result2_wrapped}")
print(f"  wrapped._base_obj.train_inputs[0].shape = {result2_base}")

# Check for issues
print("\n[Verification Results]")
if result2_wrapped == torch.Size([5, 2]) and result2_base == torch.Size([5, 2]):
    print(
        "  [ :) ] Fix effective: wrapped.train_inputs correctly returns latest data (torch.Size([5, 2]))"
    )
    sys.exit(0)
elif result2_wrapped == torch.Size([3, 2]) and result2_base == torch.Size([5, 2]):
    print(
        "  [ :( ] BUG present: wrapped.train_inputs returns stale data (torch.Size([3, 2]))"
    )
    print("     while _base_obj.train_inputs is correct (torch.Size([5, 2]))")
    sys.exit(1)
else:
    print(
        f"  [ !! ]  Unexpected result: wrapped={result2_wrapped}, base={result2_base}"
    )
    sys.exit(2)
