#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Custom Generators for AEPsych Extensions

This module contains custom generator implementations that extend AEPsych functionality.
"""

from .custom_pool_based_generator import CustomPoolBasedGenerator

# Conditionally import optional components (archived generators)
try:
    from .archive.warmup_minimal import WarmupMinimalGenerator
except ImportError:
    WarmupMinimalGenerator = None

try:
    from .archive.warmup_mixed_pool import WarmupMixedPoolGenerator
except ImportError:
    WarmupMixedPoolGenerator = None

__all__ = [
    "CustomPoolBasedGenerator",
]

if WarmupMinimalGenerator is not None:
    __all__.append("WarmupMinimalGenerator")

if WarmupMixedPoolGenerator is not None:
    __all__.append("WarmupMixedPoolGenerator")
