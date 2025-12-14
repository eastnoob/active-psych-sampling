#!/usr/bin/env python3
"""
EUR采集器Sobol预热测试
简化版本:10次Sobol + 50次EUR
"""

import sys
import io
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial.distance import cdist

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'temp_aepsych'))
sys.path.insert(0, str(PROJECT_ROOT / 'extensions'))

# Import AEPsych
from aepsych.config import Config
from aepsych.strategy import Strategy

try:
    from extensions.dynamic_eur_acquisition import DynamicEURGenerator
except ImportError:
    print("Warning: DynamicEURGenerator not available")
    DynamicEURGenerator = None


def extract_value(val):
    """Extract scalar value from list or return as-is."""
    if isinstance(val, list):
        return float(val[0])
    return float(val)


def main():
    print("=" * 80)
    print("EUR Sampling with Sobol Warmup Test")
    print("=" * 80)

    # Create result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path(__file__).parent / 'results' / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nResults directory: {result_dir}")

    # Parameters
    sobol_budget = 10
    eur_budget = 50
    total_budget = sobol_budget + eur_budget

    print(f"Sobol warmup: {sobol_budget} samples")
    print(f"EUR sampling: {eur_budget} samples")
    print(f"Total budget: {total_budget}")

    # TODO: Implement Sobol + EUR sampling
    print("\nSampling would run here...")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'sobol_budget': sobol_budget,
        'eur_budget': eur_budget,
        'total_budget': total_budget,
        'result_dir': str(result_dir)
    }

    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\nDone!")
    print(f"Results saved to: {result_dir}")


if __name__ == '__main__':
    main()
