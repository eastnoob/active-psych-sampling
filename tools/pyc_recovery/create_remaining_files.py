#!/usr/bin/env python3
"""Create the remaining three missing run scripts."""

from pathlib import Path

# run_eur_residual.py
run_eur_residual_content = r'''#!/usr/bin/env python3
"""
EUR采集器验证实验 - 简化版本
使用模型预测,避免真实交互,测试采集流程
调用modules目录中的专用模块函数
"""

import sys
import io
import argparse
from pathlib import Path
import json
from datetime import datetime

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
local_modules_path = Path(__file__).parent / 'modules'
sys.path.insert(0, str(local_modules_path))
sys.path.insert(0, str(PROJECT_ROOT))

# Imports
from loguru import logger
logger.remove()
logger.add(sys.stderr, level='INFO')

import gpytorch
import torch
import numpy as np
from aepsych.config import Config
from aepsych.strategy import Strategy

# Try to import dynamic_eur_acquisition
try:
    from extensions.dynamic_eur_acquisition import DynamicEURGenerator
except ImportError:
    logger.warning("DynamicEURGenerator not found, using placeholder")
    DynamicEURGenerator = None


class ConfigurableGaussianLikelihood(gpytorch.likelihoods.GaussianLikelihood):
    """Custom Gaussian likelihood with configurable noise priors."""

    def __init__(self, noise_prior_concentration=2.0, noise_prior_rate=1.228, noise_init=0.814, **kwargs):
        # Create noise prior
        c_val = float(noise_prior_concentration)
        r_val = float(noise_prior_rate)
        n_val = float(noise_init)

        noise_prior = gpytorch.priors.GammaPrior(
            concentration=torch.tensor(c_val, dtype=torch.float),
            rate=torch.tensor(r_val, dtype=torch.float)
        )
        noise_constraint = gpytorch.constraints.GreaterThan(0.0001)

        super().__init__(
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            **kwargs
        )
        self.noise = n_val


def main():
    parser = argparse.ArgumentParser(description='EUR Residual Sampling Test')
    parser.add_argument('--budget', type=int, default=60, help='Total sampling budget')
    parser.add_argument('--sobol_init', type=int, default=10, help='Initial Sobol samples')
    args = parser.parse_args()

    logger.info("EUR Residual Sampling Test")
    logger.info(f"Budget: {args.budget}, Sobol init: {args.sobol_init}")

    # Create result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path(__file__).parent / 'results' / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results directory: {result_dir}")

    # Load config
    config_path = PROJECT_ROOT / 'tests' / 'is_EUR_work' / 'configs' / 'eur_config.ini'
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using default")
        # Create minimal config
        config_path = result_dir / 'default_config.ini'
        with open(config_path, 'w') as f:
            f.write("[common]\n")
            f.write("parnames = [x1, x2, x3]\n")
            f.write("outcome_type = single_probit\n")

    logger.info(f"Using config: {config_path}")

    # TODO: Implement EUR sampling loop
    logger.info("EUR sampling would run here")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'budget': args.budget,
        'sobol_init': args.sobol_init,
        'result_dir': str(result_dir)
    }
    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("Done!")


if __name__ == '__main__':
    main()
'''

# run_sobol_test.py
run_sobol_test_content = r'''#!/usr/bin/env python3
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
'''

def main():
    files_to_create = [
        ('tests/is_EUR_work/00_plans/251206/scripts/run_eur_residual.py', run_eur_residual_content),
        ('tests/is_EUR_work/00_plans/251212/run_eur_residual.py', run_eur_residual_content),
        ('tests/is_EUR_work/archive/scripts/run_sobol_test.py', run_sobol_test_content),
    ]

    print("Creating files...\n")
    for filepath, content in files_to_create:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[OK] Created: {filepath} ({len(content)} bytes)")

    print("\nAll files created successfully!")

if __name__ == '__main__':
    main()
