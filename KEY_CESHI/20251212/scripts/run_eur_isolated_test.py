#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EUR Isolated Test - Standalone EUR acquisition function test

Completely self-contained test with embedded Oracle and minimal dependencies.
Designed to isolate EUR acquisition function behavior without external data files.
"""

import sys
import io
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List

# Fix encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Logging
from loguru import logger
logger.remove()
logger.add(sys.stderr, level='INFO')

import gpytorch
from aepsych.config import Config
from aepsych.server import AEPsychServer
from aepsych.server.message_handlers.handle_setup import configure
from aepsych.server.message_handlers.handle_ask import ask
from aepsych.server.message_handlers.handle_tell import tell

# Import custom components
try:
    from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator
    from extensions.dynamic_eur_acquisition.eur_anova_multi import EURAnovaMultiAcqf
    from extensions.custom_factory import CustomBaseGPResidualFactory
except ImportError as e:
    logger.warning(f"Some components not available: {e}")


class SimpleLinearOracle:
    """Simple built-in linear Oracle for isolated testing.

    No external dependencies, completely self-contained.
    """

    def __init__(self, seed=42, noise_std=0.5):
        np.random.seed(seed)
        self.seed = seed
        self.noise_std = noise_std
        # Simple weights for 6D space
        self.weights = np.array([0.3, 0.2, -0.4, 0.1, 0.25, -0.15], dtype=float)
        self.bias = 0.1
        # Interaction terms
        self.interaction_weights = {(1, 2): 0.15, (2, 4): -0.12}
        logger.info(f"SimpleLinearOracle initialized (seed={seed}, noise_std={noise_std})")

    def query(self, x: np.ndarray) -> float:
        """Query Oracle response.

        Args:
            x: Parameter vector (6,)

        Returns:
            Continuous value response
        """
        # Main effects
        linear = self.bias + np.dot(self.weights, x)

        # Interaction effects
        for (i, j), weight in self.interaction_weights.items():
            linear += weight * x[i] * x[j]

        # Add noise
        return linear + np.random.randn() * self.noise_std

    def get_model_spec(self) -> Dict:
        """Return Oracle model specification."""
        return {
            'model_type': 'linear',
            'seed': self.seed,
            'bias': float(self.bias),
            'noise_std': self.noise_std,
            'weights': self.weights.tolist(),
            'interaction_terms': {f"x{i}*x{j}": w for (i,j),w in self.interaction_weights.items()},
            'source': f'SimpleLinearOracle (seed={self.seed})'
        }


def generate_random_design_space(n_points=50, seed=42):
    """Generate random design space in [0,1]^6."""
    np.random.seed(seed)
    return np.random.uniform(0, 1, size=(n_points, 6))


def create_embedded_config():
    """Create embedded configuration string with minimal settings."""
    return """[common]
parnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
stimuli_per_trial = 1
outcome_types = [continuous]
strategy_names = [init_strat, eur_strat]
lb = [0, 0, 0, 0, 0, 0]
ub = [1, 1, 1, 1, 1, 1]

[init_strat]
min_asks = 3
generator = ManualGenerator
refit_every = 4
model = GPRegressionModel

[ManualGenerator]
stimuli_per_trial = 1
shuffle = false
points = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.2, 0.8, 0.3, 0.7, 0.4, 0.6], [0.8, 0.2, 0.7, 0.3, 0.6, 0.4]]

[eur_strat]
min_asks = 15
max_asks = 30
refit_every = 1
model = GPRegressionModel
generator = CustomPoolBasedGenerator

[GPRegressionModel]
inducing_size = 50
max_fit_time = 5.0
mean_covar_factory = CustomBaseGPResidualFactory

[CustomBaseGPResidualFactory]
mean_type = pure_residual
lengthscale_prior = lognormal
ls_loc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ls_scale = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
fixed_kernel_amplitude = False
outputscale_prior = gamma

[CustomPoolBasedGenerator]
acqf = EURAnovaMultiAcqf
allow_resampling = False
shuffle = True

[EURAnovaMultiAcqf]
variable_types_list = ordinal, ordinal, categorical, categorical, categorical, categorical
enable_main = True
enable_pairwise = True
enable_threeway = False
lambda_2 = None
use_dynamic_lambda = True
lambda_min = 0.40
lambda_max = 0.70
tau1 = 0.55
tau2 = 0.08
gamma = 0.20
use_dynamic_gamma = True
gamma_max = 0.30
gamma_min = 0.05
tau_n_min = 3
tau_n_max = 30
total_budget = 30
use_sps = True
sps_sensitivity = 8.0
sps_ema_alpha = 0.3
tau_safe = 0.5
gamma_penalty_beta = 0.3
coverage_method = min_distance
fusion_method = additive
local_num = 6
local_jitter_frac = 0.1
random_seed = 42
use_hybrid_perturbation = True
exhaustive_level_threshold = 3
exhaustive_use_cyclic_fill = True
debug_components = False
"""


def run_eur_isolated_test(budget=30, seed=42):
    """Run isolated EUR test with embedded Oracle and config.

    Args:
        budget: Total sampling budget
        seed: Random seed

    Returns:
        Results dictionary with sampling history, data, and Oracle spec
    """
    logger.info("=" * 80)
    logger.info("EUR ISOLATED TEST")
    logger.info("=" * 80)
    logger.info(f"Budget: {budget}, Seed: {seed}")

    # Create Oracle
    oracle = SimpleLinearOracle(seed=seed, noise_std=0.5)

    # Generate design space
    logger.info(f"Generating random design space (50 points x 6D)...")
    design_space = generate_random_design_space(50, seed)

    # Create embedded config
    logger.info("Creating embedded configuration...")
    config_str = create_embedded_config()

    # Inject pool points into config
    pool_points_str = str(design_space.tolist())
    config_str = config_str.replace(
        '[CustomPoolBasedGenerator]\nacqf',
        f'[CustomPoolBasedGenerator]\npool_points = {pool_points_str}\nacqf'
    )

    # Create AEPsychServer
    logger.info("Creating AEPsychServer...")
    config = Config(config_str=config_str)
    server = AEPsychServer(config=config)

    # Run sampling loop
    logger.info(f"Starting sampling loop: {budget} trials...")
    sampling_history = []
    results_data = []

    for trial_idx in range(budget):
        # Ask for next configuration
        x_config = ask(server)
        x_array = np.array([x_config[name][0] for name in server.parnames])

        # Query Oracle
        y = oracle.query(x_array)

        # Tell server
        tell(server, outcome=y, config=x_config)

        # Record
        sampling_history.append(x_array.tolist())
        results_data.append({
            'trial': trial_idx,
            'x': x_array.tolist(),
            'y': float(y)
        })

        if (trial_idx + 1) % 10 == 0:
            logger.info(f"  Trial {trial_idx + 1}/{budget} completed")

    logger.info("Sampling completed successfully!")

    return {
        'sampling_history': sampling_history,
        'results_data': results_data,
        'oracle_spec': oracle.get_model_spec(),
        'budget': budget,
        'design_space': design_space.tolist()
    }


def main():
    """Main entry point."""
    try:
        results = run_eur_isolated_test(budget=30, seed=42)
        logger.info("SUCCESS - Isolated EUR test completed")

        # Print summary
        print(f"\n=== ISOLATED EUR TEST SUMMARY ===")
        print(f"Total trials: {len(results['sampling_history'])}")
        print(f"Oracle type: {results['oracle_spec']['source']}")
        print(f"Design space size: 50 x 6D")
        print(f"Status: SUCCESS")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
