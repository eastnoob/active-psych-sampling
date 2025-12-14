#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EUR Comparison Test - Side-by-side EUR vs Random strategy comparison

Compares EUR acquisition function against random Sobol baseline using the same
Oracle, design space, and budget. Generates detailed comparison metrics and report.
"""

import sys
import io
from pathlib import Path
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Tuple

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


class PopulationConsistentOracle:
    """Oracle using fixed weights for fair EUR vs Random comparison.

    Ensures both strategies evaluate against the same ground truth.
    """

    def __init__(self, seed=42, noise_std=0.5):
        np.random.seed(seed)
        self.seed = seed
        self.noise_std = noise_std
        # Fixed weights for consistent comparison
        self.weights = np.array([0.3, 0.2, -0.4, 0.1, 0.25, -0.15], dtype=float)
        self.bias = 0.1
        self.interaction_weights = {(1, 2): 0.15, (2, 4): -0.12}
        logger.info(f"PopulationConsistentOracle initialized (seed={seed}, noise_std={noise_std})")

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
            'source': f'PopulationConsistentOracle (seed={self.seed})'
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


def run_sampling_strategy(
    strategy: str,
    oracle: PopulationConsistentOracle,
    design_space: np.ndarray,
    budget: int,
    seed: int = 42
) -> Dict:
    """Run a single sampling strategy (EUR or Random).

    Args:
        strategy: 'eur' or 'random'
        oracle: Oracle instance (shared between strategies)
        design_space: Design space array (N, 6)
        budget: Sampling budget
        seed: Random seed

    Returns:
        Results dictionary with sampling history, data, and metrics
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Running {strategy.upper()} strategy (budget={budget})")
    logger.info(f"{'='*60}")

    # Create embedded config
    config_str = create_embedded_config()

    # Inject pool points
    pool_points_str = str(design_space.tolist())
    config_str = config_str.replace(
        '[CustomPoolBasedGenerator]\nacqf',
        f'[CustomPoolBasedGenerator]\npool_points = {pool_points_str}\nacqf'
    )

    # Create server
    config = Config(config_str=config_str)
    server = AEPsychServer(config=config)

    # Initialize Sobol engine for Random strategy
    sobol_engine = None
    if strategy == 'random':
        from torch.quasirandom import SobolEngine
        sobol_engine = SobolEngine(dimension=6, scramble=True, seed=seed)
        logger.info(f"Initialized SobolEngine (seed={seed})")

    # Sampling loop
    sampling_history = []
    results_data = []
    y_values = []

    for trial_idx in range(budget):
        if strategy == 'random':
            # Random Sobol baseline
            next_x = sobol_engine.draw(1).numpy()[0]
            x_config = {name: [float(next_x[i])] for i, name in enumerate(server.parnames)}
            x_array = next_x
        else:
            # EUR strategy
            x_config = ask(server)
            x_array = np.array([x_config[name][0] for name in server.parnames])

        # Query Oracle
        y = oracle.query(x_array)
        y_values.append(y)

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

    logger.info(f"Strategy {strategy.upper()} completed!")

    # Compute metrics
    y_values = np.array(y_values)
    metrics = {
        'mean_response': float(np.mean(y_values)),
        'std_response': float(np.std(y_values)),
        'min_response': float(np.min(y_values)),
        'max_response': float(np.max(y_values)),
        'final_y': float(y_values[-1]),
        'best_y': float(np.max(y_values))  # Assumes maximization
    }

    return {
        'strategy': strategy,
        'sampling_history': sampling_history,
        'results_data': results_data,
        'y_values': y_values.tolist(),
        'metrics': metrics,
        'budget': budget
    }


def compare_strategies(
    eur_results: Dict,
    random_results: Dict
) -> Dict:
    """Compare EUR vs Random strategies.

    Args:
        eur_results: Results from EUR strategy
        random_results: Results from Random strategy

    Returns:
        Comparison metrics and analysis
    """
    comparison = {
        'experiment': 'EUR vs Random Comparison',
        'timestamp': datetime.now().isoformat(),
    }

    # Response quality comparison
    eur_y = np.array(eur_results['y_values'])
    random_y = np.array(random_results['y_values'])

    comparison['response_quality'] = {
        'EUR': eur_results['metrics'],
        'Random': random_results['metrics'],
        'difference': {
            'mean_response': float(eur_results['metrics']['mean_response'] - random_results['metrics']['mean_response']),
            'best_y': float(eur_results['metrics']['best_y'] - random_results['metrics']['best_y']),
            'final_y': float(eur_results['metrics']['final_y'] - random_results['metrics']['final_y'])
        }
    }

    # Learning curves
    eur_cummax = np.maximum.accumulate(eur_y)
    random_cummax = np.maximum.accumulate(random_y)

    comparison['learning_curves'] = {
        'EUR_best': eur_cummax.tolist(),
        'Random_best': random_cummax.tolist(),
        'EUR_final_best': float(eur_cummax[-1]),
        'Random_final_best': float(random_cummax[-1])
    }

    # Response distribution
    comparison['distribution'] = {
        'EUR': {
            'mean': float(np.mean(eur_y)),
            'std': float(np.std(eur_y)),
            'quartiles': [float(np.percentile(eur_y, q)) for q in [25, 50, 75]]
        },
        'Random': {
            'mean': float(np.mean(random_y)),
            'std': float(np.std(random_y)),
            'quartiles': [float(np.percentile(random_y, q)) for q in [25, 50, 75]]
        }
    }

    # Spatial coverage (simple: distance-based diversity)
    def compute_coverage(samples):
        """Compute average pairwise distance (diversity metric)."""
        if len(samples) < 2:
            return 0.0
        samples = np.array(samples)
        distances = []
        for i in range(len(samples)):
            for j in range(i+1, min(i+5, len(samples))):
                dist = np.linalg.norm(samples[i] - samples[j])
                distances.append(dist)
        return float(np.mean(distances)) if distances else 0.0

    eur_coverage = compute_coverage(eur_results['sampling_history'])
    random_coverage = compute_coverage(random_results['sampling_history'])

    comparison['spatial_coverage'] = {
        'EUR_diversity': eur_coverage,
        'Random_diversity': random_coverage,
        'difference': eur_coverage - random_coverage
    }

    return comparison


def main():
    """Main entry point - run both strategies and compare."""
    try:
        logger.info("=" * 80)
        logger.info("EUR COMPARISON TEST - EUR vs Random")
        logger.info("=" * 80)

        # Shared Oracle and design space
        oracle = PopulationConsistentOracle(seed=42, noise_std=0.5)
        design_space = generate_random_design_space(50, seed=42)
        budget = 30

        # Run both strategies
        eur_results = run_sampling_strategy('eur', oracle, design_space, budget, seed=42)
        random_results = run_sampling_strategy('random', oracle, design_space, budget, seed=42)

        # Compare
        comparison = compare_strategies(eur_results, random_results)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 80)
        print(f"\n=== EUR vs RANDOM COMPARISON REPORT ===")
        print(f"\nBudget: {budget}")
        print(f"Oracle: {oracle.get_model_spec()['source']}")

        print(f"\nRESPONSE QUALITY:")
        print(f"  EUR mean:    {comparison['response_quality']['EUR']['mean_response']:.4f}")
        print(f"  Random mean: {comparison['response_quality']['Random']['mean_response']:.4f}")
        print(f"  Difference:  {comparison['response_quality']['difference']['mean_response']:+.4f}")

        print(f"\nBEST RESPONSE:")
        print(f"  EUR best:    {comparison['response_quality']['EUR']['best_y']:.4f}")
        print(f"  Random best: {comparison['response_quality']['Random']['best_y']:.4f}")
        print(f"  Difference:  {comparison['response_quality']['difference']['best_y']:+.4f}")

        print(f"\nSPATIAL COVERAGE (Diversity):")
        print(f"  EUR diversity:    {comparison['spatial_coverage']['EUR_diversity']:.4f}")
        print(f"  Random diversity: {comparison['spatial_coverage']['Random_diversity']:.4f}")
        print(f"  Difference:       {comparison['spatial_coverage']['difference']:+.4f}")

        logger.info("SUCCESS - Comparison test completed")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
