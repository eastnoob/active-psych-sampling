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

# Prefer warmup's subject simulator if available
try:
    from subject_simulator_v2 import LinearSubject
except Exception as e:
    logger.warning(f"subject_simulator_v2 not available: {e}")
    LinearSubject = None

# Default oracle configuration (edit here or use CLI args in `main`) 🔧
DEFAULT_ORACLE_CONFIG = {
    'output_type': 'likert',          # 'likert' or 'continuous'
    'likert_levels': 5,               # 1..N Likert levels
    'likert_mode': 'tanh',            # 'tanh' or 'sigmoid'
    'likert_sensitivity': 2.0,        # sensitivity for mapping
    'noise_std': 0.5,                 # trial noise
    'bias': 0.1,                      # intercept for linear model
    'weights': [0.3, 0.2, -0.4, 0.1, 0.25, -0.15],
    'interaction_weights': {(1, 2): 0.15, (2, 4): -0.12},
}


class SimpleLinearOracle:
    """Simple built-in linear Oracle for isolated testing.

    No external dependencies, completely self-contained.
    """

    def __init__(self, seed=42, noise_std=0.5,
                 output_type='likert', likert_levels=5, likert_mode='tanh', likert_sensitivity=2.0):
        np.random.seed(seed)
        self.seed = seed
        self.noise_std = noise_std
        # Output settings
        self.output_type = output_type  # 'continuous' or 'likert'
        self.likert_levels = likert_levels
        self.likert_mode = likert_mode
        self.likert_sensitivity = likert_sensitivity
        # Simple weights for 6D space
        self.weights = np.array([0.3, 0.2, -0.4, 0.1, 0.25, -0.15], dtype=float)
        self.bias = 0.1
        # Interaction terms
        self.interaction_weights = {(1, 2): 0.15, (2, 4): -0.12}
        logger.info(f"SimpleLinearOracle initialized (seed={seed}, noise_std={noise_std}, output_type={output_type})")

    def query(self, x: np.ndarray):
        """Query Oracle response.

        Args:
            x: Parameter vector (6,)

        Returns:
            Continuous value (float) or Likert integer (1..likert_levels) depending on `output_type`
        """
        # Main effects
        linear = self.bias + np.dot(self.weights, x)

        # Interaction effects
        for (i, j), weight in self.interaction_weights.items():
            linear += weight * x[i] * x[j]

        # Add noise
        raw = linear + np.random.randn() * self.noise_std

        if self.output_type == 'likert':
            L = self.likert_levels
            sens = self.likert_sensitivity
            if self.likert_mode == 'tanh':
                v = np.tanh(raw * sens)
                likert_float = v * (L - 1) / 2 + (L + 1) / 2
            else:  # 'sigmoid' or other
                s = 1.0 / (1.0 + np.exp(-raw * sens))
                likert_float = s * (L - 1) + 1
            likert_int = int(np.round(likert_float))
            likert_int = int(np.clip(likert_int, 1, L))
            return likert_int
        else:
            return float(raw) 

    def get_model_spec(self) -> Dict:
        """Return Oracle model specification."""
        return {
            'model_type': 'linear',
            'seed': self.seed,
            'bias': float(self.bias),
            'noise_std': self.noise_std,
            'weights': self.weights.tolist(),
            'interaction_terms': {f"x{i}*x{j}": w for (i,j),w in self.interaction_weights.items()},
            'output_type': self.output_type,
            'likert_levels': getattr(self, 'likert_levels', None),
            'likert_mode': getattr(self, 'likert_mode', None),
            'likert_sensitivity': getattr(self, 'likert_sensitivity', None),
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
# Using continuous outcome here for compatibility with GPRegressionModel.
# If you want true ordinal modeling, change to [ordinal] and use an ordinal-aware model.
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


def run_eur_isolated_test(budget=30, seed=42, oracle_config: dict = None, subject_override=None):
    """Run isolated EUR test with embedded Oracle and config.

    Args:
        budget: Total sampling budget
        seed: Random seed
        oracle_config: dict with keys to control oracle behavior (see DEFAULT_ORACLE_CONFIG)

    Returns:
        Results dictionary with sampling history, data, and Oracle spec
    """
    logger.info("=" * 80)
    logger.info("EUR ISOLATED TEST")
    logger.info("=" * 80)
    logger.info(f"Budget: {budget}, Seed: {seed}")

    # Merge config
    cfg = dict(DEFAULT_ORACLE_CONFIG)
    if oracle_config:
        cfg.update(oracle_config)

    logger.info(f"Oracle configuration: {cfg}")

    # Use subject_override if provided (loaded from spec)
    if subject_override is not None:
        oracle = subject_override
        logger.info("Using provided subject_override as Oracle")

    else:
        # Create Oracle (prefer warmup's LinearSubject; fallback to SimpleLinearOracle)
        if LinearSubject is not None and cfg is not None:
            oracle = LinearSubject(
                weights=np.array(cfg['weights']),
                interaction_weights=cfg.get('interaction_weights', {}),
                bias=cfg.get('bias', 0.0),
                noise_std=cfg.get('noise_std', 0.0),
                likert_levels=cfg['likert_levels'] if cfg['output_type'] == 'likert' else None,
                likert_sensitivity=cfg.get('likert_sensitivity', 1.0),
                seed=seed
            )
            logger.info("Using subject_simulator_v2.LinearSubject as Oracle")
        else:
            oracle = SimpleLinearOracle(
                seed=seed,
                noise_std=cfg.get('noise_std', 0.5),
                output_type=cfg.get('output_type', 'likert'),
                likert_levels=cfg.get('likert_levels', 5),
                likert_mode=cfg.get('likert_mode', 'tanh'),
                likert_sensitivity=cfg.get('likert_sensitivity', 2.0)
            )
            # override weights/bias/interaction for fallback
            oracle.weights = np.array(cfg.get('weights', oracle.weights.tolist()))
            oracle.bias = cfg.get('bias', oracle.bias)
            oracle.interaction_weights = cfg.get('interaction_weights', oracle.interaction_weights)
            logger.info("Using fallback SimpleLinearOracle as Oracle")

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
    try:
        server = AEPsychServer(config=config)
    except TypeError:
        logger.info("AEPsychServer.__init__ doesn't accept 'config' keyword, using fallback initialization")
        server = AEPsychServer()
        try:
            server.configure(config)
        except Exception:
            # fallback to message handler configure
            configure(server, config)

    # Run sampling loop
    logger.info(f"Starting sampling loop: {budget} trials...")
    sampling_history = []
    results_data = []

    for trial_idx in range(budget):
        # Ask for next configuration
        x_config = ask(server)
        x_array = np.array([x_config[name][0] for name in server.parnames])

        # Query Oracle
        if callable(oracle):
            y = oracle(x_array)
        elif hasattr(oracle, 'query'):
            y = oracle.query(x_array)
        else:
            raise RuntimeError("Oracle object has no callable interface")

        # Tell server
        tell(server, outcome=y, config=x_config)

        # Record
        sampling_history.append(x_array.tolist())
        results_data.append({
            'trial': trial_idx,
            'x': x_array.tolist(),
            'y': int(y) if isinstance(y, (np.integer, int)) else float(y)
        })

        if (trial_idx + 1) % 10 == 0:
            logger.info(f"  Trial {trial_idx + 1}/{budget} completed")

    logger.info("Sampling completed successfully!")

    # Prepare oracle_spec (support LinearSubject.to_dict or fallback.get_model_spec)
    if hasattr(oracle, 'to_dict'):
        oracle_spec = oracle.to_dict()
    elif hasattr(oracle, 'get_model_spec'):
        oracle_spec = oracle.get_model_spec()
    else:
        oracle_spec = {'type': type(oracle).__name__}

    return {
        'sampling_history': sampling_history,
        'results_data': results_data,
        'oracle_spec': oracle_spec,
        'budget': budget,
        'design_space': design_space.tolist()
    }


def main():
    """Main entry point with optional CLI overrides for oracle behavior."""
    import argparse

    parser = argparse.ArgumentParser(description="Run isolated EUR test with configurable oracle")
    parser.add_argument("--budget", type=int, default=30, help="Total sampling budget")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-type", choices=["likert", "continuous"], default=DEFAULT_ORACLE_CONFIG['output_type'])
    parser.add_argument("--likert-levels", type=int, default=DEFAULT_ORACLE_CONFIG['likert_levels'])
    parser.add_argument("--likert-mode", choices=["tanh", "sigmoid"], default=DEFAULT_ORACLE_CONFIG['likert_mode'])
    parser.add_argument("--likert-sensitivity", type=float, default=DEFAULT_ORACLE_CONFIG['likert_sensitivity'])
    parser.add_argument("--noise-std", type=float, default=DEFAULT_ORACLE_CONFIG['noise_std'])
    parser.add_argument("--bias", type=float, default=DEFAULT_ORACLE_CONFIG['bias'])
    parser.add_argument("--weights", type=str, default=','.join(map(str, DEFAULT_ORACLE_CONFIG['weights'])), help="Comma-separated weights list")
    parser.add_argument("--cluster-dir", type=str, default=None, help="Path to generated cluster directory (loads subject_{id}_spec.json)")
    parser.add_argument("--subject-id", type=int, default=1, help="Subject id to load from cluster-dir")
    parser.add_argument("--subject-spec", type=str, default=None, help="Path to single subject spec JSON to load")
    parser.add_argument("--fixed-weights", type=str, default=None, help="Path to fixed_weights.json to override weights/bias/interactions")

    args = parser.parse_args()

    # Build oracle_config
    oracle_config = {
        'output_type': args.output_type,
        'likert_levels': args.likert_levels,
        'likert_mode': args.likert_mode,
        'likert_sensitivity': args.likert_sensitivity,
        'noise_std': args.noise_std,
        'bias': args.bias,
        'weights': [float(x) for x in args.weights.split(',')],
        'interaction_weights': DEFAULT_ORACLE_CONFIG['interaction_weights'],
    }

    # Loading priority: subject_spec > cluster_dir+subject_id > fixed_weights > CLI weights
    subject_spec_path = None
    loaded_subject = None

    if args.subject_spec:
        subject_spec_path = args.subject_spec
    elif args.cluster_dir:
        from pathlib import Path
        p = Path(args.cluster_dir) / f"subject_{args.subject_id}_spec.json"
        if p.exists():
            subject_spec_path = str(p)
        else:
            logger.warning(f"Subject spec not found at {p}; falling back to CLI/default weights")

    if subject_spec_path:
        try:
            import json
            from pathlib import Path
            sp = Path(subject_spec_path)
            spec = json.loads(sp.read_text(encoding='utf-8'))
            # if LinearSubject available, use it; else override oracle_config
            if LinearSubject is not None:
                loaded_subject = LinearSubject.from_dict(spec)
                logger.info(f"Loaded subject from spec: {subject_spec_path}")
            else:
                # extract weights/bias/interactions
                oracle_config['weights'] = spec.get('weights', oracle_config['weights'])
                oracle_config['bias'] = spec.get('bias', oracle_config['bias'])
                # parse interactions
                interactions = spec.get('interaction_weights') or spec.get('interaction_weights', {})
                # ensure proper format
                if isinstance(interactions, dict):
                    parsed = {}
                    for k, v in interactions.items():
                        if isinstance(k, str) and ',' in k:
                            i, j = k.split(',')
                            parsed[(int(i), int(j))] = float(v)
                        else:
                            parsed[k] = float(v)
                    oracle_config['interaction_weights'] = parsed
                logger.info(f"Loaded subject spec into oracle_config from: {subject_spec_path}")
        except Exception as e:
            logger.error(f"Failed to load subject spec: {e}")

    elif args.fixed_weights:
        try:
            import json
            from pathlib import Path
            fw = Path(args.fixed_weights)
            data = json.loads(fw.read_text(encoding='utf-8'))
            if 'global' in data:
                oracle_config['weights'] = data['global'][0]
            if 'interactions' in data:
                parsed = {}
                for k, v in data['interactions'].items():
                    i, j = map(int, k.split(','))
                    parsed[(i, j)] = float(v)
                oracle_config['interaction_weights'] = parsed
            if 'bias' in data:
                oracle_config['bias'] = data['bias']
            logger.info(f"Loaded fixed weights from {fw}")
        except Exception as e:
            logger.error(f"Failed to load fixed_weights: {e}")


    try:
        results = run_eur_isolated_test(budget=args.budget, seed=args.seed, oracle_config=oracle_config, subject_override=loaded_subject)
        logger.info("SUCCESS - Isolated EUR test completed")

        # Print summary
        print(f"\n=== ISOLATED EUR TEST SUMMARY ===")
        print(f"Total trials: {len(results['sampling_history'])}")
        # oracle_spec may be dict from LinearSubject/to_dict or fallback; print summary smartly
        if isinstance(results['oracle_spec'], dict):
            name = results['oracle_spec'].get('model_type') or results['oracle_spec'].get('type') or 'oracle'
            print(f"Oracle type: {name}")
            if 'likert_levels' in results['oracle_spec']:
                print(f"Likert levels: {results['oracle_spec'].get('likert_levels')}")
        print(f"Design space size: 50 x 6D")
        print(f"Status: SUCCESS")

    except Exception as e:
        logger.error(f"FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
