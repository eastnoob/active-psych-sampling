"""SingleRun behavior - Run single acquisition method test."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import numpy as np
import torch
import pandas as pd

# Add project root to path for AEPsych imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from aepsych.config import Config
from aepsych.strategy import Strategy

from core.base_behavior import BaseBehavior
from core.context import Context
from utils.ini_builder import INIBuilder
from utils.data_saver import (
    save_oracle_spec,
    save_sampling_data,
    save_iterations_csv,
    save_enhanced_summary
)


class SingleRun(BaseBehavior):
    """Single acquisition method test behavior.

    Orchestrates:
    1. Build INI config from role
    2. Create AEPsych Strategy
    3. Run warmup (Sobol initialization)
    4. Execute sampling loop with Oracle
    5. Collect results
    """

    def get_name(self) -> str:
        """Return behavior name."""
        return "single_run"

    def get_default_config(self) -> str:
        """Return default TOML configuration."""
        return """
[behavior.single_run]
budget = 50  # 总采样预算 / Total sampling budget
warmup_points = 10  # 初始Sobol点数 / Initial Sobol points
"""

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration."""
        errors = []

        budget = config.get('budget')
        if budget is None or budget <= 0:
            errors.append("budget must be positive integer")

        warmup = config.get('warmup_points', 0)
        if warmup < 0:
            errors.append("warmup_points must be non-negative")

        if warmup >= budget:
            errors.append("warmup_points must be less than budget")

        return (len(errors) == 0, errors)

    def run(self, roles: List, config: Dict[str, Any], context: Context) -> Context:
        """Execute single run experiment.

        Args:
            roles: List of roles (should be single role for SingleRun)
            config: Full configuration dictionary
            context: Shared context

        Returns:
            Updated context with results
        """
        if len(roles) != 1:
            raise ValueError(f"SingleRun expects exactly 1 role, got {len(roles)}")

        role = roles[0]
        logger.info(f"Starting SingleRun with role: {role.get_name()}")

        # Extract behavior config
        behavior_cfg = config.get('behavior', {}).get('single_run', {})
        budget = behavior_cfg.get('budget', 50)
        warmup_points = behavior_cfg.get('warmup_points', 10)

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = context.output_dir / f"{role.get_name()}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        context.output_dir = output_dir
        logger.info(f"Output directory: {output_dir}")

        # Build INI config
        ini_config = self._build_ini_config(role, config, context)
        ini_path = output_dir / "config.ini"
        with open(ini_path, 'w') as f:
            f.write(ini_config)
        context.ini_config_path = ini_path
        logger.info(f"Generated INI config: {ini_path}")

        # Create Strategy
        logger.info("Creating AEPsych Strategy...")
        aepsych_config = Config(config_str=ini_config)
        # Get strategy name from role
        strategy_name = role.get_strategy_name()
        strategy = Strategy.from_config(aepsych_config, strategy_name)
        context.strategy = strategy
        logger.info(f"Strategy '{strategy_name}' created successfully")

        # Run warmup
        if warmup_points > 0:
            logger.info(f"Running warmup: {warmup_points} Sobol points...")
            self._run_warmup(strategy, context, warmup_points)

        # Run main sampling
        main_budget = budget - warmup_points
        logger.info(f"Running main sampling: {main_budget} points...")
        self._run_sampling(strategy, context, main_budget)

        # Save results
        self._save_results(role, context)

        logger.info("SingleRun completed successfully")
        return context

    def _build_ini_config(self, role, config: Dict[str, Any], context: Context) -> str:
        """Build complete INI configuration.

        Args:
            role: Role to use
            config: Configuration dictionary
            context: Shared context

        Returns:
            Complete INI configuration string
        """
        # Get role INI content
        role_config = config.get('role', {}).get(role.get_name(), {})
        behavior_cfg = config.get('behavior', {}).get('single_run', {})
        role_config['budget'] = behavior_cfg.get('budget', 50) - behavior_cfg.get('warmup_points', 10)
        
        # Determine model based on outcome type
        if context.oracle and context.oracle.output_type in ['continuous', 'likert']:
            role_config['model'] = 'GPRegressionModel'
        else:
            role_config['model'] = 'GPClassificationModel'
            
        role_ini = role.generate_ini_section(role_config)

        # Check if INI is complete (has [common] section)
        if '[common]' in role_ini:
            logger.info("Using complete INI configuration from file")
            return role_ini

        # Otherwise, build INI from TOML + role sections
        logger.info("Building INI from TOML parameters + role strategies")
        builder = INIBuilder()

        # Get parameter definitions from config
        param_defs = config.get('parameters', {})
        if param_defs:
            parnames = list(param_defs.keys())
        elif context.design_space is not None:
            parnames = context.design_space.columns.tolist()
        else:
            parnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

        # Build lb/ub based on parameter types
        lb = []
        ub = []
        for pname in parnames:
            pdef = param_defs.get(pname, {})
            par_type = pdef.get('par_type', 'continuous')
            if par_type == 'custom_ordinal_mono':
                lb.append(0)
                ub.append(len(pdef.get('values', [])) - 1)
            elif par_type == 'categorical':
                lb.append(0)
                ub.append(len(pdef.get('choices', [])) - 1)
            else:
                lb.append(0.0)
                ub.append(1.0)

        # Get outcome type from context
        if context.oracle and context.oracle.output_type in ['continuous', 'likert']:
            outcome_type = 'continuous'
        else:
            outcome_type = 'binary'

        # Add common section
        strategy_names = [role.get_strategy_name()]
        builder.add_common_section(
            parnames=parnames,
            outcome_types=[outcome_type],
            strategy_names=strategy_names,
            lb=lb,
            ub=ub
        )

        # Add parameter sections from config
        for pname in parnames:
            pdef = param_defs.get(pname, {})
            par_type = pdef.get('par_type', 'continuous')
            if par_type == 'custom_ordinal_mono':
                builder.add_parameter_section(name=pname, par_type=par_type, values=pdef.get('values', []))
            elif par_type == 'categorical':
                builder.add_parameter_section(name=pname, par_type=par_type, choices=pdef.get('choices', []))
            else:
                builder.add_parameter_section(name=pname, par_type=par_type, lower_bound=0.0, upper_bound=1.0)

        # Add role-specific sections
        builder.add_raw_section(role_ini)

        # Add default model sections if not present
        if 'GPRegressionModel' in role_ini or role_config.get('model') == 'GPRegressionModel':
            builder.add_raw_section("\n[GPRegressionModel]\n")
        if 'GPClassificationModel' in role_ini or role_config.get('model') == 'GPClassificationModel':
            builder.add_raw_section("\n[GPClassificationModel]\n")

        return builder.build()

    def _run_warmup(self, strategy: Strategy, context: Context, n_points: int):
        """Run warmup phase with Sobol sampling.

        Args:
            strategy: AEPsych Strategy
            context: Shared context
            n_points: Number of warmup points
        """
        oracle = context.oracle
        if oracle is None:
            raise ValueError("Oracle not set in context")

        for i in range(n_points):
            # Get next point from strategy
            next_x = strategy.gen()

            # Query oracle
            x_np = next_x.numpy().flatten()
            y = oracle.query(x_np)

            # Tell strategy - pass y as list (AEPsych handles conversion)
            strategy.add_data(next_x, [y])

            if (i + 1) % 10 == 0:
                logger.info(f"  Warmup progress: {i + 1}/{n_points}")

            # Record in context
            context.sampling_history.append({
                'trial': i + 1,
                'phase': 'warmup',
                'x': x_np.tolist(),
                'y': y
            })

        logger.info(f"Warmup completed: {n_points} points")

    def _run_sampling(self, strategy: Strategy, context: Context, n_points: int):
        """Run main sampling phase.

        Args:
            strategy: AEPsych Strategy
            context: Shared context
            n_points: Number of sampling points
        """
        oracle = context.oracle
        if oracle is None:
            raise ValueError("Oracle not set in context")

        warmup_count = len([h for h in context.sampling_history if h['phase'] == 'warmup'])

        # Initialize EUR diagnostics storage
        if not hasattr(context, 'eur_diagnostics'):
            context.eur_diagnostics = {
                'lambda_t': [],
                'gamma_t': [],
                'r_t': []
            }

        for i in range(n_points):
            # Get next point from strategy
            next_x = strategy.gen()

            # Query oracle
            x_np = next_x.numpy().flatten()
            y = oracle.query(x_np)

            # Tell strategy - pass y as list (AEPsych handles conversion)
            strategy.add_data(next_x, [y])

            # Collect EUR diagnostics
            diagnostics = self._get_eur_diagnostics(strategy)
            context.eur_diagnostics['lambda_t'].append(diagnostics.get('lambda_t'))
            context.eur_diagnostics['gamma_t'].append(diagnostics.get('gamma_t'))
            context.eur_diagnostics['r_t'].append(diagnostics.get('r_t'))

            if (i + 1) % 10 == 0:
                logger.info(f"  Sampling progress: {i + 1}/{n_points}")

            # Record in context
            context.sampling_history.append({
                'trial': warmup_count + i,
                'phase': 'main',
                'x': x_np.tolist(),
                'y': y
            })

        logger.info(f"Main sampling completed: {n_points} points")

    def _get_eur_diagnostics(self, strategy: Strategy) -> Dict[str, Any]:
        """Extract EUR diagnostic parameters from strategy.

        Args:
            strategy: AEPsych Strategy

        Returns:
            Dictionary with lambda_t, gamma_t, r_t (or None if not available)
        """
        diagnostics = {
            'lambda_t': None,
            'gamma_t': None,
            'r_t': None
        }

        try:
            # Try to get acquisition function
            if hasattr(strategy, 'generator') and hasattr(strategy.generator, 'acqf'):
                acqf = strategy.generator.acqf

                # Check if acqf has get_diagnostics method
                if hasattr(acqf, 'get_diagnostics'):
                    acqf_diag = acqf.get_diagnostics()
                    diagnostics['lambda_t'] = acqf_diag.get('lambda_t')
                    diagnostics['gamma_t'] = acqf_diag.get('gamma_t')
                    diagnostics['r_t'] = acqf_diag.get('r_t')
                # Otherwise try to access weight_engine directly
                elif hasattr(acqf, 'weight_engine'):
                    try:
                        diagnostics['lambda_t'] = acqf.weight_engine.compute_lambda()
                        diagnostics['gamma_t'] = acqf.weight_engine.compute_gamma()
                        diagnostics['r_t'] = acqf.weight_engine.compute_relative_main_variance()
                    except Exception:
                        pass

        except Exception as e:
            logger.debug(f"Failed to extract EUR diagnostics: {e}")

        return diagnostics

    def _save_results(self, role, context: Context):
        """Save experiment results.

        Args:
            role: Role used
            context: Shared context
        """
        output_dir = context.output_dir

        # Save Oracle spec
        if context.oracle:
            save_oracle_spec(output_dir, context.oracle, source="LinearOracle")

        # Prepare interaction logs
        interaction_logs = []
        for entry in context.sampling_history:
            interaction_logs.append({
                'trial': entry['trial'],
                'x': entry['x'],
                'y': entry['y']
            })

        # Get EUR diagnostics if available
        eur_diagnostics = getattr(context, 'eur_diagnostics', None)

        # Save sampling data
        sampling_history = [entry['x'] for entry in context.sampling_history]
        save_sampling_data(output_dir, sampling_history, interaction_logs, eur_diagnostics)

        # Save iterations CSV
        warmup_count = len([h for h in context.sampling_history if h['phase'] == 'warmup'])
        save_iterations_csv(output_dir, interaction_logs, eur_diagnostics, warmup_count)

        # Save enhanced summary
        oracle_spec = context.oracle.get_model_spec() if context.oracle else {}
        config_info = {
            'config_name': 'keyceshi_eur_complete.ini'
        }
        save_enhanced_summary(
            output_dir,
            config_info,
            oracle_spec,
            interaction_logs,
            eur_diagnostics,
            warmup_count,
            seed=oracle_spec.get('seed')
        )

        # Also save simple CSV for backward compatibility
        history_df = pd.DataFrame(context.sampling_history)
        history_path = output_dir / "sampling_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Saved sampling history CSV: {history_path}")

        # Store in context
        context.add_result(role.get_name(), {
            'history': context.sampling_history,
            'eur_diagnostics': eur_diagnostics
        })
