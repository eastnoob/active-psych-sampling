"""MultiSubjectRun behavior - Run experiments with multiple subjects."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import numpy as np
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
from utils.oracle import LinearOracle


class MultiSubjectRun(BaseBehavior):
    """Multi-subject experiment behavior.

    Orchestrates:
    1. Create multiple subjects with different oracles
    2. Run experiment for each subject
    3. Collect and aggregate results
    """

    def get_name(self) -> str:
        """Return behavior name."""
        return "multi_subject_run"

    def get_default_config(self) -> str:
        """Return default TOML configuration."""
        return """
[behavior.multi_subject_run]
budget = 50  # 每个被试的总采样预算 / Total sampling budget per subject
warmup_points = 10  # 每个被试的初始Sobol点数 / Initial Sobol points per subject
n_subjects = 5  # 被试数量 / Number of subjects
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

        n_subjects = config.get('n_subjects', 1)
        if n_subjects <= 0:
            errors.append("n_subjects must be positive integer")

        return (len(errors) == 0, errors)

    def run(self, roles: List, config: Dict[str, Any], context: Context) -> Context:
        """Execute multi-subject experiment.

        Args:
            roles: List of roles (should be single role)
            config: Configuration dictionary
            context: Shared context

        Returns:
            Updated context with results
        """
        if len(roles) != 1:
            raise ValueError(f"MultiSubjectRun expects exactly 1 role, got {len(roles)}")

        role = roles[0]
        logger.info(f"Starting MultiSubjectRun with role: {role.get_name()}")

        # Extract config
        budget = config.get('budget', 50)
        warmup_points = config.get('warmup_points', 10)
        n_subjects = config.get('n_subjects', 5)

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = context.output_dir / f"{role.get_name()}_multi_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        context.output_dir = output_dir
        logger.info(f"Output directory: {output_dir}")

        # Store original oracle config
        original_oracle = context.oracle
        base_seed = original_oracle.seed if hasattr(original_oracle, 'seed') else 42

        # Run experiment for each subject
        all_subjects_results = []
        for subject_id in range(1, n_subjects + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Subject {subject_id}/{n_subjects}")
            logger.info(f"{'='*60}")

            # Create subject-specific oracle with different seed
            subject_seed = base_seed + subject_id
            subject_oracle = LinearOracle(
                seed=subject_seed,
                noise_std=original_oracle.noise_std,
                output_type=original_oracle.output_type,
                likert_levels=original_oracle.likert_levels,
                weights=original_oracle.weights,
                bias=original_oracle.bias,
                interactions=original_oracle.interactions
            )
            context.oracle = subject_oracle

            # Create subject output directory
            subject_dir = output_dir / f"subject_{subject_id}"
            subject_dir.mkdir(parents=True, exist_ok=True)

            # Build INI config
            ini_config = self._build_ini_config(role, config, context)
            ini_path = subject_dir / "config.ini"
            with open(ini_path, 'w') as f:
                f.write(ini_config)
            logger.info(f"Generated INI config: {ini_path}")

            # Create Strategy
            logger.info("Creating AEPsych Strategy...")
            aepsych_config = Config(config_str=ini_config)
            strategy_name = role.get_strategy_name()
            strategy = Strategy.from_config(aepsych_config, strategy_name)
            logger.info(f"Strategy '{strategy_name}' created successfully")

            # Reset sampling history for this subject
            subject_history = []

            # Run warmup
            if warmup_points > 0:
                logger.info(f"Running warmup: {warmup_points} Sobol points...")
                self._run_warmup(strategy, subject_oracle, subject_history, warmup_points)

            # Run main sampling
            main_budget = budget - warmup_points
            logger.info(f"Running main sampling: {main_budget} points...")
            self._run_sampling(strategy, subject_oracle, subject_history, main_budget, warmup_points)

            # Save subject results
            self._save_subject_results(subject_id, subject_dir, subject_history, subject_oracle)

            # Store in aggregate results
            all_subjects_results.append({
                'subject_id': subject_id,
                'seed': subject_seed,
                'history': subject_history,
                'output_dir': str(subject_dir)
            })

            logger.info(f"Subject {subject_id} completed: {len(subject_history)} trials")

        # Restore original oracle
        context.oracle = original_oracle

        # Save aggregate results
        self._save_aggregate_results(role, output_dir, all_subjects_results, config)

        logger.info(f"\n{'='*60}")
        logger.info(f"MultiSubjectRun completed: {n_subjects} subjects")
        logger.info(f"{'='*60}")

        return context

    def _build_ini_config(self, role, config: Dict[str, Any], context: Context) -> str:
        """Build complete INI configuration."""
        builder = INIBuilder()

        # Get parameter names from design space
        if context.design_space is not None:
            parnames = context.design_space.columns.tolist()
        else:
            parnames = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']

        # Add common section
        strategy_names = [role.get_strategy_name()]
        builder.add_common_section(
            parnames=parnames,
            outcome_types=['binary'],
            strategy_names=strategy_names
        )

        # Add parameter sections
        for pname in parnames:
            builder.add_parameter_section(
                name=pname,
                par_type='continuous',
                lower_bound=0.0,
                upper_bound=1.0
            )

        # Add role-specific sections
        role_config = config.get('role', {}).get(role.get_name(), {})
        role_config['budget'] = config.get('budget', 50) - config.get('warmup_points', 10)
        role_ini = role.generate_ini_section(role_config)
        builder.add_raw_section(role_ini)

        return builder.build()

    def _run_warmup(self, strategy: Strategy, oracle, history: list, n_points: int):
        """Run warmup phase with Sobol sampling."""
        for i in range(n_points):
            next_x = strategy.gen()
            x_np = next_x.numpy().flatten()
            y = oracle.query(x_np)
            strategy.add_data(next_x, [y])

            if (i + 1) % 10 == 0:
                logger.info(f"  Warmup progress: {i + 1}/{n_points}")

            history.append({
                'trial': i + 1,
                'phase': 'warmup',
                'x': x_np.tolist(),
                'y': y
            })

        logger.info(f"Warmup completed: {n_points} points")

    def _run_sampling(self, strategy: Strategy, oracle, history: list, n_points: int, warmup_count: int):
        """Run main sampling phase."""
        for i in range(n_points):
            next_x = strategy.gen()
            x_np = next_x.numpy().flatten()
            y = oracle.query(x_np)
            strategy.add_data(next_x, [y])

            if (i + 1) % 10 == 0:
                logger.info(f"  Sampling progress: {i + 1}/{n_points}")

            history.append({
                'trial': warmup_count + i + 1,
                'phase': 'main',
                'x': x_np.tolist(),
                'y': y
            })

        logger.info(f"Main sampling completed: {n_points} points")

    def _save_subject_results(self, subject_id: int, output_dir: Path, history: list, oracle):
        """Save individual subject results."""
        # Save sampling history as CSV
        history_df = pd.DataFrame(history)
        history_path = output_dir / "sampling_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Saved sampling history: {history_path}")

        # Save subject summary
        summary = {
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'total_trials': len(history),
            'warmup_trials': len([h for h in history if h['phase'] == 'warmup']),
            'main_trials': len([h for h in history if h['phase'] == 'main']),
            'oracle_spec': oracle.get_model_spec() if oracle else None,
        }
        summary_path = output_dir / "subject_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved subject summary: {summary_path}")

    def _save_aggregate_results(self, role, output_dir: Path, all_results: list, config: Dict[str, Any]):
        """Save aggregate results across all subjects."""
        # Combine all histories
        combined_history = []
        for result in all_results:
            for trial in result['history']:
                combined_history.append({
                    'subject_id': result['subject_id'],
                    **trial
                })

        combined_df = pd.DataFrame(combined_history)
        combined_path = output_dir / "combined_history.csv"
        combined_df.to_csv(combined_path, index=False)
        logger.info(f"Saved combined history: {combined_path}")

        # Save aggregate summary
        aggregate_summary = {
            'role': role.get_name(),
            'timestamp': datetime.now().isoformat(),
            'n_subjects': len(all_results),
            'budget_per_subject': config.get('budget', 50),
            'warmup_per_subject': config.get('warmup_points', 10),
            'total_trials': len(combined_history),
            'subjects': [
                {
                    'subject_id': r['subject_id'],
                    'seed': r['seed'],
                    'trials': len(r['history']),
                    'output_dir': r['output_dir']
                }
                for r in all_results
            ]
        }
        summary_path = output_dir / "aggregate_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(aggregate_summary, f, indent=2)
        logger.info(f"Saved aggregate summary: {summary_path}")
