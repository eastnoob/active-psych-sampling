"""Data saver module for rich experiment output.

Provides standardized interfaces for saving:
- Sampling history and interaction logs
- EUR diagnostics (lambda_t, gamma_t, r_t)
- Oracle specifications
- Iterations CSV with detailed metrics
- Enhanced experiment summary
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


def save_oracle_spec(
    result_dir: Path,
    oracle: Any,
    source: str = "LinearOracle"
) -> None:
    """Save Oracle model specification to oracle_spec.json.

    Args:
        result_dir: Result directory
        oracle: Oracle instance (must implement get_model_spec())
        source: Oracle source description
    """
    data_dir = result_dir / 'data_files'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Get Oracle spec
    spec = oracle.get_model_spec() if hasattr(oracle, 'get_model_spec') else {}

    # Add source info
    spec['source'] = source

    # Save
    spec_file = data_dir / 'oracle_spec.json'
    with open(spec_file, 'w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved Oracle spec: {spec_file}")


def save_sampling_data(
    result_dir: Path,
    sampling_history: List,
    interaction_logs: List[Dict],
    eur_diagnostics: Optional[Dict[str, List]] = None
) -> None:
    """Save sampling data and diagnostics.

    Args:
        result_dir: Result directory
        sampling_history: Sampling history [(x1, x2, ..., xn), ...]
        interaction_logs: Interaction logs [{trial, x, y}, ...]
        eur_diagnostics: EUR diagnostic data containing:
            - lambda_t: EUR dynamic weights
            - gamma_t: Uncertainty weights
            - r_t: Diversity weights
    """
    data_dir = result_dir / 'data_files'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save sampling history
    history_file = data_dir / 'sampling_history.npy'
    np.save(history_file, np.array(sampling_history))
    logger.info(f"Saved sampling history: {history_file}")

    # Save interaction log
    log_file = data_dir / 'interaction_log.json'
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(interaction_logs, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved interaction log: {log_file}")

    # Save EUR diagnostics
    if eur_diagnostics:
        diagnostics_dir = data_dir / 'eur_diagnostics'
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        for key, values in eur_diagnostics.items():
            diag_file = diagnostics_dir / f'{key}.npy'
            np.save(diag_file, np.array(values))
            logger.info(f"Saved EUR diagnostic '{key}': {diag_file}")


def save_iterations_csv(
    result_dir: Path,
    interaction_logs: List[Dict],
    eur_diagnostics: Optional[Dict[str, List]] = None,
    warmup_budget: int = 3
) -> None:
    """Generate iterations.csv recording detailed information for each iteration.

    Args:
        result_dir: Result directory
        interaction_logs: Interaction logs [{trial, x, y}, ...]
        eur_diagnostics: EUR diagnostic data {lambda_t: [...], gamma_t: [...], r_t: [...]}
        warmup_budget: Warmup phase budget (for determining phase)
    """
    data_dir = result_dir / 'data_files'
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for log in interaction_logs:
        trial = log['trial']
        x_array = log['x']
        y_value = log['y']

        # Determine phase
        phase = 'warmup' if trial < warmup_budget else 'eur'

        # Base row
        row = {
            'iteration': trial + 1,  # 1-indexed
            'phase': phase,
            'y_value': y_value
        }

        # EUR diagnostic data (only valid in EUR phase)
        if eur_diagnostics and trial >= warmup_budget:
            eur_idx = trial - warmup_budget
            for key in ['lambda_t', 'gamma_t', 'r_t']:
                if key in eur_diagnostics and eur_idx < len(eur_diagnostics[key]):
                    row[key] = eur_diagnostics[key][eur_idx]
                else:
                    row[key] = None
            row['n_train'] = warmup_budget + eur_idx
        else:
            # Warmup phase has no EUR diagnostic data
            for key in ['lambda_t', 'gamma_t', 'r_t', 'n_train']:
                row[key] = None

        # Parameter values (round to 1 decimal place to fix float precision)
        for i, x_val in enumerate(x_array):
            row[f'x{i}'] = round(x_val, 1)

        rows.append(row)

    # Build DataFrame and save
    df = pd.DataFrame(rows)

    # Column order
    base_cols = ['iteration', 'phase', 'y_value', 'lambda_t', 'gamma_t', 'r_t', 'n_train']
    x_cols = [f'x{i}' for i in range(len(interaction_logs[0]['x']))]
    df = df[base_cols + x_cols]

    csv_file = data_dir / 'iterations.csv'
    df.to_csv(csv_file, index=False, float_format='%.10g')
    logger.info(f"Saved iterations CSV: {csv_file}")


def save_enhanced_summary(
    result_dir: Path,
    config: Dict[str, Any],
    oracle_spec: Dict[str, Any],
    interaction_logs: List[Dict],
    eur_diagnostics: Optional[Dict[str, List]],
    warmup_budget: int,
    seed: Optional[int] = None,
    eval_results: Optional[Dict[str, Any]] = None
) -> None:
    """Save enhanced experiment summary (summary.json).

    Args:
        result_dir: Result directory
        config: Configuration info
        oracle_spec: Oracle specification
        interaction_logs: Interaction logs
        eur_diagnostics: EUR diagnostic data
        warmup_budget: Warmup budget
        seed: Random seed
        eval_results: Evaluation results (optional)
    """
    data_dir = result_dir / 'data_files'
    timestamp = result_dir.name

    # Build summary structure
    summary = {
        # Basic configuration
        "experiment": {
            "timestamp": timestamp,
            "config": config.get('config_name', 'keyceshi_eur_complete.ini'),
            "result_dir": str(result_dir)
        },

        "parameters": {
            "total_budget": len(interaction_logs),
            "warmup_budget": warmup_budget,
            "eur_budget": len(interaction_logs) - warmup_budget,
            "seed": seed
        },

        # Oracle spec
        "oracle": {
            "model_type": oracle_spec.get('model_type', 'linear'),
            "bias": oracle_spec.get('bias', 0.0),
            "noise_std": oracle_spec.get('noise_std', 0.0),
            "main_weights": oracle_spec.get('weights', []),
            "interaction_weights": oracle_spec.get('interactions', {})
        }
    }

    # Sampling trajectory statistics
    if eur_diagnostics:
        def compute_stats(values: List) -> Dict:
            """Compute statistics, filtering out None values"""
            if not values:
                return {}
            valid_values = [v for v in values if v is not None]
            if not valid_values:
                return {}
            arr = np.array(valid_values)
            return {
                'initial': float(arr[0]),
                'final': float(arr[-1]),
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr))
            }

        summary['sampling_trajectory'] = {
            'lambda_t': compute_stats(eur_diagnostics.get('lambda_t', [])),
            'gamma_t': compute_stats(eur_diagnostics.get('gamma_t', [])),
            'r_t': compute_stats(eur_diagnostics.get('r_t', []))
        }

        # Sampling diversity
        summary['sampling_trajectory']['diversity'] = {
            'unique_samples': len(interaction_logs)
        }

    # Evaluation results (if provided)
    if eval_results:
        summary['effect_recovery'] = eval_results.get('effect_recovery', {})
        summary['prediction_quality'] = eval_results.get('prediction_quality', {})

    # Save
    summary_file = data_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved enhanced summary: {summary_file}")
