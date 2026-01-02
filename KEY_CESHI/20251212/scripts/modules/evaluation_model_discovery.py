#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_v4.py - Automatic Model Comparison for Effect Discovery

Focus: Blind evaluation of effect discovery ability through automatic model comparison.
       Uses BIC/AIC to select best model from all candidate interaction combinations.
"""

import numpy as np
import statsmodels.api as sm
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
import logging
import torch
from statsmodels.stats.power import tt_solve_power

logger = logging.getLogger(__name__)


def _extract_true_effects(oracle_params: Dict) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Extract ground truth effects from oracle parameters.

    Args:
        oracle_params: Oracle specification with main_weights and interaction_weights

    Returns:
        Tuple of (significant_main_indices, significant_interaction_pairs)
    """
    # Extract significant main effects (|weight| >= 0.05)
    main_weights = np.array(oracle_params.get('main_weights', []))
    significant_main = [i for i, w in enumerate(main_weights) if abs(w) >= 0.05]

    # Extract significant interaction effects
    interaction_dict = oracle_params.get('interaction_weights', {})
    significant_interactions = []

    for key, weight in interaction_dict.items():
        if abs(weight) >= 0.05:
            # Parse "x0*x1" or "x1*x0" format
            parts = key.replace('x', '').split('*')
            i, j = int(parts[0]), int(parts[1])
            # Normalize to (min, max) order
            pair = (min(i, j), max(i, j))
            if pair not in significant_interactions:
                significant_interactions.append(pair)

    logger.info(f"True significant effects: main={significant_main}, interactions={significant_interactions}")
    return significant_main, significant_interactions


def _enumerate_candidate_models(
    train_X: np.ndarray,
    train_y: np.ndarray,
    max_interactions: int = 3
) -> List[Dict]:
    """Enumerate candidate models and fit them using statsmodels.

    Args:
        train_X: Training features (n_samples, n_features)
        train_y: Training targets (n_samples,)
        max_interactions: Maximum number of interactions to consider

    Returns:
        List of model dicts with keys: name, effects, interactions, aic, bic, rsquared_adj, model
    """
    n_samples, n_features = train_X.shape
    all_interactions = list(combinations(range(n_features), 2))

    logger.info(f"Enumerating models with up to {max_interactions} interactions from {len(all_interactions)} candidates")

    models = []

    # M0: Intercept only
    try:
        X0 = sm.add_constant(np.ones(n_samples))
        model0 = sm.OLS(train_y, X0).fit()
        models.append({
            'name': 'M0_intercept_only',
            'effects': [],
            'interactions': [],
            'aic': model0.aic,
            'bic': model0.bic,
            'rsquared_adj': model0.rsquared_adj,
            'model': model0
        })
    except Exception as e:
        logger.warning(f"Failed to fit M0: {e}")

    # M1: Main effects only
    try:
        X1 = sm.add_constant(train_X)
        model1 = sm.OLS(train_y, X1).fit()
        models.append({
            'name': 'M1_main_effects',
            'effects': list(range(n_features)),
            'interactions': [],
            'aic': model1.aic,
            'bic': model1.bic,
            'rsquared_adj': model1.rsquared_adj,
            'model': model1
        })
    except Exception as e:
        logger.warning(f"Failed to fit M1: {e}")

    # M2+: Main effects + k interactions
    for k in range(1, min(max_interactions + 1, len(all_interactions) + 1)):
        for combo in combinations(all_interactions, k):
            try:
                # Build design matrix
                interaction_cols = [train_X[:, i] * train_X[:, j] for i, j in combo]
                X_full = np.column_stack([
                    train_X,
                    *interaction_cols
                ])
                X_full = sm.add_constant(X_full)

                # Fit model
                model = sm.OLS(train_y, X_full).fit()

                models.append({
                    'name': f'M2_main+{k}int_{combo}',
                    'effects': list(range(n_features)),
                    'interactions': list(combo),
                    'aic': model.aic,
                    'bic': model.bic,
                    'rsquared_adj': model.rsquared_adj,
                    'model': model
                })
            except (np.linalg.LinAlgError, ValueError) as e:
                # Singular matrix or numerical issues, skip
                logger.debug(f"Skipped model with {combo}: {e}")
                continue

    logger.info(f"Successfully fitted {len(models)} candidate models")
    return models


def _select_best_model(models: List[Dict], criterion: str = 'bic') -> Dict:
    """Select best model based on information criterion.

    Args:
        models: List of fitted model dicts
        criterion: 'bic' or 'aic'

    Returns:
        Best model dict
    """
    if not models:
        raise ValueError("No valid models to select from")

    best_model = min(models, key=lambda m: m[criterion])
    logger.info(
        f"Best model selected: {best_model['name']} "
        f"({criterion.upper()}={best_model[criterion]:.2f}, "
        f"R²_adj={best_model['rsquared_adj']:.3f})"
    )
    return best_model


def _evaluate_structure_discovery(
    discovered_interactions: List[Tuple[int, int]],
    true_interactions: List[Tuple[int, int]]
) -> Dict[str, Any]:
    """Evaluate structure discovery performance.

    Args:
        discovered_interactions: Interactions found by model selection
        true_interactions: Ground truth interactions

    Returns:
        Dict with precision, recall, f1_score, etc.
    """
    # Convert to sets for comparison
    discovered_set = set(discovered_interactions)
    true_set = set(true_interactions)

    # Calculate metrics
    tp = len(discovered_set & true_set)
    fp = len(discovered_set - true_set)
    fn = len(true_set - discovered_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "discovered_interactions": [f"x{i}*x{j}" for i, j in discovered_interactions],
        "true_interactions": [f"x{i}*x{j}" for i, j in true_interactions],
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score)
    }


def _evaluate_statistical_power(
    fitted_model: object,
    true_main_weights: np.ndarray,
    true_interaction_weights: np.ndarray,
    n_features: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Evaluate statistical power for effect detection.

    Args:
        fitted_model: Fitted statsmodels OLS model
        true_main_weights: Ground truth main effect coefficients
        true_interaction_weights: Ground truth interaction coefficients
        n_features: Number of features
        alpha: Significance level (default: 0.05)

    Returns:
        Dict with power metrics (mean_power, min_power, underpowered_count)
    """
    # Extract all true coefficients (excluding intercept)
    all_true_coefs = np.concatenate([true_main_weights, true_interaction_weights])

    # Get fitted coefficients and standard errors (excluding intercept)
    fitted_coefs = fitted_model.params[1:]  # Skip intercept
    fitted_se = fitted_model.bse[1:]  # Skip intercept

    # Calculate effect sizes (Cohen's d = |coef| / SE)
    effect_sizes = np.abs(fitted_coefs) / (fitted_se + 1e-10)  # Avoid division by zero

    # Calculate post-hoc power for each effect
    n_samples = fitted_model.nobs
    powers = []

    for effect_size in effect_sizes:
        try:
            # Use two-tailed t-test power calculation
            power = tt_solve_power(
                effect_size=effect_size,
                nobs=n_samples,
                alpha=alpha,
                alternative='two-sided'
            )
            powers.append(float(power) if power is not None else 0.0)
        except Exception as e:
            logger.debug(f"Power calculation failed for effect_size={effect_size}: {e}")
            powers.append(0.0)

    powers = np.array(powers)

    # Calculate summary statistics
    mean_power = np.mean(powers) if len(powers) > 0 else 0.0
    min_power = np.min(powers) if len(powers) > 0 else 0.0
    underpowered_count = np.sum(powers < 0.8)  # Standard threshold: 0.8

    # Identify underpowered effects (significant true effects with low power)
    significant_true = np.abs(all_true_coefs) >= 0.05
    underpowered_effects = []
    for i, (is_sig, pwr) in enumerate(zip(significant_true, powers)):
        if is_sig and pwr < 0.8:
            effect_name = f"x{i}" if i < n_features else f"interaction_{i - n_features}"
            underpowered_effects.append({
                "effect": effect_name,
                "true_coef": float(all_true_coefs[i]),
                "power": float(pwr)
            })

    return {
        "mean_power": float(mean_power),
        "min_power": float(min_power),
        "underpowered_count": int(underpowered_count),
        "underpowered_effects": underpowered_effects,
        "all_powers": powers.tolist()
    }


def _evaluate_effect_sizes(
    fitted_model: object,
    true_main_weights: np.ndarray,
    true_interaction_weights: np.ndarray,
    n_features: int
) -> Dict[str, Any]:
    """Evaluate effect size classification accuracy.

    Args:
        fitted_model: Fitted statsmodels OLS model
        true_main_weights: Ground truth main effect coefficients
        true_interaction_weights: Ground truth interaction coefficients
        n_features: Number of features

    Returns:
        Dict with effect size classification metrics
    """
    # Cohen's d thresholds for effect size classification
    # negligible: |d| < 0.2, small: 0.2-0.5, medium: 0.5-0.8, large: >= 0.8
    def classify_effect_size(coef: float, se: float) -> str:
        """Classify effect size based on Cohen's d."""
        d = abs(coef) / (se + 1e-10)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"

    def classify_by_coefficient(coef: float) -> str:
        """Classify effect size based on raw coefficient magnitude."""
        abs_coef = abs(coef)
        if abs_coef < 0.05:
            return "negligible"
        elif abs_coef < 0.15:
            return "small"
        elif abs_coef < 0.30:
            return "medium"
        else:
            return "large"

    # Extract all true and estimated coefficients (excluding intercept)
    all_true_coefs = np.concatenate([true_main_weights, true_interaction_weights])
    fitted_coefs = fitted_model.params[1:]  # Skip intercept
    fitted_se = fitted_model.bse[1:]  # Skip intercept

    # Classify true effect sizes (by coefficient magnitude)
    true_classes = [classify_by_coefficient(coef) for coef in all_true_coefs]

    # Classify estimated effect sizes (by Cohen's d)
    estimated_classes = [
        classify_effect_size(coef, se)
        for coef, se in zip(fitted_coefs, fitted_se)
    ]

    # Calculate classification accuracy
    correct_classifications = sum(
        1 for true_cls, est_cls in zip(true_classes, estimated_classes)
        if true_cls == est_cls
    )
    accuracy = correct_classifications / len(true_classes) if len(true_classes) > 0 else 0.0

    # Build confusion matrix
    categories = ["negligible", "small", "medium", "large"]
    confusion_matrix = {true_cat: {est_cat: 0 for est_cat in categories} for true_cat in categories}

    for true_cls, est_cls in zip(true_classes, estimated_classes):
        confusion_matrix[true_cls][est_cls] += 1

    # Detailed breakdown
    effect_details = []
    for i, (true_coef, est_coef, true_cls, est_cls) in enumerate(
        zip(all_true_coefs, fitted_coefs, true_classes, estimated_classes)
    ):
        effect_name = f"x{i}" if i < n_features else f"interaction_{i - n_features}"
        effect_details.append({
            "effect": effect_name,
            "true_coef": float(true_coef),
            "estimated_coef": float(est_coef),
            "true_class": true_cls,
            "estimated_class": est_cls,
            "match": true_cls == est_cls
        })

    return {
        "classification_accuracy": float(accuracy),
        "correct_count": int(correct_classifications),
        "total_effects": len(true_classes),
        "confusion_matrix": confusion_matrix,
        "effect_details": effect_details
    }


def track_effect_discovery_timeline(
    model_history: List[object],
    oracle_params: Dict,
    pairs_to_eval: List[Tuple[int, int]],
    alpha: float = 0.05
) -> Dict[str, Optional[int]]:
    """Track when each effect first becomes statistically significant.

    Args:
        model_history: List of model snapshots from warmup to final
        oracle_params: Oracle specification with main_weights and interaction_weights
        pairs_to_eval: List of interaction pairs to evaluate
        alpha: Significance threshold (default 0.05)

    Returns:
        Dictionary mapping effect names to first significant iteration
        Example: {"x0": 5, "x1": 3, "x0*x5": 12, "x1*x2": None}
    """
    logger.info(f"Tracking effect discovery timeline (alpha={alpha})")

    # Initialize discovery timeline (None = never significant)
    timeline = {}
    for i in range(6):  # 6 main effects
        timeline[f"x{i}"] = None
    for i, j in pairs_to_eval:
        timeline[f"x{i}*x{j}"] = None

    # Track discovery for each iteration
    for iteration_idx, model in enumerate(model_history):
        try:
            # Extract training data
            if not hasattr(model, 'train_inputs') or model.train_inputs is None or len(model.train_inputs) == 0:
                logger.debug(f"Skipping iteration {iteration_idx}: model has no train_inputs")
                continue

            train_X = model.train_inputs[0].detach().cpu().numpy()
            train_y_raw = model.train_targets.detach().cpu().numpy()

            # Need at least n_params samples to fit OLS
            n_params = 1 + 6 + len(pairs_to_eval)  # intercept + main + interactions
            if len(train_X) < n_params:
                logger.debug(f"Skipping iteration {iteration_idx}: insufficient samples ({len(train_X)} < {n_params})")
                continue

            # Handle ordinal models
            if hasattr(model.likelihood, "cutpoints"):
                with torch.no_grad():
                    probs = model.predict_probs(torch.from_numpy(train_X).to(torch.float64))
                    probs_np = probs.cpu().numpy()
                    train_y = np.array([
                        np.sum(np.arange(len(probs_np[i])) * probs_np[i])
                        for i in range(len(probs_np))
                    ])
            else:
                train_y = train_y_raw.flatten() if hasattr(train_y_raw, 'flatten') else train_y_raw

            # Build design matrix: intercept + main effects + interactions
            n_train = len(train_X)
            X_design = np.column_stack([
                np.ones(n_train),  # intercept
                train_X,           # main effects (x0-x5)
                *(train_X[:, i] * train_X[:, j] for i, j in pairs_to_eval)  # interactions
            ])

            # Fit OLS model to get p-values
            ols_model = sm.OLS(train_y, X_design)
            ols_result = ols_model.fit()

            # Extract p-values (skip intercept)
            pvalues = ols_result.pvalues[1:]

            # Check main effects (x0-x5)
            for i in range(6):
                effect_name = f"x{i}"
                if timeline[effect_name] is None and pvalues[i] < alpha:
                    timeline[effect_name] = iteration_idx + 1  # 1-indexed
                    logger.info(f"  Effect {effect_name} discovered at iteration {iteration_idx + 1}")

            # Check interaction effects
            for idx, (i, j) in enumerate(pairs_to_eval):
                effect_name = f"x{i}*x{j}"
                pvalue_idx = 6 + idx  # After main effects
                if timeline[effect_name] is None and pvalues[pvalue_idx] < alpha:
                    timeline[effect_name] = iteration_idx + 1  # 1-indexed
                    logger.info(f"  Interaction {effect_name} discovered at iteration {iteration_idx + 1}")

        except Exception as e:
            logger.warning(f"Failed to process iteration {iteration_idx}: {e}")
            continue

    logger.info(f"Effect discovery timeline complete. Discovered: {sum(1 for v in timeline.values() if v is not None)}/{len(timeline)}")
    return timeline


def evaluate_effect_capture_v4(
    model: object,
    oracle_params: Dict,
    evaluation_mode: str = 'auto_discovery',
    max_interactions: int = 3,
    criterion: str = 'bic',
    pairs_to_eval: Optional[List[Tuple[int, int]]] = None,
    design_space: Optional[np.ndarray] = None,
    oracle: Optional[object] = None
) -> Dict:
    """Main entry function for automatic model comparison evaluation.

    Args:
        model: Fitted GP model with train_inputs and train_targets
        oracle_params: Ground truth oracle parameters
        evaluation_mode: 'auto_discovery' (blind) or 'known_structure' (open-book)
        max_interactions: Maximum number of interactions to consider
        criterion: 'bic' or 'aic' for model selection
        pairs_to_eval: Pre-specified interaction pairs (for known_structure mode)
        design_space: Full design space (N, d) for computing test set performance
        oracle: Oracle instance for getting true y values on design space

    Returns:
        Evaluation results dict
    """
    logger.info(f"Starting v4 evaluation (mode={evaluation_mode}, criterion={criterion})")

    # Extract training data
    train_X = model.train_inputs[0].detach().cpu().numpy()
    train_y_raw = model.train_targets.detach().cpu().numpy()

    # Validate shape consistency
    if len(train_X) != len(train_y_raw):
        logger.warning(
            f"Data mismatch: train_X={len(train_X)}, train_y={len(train_y_raw)}. "
            f"Using first {min(len(train_X), len(train_y_raw))} samples."
        )
        min_samples = min(len(train_X), len(train_y_raw))
        train_X = train_X[:min_samples]
        train_y_raw = train_y_raw[:min_samples]

    # Handle ordinal models
    if model is not None and hasattr(model.likelihood, "cutpoints"):
        logger.info("Ordinal model detected, converting to expected ratings")
        with torch.no_grad():
            probs = model.predict_probs(torch.from_numpy(train_X).to(torch.float64))
            probs_np = probs.cpu().numpy()
            train_y = np.array([
                np.sum(np.arange(len(probs_np[i])) * probs_np[i])
                for i in range(len(probs_np))
            ])
    else:
        train_y = train_y_raw.flatten() if hasattr(train_y_raw, 'flatten') else train_y_raw

    # Extract ground truth
    true_main_indices, true_interactions = _extract_true_effects(oracle_params)

    # Extract true weights for power and effect size analysis
    true_main_weights = np.array(oracle_params.get('main_weights', []))
    true_interaction_dict = oracle_params.get('interaction_weights', {})

    if evaluation_mode == 'auto_discovery':
        # === Blind Mode: Automatic Model Comparison ===
        logger.info("Auto-discovery mode: enumerating candidate models...")

        # Enumerate and fit all candidate models
        candidate_models = _enumerate_candidate_models(
            train_X, train_y, max_interactions
        )

        # Select best model
        best_model = _select_best_model(candidate_models, criterion)

        # Evaluate structure discovery
        structure_discovery = _evaluate_structure_discovery(
            best_model['interactions'],
            true_interactions
        )

        # Model comparison table with detailed metrics
        best_aic = min(m['aic'] for m in candidate_models)
        best_bic = min(m['bic'] for m in candidate_models)

        model_comparison = []
        for m in sorted(candidate_models, key=lambda m: m[criterion])[:10]:  # Top 10
            # Calculate RMSE and MAE on design space if available, else on training set
            if design_space is not None and oracle is not None:
                # Test set performance (design space)
                y_oracle = np.array([oracle.query(x) for x in design_space])
                # Build design matrix for OLS prediction
                n_test = len(design_space)
                X_design_test = np.column_stack([
                    np.ones(n_test),  # intercept
                    design_space,      # main effects
                    *(design_space[:, i] * design_space[:, j] for i, j in m['interactions'])
                ])
                y_pred_test = m['model'].predict(X_design_test)
                rmse = np.sqrt(np.mean((y_oracle - y_pred_test)**2))
                mae = np.mean(np.abs(y_oracle - y_pred_test))
            else:
                # Training set performance (fallback)
                y_pred = m['model'].fittedvalues
                y_true = train_y
                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                mae = np.mean(np.abs(y_true - y_pred))

            # Count significant coefficients (p < 0.05)
            pvalues = m['model'].pvalues[1:]  # Skip intercept
            significant_count = np.sum(pvalues < 0.05)
            total_params = len(pvalues)
            significant_ratio = significant_count / total_params if total_params > 0 else 0

            model_comparison.append({
                'name': m['name'],
                'aic': float(m['aic']),
                'bic': float(m['bic']),
                'delta_aic': float(m['aic'] - best_aic),
                'delta_bic': float(m['bic'] - best_bic),
                'rsquared_adj': float(m['rsquared_adj']),
                'rmse': float(rmse),
                'mae': float(mae),
                'n_params': int(total_params),
                'significant_coefs_ratio': float(significant_ratio),
                'interactions': [f"x{i}*x{j}" for i, j in m['interactions']]
            })

        # === Enhanced Metrics: Statistical Power & Effect Size ===
        # Prepare true interaction weights in the same order as best model
        true_interaction_weights = []
        for i, j in best_model['interactions']:
            key1 = f"x{i}*x{j}"
            key2 = f"x{j}*x{i}"
            true_interaction_weights.append(
                true_interaction_dict.get(key1, true_interaction_dict.get(key2, 0.0))
            )
        true_interaction_weights = np.array(true_interaction_weights)

        # Calculate statistical power
        n_features = train_X.shape[1]
        statistical_power = _evaluate_statistical_power(
            best_model['model'],
            true_main_weights,
            true_interaction_weights,
            n_features
        )

        # Calculate effect size classification accuracy
        effect_size_comparison = _evaluate_effect_sizes(
            best_model['model'],
            true_main_weights,
            true_interaction_weights,
            n_features
        )

        results = {
            "evaluation_mode": "auto_discovery",
            "structure_discovery": structure_discovery,
            "best_model": {
                "name": best_model['name'],
                "criterion": criterion,
                "criterion_value": float(best_model[criterion]),
                "rsquared_adj": float(best_model['rsquared_adj']),
                "selected_interactions": [f"x{i}*x{j}" for i, j in best_model['interactions']]
            },
            "model_comparison_table": model_comparison,
            "statistical_power": statistical_power,
            "effect_size_comparison": effect_size_comparison
        }

    else:
        # === Open-book Mode: Known Structure (backward compatibility with v3) ===
        logger.info(f"Known-structure mode: using pre-specified pairs {pairs_to_eval}")

        if pairs_to_eval is None:
            raise ValueError("pairs_to_eval must be provided in known_structure mode")

        # Fit model with known interactions
        interaction_cols = [train_X[:, i] * train_X[:, j] for i, j in pairs_to_eval]
        X_full = np.column_stack([train_X, *interaction_cols])
        X_full = sm.add_constant(X_full)

        fitted_model = sm.OLS(train_y, X_full).fit()

        results = {
            "evaluation_mode": "known_structure",
            "structure_discovery": None,  # N/A in open-book mode
            "model_fit": {
                "aic": float(fitted_model.aic),
                "bic": float(fitted_model.bic),
                "rsquared_adj": float(fitted_model.rsquared_adj),
                "specified_interactions": [f"x{i}*x{j}" for i, j in pairs_to_eval]
            }
        }

    logger.info("v4 evaluation completed")
    return results
