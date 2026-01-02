
"""
evaluation_v3.py - Effect Capture Evaluation Module

Focus: Precisely measure ability to capture main effects and interaction effects
       in limited sampling scenarios (not prediction accuracy).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple, Optional, Any
import logging
import torch

logger = logging.getLogger(__name__)


def _fit_linear_model(
    train_X: np.ndarray,
    train_y_raw: np.ndarray,
    pairs_to_eval: List[Tuple[int, int]],
    model: Optional[object] = None
) -> Tuple[np.ndarray, np.ndarray, float, object]:
    """Fit linear regression model with main effects + interaction terms."""
    n_train = len(train_X)
    X_design = np.column_stack([
        np.ones(n_train),  # intercept
        train_X,           # main effects
        *(train_X[:, i] * train_X[:, j] for i, j in pairs_to_eval)  # interactions
    ])

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

    # Check for shape mismatch and fix
    if len(train_y) != len(train_X):
        logger.warning(
            f"Shape mismatch detected: train_X has {len(train_X)} samples, "
            f"train_y has {len(train_y)} samples. Using first {min(len(train_X), len(train_y))} samples."
        )
        min_samples = min(len(train_X), len(train_y))
        train_X = train_X[:min_samples]
        train_y = train_y[:min_samples]
        # Rebuild design matrix with correct number of samples
        X_design = np.column_stack([
            np.ones(min_samples),
            train_X,
            *(train_X[:, i] * train_X[:, j] for i, j in pairs_to_eval)
        ])

    lr_model = LinearRegression()
    lr_model.fit(X_design, train_y)

    intercept = lr_model.intercept_
    main_coefs = lr_model.coef_[1:7]  # x0-x5
    interaction_coefs = lr_model.coef_[7:]  # interactions

    return main_coefs, interaction_coefs, intercept, lr_model


def _evaluate_detection_rate(
    estimated_coefs: np.ndarray,
    true_coefs: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """Calculate effect detection rate metrics."""
    true_significant = np.abs(true_coefs) >= threshold
    estimated_significant = np.abs(estimated_coefs) >= threshold
    
    tp = np.sum(true_significant & estimated_significant)
    fp = np.sum(~true_significant & estimated_significant)
    fn = np.sum(true_significant & ~estimated_significant)
    tn = np.sum(~true_significant & ~estimated_significant)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score)
    }


def _evaluate_sign_accuracy(
    estimated_coefs: np.ndarray,
    true_coefs: np.ndarray,
    threshold: float = 0.05
) -> Dict[str, Any]:
    """Calculate sign (direction) accuracy for significant effects."""
    true_significant = np.abs(true_coefs) >= threshold
    
    if not np.any(true_significant):
        return {
            "sign_accuracy": None,
            "correct_signs": 0,
            "total_significant": 0
        }
    
    sign_match = (np.sign(estimated_coefs[true_significant]) == np.sign(true_coefs[true_significant]))
    correct_signs = np.sum(sign_match)
    total_significant = np.sum(true_significant)
    
    return {
        "sign_accuracy": float(correct_signs / total_significant),
        "correct_signs": int(correct_signs),
        "total_significant": int(total_significant)
    }


def _evaluate_top_k_ranking(
    estimated_coefs: np.ndarray,
    true_coefs: np.ndarray,
    k: int = 3
) -> Dict[str, Any]:
    """Calculate Top-k ranking accuracy."""
    true_abs = np.abs(true_coefs)
    estimated_abs = np.abs(estimated_coefs)
    
    true_top_k = set(np.argsort(true_abs)[-k:])
    estimated_top_k = set(np.argsort(estimated_abs)[-k:])
    
    overlap = len(true_top_k & estimated_top_k)
    overlap_ratio = overlap / k
    
    try:
        tau, p_value = kendalltau(true_abs, estimated_abs)
    except Exception as e:
        logger.warning(f"Kendall tau calculation failed: {e}")
        tau, p_value = None, None
    
    return {
        "top_k_overlap": float(overlap_ratio),
        "overlap_count": int(overlap),
        "k": k,
        "kendall_tau": float(tau) if tau is not None else None,
        "kendall_pvalue": float(p_value) if p_value is not None else None
    }


def _compute_correlations(
    estimated_coefs: np.ndarray,
    true_coefs: np.ndarray
) -> Dict[str, Any]:
    """Compute correlation metrics (reused from v2)."""
    try:
        rho, p_value = spearmanr(true_coefs, estimated_coefs)
        return {
            "spearman": float(rho) if not np.isnan(rho) else None,
            "spearman_pvalue": float(p_value) if not np.isnan(p_value) else None
        }
    except Exception as e:
        logger.warning(f"Spearman correlation failed: {e}")
        return {"spearman": None, "spearman_pvalue": None}


def _compute_errors(
    estimated_coefs: np.ndarray,
    true_coefs: np.ndarray
) -> Dict[str, float]:
    """Compute error metrics (reused from v2)."""
    mae = np.mean(np.abs(estimated_coefs - true_coefs))
    rmse = np.sqrt(np.mean((estimated_coefs - true_coefs) ** 2))
    
    return {
        "mae": float(mae),
        "rmse": float(rmse)
    }


def _evaluate_sampling_support(
    train_X: np.ndarray,
    pairs_to_eval: List[Tuple[int, int]]
) -> Dict[str, Any]:
    """Evaluate whether sampling covers interaction term estimation requirements."""
    interaction_coverage = {}
    
    for i, j in pairs_to_eval:
        Xi = train_X[:, i]
        Xj = train_X[:, j]
        
        Xi_min, Xi_max = Xi.min(), Xi.max()
        Xj_min, Xj_max = Xj.min(), Xj.max()
        
        if Xi_max > Xi_min and Xj_max > Xj_min:
            Xi_norm = (Xi - Xi_min) / (Xi_max - Xi_min)
            Xj_norm = (Xj - Xj_min) / (Xj_max - Xj_min)
            
            corners = [(0, 0), (0, 1), (1, 0), (1, 1)]
            corner_coverage = 0
            for ci, cj in corners:
                distances = np.sqrt((Xi_norm - ci)**2 + (Xj_norm - cj)**2)
                if np.any(distances < 0.2):
                    corner_coverage += 1
            
            range_coverage_i = (Xi_max - Xi_min)
            range_coverage_j = (Xj_max - Xj_min)
        else:
            corner_coverage = 0
            range_coverage_i = 0.0
            range_coverage_j = 0.0
        
        interaction_coverage[f"x{i}*x{j}"] = {
            "corner_count": int(corner_coverage),
            "range_i": float(range_coverage_i),
            "range_j": float(range_coverage_j)
        }
    
    avg_corner_coverage = np.mean([v["corner_count"] for v in interaction_coverage.values()]) / 4.0
    
    return {
        "interaction_coverage": interaction_coverage,
        "overall_support": float(avg_corner_coverage)
    }


def _compute_condition_number(
    X_design: np.ndarray
) -> float:
    """Compute condition number of design matrix."""
    try:
        cond = np.linalg.cond(X_design)
        return float(cond)
    except Exception as e:
        logger.warning(f"Condition number calculation failed: {e}")
        return None


def evaluate_effect_capture(
    model: object,
    oracle_params: Dict,
    pairs_to_eval: List[Tuple[int, int]],
    detection_threshold: float = 0.05,
    top_k: int = 3
) -> Dict:
    """Main entry function for effect capture evaluation."""
    logger.info("Starting effect capture evaluation (v3)")
    
    # Extract training data from model
    train_X = model.train_inputs[0].detach().cpu().numpy()
    train_y_raw = model.train_targets.detach().cpu().numpy()

    # DEBUG: Print data shapes immediately after extraction
    logger.warning(f"[DEBUG] Data extraction from model:")
    logger.warning(f"        train_inputs[0] shape: {model.train_inputs[0].shape}")
    logger.warning(f"        train_targets shape: {model.train_targets.shape}")
    logger.warning(f"        train_X shape: {train_X.shape}")
    logger.warning(f"        train_y_raw shape: {train_y_raw.shape}")

    # Validate shape consistency
    if len(train_X) != len(train_y_raw):
        logger.warning(
            f"Data mismatch at model level: train_X={len(train_X)}, train_y={len(train_y_raw)}. "
            f"Using first {min(len(train_X), len(train_y_raw))} samples."
        )
        min_samples = min(len(train_X), len(train_y_raw))
        train_X = train_X[:min_samples]
        train_y_raw = train_y_raw[:min_samples]

    # Fit linear model
    main_coefs, interaction_coefs, intercept, lr_model = _fit_linear_model(
        train_X, train_y_raw, pairs_to_eval, model
    )
    
    # Extract ground truth
    true_main = np.array(oracle_params["main_weights"])
    true_interaction_dict = oracle_params.get("interaction_weights", {})
    
    # Match interaction coefficient order
    true_interaction = []
    for i, j in pairs_to_eval:
        key1 = f"x{i}*x{j}"
        key2 = f"x{j}*x{i}"
        true_interaction.append(true_interaction_dict.get(key1, true_interaction_dict.get(key2, 0.0)))
    true_interaction = np.array(true_interaction)
    
    # Concatenate all effects for unified evaluation
    all_estimated = np.concatenate([main_coefs, interaction_coefs])
    all_true = np.concatenate([true_main, true_interaction])
    
    # === Core Metrics ===
    detection_rate = _evaluate_detection_rate(all_estimated, all_true, detection_threshold)
    sign_accuracy = _evaluate_sign_accuracy(all_estimated, all_true, detection_threshold)
    top_k_ranking = _evaluate_top_k_ranking(all_estimated, all_true, top_k)
    
    # === Quality Metrics ===
    correlations = _compute_correlations(all_estimated, all_true)
    errors = _compute_errors(all_estimated, all_true)
    
    # === Reference Metrics ===
    sampling_support = _evaluate_sampling_support(train_X, pairs_to_eval)
    
    # Design matrix for condition number
    n_train = len(train_X)
    X_design = np.column_stack([
        np.ones(n_train),
        train_X,
        *(train_X[:, i] * train_X[:, j] for i, j in pairs_to_eval)
    ])
    condition_number = _compute_condition_number(X_design)
    
    # === Assemble Results ===
    results = {
        "core_metrics": {
            "detection_rate": detection_rate,
            "sign_accuracy": sign_accuracy,
            "top_k_ranking": top_k_ranking
        },
        "quality_metrics": {
            "correlations": correlations,
            "errors": errors
        },
        "reference_metrics": {
            "sampling_support": sampling_support,
            "condition_number": condition_number
        },
        "model_fit": {
            "intercept": float(intercept),
            "main_coefs": main_coefs.tolist(),
            "interaction_coefs": interaction_coefs.tolist()
        }
    }
    
    logger.info("Effect capture evaluation completed")
    return results
