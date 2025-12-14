#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for extended evaluation_v4 functionality.
"""

import sys
import numpy as np
import torch
import json
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.evaluation_model_discovery import evaluate_effect_capture_v4


class MockGPModel:
    """Mock GP model for testing."""
    def __init__(self, train_X, train_y):
        self.train_inputs = [torch.tensor(train_X, dtype=torch.float64)]
        self.train_targets = torch.tensor(train_y, dtype=torch.float64)
        self.likelihood = None  # Not ordinal


def test_v4_extended():
    """Test the extended v4 evaluation with statistical power and effect size."""
    print("=" * 60)
    print("Testing Extended evaluation_v4 with Power & Effect Size")
    print("=" * 60)

    # Create synthetic data (6 features, 10 samples)
    np.random.seed(42)
    n_samples = 10
    n_features = 6

    train_X = np.random.rand(n_samples, n_features)

    # Generate synthetic y with known effects
    # Main effects: x0=0.3, x1=0.1, x2=0.05, x3=0.4, x4=-0.05, x5=-0.05
    # Interaction: x1*x2=0.2
    main_weights = np.array([0.3, 0.1, 0.05, 0.4, -0.05, -0.05])
    interaction_weight = 0.2

    train_y = (
        0.5 +  # intercept
        train_X @ main_weights +
        interaction_weight * train_X[:, 1] * train_X[:, 2] +
        np.random.normal(0, 0.1, n_samples)
    )

    # Create mock model
    model = MockGPModel(train_X, train_y)

    # Oracle params
    oracle_params = {
        'main_weights': main_weights.tolist(),
        'interaction_weights': {
            'x1*x2': interaction_weight
        }
    }

    # Run evaluation
    print("\nRunning evaluation_v4 in auto_discovery mode...")
    results = evaluate_effect_capture_v4(
        model=model,
        oracle_params=oracle_params,
        evaluation_mode='auto_discovery',
        max_interactions=2,
        criterion='bic'
    )

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print("\n1. Structure Discovery:")
    print(json.dumps(results['structure_discovery'], indent=2))

    print("\n2. Best Model:")
    print(json.dumps(results['best_model'], indent=2))

    print("\n3. Statistical Power:")
    power_metrics = results['statistical_power']
    print(f"   Mean Power: {power_metrics['mean_power']:.3f}")
    print(f"   Min Power: {power_metrics['min_power']:.3f}")
    print(f"   Underpowered Count: {power_metrics['underpowered_count']}")
    if power_metrics['underpowered_effects']:
        print("   Underpowered Effects:")
        for eff in power_metrics['underpowered_effects']:
            print(f"     - {eff['effect']}: power={eff['power']:.3f}, true_coef={eff['true_coef']:.3f}")

    print("\n4. Effect Size Comparison:")
    effect_size = results['effect_size_comparison']
    print(f"   Classification Accuracy: {effect_size['classification_accuracy']:.3f}")
    print(f"   Correct: {effect_size['correct_count']}/{effect_size['total_effects']}")
    print("\n   Confusion Matrix:")
    for true_cat, est_cats in effect_size['confusion_matrix'].items():
        print(f"     {true_cat:>12s}: {est_cats}")

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

    return results


if __name__ == '__main__':
    try:
        results = test_v4_extended()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
