import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add project root and parent to sys.path
ROOT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = ROOT_DIR.parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from core.context import Context
from modules.step1 import Step1Module
from modules.step1_5 import Step1_5Module
from modules.step2 import Step2Module
from modules.step3 import Step3Module

def test_full_flow_with_review_artifacts():
    print("Starting Full Flow Verification...")
    
    # Setup
    context = Context()
    design_path = r"D:\ENVS\active-psych-sampling\data\i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA_mapped_normalized_zscore_v2.csv"
    
    config = {
        "step1": {
            "design_csv": design_path,
            "n_subjects": 2,
            "trials_per_subject": 10,
            "interaction_mode": "hybrid",
            "interaction_pairs": [[0, 1]],
            "output_dir": "test_verify_flow",
            "merge": False,
            "auto_confirm": True,
            "skip_interaction": False
        },
        "step1_5": {
            "population_mean": 0.0,
            "population_std": 0.1,
            "individual_std_percent": 0.2,
            "interaction_pairs": [[0, 1]],
            "interaction_scale": 5.0, # Strong interaction to ensure discovery
            "output_type": "continuous",
            "response_col": "y",
            "seed": 42
        },
        "step2": {
            "subject_col": "subject_id",
            "response_col": "y",
            "max_pairs": 3,
            "min_pairs": 1,
            "selection_method": "elbow",
            "suspected_pairs": []
        },
        "step3": {
            "max_iters": 5,
            "learning_rate": 0.1,
            "use_cuda": False,
            "ensure_diversity": True,
            "subject_col": "subject_id",
            "response_col": "y",
            "kernel_type": "default", # Should be overridden by Step 2
            "dim_specs": [
                {"name": "x1", "type": "continuous"},
                {"name": "x2", "type": "continuous"},
                {"name": "x3", "type": "continuous"},
                {"name": "x4", "type": "continuous"},
                {"name": "x5", "type": "continuous"},
                {"name": "x6", "type": "continuous"}
            ]
        }
    }

    # Step 1: Sampling
    print("\n--- Running Step 1 ---")
    s1 = Step1Module()
    context = s1.run(config, context)
    
    # Step 1.5: Simulation
    print("\n--- Running Step 1.5 ---")
    s15 = Step1_5Module()
    context = s15.run(config, context)
    
    # Step 2: Analysis
    print("\n--- Running Step 2 ---")
    s2 = Step2Module()
    context = s2.run(config, context)
    
    # Verify Step 2 Artifacts
    spec_path = Path(context.model_spec_path)
    summary_path = Path(context.model_summary_path)
    
    assert spec_path.exists(), "model_spec.json missing!"
    assert summary_path.exists(), "model_spec_summary.md missing!"
    
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
        print(f"Discovered Kernel: {spec['kernel_type']}")
        print(f"Discovered Pairs: {spec['interaction_pairs']}")
        
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = f.read()
        assert "# Phase 2 Model Configuration Proposal" in summary
        print("Summary report verified (English).")

    # Step 3: Base GP
    print("\n--- Running Step 3 ---")
    s3 = Step3Module()
    context = s3.run(config, context)
    
    print("\nFull Flow Verification Successful!")

if __name__ == "__main__":
    try:
        test_full_flow_with_review_artifacts()
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
