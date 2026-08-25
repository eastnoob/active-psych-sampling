import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import os

from core.context import Context
from modules.step1 import Step1Module
from modules.step1_5 import Step1_5Module
from modules.step2 import Step2Module
from modules.step3 import Step3Module

class TestWarmupFlow(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_output_dir")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a dummy design CSV
        self.design_csv = self.test_dir / "dummy_design.csv"
        df = pd.DataFrame({
            "x1": np.linspace(0, 1, 10),
            "x2": np.linspace(0, 1, 10),
            "cat1": ["A", "B"] * 5
        })
        df.to_csv(self.design_csv, index=False)
        
        self.context = Context(output_root=self.test_dir)
        self.config = {
            "step1": {
                "design_csv": str(self.design_csv),
                "n_subjects": 2,
                "trials_per_subject": 5,
                "interaction_mode": "free",
                "skip_interaction": True,
                "merge": False,
                "auto_confirm": True
            },
            "step1_5": {
                "seed": 42,
                "population_mean": 0.0,
                "population_std": 0.1,
                "individual_std_percent": 0.5,
                "output_type": "continuous",
                "likert_levels": 5,
                "likert_mode": "tanh",
                "likert_sensitivity": 2.0,
                "interaction_scale": 0.25,
                "response_col": "y"
            },
            "step2": {
                "subject_col": "subject_id",
                "response_col": "y",
                "max_pairs": 2,
                "min_pairs": 1,
                "selection_method": "elbow",
                "lambda_adjustment": 1.0
            },
            "step3": {
                "max_iters": 10,
                "learning_rate": 0.1,
                "use_cuda": False,
                "ensure_diversity": True,
                "subject_col": "subject_id",
                "response_col": "y"
            }
        }

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_full_flow(self):
        # Step 1
        s1 = Step1Module()
        self.context = s1.run(self.config, self.context)
        self.assertEqual(len(self.context.subject_files), 2)
        
        # Step 1.5
        s15 = Step1_5Module()
        self.context = s15.run(self.config, self.context)
        for f in self.context.subject_files:
            df = pd.read_csv(f)
            self.assertIn('y', df.columns)
            
        # Step 2
        s2 = Step2Module()
        self.context = s2.run(self.config, self.context)
        self.assertIn('lambda_init', self.context.analysis_results)
        
        # Step 3
        s3 = Step3Module()
        self.context = s3.run(self.config, self.context)
        self.assertTrue(Path(self.context.model_path).exists())

if __name__ == "__main__":
    unittest.main()
