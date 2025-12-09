import json
from pathlib import Path
import tempfile
import pandas as pd
import sys

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from extensions.warmup_budget_check.core.phase1_step3_base_gp_v2 import (
    process_step3,
)


def test_smart_factor_selection(tmp_path: Path):
    # Prepare a simple sample dataset with x1..x3 and a response
    df = pd.DataFrame(
        {
            "x1": [0, 1, 0, 1],
            "x2": [1, 2, 1, 2],
            "x3": [0.0, 0.25, 0.5, 0.75],
        }
    )
    df["subject_id"] = ["s1", "s1", "s2", "s2"]
    df["response"] = df["x1"] + df["x2"] + df["x3"]
    data_csv = tmp_path / "phase1_test.csv"
    df.to_csv(data_csv, index=False)

    # Create a simple design_space csv (matching columns x1..x3)
    design_df = pd.DataFrame({"x1": [0, 1], "x2": [1, 2], "x3": [0.0, 0.75]})
    design_csv = tmp_path / "design_space_test.csv"
    design_df.to_csv(design_csv, index=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    res = process_step3(
        data_csv_path=str(data_csv),
        design_space_csv=str(design_csv),
        subject_col="subject_id",
        response_col="response",
        output_dir=str(out_dir),
        model_type="continuous",
        max_iters=1,
        use_cuda=False,
    )

    # Check that out file contains factor names matching x1,x2,x3
    lengthscales_json = out_dir / "base_gp_lengthscales.json"
    assert lengthscales_json.exists()
    payload = json.loads(lengthscales_json.read_text(encoding="utf-8"))
    assert payload["factor_names"] == ["x1", "x2", "x3"]
