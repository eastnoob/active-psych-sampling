import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_design_space_module():
    # Ascend until we reach repo root named "active-psych-sampling" to be resilient to pytest rootdir
    path = Path(__file__).resolve()
    repo_root = next(p for p in path.parents if p.name == "active-psych-sampling")
    module_path = (
        repo_root
        / "tests"
        / "is_EUR_work"
        / "00_plans"
        / "251206"
        / "scripts"
        / "modules"
        / "design_space.py"
    )
    spec = importlib.util.spec_from_file_location("design_space", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transform_to_numeric_maps_to_indices():
    design_space = pd.DataFrame(
        {
            "x1_CeilingHeight": [2.8, 4.0, 8.5],
            "x2_GridModule": [6.5, 8.0, 6.5],
            "x3_OuterFurniture": ["Chaos", "Rotated", "Strict"],
            "x4_VisualBoundary": ["Color", "Solid", "Translucent"],
            "x5_PhysicalBoundary": ["Closed", "Open", "Closed"],
            "x6_InnerFurniture": ["Chaos", "Rotated", "Strict"],
        }
    )

    design_space_module = _load_design_space_module()
    numeric = design_space_module.transform_to_numeric(design_space)

    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [2, 0, 2, 2, 0, 2],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(numeric, expected)


def test_transformed_values_within_valid_index_ranges():
    design_space = pd.DataFrame(
        {
            "x1_CeilingHeight": [2.8, 4.0, 8.5],
            "x2_GridModule": [6.5, 8.0, 6.5],
            "x3_OuterFurniture": ["Chaos", "Rotated", "Strict"],
            "x4_VisualBoundary": ["Color", "Solid", "Translucent"],
            "x5_PhysicalBoundary": ["Closed", "Open", "Closed"],
            "x6_InnerFurniture": ["Chaos", "Rotated", "Strict"],
        }
    )

    design_space_module = _load_design_space_module()
    numeric = design_space_module.transform_to_numeric(design_space)

    # Each column should be integer indices and within its categorical bounds.
    allowed = [
        {0, 1, 2},  # x1
        {0, 1},  # x2
        {0, 1, 2},  # x3
        {0, 1, 2},  # x4
        {0, 1},  # x5
        {0, 1, 2},  # x6
    ]

    for col_idx, allowed_set in enumerate(allowed):
        col = numeric[:, col_idx]
        assert np.all(col == col.astype(int)), f"column {col_idx} not integral: {col}"
        assert set(col.tolist()).issubset(
            allowed_set
        ), f"column {col_idx} out of range: {col}"
