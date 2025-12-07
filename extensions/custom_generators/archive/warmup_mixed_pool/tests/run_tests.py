import sys
import os

# Allow running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import torch
from extensions.custom_generators.warmup_mixed_pool import WarmupMixedPoolGenerator


def build_mixed_pool(N: int = 300):
    # Mixed variables: [continuous, integer, ordinal(3 levels), categorical(4 levels), boolean]
    cont = torch.rand(N, 1) * 10.0 - 5.0  # [-5,5]
    integer = torch.randint(low=0, high=6, size=(N, 1)).to(torch.float32)  # 0..5
    ordinal = torch.randint(low=1, high=4, size=(N, 1)).to(torch.float32)  # 1..3
    categorical = torch.randint(low=0, high=4, size=(N, 1)).to(
        torch.float32
    )  # 0..3 as levels
    boolean = torch.randint(low=0, high=2, size=(N, 1)).to(torch.float32)  # 0/1
    P = torch.cat([cont, integer, ordinal, categorical, boolean], dim=1)

    schema = [
        {"type": "continuous", "lb": -5.0, "ub": 5.0},
        {"type": "integer", "lb": 0.0, "ub": 5.0},
        {"type": "ordinal", "levels": [1, 2, 3]},
        {"type": "categorical", "levels": [0, 1, 2, 3]},
        {"type": "boolean"},
    ]
    return P, schema


def matrix_rank(X: torch.Tensor) -> int:
    return int(torch.linalg.matrix_rank(X).item())


def test_mixed_pool_basic():
    P, schema = build_mixed_pool(240)
    gen = WarmupMixedPoolGenerator(
        pool_points=P, schema=schema, n_runs=None, add_center=True
    )
    # Drain
    pts = []
    while not gen.finished:
        pts.append(gen.gen(5))
    Xsel = torch.cat(pts, dim=0)

    # Size should be >= columns (main effects + intercept)
    # Rebuild model matrix to check rank
    X_all, _ = gen._build_main_effect_matrix(P, schema)
    # Shape of selected
    idx = gen._selected_indices
    X_sub = X_all[idx]
    r = matrix_rank(X_sub)

    assert Xsel.shape[0] == idx.numel()
    assert (
        r == X_sub.shape[1]
    ), f"Selected design not full rank: rank {r} vs p {X_sub.shape[1]}"

    # Uniqueness and pool membership
    # Compare rows presence via exact match (here numeric)
    # Build a set of tuples for selected
    sel_set = {tuple(row.tolist()) for row in Xsel}
    assert len(sel_set) == Xsel.shape[0]


if __name__ == "__main__":
    test_mixed_pool_basic()
    print("ALL TESTS PASSED (mixed pool)")
