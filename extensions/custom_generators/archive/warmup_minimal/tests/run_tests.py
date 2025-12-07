import sys
import os

# Allow running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

import torch
from extensions.custom_generators.warmup_minimal import WarmupMinimalGenerator


def matrix_rank(X: torch.Tensor) -> int:
    return int(torch.linalg.matrix_rank(X).item())


def to_coded(X: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    """Map values back to {-1,0,+1} using midpoints as thresholds (approximate).
    Center rows map to 0.
    """
    mid = (lb + ub) / 2.0
    coded = torch.sign(
        torch.round((X - mid.unsqueeze(0)) / ((ub - lb).unsqueeze(0) / 2.0))
    )
    # clamp to -1..+1
    coded = torch.clamp(coded, -1, 1)
    return coded


def assert_within_bounds(points: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor):
    assert torch.all(
        points >= lb
    ), f"Points below lb: {points.min().item()} < {lb.min().item()}"
    assert torch.all(
        points <= ub
    ), f"Points above ub: {points.max().item()} > {ub.max().item()}"


def test_case(k: int, lb: float = 0.0, ub: float = 1.0, add_center: bool = True):
    lb_vec = torch.full((k,), lb, dtype=torch.float32)
    ub_vec = torch.full((k,), ub, dtype=torch.float32)

    gen = WarmupMinimalGenerator(lb=lb_vec, ub=ub_vec, add_center=add_center)

    # Drain all points
    out = []
    while not gen.finished:
        out.append(gen.gen(1))
    X = torch.cat(out, dim=0)

    # Basic checks
    assert X.shape[0] >= (
        4
        if k <= 3
        else (8 if k <= 7 else (12 if k <= 12 else (16 if k + 1 <= 16 else 20)))
    )
    assert X.shape[1] == k
    assert_within_bounds(X, lb_vec, ub_vec)

    # Rank check for non-center rows
    coded = to_coded(X, lb_vec, ub_vec)
    # drop any center rows (zeros)
    mask_noncenter = coded.abs().sum(dim=1) > 0
    coded_nc = coded[mask_noncenter]
    r = matrix_rank(coded_nc)
    assert r >= min(
        coded_nc.shape[0], k
    ), f"Rank too low: {r} vs expected {min(coded_nc.shape[0], k)}"


def main():
    # Small k
    test_case(2, add_center=True)
    test_case(3, add_center=True)
    # Mid k (OA 8-run)
    test_case(5, add_center=True)
    # PB12 path
    test_case(10, add_center=True)
    # 16-run path (if k+1 <= 16)
    test_case(15, add_center=False)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
