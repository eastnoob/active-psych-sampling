import numpy as np
import json

with open(
    "tests/is_EUR_work/00_plans/251206/scripts/results/20251210_185924_test_balance_full/data_files/summary.json",
    "r",
) as f:
    summary1 = json.load(f)

# Load second run (fixed)
with open(
    "tests/is_EUR_work/00_plans/251206/scripts/results/20251210_191336_test_balance_reproducible_fixed/data_files/summary.json",
    "r",
) as f:
    summary2 = json.load(f)

print("=== First Run (20251210_185924) ===")
print(
    f'Lambda stats: min={summary1["lambda_statistics"]["min"]:.3f}, max={summary1["lambda_statistics"]["max"]:.3f}, mean={summary1["lambda_statistics"]["mean"]:.3f}'
)
print(
    f'Gamma stats: min={summary1["gamma_statistics"]["min"]:.3f}, max={summary1["gamma_statistics"]["max"]:.3f}, mean={summary1["gamma_statistics"]["mean"]:.3f}'
)

print("\n=== Second Run (20251210_191336_fixed) ===")
print(
    f'Lambda stats: min={summary2["lambda_statistics"]["min"]:.3f}, max={summary2["lambda_statistics"]["max"]:.3f}, mean={summary2["lambda_statistics"]["mean"]:.3f}'
)
print(
    f'Gamma stats: min={summary2["gamma_statistics"]["min"]:.3f}, max={summary2["gamma_statistics"]["max"]:.3f}, mean={summary2["gamma_statistics"]["mean"]:.3f}'
)

# Load sampling histories
history1 = np.load(
    "tests/is_EUR_work/00_plans/251206/scripts/results/20251210_185924_test_balance_full/data_files/sampling_history.npy"
)
history2 = np.load(
    "tests/is_EUR_work/00_plans/251206/scripts/results/20251210_191336_test_balance_reproducible_fixed/data_files/sampling_history.npy"
)

print(f"\n=== Sampling History Comparison ===")
print(f"First run shape: {history1.shape}")
print(f"Second run shape: {history2.shape}")

# Check if arrays are close (accounting for floating point precision)
close = np.allclose(history1, history2, rtol=1e-6, atol=1e-8)
print(f"Sampling histories identical (within tolerance): {close}")

if not close:
    # Find which samples differ
    diff_mask = ~np.isclose(history1, history2, rtol=1e-6, atol=1e-8)
    diff_samples = np.any(diff_mask, axis=1)
    diff_indices = np.where(diff_samples)[0]
    print(f"Number of different samples: {len(diff_indices)}")
    if len(diff_indices) > 0:
        print("First few differences:")
        for i in diff_indices[:3]:
            print(f"  Sample {i}: {history1[i]} vs {history2[i]}")
else:
    print("✓ Perfect reproducibility achieved!")
