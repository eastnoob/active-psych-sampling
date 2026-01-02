#!/usr/bin/env python3
"""Generate a cluster of linear subjects and save specs/CSVs.

Usage:
    python generate_cluster.py --n-subjects 50 --out-dir ./runs/run1
"""
from pathlib import Path
import argparse
import time
import json
import numpy as np
import sys

# Ensure project root is on sys.path so subject_simulator_v2 (in tools/) can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
# Add tools/ to sys.path so `subject_simulator_v2` (in tools/) is importable
TOOLS_PATH = PROJECT_ROOT / 'tools'
sys.path.insert(0, str(TOOLS_PATH))

from subject_simulator_v2 import ClusterGenerator

# Local helper to save design_space

def load_or_generate_design_space(design_space_path: str = None, n_points=50, n_features=6, seed=42):
    """Load design space from CSV or generate a numeric design space.

    If CSV contains categorical columns, map them to integer codes and return
    (design_space_np, mappings_dict, df).
    """
    import pandas as pd

    if design_space_path:
        df = pd.read_csv(design_space_path)

        mappings = {}
        # Convert categorical/object columns to integer codes
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                uniques = sorted(df[col].unique())
                mapping = {str(val): idx for idx, val in enumerate(uniques)}
                mappings[col] = mapping
                df[col] = df[col].map(mapping)

        return df.values.astype(float), mappings, df

    # fallback: random numeric design space
    np.random.seed(seed)
    arr = np.random.uniform(0, 1, size=(n_points, n_features))
    df = pd.DataFrame(arr)
    return arr, {}, df


def main():
    parser = argparse.ArgumentParser(description="Generate a cluster of linear subjects")
    parser.add_argument("--n-subjects", type=int, default=50)
    parser.add_argument("--population-std", type=float, default=0.25)
    parser.add_argument("--individual-std-percent", type=float, default=0.4,
                        help="Proportion of population std used as individual std (0.3-0.5 realistic)")
    parser.add_argument("--interaction-pairs", type=str, default="3,4;0,1",
                        help="Semicolon-separated pairs like '3,4;0,1'")
    parser.add_argument("--out-dir", type=str, default=str(Path(__file__).parent / "runs" / time.strftime("%Y%m%d%H%M%S")))
    parser.add_argument("--design-space", type=str, default=None, help="Path to design space CSV to use instead of generating one")
    parser.add_argument("--jitter-sd", type=float, default=0.0, help='Relative jitter sd (fraction of column range), e.g. 0.01')
    parser.add_argument("--likert-levels", type=int, default=5, help='Number of Likert levels for simulated subjects (e.g., 5 or 6)')
    parser.add_argument("--likert-sensitivity", type=float, default=2.0, help='Sensitivity for mapping continuous outputs to likert levels')
    parser.add_argument("--normalize", action='store_true', help='Normalize numeric columns (non-mapped) to [0,1] or zscore before generation')
    parser.add_argument("--normalize-method", choices=['minmax','zscore'], default='zscore', help='Normalization method to use when --normalize is specified')
    parser.add_argument("--min-unique-to-normalize", type=int, default=3, help='Minimum unique values required to normalize a numeric column')
    # Normality check parameters
    parser.add_argument('--normality-min-coverage', type=int, default=3, help='min coverage for validators.check_normality')
    parser.add_argument('--normality-max-single-ratio', type=float, default=0.6, help='max single-level ratio for validators.check_normality')
    parser.add_argument('--normality-mean-range', type=str, default=None, help='mean range tuple for validators.check_normality, e.g. 2.0,4.0. If omitted we compute it from --likert-levels (center +/- 1)')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build or load design space
    if args.design_space:
        design_space_np, mappings, df = load_or_generate_design_space(design_space_path=args.design_space)

        # If mappings were not detected from the CSV (e.g., already mapped CSV),
        # try loading a companion mappings JSON file next to the CSV. We try a few
        # candidate names to handle suffixes like '_mapped'.
        if not mappings:
            ds_path = Path(args.design_space)
            dirp = ds_path.parent
            stem = ds_path.stem
            tried = []
            # Candidate 1: exact stem + '_mappings.json'
            candidate = dirp / f"{stem}_mappings.json"
            tried.append(candidate)
            if not candidate.exists() and stem.endswith('_mapped'):
                candidate2 = dirp / f"{stem.replace('_mapped','')}_mappings.json"
                tried.append(candidate2)
                candidate = candidate2 if candidate2.exists() else candidate
            # Fallback: any *_mappings.json in same dir that contains base stem without suffix
            if not candidate.exists():
                for p in dirp.glob('*_mappings.json'):
                    # If original stem (or trimmed stem) is substring of filename, prefer it
                    if stem.replace('_mapped','') in p.stem:
                        candidate = p
                        tried.append(p)
                        break
            if candidate.exists():
                with open(candidate, 'r', encoding='utf-8') as f:
                    mappings = json.load(f)
                print(f"Loaded existing mappings from: {candidate}")
            else:
                print(f"No companion mappings file found (tried: {tried})")

        # Normalize numeric columns if requested (do this before jitter)
        normalize_info = {}
        if args.normalize:
            numeric_cols = [col for col in df.columns if col not in mappings]
            for col in numeric_cols:
                unique_vals = df[col].nunique()
                if unique_vals < args.min_unique_to_normalize:
                    print(f"Skipping normalization for '{col}' (unique={unique_vals} < {args.min_unique_to_normalize}) to preserve numeric meaning")
                    continue

                if args.normalize_method == 'minmax':
                    col_min = float(df[col].min())
                    col_max = float(df[col].max())
                    rng = col_max - col_min
                    if rng > 0:
                        df[col] = (df[col].astype(float) - col_min) / rng
                        normalize_info[col] = {'method': 'minmax', 'min': col_min, 'max': col_max}

                elif args.normalize_method == 'zscore':
                    mean = float(df[col].astype(float).mean())
                    std = float(df[col].astype(float).std(ddof=0))
                    if std > 0:
                        df[col] = (df[col].astype(float) - mean) / std
                        normalize_info[col] = {'method': 'zscore', 'mean': mean, 'std': std}

            if normalize_info:
                print(f"Normalized columns: {list(normalize_info.keys())}")

        # Apply jitter if requested (only to numeric columns, excluding mapped categorical columns)
        if args.jitter_sd and args.jitter_sd > 0.0:
            applied = []
            # columns not in mappings are treated as numeric (safe to jitter)
            numeric_cols = [col for col in df.columns if col not in mappings]
            for col in numeric_cols:
                col_max = df[col].max()
                col_min = df[col].min()
                col_range = col_max - col_min
                sigma = args.jitter_sd * col_range if col_range > 0 else 0.0
                if sigma > 0:
                    noise = np.random.normal(0, sigma, size=len(df))
                    df[col] = df[col].astype(float) + noise
                    applied.append((col, float(sigma)))
            if applied:
                print(f"Applied jitter (sd factor={args.jitter_sd}) to columns: {applied}")

        # Save the processed (numeric) design space
        df.to_csv(out_path / "design_space.csv", index=False)
        # Save mappings if any
        if mappings:
            with open(out_path / "design_space_mappings.json", 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)
        design_space_np = df.values.astype(float)
    else:
        design_space_np, _, df = load_or_generate_design_space(n_points=50, n_features=6, seed=args.seed)
        # if jitter requested, apply to all columns
        if args.jitter_sd and args.jitter_sd > 0.0:
            applied = []
            for col in df.columns:
                col_max = df[col].max()
                col_min = df[col].min()
                col_range = col_max - col_min
                sigma = args.jitter_sd * col_range if col_range > 0 else 0.0
                if sigma > 0:
                    df[col] = df[col].astype(float) + np.random.normal(0, sigma, size=len(df))
                    applied.append((col, float(sigma)))
            if applied:
                print(f"Applied jitter (sd factor={args.jitter_sd}) to generated numeric columns: {applied}")
        df.to_csv(out_path / "design_space.csv", index=False)
    design_space = design_space_np
    # Parse interaction pairs
    pairs = []
    if args.interaction_pairs:
        for token in args.interaction_pairs.split(';'):
            token = token.strip()
            if not token:
                continue
            i, j = token.split(',')
            pairs.append((int(i), int(j)))

    population_std = float(args.population_std)
    individual_std = population_std * float(args.individual_std_percent)

    # parse mean range for normality check
    if args.normality_mean_range:
        mr = tuple([float(s) for s in args.normality_mean_range.split(',')])
    else:
        # compute symmetric default around the likert midpoint (center +/- 1)
        center = (int(args.likert_levels) + 1.0) / 2.0
        mr = (center - 1.0, center + 1.0)
        print(f"Computed normality_mean_range={mr} from likert_levels={args.likert_levels}")

    gen = ClusterGenerator(
        design_space=design_space,
        n_subjects=int(args.n_subjects),
        population_mean=0.0,
        population_std=population_std,
        individual_std=individual_std,
        interaction_pairs=pairs,
        interaction_scale=0.25,
        bias=0.0,
        noise_std=0.0,
        likert_levels=int(args.likert_levels),
        likert_sensitivity=float(args.likert_sensitivity) if hasattr(args, 'likert_sensitivity') else 2.0,
        ensure_normality=True,
        max_retries=20,
        normality_min_coverage=int(args.normality_min_coverage),
        normality_max_single_ratio=float(args.normality_max_single_ratio),
        normality_mean_range=mr,
        seed=int(args.seed),
    )

    result = gen.generate_cluster(str(out_path))

    # Save run metadata
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_subjects": args.n_subjects,
        "population_std": population_std,
        "individual_std": individual_std,
        "interaction_pairs": [f"{i},{j}" for (i, j) in pairs],
        "seed": args.seed,
        "jitter_sd": float(args.jitter_sd),
        "normalize": bool(args.normalize),
        "normalize_method": args.normalize_method if 'normalize_method' in args else None,
        "normalize_info": normalize_info if 'normalize_info' in locals() else {},
        "normality_min_coverage": int(args.normality_min_coverage),
        "normality_max_single_ratio": float(args.normality_max_single_ratio),
        "normality_mean_range": mr,
        "likert_levels": int(args.likert_levels),
        "likert_sensitivity": float(args.likert_sensitivity),
    }

    with open(out_path / "run_meta.json", 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print("Cluster generated at:", out_path)


if __name__ == '__main__':
    main()
