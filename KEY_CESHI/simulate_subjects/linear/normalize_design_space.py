#!/usr/bin/env python3
"""Normalize numeric columns (excluding mapped categorical columns) in a design-space CSV and save results.

Usage:
  python normalize_design_space.py --input data/your_mapped.csv --output data/your_mapped_normalized.csv
"""
from pathlib import Path
import argparse
import json
import pandas as pd


def normalize_design_space(input_path: Path, output_path: Path, info_path: Path = None, min_unique_to_normalize: int = 3, method: str = 'zscore', verbose=True):
    df = pd.read_csv(input_path)

    # Try to load mappings to detect categorical columns
    mappings = {}
    mapping_candidate = input_path.with_name(input_path.stem.replace('_mapped','') + '_mappings.json')
    alt_candidate = input_path.with_name(input_path.stem + '_mappings.json')
    if mapping_candidate.exists():
        with open(mapping_candidate, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        if verbose:
            print(f"Loaded mappings from: {mapping_candidate}")
    elif alt_candidate.exists():
        with open(alt_candidate, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        if verbose:
            print(f"Loaded mappings from: {alt_candidate}")
    else:
        if verbose:
            print("No mappings file found; treating all columns as numeric")

    normalize_info = {}
    # Normalize numeric columns only (columns not present in mappings)
    for col in df.columns:
        if col in mappings:
            if verbose:
                print(f"Skipping categorical-mapped column '{col}'")
            continue
        # Skip normalization for low-cardinality numeric columns to preserve numeric meaning
        unique_vals = df[col].nunique()
        if unique_vals < min_unique_to_normalize:
            if verbose:
                print(f"Skipping column '{col}' (unique values={unique_vals} < {min_unique_to_normalize}); preserving original numeric values")
            continue

        if method == 'minmax':
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            rng = col_max - col_min
            if rng > 0:
                df[col] = (df[col].astype(float) - col_min) / rng
                normalize_info[col] = {'method': 'minmax', 'min': col_min, 'max': col_max}
                if verbose:
                    print(f"MinMax normalized column '{col}' (min={col_min}, max={col_max})")
            else:
                if verbose:
                    print(f"Column '{col}' has zero range; left unchanged")

        elif method == 'zscore':
            mean = float(df[col].astype(float).mean())
            std = float(df[col].astype(float).std(ddof=0))
            if std > 0:
                df[col] = (df[col].astype(float) - mean) / std
                normalize_info[col] = {'method': 'zscore', 'mean': mean, 'std': std}
                if verbose:
                    print(f"Z-score normalized column '{col}' (mean={mean}, std={std})")
            else:
                if verbose:
                    print(f"Column '{col}' has zero std; left unchanged")

        else:
            raise ValueError(f"Unknown normalization method: {method}")

    df.to_csv(output_path, index=False)

    if info_path:
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(normalize_info, f, indent=2, ensure_ascii=False)

    return normalize_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=False)
    parser.add_argument('--info', required=False)
    parser.add_argument('--min-unique-to-normalize', type=int, default=3, help='Minimum number of unique values a numeric column must have to be normalized')
    parser.add_argument('--method', choices=['minmax', 'zscore'], default='zscore', help='Normalization method to apply to numeric columns')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + f'_normalized_{args.method}.csv')
    info_path = Path(args.info) if args.info else input_path.with_name(input_path.stem + f'_normalize_info_{args.method}.json')

    info = normalize_design_space(input_path, output_path, info_path, min_unique_to_normalize=args.min_unique_to_normalize, method=args.method)
    print('\nSaved normalized design space to:', output_path)
    print('Saved normalize info to:', info_path)

if __name__ == '__main__':
    main()
