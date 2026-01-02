#!/usr/bin/env python3
"""Map categorical columns in a design-space CSV to integer codes and save results.

Usage:
  python map_design_space.py --input data/your.csv --output data/your_mapped.csv
"""
from pathlib import Path
import argparse
import json
import pandas as pd

def map_design_space(input_path: Path, output_path: Path, mappings_path: Path = None, verbose=True):
    df = pd.read_csv(input_path)
    mappings = {}
    for col in df.columns:
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
            uniques = sorted(df[col].unique())
            mapping = {str(val): idx for idx, val in enumerate(uniques)}
            mappings[col] = mapping
            df[col] = df[col].map(lambda v: mapping.get(str(v), None))
            if verbose:
                print(f"Mapped column '{col}': {len(uniques)} categories -> 0..{len(uniques)-1}")
    df.to_csv(output_path, index=False)
    if mappings_path:
        with open(mappings_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
    return mappings


def main():
    parser = argparse.ArgumentParser(description="Map categorical columns to integer codes")
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output', required=False, help='Path to output mapped CSV')
    parser.add_argument('--mappings', required=False, help='Path to save mappings JSON')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + '_mapped.csv')
    mappings_path = Path(args.mappings) if args.mappings else input_path.with_name(input_path.stem + '_mappings.json')

    mappings = map_design_space(input_path, output_path, mappings_path)
    print('\nSaved mapped design space to:', output_path)
    print('Saved mappings to:', mappings_path)

if __name__ == '__main__':
    main()
