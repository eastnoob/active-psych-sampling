#!/usr/bin/env python3
"""Inspect normality check results for a generated run.
Reads subject CSVs (subject_*.csv) in a run directory and reports
how many failed and why according to validators.check_normality.
"""
from pathlib import Path
import json
import numpy as np
import sys

# ensure tools on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOLS_PATH = PROJECT_ROOT / 'tools'
sys.path.insert(0, str(TOOLS_PATH))

from subject_simulator_v2.validators import check_normality, get_distribution_stats


def inspect_run(run_dir: str):
    p = Path(run_dir)
    if not p.exists():
        raise SystemExit(f"Run dir not found: {p}")

    csvs = sorted(p.glob('subject_*.csv'))
    total = 0
    failed = 0
    reasons = {}
    coverages = []
    max_ratios = []
    means = []

    for csv in csvs:
        total += 1
        import pandas as pd
        df = pd.read_csv(csv)
        if 'y' not in df.columns:
            continue
        responses = df['y'].dropna().astype(int).tolist()
        res = check_normality(responses)
        if not res['passed']:
            failed += 1
            reasons.setdefault(res['reason'], 0)
            reasons[res['reason']] += 1
        coverages.append(res.get('coverage', 0))
        max_ratios.append(res.get('max_ratio', 0.0))
        means.append(res.get('mean', 0.0))

    summary = {
        'run_dir': str(p),
        'n_subjects': total,
        'n_failed': failed,
        'failed_frac': failed / total if total else None,
        'failure_reasons': reasons,
        'mean_coverage': float(np.mean(coverages)) if coverages else None,
        'median_coverage': float(np.median(coverages)) if coverages else None,
        'mean_max_ratio': float(np.mean(max_ratios)) if max_ratios else None,
        'mean_response_mean': float(np.mean(means)) if means else None,
    }

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    args = parser.parse_args()
    inspect_run(args.run_dir)
