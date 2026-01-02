#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path


def summarize(run_dir):
    p = Path(run_dir)
    with open(p / 'cluster_summary.json', 'r', encoding='utf-8') as f:
        summary = json.load(f)
    pop_weights = np.array(summary['population_weights'])
    specs = sorted(list(p.glob('subject_*_spec.json')))
    n = len(specs)
    fallback = 0
    coverages = []
    max_ratios = []
    means = []
    for s in specs:
        data = json.load(open(s, 'r', encoding='utf-8'))
        w = np.array(data['weights'])
        if np.allclose(w, pop_weights, atol=1e-8):
            fallback += 1
        stats = data.get('response_statistics', {})
        coverages.append(stats.get('coverage', 0))
        max_ratios.append(stats.get('max_ratio', 0.0))
        means.append(stats.get('mean', 0.0))
    return {
        'run': str(p),
        'n_subjects': n,
        'fallback_count': fallback,
        'fallback_frac': fallback / n if n else None,
        'mean_coverage': float(np.mean(coverages)) if coverages else None,
        'mean_max_ratio': float(np.mean(max_ratios)) if max_ratios else None,
        'mean_response_mean': float(np.mean(means)) if means else None,
    }


if __name__ == '__main__':
    runs = [
        'KEY_CESHI/simulate_subjects/linear/runs/50_variance_nojitter',
        'KEY_CESHI/simulate_subjects/linear/runs/50_variance_jitter3',
        'KEY_CESHI/simulate_subjects/linear/runs/50_variance_norm_jitter',
    ]
    for r in runs:
        print(json.dumps(summarize(r), indent=2))
