#!/usr/bin/env python3
"""Test effect of likert_sensitivity on cluster generation normality checks.

Produces a small JSON summary for each sensitivity value.
"""
from pathlib import Path
import json
import numpy as np
import sys

# Ensure tools/ is importable (same approach as other scripts)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOLS_PATH = PROJECT_ROOT / 'tools'
sys.path.insert(0, str(TOOLS_PATH))

from subject_simulator_v2 import ClusterGenerator


def run_sensitivity_test(design_space_csv, sensitivities, n_subjects=20, out_dir_base=None, seed=20251221):
    design_space = np.loadtxt(design_space_csv, delimiter=',', skiprows=1)

    results = {}

    for sens in sensitivities:
        out_dir = Path(out_dir_base) / f"sensitivity_{sens}"
        out_dir.mkdir(parents=True, exist_ok=True)

        gen = ClusterGenerator(
            design_space=design_space,
            n_subjects=n_subjects,
            population_mean=0.0,
            population_std=0.25,
            individual_std=0.25 * 0.4,
            interaction_pairs=[(3,4), (0,1)],
            interaction_scale=0.25,
            bias=0.0,
            noise_std=0.0,
            likert_levels=5,
            likert_sensitivity=float(sens),
            ensure_normality=True,
            max_retries=20,
            seed=seed,
        )

        res = gen.generate_cluster(str(out_dir))

        # Load cluster summary and subject specs
        with open(out_dir / 'cluster_summary.json', 'r', encoding='utf-8') as f:
            summary = json.load(f)

        pop_weights = np.array(summary['population_weights'])

        subject_files = sorted(list(out_dir.glob('subject_*_spec.json')))
        fallback_count = 0
        coverages = []
        max_ratios = []
        means = []

        for sf in subject_files:
            with open(sf, 'r', encoding='utf-8') as fh:
                spec = json.load(fh)
            w = np.array(spec['weights'])
            # fallback if weights equal population weights
            if np.allclose(w, pop_weights, atol=1e-8):
                fallback_count += 1
            stats = spec.get('response_statistics', {})
            coverages.append(stats.get('coverage', 0))
            max_ratios.append(stats.get('max_ratio', 0.0))
            means.append(stats.get('mean', 0.0))

        results[str(sens)] = {
            'fallback_count': fallback_count,
            'n_subjects': n_subjects,
            'fallback_fraction': fallback_count / n_subjects,
            'mean_coverage': float(np.mean(coverages)) if coverages else None,
            'mean_max_ratio': float(np.mean(max_ratios)) if max_ratios else None,
            'mean_response_mean': float(np.mean(means)) if means else None,
            'out_dir': str(out_dir)
        }

    # Save results
    out_json = Path(out_dir_base) / 'sensitivity_scan_results.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--design-space', required=True)
    parser.add_argument('--out-dir', required=False, default=str(Path(__file__).parent / 'runs' / 'sensitivity_test'))
    parser.add_argument('--n-subjects', type=int, default=20)
    args = parser.parse_args()

    sensitivities = [0.5, 1.0, 2.0, 4.0]
    results = run_sensitivity_test(
        design_space_csv=args.design_space,
        sensitivities=sensitivities,
        n_subjects=args.n_subjects,
        out_dir_base=args.out_dir,
    )

    print(json.dumps(results, indent=2))
