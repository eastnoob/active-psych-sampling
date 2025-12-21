#!/usr/bin/env python3
"""Grid search over population_std, individual_std_percent, jitter_sd to minimize fallback fraction.
Outputs a run folder with per-run summaries and a grid summary JSON.
"""
from pathlib import Path
import time
import json
import numpy as np
import sys

# Make sure subject_simulator_v2 is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOLS_PATH = PROJECT_ROOT / 'tools'
sys.path.insert(0, str(TOOLS_PATH))

from subject_simulator_v2 import ClusterGenerator


def run_grid(design_space_csv, out_base, population_stds, individual_pct, jitter_sds, n_subjects=50, seed=20251221, likert_levels=5, likert_sensitivity=2.0, normality_min_coverage=3, normality_max_single_ratio=0.6, normality_mean_range=None):
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    # load design space
    import pandas as pd
    df = pd.read_csv(design_space_csv)
    design_space = df.values.astype(float)

    grid_results = []
    run_idx = 0
    timestamp = time.strftime('%Y%m%d%H%M%S')
    grid_dir = out_base / f'grid_{timestamp}'
    grid_dir.mkdir(parents=True)

    for pop in population_stds:
        for ind_pct in individual_pct:
            for jitter in jitter_sds:
                run_idx += 1
                out_dir = grid_dir / f'run_{run_idx:03d}_pop{pop}_indpct{ind_pct}_j{jitter}'
                out_dir.mkdir(parents=True, exist_ok=True)

                individual_std = pop * float(ind_pct)

                # adapt mean_range if not supplied
                if normality_mean_range is None:
                    center = (float(likert_levels) + 1.0) / 2.0
                    mr = (center - 1.0, center + 1.0)
                else:
                    mr = tuple([float(s) for s in str(normality_mean_range).split(',')])

                gen = ClusterGenerator(
                    design_space=design_space,
                    n_subjects=n_subjects,
                    population_mean=0.0,
                    population_std=float(pop),
                    individual_std=float(individual_std),
                    interaction_pairs=[(3,4), (0,1)],
                    interaction_scale=0.25,
                    bias=0.0,
                    noise_std=0.0,
                    likert_levels=int(likert_levels),
                    likert_sensitivity=float(likert_sensitivity),
                    ensure_normality=True,
                    max_retries=20,
                    normality_min_coverage=int(normality_min_coverage),
                    normality_max_single_ratio=float(normality_max_single_ratio),
                    normality_mean_range=mr,
                    seed=int(seed),
                )

                # if jitter > 0, apply jitter to a copy of df
                df_work = df.copy()
                if jitter and jitter > 0:
                    # apply jitter to numeric columns only (non-mapped assumed to be numeric)
                    for col in df_work.columns:
                        try:
                            vals = df_work[col].astype(float)
                        except Exception:
                            continue
                        col_range = float(vals.max() - vals.min())
                        sigma = float(jitter) * col_range if col_range > 0 else 0.0
                        if sigma > 0:
                            df_work[col] = vals + np.random.RandomState(seed).normal(0, sigma, size=len(vals))
                    ds = df_work.values.astype(float)
                    gen.design_space = ds

                res = gen.generate_cluster(str(out_dir))

                # compute fallback fraction
                pop_weights = res['population_weights']
                subject_files = sorted((out_dir).glob('subject_*_spec.json'))
                fallback = 0
                for sf in subject_files:
                    spec = json.load(open(sf, 'r', encoding='utf-8'))
                    w = np.array(spec['weights'])
                    if np.allclose(w, pop_weights, atol=1e-8):
                        fallback += 1

                fallback_frac = fallback / float(n_subjects)

                run_summary = {
                    'run_idx': run_idx,
                    'out_dir': str(out_dir),
                    'population_std': float(pop),
                    'individual_std_percent': float(ind_pct),
                    'individual_std': float(individual_std),
                    'jitter_sd': float(jitter),
                    'fallback_count': int(fallback),
                    'fallback_frac': float(fallback_frac),
                }

                print('RUN', run_idx, run_summary)
                grid_results.append(run_summary)

                # save per-run summary
                with open(out_dir / 'grid_run_summary.json', 'w', encoding='utf-8') as f:
                    json.dump(run_summary, f, indent=2)

    # save grid summary
    grid_summary = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'design_space': str(design_space_csv),
        'n_subjects': n_subjects,
        'grid_results': grid_results
    }
    with open(grid_dir / 'grid_summary.json', 'w', encoding='utf-8') as f:
        json.dump(grid_summary, f, indent=2)

    # sort and return
    sorted_runs = sorted(grid_results, key=lambda x: x['fallback_frac'])
    return grid_dir, sorted_runs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--design-space', required=True)
    parser.add_argument('--out-base', required=False, default=str(Path(__file__).parent / 'runs'))
    parser.add_argument('--n-subjects', type=int, default=50)
    parser.add_argument('--likert-levels', type=int, default=6)
    parser.add_argument('--likert-sensitivity', type=float, default=2.0)
    parser.add_argument('--normality-min-coverage', type=int, default=2)
    parser.add_argument('--normality-max-single-ratio', type=float, default=0.75)
    parser.add_argument('--normality-mean-range', type=str, default=None, help='Comma separated, e.g. 2.0,4.0; if omitted uses center +/- 1')
    args = parser.parse_args()

    # grid (adjust ranges if you want broader search)
    population_stds = [0.35, 0.4, 0.45]
    individual_pct = [0.3, 0.4, 0.5]
    jitter_sds = [0.0, 0.005, 0.01]

    grid_dir, sorted_runs = run_grid(
        design_space_csv=args.design_space,
        out_base=args.out_base,
        population_stds=population_stds,
        individual_pct=individual_pct,
        jitter_sds=jitter_sds,
        n_subjects=args.n_subjects,
        seed=20251221,
        likert_levels=args.likert_levels,
        likert_sensitivity=args.likert_sensitivity,
        normality_min_coverage=args.normality_min_coverage,
        normality_max_single_ratio=args.normality_max_single_ratio,
        normality_mean_range=args.normality_mean_range
    )

    print('\nBest runs:')
    for r in sorted_runs[:5]:
        print(r)

    print('\nFull grid saved to:', grid_dir)
