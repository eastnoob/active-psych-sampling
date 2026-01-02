#!/usr/bin/env python3
"""Monte Carlo sampling of individual deviations to estimate validation failure rates and reasons."""
from pathlib import Path
import json
import numpy as np
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
TOOLS_PATH = PROJECT_ROOT / 'tools'
sys.path.insert(0, str(TOOLS_PATH))

from subject_simulator_v2.validators import check_normality
from subject_simulator_v2.linear import LinearSubject


def load_design_space(path):
    import pandas as pd
    df = pd.read_csv(path)
    return df.values.astype(float), df


def run_sim(design_space_csv, pop_weights, interaction_weights, bias=0.0, noise_std=0.0, likert_levels=5, likert_sensitivity=2.0, individual_std=0.25, n_draws=1000, seed=0):
    ds, df = load_design_space(design_space_csv)
    rng = np.random.RandomState(seed)
    reasons = {}
    passed = 0
    for i in range(n_draws):
        dev = rng.normal(0, individual_std, size=len(pop_weights))
        weights = pop_weights + dev
        subj = LinearSubject(weights=weights, interaction_weights=interaction_weights, bias=bias, noise_std=noise_std, likert_levels=likert_levels, likert_sensitivity=likert_sensitivity, seed=None)
        responses = [subj(x) for x in ds]
        val = check_normality(responses)
        if val['passed']:
            passed += 1
        else:
            reasons[val['reason']] = reasons.get(val['reason'], 0) + 1
    return {'n_draws': n_draws, 'passed': passed, 'failed': n_draws-passed, 'reasons': reasons}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--design-space', required=True)
    parser.add_argument('--pop-summary', required=True, help='Path to cluster_summary.json to get population_weights and interactions')
    parser.add_argument('--individual-std', type=float, default=0.25)
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()

    s = json.load(open(args.pop_summary, 'r', encoding='utf-8'))
    pop_weights = np.array(s['population_weights'])
    inter = {tuple(map(int,k.split(','))):v for k,v in s['interaction_weights'].items()}

    res = run_sim(args.design_space, pop_weights, inter, individual_std=args.individual_std, n_draws=args.n)
    print(json.dumps(res, indent=2))
