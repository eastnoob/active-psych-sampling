#!/usr/bin/env python3
"""
Run simulation on sample subject CSVs and write results to 'result' folder.

Usage:
    ap.add_argument(
        "--individual_corr",
        required=False,
        type=float,
        default=0.0,
        help=(
            "If set, introduces feature-wise correlation of individual deviations (0 no-corr). "
            "Value must be between -1/(p-1) and 1 for stability (p=number of features)."
        ),
    )
  python run_simulation.py --input_dir ".../202511161637" --seed 42

This script will:
- Read subject_*.csv
- Convert categorical columns to numeric
- Create a shared population_weights for all subjects
- Instantiate a SingleOutputLatentSubject per subject (unique seed per subject)
- Simulate y for each row and write a subject_X_result.csv containing a new column 'y'
- Also write combined_results.csv aggregating all subjects
"""

# ---------------------------------------------------------------------------
# Quick configuration block: mimic `quick_start.py` so you can edit these
# values at top-of-file and re-run this script directly.
# ---------------------------------------------------------------------------
# MODE: 'step1' (generate samples), 'step2' (simulate with existing samples),
# or 'both' (generate samples then simulate).
MODE = "both"

# Sampling config (Step 1)
STEP1_CONFIG = {
    "design_csv_path": r"D:\WORKSPACE\python\aepsych-source\data\only_independences\data\only_independences\6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
    "n_subjects": 5,  # 202511172026 has 5 subjects
    "trials_per_subject": 25,
    "skip_interaction": True,
    "output_dir": r"extensions\warmup_budget_check\sample\202511172026",
    "merge": False,
}

# Simulation config (Step 2) - controls how this script simulates y
STEP2_CONFIG = {
    "output_mode": "combined",  # individual|combined|both
    "use_latent": False,
    "output_type": "likert",  # continuous|likert
    "likert_levels": 5,
    "likert_mode": "percentile",  # tanh|percentile
    "likert_sensitivity": 0.5,
    "clean": True,
    "population_std": 0.4,  # ← 增加权重的群体分布宽度（从0.05到0.4）
    "population_mean": 0.0,
    "individual_std_percent": 1.0,  # ← 个体偏差 = 1.0 * 0.4 = 0.4
    # 交互效应参数
    "interaction_pairs": [(3, 4), (0, 1), (1, 3)],  # 显式指定交互对索引
    "num_interactions": 0,  # 额外随机生成的交互项数量
    "interaction_scale": 0.4,  # 交互项权重的尺度
}

# ---------------------------------------------------------------------------

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import os

# Use pandas when available for easier IO; fallback to csv module
try:
    import pandas as pd
except Exception:
    pd = None


def import_subject_module():
    """Import MixedEffectsLatentSubject robustly by searching upward for the project root.

    Walk parents until a `subject` directory is found, then add it to sys.path.
    """

    try:
        from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject

        return MixedEffectsLatentSubject
    except Exception:
        pass

    # Walk upwards until we find a 'subject' directory or reach root
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "subject").is_dir():
            sys.path.insert(0, str(p))
            try:
                from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject

                return MixedEffectsLatentSubject
            except Exception:
                # continue searching in case of partial matches
                continue

    # Try models location: tools/archive/simulate_subject/models (推荐最新版本)
    for p in [here] + list(here.parents):
        models_subject = p / "tools" / "archive" / "simulate_subject" / "models"
        if models_subject.is_dir():
            sys.path.insert(0, str(models_subject))
            try:
                from subject_models.MixedEffectsLatentSubject import MixedEffectsLatentSubject
                return MixedEffectsLatentSubject
            except Exception:
                pass

    # Fallback: Try older archive location
    for p in [here] + list(here.parents):
        archive_subject = p / "tools" / "archive" / "simulate_subject" / "archive"
        if archive_subject.is_dir():
            sys.path.insert(0, str(archive_subject))
            try:
                from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject
                return MixedEffectsLatentSubject
            except Exception:
                pass

    # Last attempt: add repository root by walking up to find it
    here = Path(__file__).resolve()
    # From core/simulation_runner.py -> core -> warmup_budget_check -> extensions -> aepsych-source
    # Try different levels
    for level in range(2, 8):
        try:
            fallback = here.parents[level]
            if (fallback / "subject").is_dir():
                sys.path.insert(0, str(fallback))
                from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject
                return MixedEffectsLatentSubject
        except (IndexError, ImportError):
            continue

    # If still not found, raise error
    raise ImportError("无法找到subject.MixedEffectsLatentSubject模块。请确保项目根目录包含subject/文件夹")


# Import subject module robustly
MixedEffectsLatentSubject = import_subject_module()

# Import our single output wrapper
try:
    from single_output_subject import SingleOutputLatentSubject
except Exception:
    # If the module path isn't found, add this file's directory to path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from single_output_subject import SingleOutputLatentSubject


# Column mapping for categorical features
X4_MAP = {"low": 0, "mid": 1, "high": 2, "max": 3}
X5_MAP = {"A": 0, "B": 1, "C": 2}

# Column mappings for i9csy65bljq14ovww2v91 format
X3_OUTER_MAP = {"Strict": 0, "Rotated": 1, "Chaos": 2}
X4_VISUAL_MAP = {"Solid": 0, "Translucent": 1, "Color": 2}
X5_PHYSICAL_MAP = {"Closed": 0, "Open": 1}
X6_INNER_MAP = {"Strict": 0, "Rotated": 1, "Chaos": 2}


def convert_row_to_features(row):
    """
    Convert a CSV row (dict or pd.Series) to a numeric feature vector X with 6 features.
    Auto-detects format based on column names.

    Supports two formats:
    1. i9csy65bljq14ovww2v91: x1_CeilingHeight, x2_GridModule, x3_OuterFurniture,
                               x4_VisualBoundary, x5_PhysicalBoundary, x6_InnerFurniture
    2. 6vars: x1_binary, x2_5level_discrete, x3_5level_decimal,
              x4_4level_categorical, x5_3level_categorical, x6_binary
    """

    # Accept both pandas Series and dict
    def val(k):
        return row[k]

    # Detect format based on available columns
    if hasattr(row, 'index'):
        columns = list(row.index)
    else:
        columns = list(row.keys())

    if 'x1_CeilingHeight' in columns:
        # Format 1: i9csy65bljq14ovww2v91
        x1 = float(val("x1_CeilingHeight"))
        x2 = float(val("x2_GridModule"))

        x3 = val("x3_OuterFurniture")
        if isinstance(x3, str):
            x3 = X3_OUTER_MAP.get(x3, 0)
        else:
            x3 = float(x3)

        x4 = val("x4_VisualBoundary")
        if isinstance(x4, str):
            x4 = X4_VISUAL_MAP.get(x4, 0)
        else:
            x4 = float(x4)

        x5 = val("x5_PhysicalBoundary")
        if isinstance(x5, str):
            x5 = X5_PHYSICAL_MAP.get(x5, 0)
        else:
            x5 = float(x5)

        x6 = val("x6_InnerFurniture")
        if isinstance(x6, str):
            x6 = X6_INNER_MAP.get(x6, 0)
        else:
            x6 = float(x6)

    elif 'x1_binary' in columns:
        # Format 2: 6vars
        x1 = int(val("x1_binary"))
        x2 = float(val("x2_5level_discrete"))
        x3 = float(val("x3_5level_decimal"))

        x4 = val("x4_4level_categorical")
        if isinstance(x4, str):
            x4 = X4_MAP.get(x4, 0)
        else:
            x4 = float(x4)

        x5 = val("x5_3level_categorical")
        if isinstance(x5, str):
            x5 = X5_MAP.get(x5, 0)
        else:
            x5 = float(x5)

        x6 = val("x6_binary")
        if isinstance(x6, str):
            x6 = x6.strip().lower() in {"true", "1", "t", "yes"}
        x6 = 1 if bool(x6) else 0
    else:
        raise ValueError(f"无法识别CSV格式，列名：{columns}")

    return np.array([x1, x2, x3, x4, x5, x6], dtype=float)


def read_csv_to_dataframe(path: Path):
    if pd is not None:
        return pd.read_csv(path)
    # Fallback naive reader
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def write_result_df(df, path: Path):
    if pd is not None and hasattr(df, "to_csv"):
        df.to_csv(path, index=False)
    else:
        # df is list of dicts
        if not df:
            return
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(df[0].keys()))
            writer.writeheader()
            for r in df:
                writer.writerow(r)


def run(
    input_dir: Path,
    seed: int = 42,
    output_mode: str = "individual",
    clean: bool = False,
    interaction_pairs=None,
    num_interactions: int = 0,
    interaction_scale: float = 1.0,
    # Additional parameters from STEP1_5_CONFIG
    use_latent: bool = False,
    output_type: str = "continuous",
    likert_levels: int = 5,
    likert_mode: str = "tanh",
    likert_sensitivity: float = 1.0,
    population_mean: float = 0.0,
    population_std: float = 0.05,
    individual_std_percent: float = 0.8,
    individual_corr: float = 0.0,
    fixed_weights_file: str = None,
    # Model display parameters (new)
    print_model: bool = False,
    save_model_summary: bool = False,
    model_summary_format: str = "txt",
):
    assert input_dir.exists(), f"Input dir does not exist: {input_dir}"

    # Create a local args object from function parameters (for compatibility)
    class Args:
        pass

    args = Args()
    args.seed = seed
    args.output_mode = output_mode
    args.use_latent = str(use_latent).lower()
    args.output_type = output_type
    args.likert_levels = likert_levels
    args.likert_mode = likert_mode
    args.likert_sensitivity = likert_sensitivity
    args.population_mean = population_mean
    args.population_std = population_std
    args.individual_std_percent = individual_std_percent
    args.individual_corr = individual_corr
    args.fixed_weights_file = fixed_weights_file
    args.interaction_pairs = interaction_pairs
    args.num_interactions = num_interactions
    args.interaction_scale = interaction_scale

    # DEBUG: Print interaction parameters
    print(f"[DEBUG] run() called with:")
    print(f"  - interaction_pairs: {interaction_pairs}")
    print(f"  - num_interactions: {num_interactions}")
    print(f"  - interaction_scale: {interaction_scale}")
    print(f"[DEBUG] args object attributes:")
    print(
        f"  - args.interaction_pairs: {getattr(args, 'interaction_pairs', 'NOT SET')}"
    )
    print(f"  - args.num_interactions: {getattr(args, 'num_interactions', 'NOT SET')}")
    print(
        f"  - args.interaction_scale: {getattr(args, 'interaction_scale', 'NOT SET')}"
    )

    result_dir = input_dir / "result"
    result_dir.mkdir(exist_ok=True)

    # Optionally clean previous generated result artifacts
    if clean:
        patterns = [
            "subject_*_result.csv",
            "subject_*_model.md",
            "combined_results.csv",
            "model_spec.txt",
            "subjects_parameters_summary.md",
        ]
        for pat in patterns:
            for p in result_dir.glob(pat):
                try:
                    p.unlink()
                    print("Removed", p.name)
                except Exception:
                    print("Could not remove", p)

    # Find subject CSV files
    csvs = sorted(list(input_dir.glob("subject_*.csv")))

    if not csvs:
        print("No subject CSV files found in", input_dir)
        return

    # NOTE: Each subject will generate its own population_weights (群体固定效应)
    # and individual_deviation (随机效应) independently using their unique seed.
    # This ensures all 5 subjects have DIFFERENT parameters despite sharing the same
    # population distribution (population_mean, population_std, individual_std_percent).
    #
    # For reference, create a "base" instance just to get default parameters
    base = MixedEffectsLatentSubject(
        num_features=6,
        num_observed_vars=1,
        seed=seed,
        population_mean=args.population_mean,
        population_std=args.population_std,
        individual_std_percent=args.individual_std_percent,
        individual_corr=args.individual_corr,
    )
    print(
        f"Each subject will independently generate population_weights and individual_deviations."
    )
    print(
        f"Population distribution: mean={args.population_mean}, std={args.population_std}"
    )
    print(
        f"Individual std: {args.individual_std_percent * args.population_std:.4f} (percent={args.individual_std_percent})"
    )

    # Fixed weights import (optional)
    fixed_weights_map = None
    if args.fixed_weights_file:
        fw_path = Path(args.fixed_weights_file)
        if not fw_path.exists():
            print("Fixed weights file not found:", fw_path)
            fw_path = None
        else:
            if fw_path.suffix.lower() in {".json"}:
                import json

                with fw_path.open("r", encoding="utf-8") as jf:
                    data = json.load(jf)
                # Support dict of subject -> weights, or direct weights array
                if isinstance(data, dict):
                    fixed_weights_map = data
                else:
                    fixed_weights_map = {"global": data}
            elif fw_path.suffix.lower() in {".npy"}:
                fixed_weights_map = {"global": np.load(str(fw_path)).tolist()}
            else:
                print("Unsupported fixed_weights file type, use .json or .npy")
                fixed_weights_map = None
    else:
        # If not provided, create a default global fixed_weights using the seed for reproducibility
        default_fw = (
            np.random.RandomState(seed)
            .uniform(
                -base.weight_range,
                base.weight_range,
                size=(base.num_observed_vars or 1, base.num_features),
            )
            .tolist()
        )
        default_path = result_dir / "fixed_weights_auto.json"
        import json

        with default_path.open("w", encoding="utf-8") as jf:
            json.dump({"global": default_fw}, jf)
        fixed_weights_map = {"global": default_fw}
        print("Wrote default fixed_weights to", default_path)

    combined_rows = []
    subject_specs = []
    header_written = False
    subject_outputs = {}

    for idx, csv_path in enumerate(csvs, start=1):
        df = read_csv_to_dataframe(csv_path)
        using_pandas = pd is not None and isinstance(df, pd.DataFrame)

        # Determine subject seed offset for independent parameter generation
        subject_seed = seed + idx

        # In deterministic mode with individual differences: generate subject-specific fixed_weights
        # by adding individual deviations to the base fixed_weights
        subject_fixed_weights = None
        if (
            args.use_latent.lower() == "false"
            and args.individual_std_percent is not None
            and args.individual_std_percent > 0
        ):
            # Generate subject-specific fixed weights with individual deviations
            rng_subj = np.random.RandomState(subject_seed)
            base_fw = (
                fixed_weights_map.get("global")
                if fixed_weights_map is not None
                else None
            )
            if base_fw is None:
                base_fw = [
                    [0.1 * i - 0.25 for i in range(6)] for _ in range(1)
                ]  # fallback

            # Add individual deviations to fixed weights
            base_fw_arr = np.array(base_fw, dtype=float)
            indiv_std_val = args.individual_std_percent * args.population_std
            deviation = rng_subj.normal(0, indiv_std_val, size=base_fw_arr.shape)
            subject_fixed_weights = base_fw_arr + deviation

        subj = SingleOutputLatentSubject(
            num_features=6,
            seed=subject_seed,
            # Don't pass base's params; let each subject independently sample from distributions
            individual_std=0.0,  # Will be overridden by individual_std_percent in _initialize_parameters
            noise_std=base.noise_std,
            population_weights=None,  # Force each subject to generate their own
            weight_range=base.weight_range,
            num_latent_vars=base.num_latent_vars,
            factor_loadings=None,  # Let subject generate its own
            item_biases=None,  # Let subject generate its own
            item_noises=None,  # Let subject generate its own
            population_mean=args.population_mean,
            population_std=args.population_std,
            individual_std_percent=args.individual_std_percent,
            individual_corr=args.individual_corr,
            use_latent=(args.use_latent.lower() == "true"),
            fixed_weights=subject_fixed_weights,  # Pass subject-specific weights
            # 交互效应参数
            interaction_pairs=getattr(args, "interaction_pairs", None),
            num_interactions=getattr(args, "num_interactions", 0),
            interaction_scale=getattr(args, "interaction_scale", 1.0),
            # Likert levels only used when output_type=likert AND mode is 'tanh'
            likert_levels=(
                args.likert_levels
                if args.output_type == "likert"
                and getattr(args, "likert_mode", "tanh") == "tanh"
                else None
            ),
            likert_sensitivity=(
                args.likert_sensitivity
                if args.output_type == "likert"
                and getattr(args, "likert_mode", "tanh") == "tanh"
                else None
            ),
        )

        # DEBUG: Print interaction parameters passed to SingleOutputLatentSubject
        if idx == 1:
            print(
                f"[DEBUG] Subject {idx} - interaction_pairs: {getattr(args, 'interaction_pairs', None)}"
            )

        # Prepare output list
        out_rows = []

        if using_pandas:
            for _, row in df.iterrows():
                X = convert_row_to_features(row)
                y = subj(X)
                out_rows.append(dict(**row.to_dict(), y=y))
        else:
            for row in df:
                X = convert_row_to_features(row)
                y = subj(X)
                out_rows.append({**row, "y": y})

        # Save per-subject outputs into a dict for optional post-processing
        subject_outputs[csv_path.stem] = out_rows

        # Add to combined
        combined_rows.extend([{"subject": csv_path.stem, **r} for r in out_rows])
        # Save this subject's model spec (in markdown for readability)
        spec = subj.get_model_spec()
        subject_specs.append(
            {
                "name": csv_path.stem,
                "individual_deviation": np.array(
                    spec.get("individual_deviation") or []
                ),
                "population_weights": np.array(spec.get("population_weights") or []),
                "fixed_weights": np.array(spec.get("fixed_weights") or []),
            }
        )
        spec_md_path = result_dir / f"{csv_path.stem}_model.md"
        with spec_md_path.open("w", encoding="utf-8") as md:
            md.write(f"# Model spec for {csv_path.stem}\n\n")
            md.write(f"- seed: {subject_seed}\n")
            md.write(f"- num_features: {spec.get('num_features')}\n")
            md.write(f"- use_latent: {spec.get('use_latent', True)}\n")
            if spec.get("use_latent", True):
                md.write(f"- num_latent_vars: {spec.get('num_latent_vars')}\n")
            md.write(f"- individual_std: {spec.get('individual_std')}\n")
            md.write(f"- weight_range: {spec.get('weight_range')}\n\n")

            # If using latent variables, show population/individual decompositions
            if spec.get("use_latent", True):
                md.write("## Population Weights (fixed effects)\n\n")
                for lv_idx, weights in enumerate(
                    spec.get("population_weights") or [], start=1
                ):
                    md.write(f"### Latent variable {lv_idx}\n\n")
                    for feat_idx, w in enumerate(weights, start=1):
                        md.write(f"- x{feat_idx}: {w:.5f}\n")
                    md.write("\n")

                md.write("## Individual Deviation (random effects)\n\n")
                for lv_idx, devs in enumerate(
                    spec.get("individual_deviation") or [], start=1
                ):
                    md.write(f"### Latent variable {lv_idx}\n\n")
                    for feat_idx, d in enumerate(devs, start=1):
                        md.write(f"- x{feat_idx}: {d:+.5f}\n")
                    md.write("\n")

                md.write("## Individual Weights (population + deviation)\n\n")
                for lv_idx, weights in enumerate(
                    spec.get("latent_weights") or [], start=1
                ):
                    md.write(f"### Latent variable {lv_idx}\n\n")
                    for feat_idx, w in enumerate(weights, start=1):
                        md.write(f"- x{feat_idx}: {w:.5f}\n")
                    md.write("\n")
            else:
                md.write("## Fixed Weights (deterministic outputs)\n\n")
                fixed = spec.get("fixed_weights") or []
                for obs_idx, weights in enumerate(fixed, start=1):
                    md.write(f"### Observed var {obs_idx}\n\n")
                    for feat_idx, w in enumerate(weights, start=1):
                        md.write(f"- x{feat_idx}: {w:.5f}\n")
                    md.write("\n")

        print(f"Wrote model description for {csv_path.name} -> {spec_md_path.name}")

    # If percentile mapping is requested for Likert, remap y across combined data
    if (
        args.output_type == "likert"
        and getattr(args, "likert_mode", "tanh") == "percentile"
    ):
        all_y = [float(r.get("y")) for r in combined_rows]
        if len(all_y) > 0:
            levels = int(args.likert_levels)
            cuts = [
                np.percentile(all_y, pct)
                for pct in np.linspace(0, 100, levels + 1)[1:-1]
            ]

            def perc_map(y):
                idx = int(np.digitize([y], cuts)[0])
                return idx + 1

            for r in combined_rows:
                r["y"] = perc_map(float(r.get("y")))

            # propagate to individual subject outputs
            for k, rows in subject_outputs.items():
                for r in rows:
                    r["y"] = perc_map(float(r.get("y")))

    # Conditionally write combined file
    combined_path = result_dir / "combined_results.csv"
    # Optionally write per-subject files (after mapping, if any)
    if output_mode in ("individual", "both"):
        for subj_name, rows in subject_outputs.items():
            out_path = result_dir / f"{subj_name}_result.csv"
            write_result_df(rows, out_path)
            print(f"Wrote result for {subj_name} -> {out_path.name}")
    if output_mode in ("combined", "both"):
        write_result_df(combined_rows, combined_path)
        print("Combined results written to", combined_path)
    else:
        if combined_path.exists():
            try:
                combined_path.unlink()
            except Exception:
                pass

    # Also save model spec for the final subject used (mostly for reproducibility)
    spec_path = result_dir / "model_spec.txt"
    with spec_path.open("w", encoding="utf-8") as sf:
        sf.write(str(subj.get_model_spec()))
    print("Saved model spec to", spec_path)

    # Summarize parameter differences across subjects
    summary_md = result_dir / "subjects_parameters_summary.md"
    with summary_md.open("w", encoding="utf-8") as sm:
        sm.write("# Subjects parameter summary\n\n")
        sm.write(
            "This file summarizes whether subjects share population weights and whether individual deviations differ.\n\n"
        )

        # Compare population weights
        sm.write("## Population weights comparison\n\n")
        # Determine whether latent model was used for any subject by checking for non-empty individual deviations
        has_latent = any(
            s.get("individual_deviation") is not None
            and s.get("individual_deviation").size
            for s in subject_specs
        )
        has_fixed = any(
            s.get("fixed_weights") is not None and s.get("fixed_weights").size
            for s in subject_specs
        )
        if has_fixed:
            sm.write(
                "This simulation used deterministic fixed-weight outputs for subjects (no latent variables).\n\n"
            )
        elif has_latent:
            sm.write(
                "Population weights are shared across subjects in this simulation by design.\n\n"
            )
        elif has_fixed:
            sm.write(
                "This simulation used deterministic fixed-weight outputs for subjects (no latent variables).\n\n"
            )
        else:
            sm.write("No population weights or fixed weights detected.\n\n")

        # Compare individual deviations and show if unique
        sm.write("## Individual deviations check\n\n")
        sm.write(
            "Below is a list of per-subject `individual_deviation` filenames and whether they are identical across all subjects.\n\n"
        )

        # Collect deviations
        deviations = []
        subj_names = []
        for p in sorted(list(result_dir.glob("subject_*_model.md"))):
            subj_names.append(p.stem.replace("_model", ""))
            # read and store numeric deviations by parsing get_model_spec instead
            # read original model_spec file for this subject
            try:
                # get spec by reading JSON-like repr from subj.get_model_spec if present
                # we have individual md with items; instead simply record file path
                deviations.append(p)
            except Exception:
                deviations.append(p)

        for i, nm in enumerate(subj_names, start=1):
            sm.write(f"- {nm}: {deviations[i-1].name}\n")

        sm.write("\n## Pairwise subject deviation distances\n\n")
        # Compute Euclidean distances between flattened deviation arrays
        import math

        # Prefer individual_deviation (latent) for distance; fallback to fixed_weights
        devs = []
        for s in subject_specs:
            iv = s.get("individual_deviation")
            if iv is not None and iv.size:
                devs.append(iv)
                continue
            fw = s.get("fixed_weights")
            if fw is not None and fw.size:
                devs.append(fw)
            else:
                devs.append(None)
        usable = [d for d in devs if d is not None]
        if usable and len(usable) > 1:
            flat = [d.flatten() for d in usable]
            n = len(flat)
            sm.write("Pairwise Euclidean distances between individual deviations:\n\n")
            sm.write("| Subject A | Subject B | Distance |\n")
            sm.write("| --- | --- | ---: |\n")
            # Map subjects with usable vectors
            names = [
                s["name"]
                for s in subject_specs
                if s.get("individual_deviation") is not None
                or s.get("fixed_weights") is not None
            ]
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(np.linalg.norm(flat[i] - flat[j]))
                    sm.write(f"| {names[i]} | {names[j]} | {dist:.6f} |\n")
        else:
            sm.write("Not enough subjects for pairwise distance summary.\n")

        sm.write("\n## Quick conclusion\n\n")
        if has_fixed:
            sm.write(
                "- Subjects use deterministic fixed_weights; pairwise distances above are computed across fixed_weights when latent weights are not available.\n"
            )
        elif has_latent:
            sm.write(
                "- Each subject receives the same population_weights; the individual deviations are generated using different seeds so subjects differ in their latent weights.\n"
            )
        else:
            sm.write("- No per-subject deviations detected.\n")

    print("Wrote subjects parameters summary to", summary_md)

    # If Likert output was used, save a small distribution report for y
    if args.output_type == "likert":
        from collections import Counter

        y_vals = []
        for r in combined_rows:
            try:
                yv = int(float(r.get("y")))
            except Exception:
                yv = r.get("y")
            y_vals.append(yv)

        counter = Counter(y_vals)
        dist_path = result_dir / "likert_distribution.txt"
        with dist_path.open("w", encoding="utf-8") as df:
            df.write("Likert distribution summary\n")
            df.write("========================\n\n")
            for level in sorted(counter.keys()):
                df.write(f"Level {level}: {counter[level]}\n")
            df.write("\nTotal: %d\n" % (sum(counter.values())))

        print("Wrote Likert distribution summary to", dist_path)

    # Generate comprehensive simulation report/documentation
    report_path = result_dir / "SIMULATION_REPORT.md"
    with report_path.open("w", encoding="utf-8") as rp:
        rp.write("# Simulation Report\n\n")
        rp.write(
            f"**Generated:** {__import__('datetime').datetime.now().isoformat()}\n\n"
        )

        # Configuration summary
        rp.write("## Configuration Summary\n\n")
        rp.write("### Step 1: Sampling Configuration\n\n")
        rp.write(f"- Design CSV path: `{STEP1_CONFIG.get('design_csv_path')}`\n")
        rp.write(f"- Number of subjects: {STEP1_CONFIG.get('n_subjects')}\n")
        rp.write(f"- Trials per subject: {STEP1_CONFIG.get('trials_per_subject')}\n")
        rp.write(
            f"- Skip interaction (in budget evaluation): {STEP1_CONFIG.get('skip_interaction', False)}\n"
        )
        rp.write(f"- Output directory: `{STEP1_CONFIG.get('output_dir')}`\n")
        rp.write(
            f"- Merge subjects in sampling: {STEP1_CONFIG.get('merge', False)}\n\n"
        )

        rp.write("### Step 2: Simulation Configuration\n\n")
        rp.write(f"- Random seed: {args.seed}\n")
        rp.write(f"- Output mode: {args.output_mode}\n")
        rp.write(f"- Use latent variables: {args.use_latent}\n")
        rp.write(f"- Output type: {args.output_type}\n")
        if args.output_type == "likert":
            rp.write(f"- Likert levels: {args.likert_levels}\n")
            rp.write(f"- Likert mapping mode: {args.likert_mode}\n")
            rp.write(f"- Likert sensitivity: {args.likert_sensitivity}\n")
        rp.write(f"- Population mean: {args.population_mean}\n")
        rp.write(f"- Population std: {args.population_std}\n")
        rp.write(f"- Individual std (percent): {args.individual_std_percent}\n")
        rp.write(f"- Individual correlation: {args.individual_corr}\n")
        if args.fixed_weights_file:
            rp.write(f"- Fixed weights file: `{args.fixed_weights_file}`\n")
        rp.write(f"- Clean prior results: {clean}\n\n")

        # Data summary
        rp.write("## Data Summary\n\n")
        rp.write(f"- Total number of subject files processed: {len(csvs)}\n")
        rp.write(
            f"- Total observations (rows across all subjects): {len(combined_rows)}\n"
        )
        if args.output_type == "likert":
            rp.write(f"- Output range: Likert 1-{args.likert_levels}\n")
        else:
            rp.write("- Output type: Continuous (unbounded)\n")
        rp.write("\n")

        rp.write("### Subject Files\n\n")
        for i, csv_path in enumerate(csvs, start=1):
            df = read_csv_to_dataframe(csv_path)
            n_rows = len(df) if isinstance(df, list) else len(df)
            rp.write(f"- {csv_path.name}: {n_rows} rows\n")
        rp.write("\n")

        # Output artifacts
        rp.write("## Generated Output Files\n\n")
        rp.write("### Result Files\n\n")
        if Path(combined_path).exists():
            rp.write(
                f"- **combined_results.csv**: Aggregated results from all subjects (total: {len(combined_rows)} rows)\n"
            )
        for i, csv_path in enumerate(csvs, start=1):
            result_file = result_dir / f"{csv_path.stem}_result.csv"
            if result_file.exists():
                rows = len(read_csv_to_dataframe(result_file))
                rp.write(
                    f"- **{result_file.name}**: Per-subject results ({rows} rows)\n"
                )
        rp.write("\n")

        rp.write("### Model Documentation Files\n\n")
        for p in sorted(list(result_dir.glob("subject_*_model.md"))):
            rp.write(f"- **{p.name}**: Model specification and weights for subject\n")
        if Path(spec_path).exists():
            rp.write(f"- **model_spec.txt**: Final model specification snapshot\n")
        rp.write("\n")

        rp.write("### Summary & Analysis Files\n\n")
        if Path(summary_md).exists():
            rp.write(
                f"- **subjects_parameters_summary.md**: Summary of subject parameter differences and pairwise distances\n"
            )
        if args.output_type == "likert" and Path(dist_path).exists():
            rp.write(
                f"- **likert_distribution.txt**: Distribution of Likert values across output\n"
            )
        if Path(result_dir / "fixed_weights_auto.json").exists():
            rp.write(
                f"- **fixed_weights_auto.json**: Auto-generated fixed weights (if not imported)\n"
            )
        rp.write(f"- **SIMULATION_REPORT.md**: This file\n\n")

        # Results & statistics
        rp.write("## Results & Statistics\n\n")
        if args.output_type == "likert":
            rp.write("### Likert Distribution\n\n")
            from collections import Counter

            y_vals = [int(float(r.get("y"))) for r in combined_rows]
            counter = Counter(y_vals)
            rp.write("| Likert Level | Count | Percentage |\n")
            rp.write("| --- | ---: | ---: |\n")
            total = sum(counter.values())
            for level in sorted(counter.keys()):
                count = counter[level]
                pct = 100 * count / total if total > 0 else 0
                rp.write(f"| {level} | {count} | {pct:.1f}% |\n")
            rp.write(f"| **Total** | **{total}** | **100%** |\n\n")
        else:
            rp.write("### Continuous Output Statistics\n\n")
            y_vals = [float(r.get("y")) for r in combined_rows]
            rp.write(f"- Mean: {np.mean(y_vals):.6f}\n")
            rp.write(f"- Std Dev: {np.std(y_vals):.6f}\n")
            rp.write(f"- Min: {np.min(y_vals):.6f}\n")
            rp.write(f"- Max: {np.max(y_vals):.6f}\n")
            rp.write(f"- Median: {np.median(y_vals):.6f}\n\n")

        # Subject parameters summary
        rp.write("## Subject Parameters Summary\n\n")
        rp.write(f"- Number of subjects: {len(subject_specs)}\n")
        rp.write(
            f"- Shared population weights: Yes (all subjects use same pop_weights)\n"
        )
        if any(
            s.get("individual_deviation") is not None
            and s.get("individual_deviation").size
            for s in subject_specs
        ):
            rp.write(
                f"- Individual deviations: Present (generated per-subject with unique seeds)\n"
            )
        elif any(
            s.get("fixed_weights") is not None and s.get("fixed_weights").size
            for s in subject_specs
        ):
            rp.write(
                f"- Fixed weights mode: Deterministic outputs (no latent variables)\n"
            )
        else:
            rp.write(
                f"- Individual parameters: Minimal (deterministic mode with no deviations)\n"
            )
        rp.write("\n")

        # Model details section
        rp.write("## Model Details\n\n")
        if args.use_latent == "true":
            rp.write("### Latent Variable Model\n\n")
            rp.write("Each subject has:\n")
            rp.write(
                "- **Population weights**: Shared fixed effects across all subjects\n"
            )
            rp.write(
                "- **Individual deviations**: Subject-specific random effects added to population weights\n"
            )
            rp.write(
                "- **Observed output**: Deterministic mapping from input features through latent variables\n"
            )
            rp.write(
                "- **Noise**: Optional measurement noise added based on noise_std parameter\n\n"
            )
        else:
            rp.write("### Deterministic Fixed-Weight Model\n\n")
            rp.write("Each subject has:\n")
            rp.write(
                "- **Fixed weights**: Direct linear mapping from input features to output\n"
            )
            rp.write(
                "- **No latent variables**: Output is computed as y = X · w (dot product)\n"
            )
            rp.write(
                "- **Per-subject weights**: May vary across subjects if individually specified\n"
            )
            if any(
                s.get("individual_deviation") is not None
                and s.get("individual_deviation").size
                for s in subject_specs
            ):
                rp.write(
                    "- **Individual deviations**: Applied to fixed weights (subject-specific perturbations)\n"
                )
            rp.write("\n")

        # Notes for reproducibility
        rp.write("## Reproducibility Notes\n\n")
        rp.write(
            f"- Random seed: **{args.seed}** (use same seed to reproduce results)\n"
        )
        rp.write(
            f"- Subject-specific seeds: seed + subject_index (e.g., seed={args.seed} → subject_1 uses {args.seed + 1})\n"
        )
        rp.write(
            f"- Fixed weights reproducibility: Auto-generated weights use population seed {args.seed}\n"
        )
        rp.write(
            f"- Percentile binning: Uses global quantiles computed across all {len(combined_rows)} observations\n\n"
        )

        # Footer
        rp.write("---\n\n")
        rp.write(
            "*For detailed model specifications per subject, see `subject_*_model.md` files.*\n"
        )
        rp.write(
            "*For per-subject parameter comparisons, see `subjects_parameters_summary.md`.*\n"
        )

    print("Wrote comprehensive simulation report to", report_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input_dir", required=False, default=".", help="Input folder path"
    )
    ap.add_argument("--seed", required=False, type=int, default=42)
    ap.add_argument(
        "--population_std",
        required=False,
        type=float,
        default=0.05,
        help="If set, population weights are sampled from N(population_mean, population_std)",
    )
    ap.add_argument(
        "--output_mode",
        required=False,
        choices=("individual", "combined", "both"),
        default="individual",
        help=(
            "Control whether to write per-subject results ('individual'), a single combined file "
            "with a subject column ('combined'), or both ('both'). Defaults to 'individual'."
        ),
    )
    ap.add_argument(
        "--clean",
        required=False,
        action="store_true",
        help=(
            "If set, remove previously-generated result files (subject_*.csv, *_model.md, "
            "combined_results.csv, etc.) from the result folder prior to running the simulation."
        ),
    )
    ap.add_argument(
        "--population_mean",
        required=False,
        type=float,
        default=0.0,
        help="Mean for the population weight normal distribution",
    )
    ap.add_argument(
        "--individual_corr",
        required=False,
        type=float,
        default=0.0,
        help=(
            "If set, introduces feature-wise correlation of individual deviations (0 no-corr). "
            "Range ~ (-1/(p-1), 1)."
        ),
    )
    ap.add_argument(
        "--individual_std_percent",
        required=False,
        type=float,
        default=0.8,
        help=(
            "If set, individual std is individual_std_percent * base population std; "
            "if base is not provided, uses std(population_weights)."
        ),
    )
    ap.add_argument(
        "--use_latent",
        required=False,
        choices=("true", "false"),
        default="true",
        help="Whether to use latent variable model (true) or deterministic fixed-weight outputs (false).",
    )
    ap.add_argument(
        "--output_type",
        required=False,
        choices=("continuous", "likert"),
        default="continuous",
        help="If 'likert', map outputs to Likert scale using --likert_levels; otherwise output continuous values.",
    )
    ap.add_argument(
        "--likert_levels",
        required=False,
        type=int,
        default=5,
        help="Number of Likert levels (only used when --output_type=likert).",
    )
    ap.add_argument(
        "--likert_sensitivity",
        required=False,
        type=float,
        default=1.0,
        help=(
            "Sensitivity for Likert mapping; lower values push outputs to extremes, "
            "higher values centralize. Use <1 for broader spread."
        ),
    )
    ap.add_argument(
        "--likert_mode",
        required=False,
        choices=("tanh", "percentile"),
        default="tanh",
        help=(
            "Mapping mode for Likert: 'tanh' uses tanh normalization (default). "
            "'percentile' bins outputs into equal-count buckets across the dataset."
        ),
    )
    ap.add_argument(
        "--interaction_pairs",
        required=False,
        type=str,
        default=None,
        help=(
            "交互项对的索引列表，格式如 '(0,1) (1,2) (3,4)' 或 '[(0,1), (1,2)]'. "
            "若不提供，将使用STEP2_CONFIG中的默认值。"
        ),
    )
    ap.add_argument(
        "--num_interactions",
        required=False,
        type=int,
        default=0,
        help="额外随机生成的交互项数量（默认0）",
    )
    ap.add_argument(
        "--interaction_scale",
        required=False,
        type=float,
        default=1.0,
        help="交互项权重的尺度系数（默认1.0）",
    )
    ap.add_argument(
        "--fixed_weights_file",
        required=False,
        default=None,
        help=(
            "Optional path to a JSON or .npy file with fixed_weights. If supplied and "
            "--use_latent=false, these weights will be used as deterministic outputs. "
            "JSON should contain either a list-of-lists (num_obs x num_features) or "
            "a dict mapping subject stem to list-of-lists."
        ),
    )
    args = ap.parse_args()

    # Allow INDIVIDUAL_CORR to be supplied via env var when CLI parsing doesn't accept it
    env_corr = os.environ.get("INDIVIDUAL_CORR")
    if env_corr is not None:
        try:
            args.individual_corr = float(env_corr)
        except Exception:
            print(
                "Warning: INDIVIDUAL_CORR environment variable could not be parsed as float"
            )

    # If user opted to use top-of-file configuration (quick start style)
    if MODE in {"step1", "both"}:
        # Generate samples using the WarmupSampler
        # Ensure package root for warmup_sampler is on path
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from warmup_sampler import WarmupSampler

        design_csv = STEP1_CONFIG.get("design_csv_path")
        sampler = WarmupSampler(design_csv)
        adequacy, budget = sampler.evaluate_budget(
            n_subjects=STEP1_CONFIG.get("n_subjects"),
            trials_per_subject=STEP1_CONFIG.get("trials_per_subject"),
            skip_interaction=STEP1_CONFIG.get("skip_interaction", False),
        )

        # If budget OK, generate samples
        if adequacy not in ["预算不足", "严重不足"]:
            sampler.generate_samples(
                budget=budget,
                output_dir=STEP1_CONFIG.get("output_dir"),
                merge=STEP1_CONFIG.get("merge", False),
            )
        else:
            print("Budget insufficient; generated no samples. Check STEP1_CONFIG.")

    # Decide input_dir for simulation
    if MODE in {"step2", "both"}:
        sim_input_dir = Path(STEP1_CONFIG.get("output_dir"))
    else:
        sim_input_dir = Path(args.input_dir)

    # Compose final args from Step2 config (allow CLI to override when explicitly provided)
    def _cli_override(arg_name, config_val):
        flag = f"--{arg_name.replace('_','-')}"
        if flag in sys.argv:
            return getattr(args, arg_name)
        return config_val

    final_output_mode = _cli_override("output_mode", STEP2_CONFIG.get("output_mode"))
    final_use_latent = _cli_override("use_latent", STEP2_CONFIG.get("use_latent"))
    final_output_type = _cli_override("output_type", STEP2_CONFIG.get("output_type"))
    final_likert_levels = _cli_override(
        "likert_levels", STEP2_CONFIG.get("likert_levels")
    )
    final_likert_mode = _cli_override("likert_mode", STEP2_CONFIG.get("likert_mode"))
    final_likert_sensitivity = _cli_override(
        "likert_sensitivity", STEP2_CONFIG.get("likert_sensitivity")
    )
    final_clean = _cli_override("clean", STEP2_CONFIG.get("clean"))

    # 处理交互参数
    final_interaction_pairs = _cli_override(
        "interaction_pairs", STEP2_CONFIG.get("interaction_pairs")
    )
    final_num_interactions = _cli_override(
        "num_interactions", STEP2_CONFIG.get("num_interactions", 0)
    )
    final_interaction_scale = _cli_override(
        "interaction_scale", STEP2_CONFIG.get("interaction_scale", 1.0)
    )

    # 解析交互对字符串
    if isinstance(final_interaction_pairs, str):
        try:
            # 尝试eval(支持 "[(0,1), (1,2)]" 格式)
            final_interaction_pairs = eval(final_interaction_pairs)
        except Exception:
            # 尝试解析 "(0,1) (1,2)" 格式
            try:
                pairs_strs = final_interaction_pairs.strip().split()
                final_interaction_pairs = [
                    tuple(map(int, p.strip("()").split(","))) for p in pairs_strs
                ]
            except Exception:
                print(
                    f"Warning: Could not parse interaction_pairs: {final_interaction_pairs}"
                )
                final_interaction_pairs = None

    # Map 'true'/'false' to booleans if value was taken from STEP2_CONFIG
    if isinstance(final_use_latent, str):
        final_use_latent = final_use_latent.lower() == "true"

    # Run simulation using computed values
    # Set args so run() uses the final decisions for mapping
    args.output_mode = final_output_mode
    args.use_latent = str(final_use_latent)
    args.output_type = final_output_type
    args.likert_levels = int(final_likert_levels)
    args.likert_mode = final_likert_mode
    args.likert_sensitivity = float(final_likert_sensitivity)
    args.clean = bool(final_clean)
    args.interaction_pairs = final_interaction_pairs
    args.num_interactions = int(final_num_interactions)
    args.interaction_scale = float(final_interaction_scale)

    run(
        Path(sim_input_dir).resolve(),
        seed=args.seed,
        output_mode=final_output_mode,
        clean=final_clean,
        interaction_pairs=final_interaction_pairs,
        num_interactions=int(final_num_interactions),
        interaction_scale=float(final_interaction_scale),
    )
