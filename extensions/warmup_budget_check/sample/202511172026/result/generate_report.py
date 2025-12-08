#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model Parameter and Error Report Generator
Support both Markdown (md) and Plain Text (txt) formats
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime


def generate_model_report(subjects_info, output_dir, report_format="md"):
    """
    Generate model parameters and error reports for each subject and group

    Inputs:
      subjects_info: list of dict with subject information
        {
          "subject": "subject_1",
          "n_trials": 10,
          "y_mean": 2.5,
          "y_std": 1.2,
          "y_min": 0.5,
          "y_max": 4.8,
          "model_spec": {...}
        }
      output_dir: Path output directory
      report_format: str, "md" (Markdown) or "txt" (plain text), default "md"

    Outputs:
      - model_report.csv - summary table
      - model_report.md or model_report.txt - detailed report
      - group_statistics.md or group_statistics.txt - group statistics
      - individual_specs/ - individual parameter files
    """

    if not subjects_info:
        print("WARNING: No subject information, skipping report generation")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Validate format parameter
    if report_format not in ("md", "txt"):
        report_format = "md"

    print(f"\n[Generating Model Reports] (Format: {report_format.upper()})")

    # ========================================================================
    # 1. Generate summary table (CSV)
    # ========================================================================

    summary_data = []
    for info in subjects_info:
        summary_data.append(
            {
                "Subject": info["subject"],
                "N_Trials": info["n_trials"],
                "Y_Mean": f"{info['y_mean']:.4f}",
                "Y_Std": f"{info['y_std']:.4f}",
                "Y_Min": f"{info['y_min']:.4f}",
                "Y_Max": f"{info['y_max']:.4f}",
            }
        )

    summary_df = pd.DataFrame(summary_data)
    summary_csv = output_dir / "model_report.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"  - {summary_csv.name}")

    # ========================================================================
    # 2. Generate detailed model parameter report
    # ========================================================================

    if report_format == "md":
        _generate_markdown_report(subjects_info, output_dir)
    else:
        _generate_text_report(subjects_info, output_dir)

    # ========================================================================
    # 3. Save individual parameter files
    # ========================================================================

    specs_dir = output_dir / "individual_specs"
    specs_dir.mkdir(exist_ok=True)

    for info in subjects_info:
        spec_file = specs_dir / f"{info['subject']}_spec.json"
        spec = info.get("model_spec", {})

        # Convert numpy arrays for JSON serialization
        spec_serializable = _make_serializable(spec)

        with open(spec_file, "w", encoding="utf-8") as f:
            json.dump(spec_serializable, f, indent=2, ensure_ascii=False)

    print(f"  - individual_specs/ ({len(subjects_info)} files)")

    print("\n[Report Generation Complete]")


def _generate_markdown_report(subjects_info, output_dir):
    """Generate Markdown format reports"""

    # Generate detailed parameter report (Markdown)
    report_path = output_dir / "model_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Model Parameters and Error Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Subjects**: {len(subjects_info)}\n\n")

        f.write("---\n\n")

        # Detailed information for each subject
        for idx, info in enumerate(subjects_info, start=1):
            f.write(f"## Subject {idx}: {info['subject']}\n\n")
            f.write(f"**Total Trials**: {info['n_trials']}\n\n")
            f.write("### Response Statistics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Mean | {info['y_mean']:.6f} |\n")
            f.write(f"| Std Dev | {info['y_std']:.6f} |\n")
            f.write(f"| Min | {info['y_min']:.6f} |\n")
            f.write(f"| Max | {info['y_max']:.6f} |\n")
            f.write(f"| Range | {info['y_max'] - info['y_min']:.6f} |\n\n")

            # Model parameters
            spec = info.get("model_spec", {})
            if spec:
                f.write("### Model Parameters\n\n")

                # Population weights
                if "population_weights" in spec:
                    weights = spec["population_weights"]
                    f.write("#### Population Weights\n\n")
                    if (
                        isinstance(weights, list)
                        and len(weights) > 0
                        and isinstance(weights[0], list)
                    ):
                        for i, w in enumerate(weights[0]):
                            f.write(f"- w{i + 1}: {w:.6f}\n")
                    elif isinstance(weights, list):
                        for i, w in enumerate(weights):
                            f.write(f"- w{i + 1}: {w:.6f}\n")
                    f.write("\n")

                # Individual deviation
                if "individual_deviation" in spec:
                    dev = spec["individual_deviation"]
                    f.write(f"#### Individual Deviation\n\n")
                    f.write(f"```\n{dev}\n```\n\n")

                # Interaction terms
                if "interaction_terms" in spec:
                    interactions = spec["interaction_terms"]
                    if interactions:
                        f.write("#### Interaction Terms\n\n")
                        for k, v in interactions.items():
                            f.write(f"- {k}: {v:.6f}\n")
                        f.write("\n")

                # Noise
                if "noise_std" in spec:
                    f.write(f"#### Noise Std Dev: {spec['noise_std']:.6f}\n\n")

                # Other parameters
                other_params = {
                    k: v
                    for k, v in spec.items()
                    if k
                    not in [
                        "population_weights",
                        "individual_deviation",
                        "interaction_terms",
                        "noise_std",
                    ]
                }
                if other_params:
                    f.write("#### Other Parameters\n\n")
                    for k, v in other_params.items():
                        f.write(f"- {k}: {v}\n")
                    f.write("\n")

            f.write("---\n\n")

    print(f"  - {report_path.name}")

    # Generate group statistics report (Markdown)
    group_stat_path = output_dir / "group_statistics.md"
    with open(group_stat_path, "w", encoding="utf-8") as f:
        f.write("# Group Statistics Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Collect statistics for all subjects
        y_means = [info["y_mean"] for info in subjects_info]
        y_stds = [info["y_std"] for info in subjects_info]
        y_mins = [info["y_min"] for info in subjects_info]
        y_maxs = [info["y_max"] for info in subjects_info]
        y_ranges = [info["y_max"] - info["y_min"] for info in subjects_info]

        f.write("## Summary Statistics\n\n")
        f.write(f"**Number of Subjects**: {len(subjects_info)}\n\n")

        f.write("### Mean\n\n")
        f.write(f"- Group Average: {np.mean(y_means):.6f}\n")
        f.write(f"- Group Std Dev: {np.std(y_means):.6f}\n")
        f.write(f"- Range: [{np.min(y_means):.6f}, {np.max(y_means):.6f}]\n\n")

        f.write("### Std Dev\n\n")
        f.write(f"- Group Average: {np.mean(y_stds):.6f}\n")
        f.write(f"- Group Std Dev: {np.std(y_stds):.6f}\n")
        f.write(f"- Range: [{np.min(y_stds):.6f}, {np.max(y_stds):.6f}]\n\n")

        f.write("### Minimum\n\n")
        f.write(f"- Group Average: {np.mean(y_mins):.6f}\n")
        f.write(f"- Range: [{np.min(y_mins):.6f}, {np.max(y_mins):.6f}]\n\n")

        f.write("### Maximum\n\n")
        f.write(f"- Group Average: {np.mean(y_maxs):.6f}\n")
        f.write(f"- Range: [{np.min(y_maxs):.6f}, {np.max(y_maxs):.6f}]\n\n")

        f.write("### Range\n\n")
        f.write(f"- Group Average: {np.mean(y_ranges):.6f}\n")
        f.write(f"- Range: [{np.min(y_ranges):.6f}, {np.max(y_ranges):.6f}]\n\n")

        f.write("---\n\n")
        f.write("## Subject Comparison\n\n")
        f.write("Ranked by mean response (highest to lowest):\n\n")

        # Rank subjects by mean
        sorted_info = sorted(subjects_info, key=lambda x: x["y_mean"], reverse=True)
        f.write("|Rank|Subject|Mean|Std Dev|Min|Max|\n")
        f.write("|:---:|---|:---:|:---:|:---:|:---:|\n")
        for rank, info in enumerate(sorted_info, start=1):
            f.write(
                f"|{rank}|{info['subject']}|{info['y_mean']:.4f}|{info['y_std']:.4f}|{info['y_min']:.4f}|{info['y_max']:.4f}|\n"
            )

    print(f"  - {group_stat_path.name}")


def _generate_text_report(subjects_info, output_dir):
    """Generate plain text format reports"""

    # Generate detailed parameter report (TXT)
    report_path = output_dir / "model_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Model Parameters and Error Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Subjects: {len(subjects_info)}\n")
        f.write("=" * 80 + "\n\n")

        # Detailed information for each subject
        for idx, info in enumerate(subjects_info, start=1):
            f.write(f"\nSubject {idx}: {info['subject']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Trials: {info['n_trials']}\n")
            f.write(f"\nResponse Statistics:\n")
            f.write(f"  Mean: {info['y_mean']:.6f}\n")
            f.write(f"  Std Dev: {info['y_std']:.6f}\n")
            f.write(f"  Min: {info['y_min']:.6f}\n")
            f.write(f"  Max: {info['y_max']:.6f}\n")
            f.write(f"  Range: {info['y_max'] - info['y_min']:.6f}\n")

            # Model parameters
            spec = info.get("model_spec", {})
            if spec:
                f.write(f"\nModel Parameters:\n")

                # Population weights
                if "population_weights" in spec:
                    weights = spec["population_weights"]
                    f.write(f"  Population Weights:\n")
                    if (
                        isinstance(weights, list)
                        and len(weights) > 0
                        and isinstance(weights[0], list)
                    ):
                        for i, w in enumerate(weights[0]):
                            f.write(f"    w{i + 1}: {w:.6f}\n")
                    elif isinstance(weights, list):
                        for i, w in enumerate(weights):
                            f.write(f"    w{i + 1}: {w:.6f}\n")

                # Individual deviation
                if "individual_deviation" in spec:
                    dev = spec["individual_deviation"]
                    f.write(f"  Individual Deviation: {dev}\n")

                # Interaction terms
                if "interaction_terms" in spec:
                    interactions = spec["interaction_terms"]
                    if interactions:
                        f.write(f"  Interaction Terms:\n")
                        for k, v in interactions.items():
                            f.write(f"    {k}: {v:.6f}\n")
                    else:
                        f.write(f"  Interaction Terms: None\n")

                # Noise
                if "noise_std" in spec:
                    f.write(f"  Noise Std Dev: {spec['noise_std']:.6f}\n")

                # Other parameters
                other_params = {
                    k: v
                    for k, v in spec.items()
                    if k
                    not in [
                        "population_weights",
                        "individual_deviation",
                        "interaction_terms",
                        "noise_std",
                    ]
                }
                if other_params:
                    f.write(f"  Other Parameters:\n")
                    for k, v in other_params.items():
                        f.write(f"    {k}: {v}\n")

            f.write("\n")

    print(f"  - {report_path.name}")

    # Generate group statistics report (TXT)
    group_stat_path = output_dir / "group_statistics.txt"
    with open(group_stat_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Group Statistics Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        # Collect statistics for all subjects
        y_means = [info["y_mean"] for info in subjects_info]
        y_stds = [info["y_std"] for info in subjects_info]
        y_mins = [info["y_min"] for info in subjects_info]
        y_maxs = [info["y_max"] for info in subjects_info]
        y_ranges = [info["y_max"] - info["y_min"] for info in subjects_info]

        f.write("Summary Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Number of Subjects: {len(subjects_info)}\n\n")

        f.write("Mean:\n")
        f.write(f"  Group Average: {np.mean(y_means):.6f}\n")
        f.write(f"  Group Std Dev: {np.std(y_means):.6f}\n")
        f.write(f"  Range: [{np.min(y_means):.6f}, {np.max(y_means):.6f}]\n\n")

        f.write("Std Dev:\n")
        f.write(f"  Group Average: {np.mean(y_stds):.6f}\n")
        f.write(f"  Group Std Dev: {np.std(y_stds):.6f}\n")
        f.write(f"  Range: [{np.min(y_stds):.6f}, {np.max(y_stds):.6f}]\n\n")

        f.write("Minimum:\n")
        f.write(f"  Group Average: {np.mean(y_mins):.6f}\n")
        f.write(f"  Range: [{np.min(y_mins):.6f}, {np.max(y_mins):.6f}]\n\n")

        f.write("Maximum:\n")
        f.write(f"  Group Average: {np.mean(y_maxs):.6f}\n")
        f.write(f"  Range: [{np.min(y_maxs):.6f}, {np.max(y_maxs):.6f}]\n\n")

        f.write("Range:\n")
        f.write(f"  Group Average: {np.mean(y_ranges):.6f}\n")
        f.write(f"  Range: [{np.min(y_ranges):.6f}, {np.max(y_ranges):.6f}]\n\n")

        f.write("=" * 80 + "\n")
        f.write("Subject Comparison\n")
        f.write("=" * 80 + "\n\n")

        # Rank subjects by mean
        sorted_info = sorted(subjects_info, key=lambda x: x["y_mean"], reverse=True)
        f.write("Ranked by mean response (highest to lowest):\n")
        f.write("-" * 80 + "\n")
        for rank, info in enumerate(sorted_info, start=1):
            f.write(
                f"{rank:2d}. {info['subject']:15s} | "
                f"Mean={info['y_mean']:8.4f} | "
                f"Std={info['y_std']:8.4f} | "
                f"Range=[{info['y_min']:8.4f}, {info['y_max']:8.4f}]\n"
            )

    print(f"  - {group_stat_path.name}")


def _make_serializable(obj):
    """
    Recursively convert numpy objects to JSON-serializable Python objects
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    else:
        return obj
