#!/usr/bin/env python3
"""
SCOUT Phase-1 Warmup Sampling Runner

This script orchestrates the complete Phase-1 warmup sampling workflow:
1. Load design space from CSV
2. Coordinate multi-subject batch planning
3. Generate per-subject trial schedules with adaptive binning
4. Save results with comprehensive metrics

Usage:
    python run_warmup_sampling.py \\
        --design_csv <path> \\
        --n_subjects 5 \\
        --trials_per_subject 25 \\
        --output_dir <path>

Example:
    python run_warmup_sampling.py \\
        --design_csv "../../data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv" \\
        --n_subjects 5 \\
        --trials_per_subject 25
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from scout_warmup_generator import WarmupAEPsychGenerator
from study_coordinator import StudyCoordinator


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    log_file = output_dir / "warmup_sampling.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger("warmup_sampling")


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate command-line arguments."""
    if args.n_subjects < 1:
        raise ValueError(f"n_subjects must be ≥1, got {args.n_subjects}")
    if args.trials_per_subject < 8:
        raise ValueError(
            f"trials_per_subject must be ≥8 (for Core-1), got {args.trials_per_subject}"
        )
    if not Path(args.design_csv).exists():
        raise FileNotFoundError(f"Design CSV not found: {args.design_csv}")


def load_design_space(csv_path: str) -> tuple[pd.DataFrame, dict]:
    """
    Load design space from CSV and encode all factors as integers.

    Returns:
        (encoded_design_df, encoding_map)
        encoded_design_df: Design space with all numeric columns (0, 1, 2, ...)
        encoding_map: Dict mapping factor -> {original_value -> encoded_int}
    """
    design_df = pd.read_csv(csv_path)

    # Rename columns to standard format (f1, f2, ...)
    factor_names = [f"f{i+1}" for i in range(len(design_df.columns))]
    design_df.columns = factor_names

    # Build encoding map and encode all factors
    encoding_map = {}
    encoded_df = design_df.copy()

    for factor in factor_names:
        # Get unique values in order of appearance
        unique_vals = design_df[factor].unique()
        # Create mapping: original_value -> integer code
        val_to_int = {val: i for i, val in enumerate(unique_vals)}
        int_to_val = {i: val for val, i in val_to_int.items()}

        encoding_map[factor] = {
            "val_to_int": val_to_int,
            "int_to_val": int_to_val,
            "unique_values": list(unique_vals),
        }

        # Encode this factor
        encoded_df[factor] = design_df[factor].map(val_to_int).astype(int)

    print(
        f"[OK] Loaded design space: {len(design_df)} combinations x {len(factor_names)} factors"
    )
    print(f"  Factors: {', '.join(factor_names)}")
    print(f"  Encoding: All factors converted to integers (0, 1, 2, ...)")
    for factor, enc_info in encoding_map.items():
        print(f"    {factor}: {len(enc_info['unique_values'])} levels")

    return encoded_df, encoding_map


def decode_trial_schedule(trial_df: pd.DataFrame, encoding_map: dict) -> pd.DataFrame:
    """
    Decode trial schedule from integer representation back to original values.

    Args:
        trial_df: Trial schedule with integer-encoded factors
        encoding_map: Encoding map from load_design_space()

    Returns:
        Trial schedule with original factor values
    """
    decoded_df = trial_df.copy()

    for factor, enc_info in encoding_map.items():
        if factor in decoded_df.columns:
            int_to_val = enc_info["int_to_val"]
            # Map integer codes back to original values
            decoded_df[factor] = decoded_df[factor].map(int_to_val)

    return decoded_df


def create_output_directory(base_output_dir: str) -> Path:
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_warmup_sampling(
    design_df: pd.DataFrame,
    encoding_map: dict,
    n_subjects: int,
    n_batches: int,
    trials_per_subject: int,
    output_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Execute the complete warmup sampling workflow.

    Args:
        design_df: Integer-encoded design space DataFrame
        encoding_map: Mapping to decode factors back to original values
        n_subjects: Number of subjects
        n_batches: Number of batches to divide subjects across
        trials_per_subject: Trials per subject in Phase-1
        output_dir: Output directory for results
        logger: Logger instance

    Returns:
        Summary statistics dictionary
    """
    logger.info(
        f"Starting warmup sampling: {n_subjects} subjects × {trials_per_subject} trials each"
    )

    # Step 1: Initialize coordinator
    total_budget = n_subjects * trials_per_subject
    logger.info(f"Total budget: {total_budget} trials")

    coordinator = StudyCoordinator(
        design_df=design_df,
        n_subjects=n_subjects,
        n_batches=n_batches,  # Use the provided n_batches parameter
        total_budget=total_budget,
        seed=42,
    )

    # Step 2: Fit global plan
    logger.info(
        "Fitting global plan (Core-1 selection, interaction pairs, boundary library)..."
    )
    coordinator.fit_initial_plan()

    # Step 3: Generate plans for all subjects
    logger.info(f"Allocating per-subject budgets...")
    subject_plans = {}
    for subject_id in range(n_subjects):
        # Let coordinator assign batch_id based on subject distribution
        batch_id = subject_id % n_batches
        plan = coordinator.allocate_subject_plan(subject_id=subject_id, batch_id=batch_id)
        subject_plans[subject_id] = plan
        quotas = plan["quotas"]
        logger.info(
            f"  Subject {subject_id}: {quotas.get('core1', 0)} Core-1, "
            f"{quotas.get('core2_main_effects', 0)} Core-2 main, "
            f"{quotas.get('core2_interactions', 0)} Core-2 interaction, "
            f"{quotas.get('individual', 0)} individual, "
            f"{quotas.get('boundary', 0)} boundary, "
            f"{quotas.get('lhs', 0)} LHS"
        )

    # Step 4: Generate trials for ALL subjects at once
    # This allows scout_warmup_generator to properly allocate Core-2 trials across subjects
    logger.info("Generating trial schedules for all subjects...")
    
    all_trials = []
    
    # Create a single shared generator instance for fit_planning, then use per-subject generators
    # This prevents each subject from regenerating the planning space
    shared_generator = WarmupAEPsychGenerator(
        design_df=design_df,
        n_subjects=n_subjects,
        n_batches=coordinator.n_batches,
        seed=42,
    )
    shared_generator.fit_planning()
    
    # Generate trials for each subject using their specific plans
    for subject_id in range(n_subjects):
        plan = subject_plans[subject_id]
        
        # Create a per-subject generator
        subject_generator = WarmupAEPsychGenerator(
            design_df=design_df,
            n_subjects=n_subjects,  # Keep real n_subjects for correct distribution
            n_batches=coordinator.n_batches,
            seed=42 + subject_id,  # Per-subject seed for uniqueness
        )
        
        # Reuse the shared planning state (d, binary/continuous detection)
        subject_generator.fit_planning()
        
        # Apply this subject's specific plan
        subject_generator.apply_plan(plan)
        
        # Generate trials for this subject
        subject_generator.generate_trials()
        
        # Get trial schedule
        trial_df = subject_generator.trial_schedule_df.copy()
        
        # Decode factors back to original values
        trial_df = decode_trial_schedule(trial_df, encoding_map)
        
        # Ensure subject_id is correctly set
        trial_df["subject_id"] = subject_id
        
        # Limit to trials_per_subject
        trial_df = trial_df.head(trials_per_subject)
        
        all_trials.append(trial_df)
        
        logger.info(
            f"  Subject {subject_id}: {len(trial_df)} trials allocated "
            f"(requested {trials_per_subject})"
        )

    # Step 5: Combine and save results
    logger.info("Combining results and saving outputs...")

    # Combine all trials
    combined_trials = pd.concat(all_trials, ignore_index=True)
    combined_trials = combined_trials.sort_values(
        ["subject_id", "batch_id"]
    ).reset_index(drop=True)

    # Fix design_row_id type (convert to int) and fill in missing factor values for individual/LHS trials
    # For rows with design_row_id but missing factor values, merge from design_df
    if "design_row_id" in combined_trials.columns:
        combined_trials["design_row_id"] = (
            pd.to_numeric(combined_trials["design_row_id"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )

        # Identify rows with valid design_row_id but missing factor values
        valid_design_ids = combined_trials["design_row_id"] >= 0
        missing_factors = (
            combined_trials[valid_design_ids].iloc[:, :].isnull().any(axis=1)
        )

        # For each missing row, fill factor values from design_df
        factor_cols = [col for col in combined_trials.columns if col.startswith("f")]
        for factor in factor_cols:
            mask = (valid_design_ids) & (combined_trials[factor].isna())
            if mask.any():
                # Map design_row_id to design_df index and fill values
                for idx in combined_trials[mask].index:
                    design_id = int(combined_trials.loc[idx, "design_row_id"])
                    if design_id < len(design_df):
                        combined_trials.loc[idx, factor] = design_df.iloc[design_id][
                            factor
                        ]

    # Save trial schedule
    trials_csv = output_dir / "trial_schedule.csv"
    combined_trials.to_csv(trials_csv, index=False)
    logger.info(f"[OK] Trial schedule: {trials_csv}")

    # Save per-subject summaries (simplified - from combined trials)
    summaries_json = output_dir / "subject_summaries.json"
    summaries_to_save = {}
    for subject_id in range(n_subjects):
        subject_trials = combined_trials[combined_trials["subject_id"] == subject_id]
        summaries_to_save[str(subject_id)] = {
            "n_trials": len(subject_trials),
            "block_types": subject_trials["block_type"].value_counts().to_dict(),
            "batch_ids": sorted(subject_trials["batch_id"].unique().tolist()),
        }

    with open(summaries_json, "w") as f:
        json.dump(summaries_to_save, f, indent=2, default=str)
    logger.info(f"[OK] Subject summaries: {summaries_json}")

    # Generate execution summary
    n_trials_by_type = combined_trials.groupby("block_type").size().to_dict()
    execution_summary = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "n_subjects": n_subjects,
            "trials_per_subject": trials_per_subject,
            "total_budget": total_budget,
            "design_space_size": len(design_df),
            "n_factors": design_df.shape[1],
        },
        "results": {
            "total_trials_generated": len(combined_trials),
            "trials_by_type": n_trials_by_type,
            "avg_coverage": np.nan,  # Would require per-subject metrics
            "avg_gini": np.nan,
            "avg_core1_repeat_rate": np.nan,
        },
        "output_files": {
            "trial_schedule_csv": str(trials_csv.relative_to(output_dir.parent)),
            "subject_summaries_json": str(
                summaries_json.relative_to(output_dir.parent)
            ),
        },
    }

    summary_json = output_dir / "execution_summary.json"
    with open(summary_json, "w") as f:
        json.dump(execution_summary, f, indent=2)
    logger.info(f"[OK] Execution summary: {summary_json}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("WARMUP SAMPLING EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Timestamp: {execution_summary['timestamp']}")
    print(f"Configuration: {n_subjects} subjects × {trials_per_subject} trials/subject")
    print(
        f"Total trials generated: {execution_summary['results']['total_trials_generated']}"
    )
    print(f"Trials by type: {n_trials_by_type}")
    print(
        f"Average coverage: {execution_summary['results']['avg_coverage']:.4f} (target >0.10)"
    )
    print(
        f"Average Gini: {execution_summary['results']['avg_gini']:.4f} (target <0.40)"
    )
    print(
        f"Average Core-1 repeat rate: {execution_summary['results']['avg_core1_repeat_rate']:.4f} (target ≥0.50)"
    )
    print(f"\nOutput directory: {output_dir}")
    print("=" * 70 + "\n")

    return execution_summary


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SCOUT Phase-1 Warmup Sampling Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 5 subjects × 25 trials each
  python run_warmup_sampling.py --design_csv <path> --n_subjects 5 --trials_per_subject 25
  
  # Custom configuration
  python run_warmup_sampling.py --design_csv <path> --n_subjects 10 --trials_per_subject 50 --output_dir ./custom_results
        """,
    )

    parser.add_argument(
        "--design_csv",
        type=str,
        default="../../data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        help="Path to design space CSV file",
    )
    parser.add_argument(
        "--n_subjects", type=int, default=5, help="Number of subjects (default: 5)"
    )
    parser.add_argument(
        "--n_batches",
        type=int,
        default=3,
        help="Number of batches to divide trials across (default: 3)",
    )
    parser.add_argument(
        "--trials_per_subject",
        type=int,
        default=25,
        help="Number of trials per subject (default: 25)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Base output directory for timestamped results (default: ../results)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate inputs
    try:
        validate_inputs(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"[ERROR] Input validation failed: {e}")
        sys.exit(1)

    # Create output directory
    output_dir = create_output_directory(args.output_dir)

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("SCOUT Phase-1 Warmup Sampling Started")
    logger.info("=" * 70)
    logger.info(f"Design CSV: {args.design_csv}")
    logger.info(
        f"Configuration: {args.n_subjects} subjects × {args.trials_per_subject} trials each"
    )
    logger.info(f"Output directory: {output_dir}")

    try:
        # Load design space (with encoding)
        design_df, encoding_map = load_design_space(args.design_csv)

        # Run warmup sampling
        summary = run_warmup_sampling(
            design_df=design_df,
            encoding_map=encoding_map,
            n_subjects=args.n_subjects,
            n_batches=args.n_batches,
            trials_per_subject=args.trials_per_subject,
            output_dir=output_dir,
            logger=logger,
        )

        # Auto-generate per-subject CSV files
        logger.info("=" * 70)
        logger.info("Generating per-subject CSV files...")
        logger.info("=" * 70)

        trial_schedule_csv = output_dir / "trial_schedule.csv"
        from generate_subject_csvs import generate_subject_csvs

        subject_files = generate_subject_csvs(
            trial_schedule_csv=str(trial_schedule_csv),
            output_dir=str(output_dir),
            logger=logger,
            n_subjects=args.n_subjects,  # Filter to only requested subjects
        )

        logger.info(f"Generated {len(subject_files)} per-subject CSV files")

        # Success
        logger.info("=" * 70)
        logger.info("[OK] SCOUT Phase-1 Warmup Sampling Completed Successfully")
        logger.info("=" * 70)
        print("[OK] Warmup sampling completed successfully!")
        print(f"[OK] {len(subject_files)} per-subject CSV files generated!")

    except Exception as e:
        logger.error(f"[ERROR] Error during execution: {e}", exc_info=True)
        print(f"[ERROR] Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
