#!/usr/bin/env python3
"""
Quick-start SCOUT warmup sampling with 5 subjects × 25 trials each.

This is a simplified wrapper around run_warmup_sampling.py with default parameters.
Just run: python quick_start.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).parent
    runner_script = script_dir / "run_warmup_sampling.py"

    if not runner_script.exists():
        print(f"❌ Runner script not found: {runner_script}")
        sys.exit(1)

    # Run with test parameters: 61 subjects, 30 trials each
    # Using test CSV by default; can be customized by editing this file
    cmd = [
        sys.executable,
        str(runner_script),
        "--design_csv",
        "../test/sample_design.csv",
        "--n_subjects",
        "61",
        "--trials_per_subject",
        "30",
        "--output_dir",
        "../results",
    ]

    print("Starting SCOUT Phase-1 Warmup Sampling...")
    print(f"Configuration: 61 subjects × 30 trials/subject = 1830 total trials")
    print()

    result = subprocess.run(cmd, cwd=str(script_dir))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
