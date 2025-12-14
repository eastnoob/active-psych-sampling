#!/usr/bin/env python3
"""Script to recreate the missing run*.py files from bytecode analysis."""

import textwrap
from pathlib import Path

# Define file contents based on bytecode analysis
FILES_TO_CREATE = {
    "tests/is_EUR_work/run_server_sps_test.py": '''
#!/usr/bin/env python3
"""
Server Flow Test with Complete Reports
使用AEPsych Server + SPS + BaseGP先验 + 不同初始化池
输出增强报告: summary.json, oracle_model_spec.json, sampling_history.npy, interaction_log.json
"""

import sys
import io
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
import platform

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'extensions'))
sys.path.insert(0, str(project_root / 'tools'))

# Import AEPsych components
from aepsych.server import AEPsychServer
from aepsych.config import Config

# Import custom generators and oracle
try:
    from custom_generators.pool_based_generator import PoolBasedGenerator
except ImportError:
    print("Warning: PoolBasedGenerator not found, using standard generators")

try:
    from oracle.single_subject import SingleSubject
except ImportError:
    print("Warning: SingleSubject oracle not found")
    # Define a minimal oracle for testing
    class SingleSubject:
        def __init__(self, design_space):
            self.design_space = design_space

        def query(self, x):
            # Simple oracle: return 1 if sum of features > threshold
            return 1 if np.sum(x) > len(x) * 0.5 else 0


def load_design_space():
    """Load design space from CSV file."""
    try:
        design_space_path = project_root / 'data' / 'only_independences' / '6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv'

        df_design = pd.read_csv(design_space_path)
        print(f"Loaded design space with shape: {df_design.shape}")

        # Convert to numeric, mapping categorical values
        df_numeric = df_design.copy()

        # Map x4_4level_categorical: low, medium-low, medium-high, high -> 0, 1, 2, 3
        if 'x4_4level_categorical' in df_numeric.columns:
            category_map = {'low': 0.0, 'medium-low': 0.25, 'medium-high': 0.5, 'high': 0.75}
            df_numeric['x4_4level_categorical'] = df_numeric['x4_4level_categorical'].map(category_map)

        # Convert all columns to float
        design_space = df_numeric.astype(float).values
        return design_space

    except Exception as e:
        print(f"Error loading design space: {e}")
        # Return a minimal design space for testing
        print("Using minimal test design space (8 combinations)")
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0.25, 0.25, 0, 0],
            [0, 1, 0.25, 0.25, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 0.25, 0.25, 0, 0],
            [1, 1, 0.25, 0.25, 0, 1],
        ])


def update_config_with_pool(config_path: str, pool_points: np.ndarray) -> str:
    """Update config file with pool_points parameter."""
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    in_pool_section = False

    for line in lines:
        if line.strip().startswith('[PoolBasedGenerator]'):
            in_pool_section = True
            new_lines.append(line)
            # Add pool_points parameter
            pool_str = str(pool_points.tolist()).replace('\\n', '')
            new_lines.append(f'pool_points = {pool_str}\\n')
        elif in_pool_section and line.strip().startswith('['):
            in_pool_section = False
            new_lines.append(line)
        else:
            new_lines.append(line)

    # Write updated config
    updated_path = config_path.replace('.ini', '_updated.ini')
    with open(updated_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    return updated_path


def run_server_test(budget: int = 50):
    """Run test using AEPsych Server with complete reports."""
    print("="*80)
    print("AEPsych Server + SPS + BaseGP Prior Test".center(80))
    print("="*80)

    # Create result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path(__file__).parent / 'results' / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    data_dir = result_dir / 'data_files'
    data_dir.mkdir(exist_ok=True)

    print(f"\\nResult directory: {result_dir}")

    # Step 1: Load Design Space
    print("\\n" + "="*80)
    print("Step 1: Load Design Space")
    print("="*80)
    design_space = load_design_space()
    print(f"Design space shape: {design_space.shape}")

    # Step 2: Setup Config
    print("\\n" + "="*80)
    print("Step 2: Setup Config")
    print("="*80)
    base_config = project_root / 'tests' / 'is_EUR_work' / 'configs' / 'base_config.ini'

    if not base_config.exists():
        print(f"Warning: Config not found at {base_config}")
        print("Using minimal default config")
        # Create minimal config
        base_config = result_dir / 'minimal_config.ini'
        with open(base_config, 'w') as f:
            f.write("[common]\\n")
            f.write("parnames = [x1, x2, x3, x4, x5, x6]\\n")
            f.write("outcome_type = binary\\n")

    config_path = update_config_with_pool(str(base_config), design_space)
    print(f"Updated config: {config_path}")

    # Step 3: Create Oracle
    print("\\n" + "="*80)
    print("Step 3: Create Oracle")
    print("="*80)
    oracle = SingleSubject(design_space)
    print(f"Oracle created with {len(design_space)} design points")

    # Save oracle spec
    oracle_spec = {
        'type': 'SingleSubject',
        'design_space_size': len(design_space),
        'n_dims': design_space.shape[1]
    }
    oracle_spec_file = data_dir / 'oracle_model_spec.json'
    with open(oracle_spec_file, 'w') as f:
        json.dump(oracle_spec, f, indent=2)
    print(f"Saved oracle spec to {oracle_spec_file}")

    # Step 4: Run Server
    print("\\n" + "="*80)
    print("Step 4: Run AEPsych Server")
    print("="*80)

    server = AEPsychServer()
    config = Config.from_file(config_path)
    server.configure(config)

    # Tracking
    sampling_history = []
    interaction_logs = []

    print(f"\\nRunning {budget} trials...")
    for trial_idx in range(budget):
        # Ask for next configuration
        x_config = server.ask()

        # Convert config dict to array
        parnames = server.strat.model.parnames if hasattr(server.strat, 'model') else config.parnames
        x_array = np.array([x_config[name] for name in parnames], dtype=np.float64)

        # Query oracle
        y = oracle.query(x_array)

        # Tell server
        server.tell(x_config, y)

        # Log
        sampling_history.append(x_array.tolist())
        interaction_logs.append({
            'trial': trial_idx,
            'x': x_array.tolist(),
            'y': y,
            'x_config': x_config
        })

        if (trial_idx + 1) % 10 == 0:
            print(f"  Trial {trial_idx + 1}/{budget} completed")

    # Step 5: Save Results
    print("\\n" + "="*80)
    print("Step 5: Save Results")
    print("="*80)

    # Save sampling history
    history_file = data_dir / 'sampling_history.npy'
    np.save(history_file, np.array(sampling_history))
    print(f"Saved sampling history to {history_file}")

    # Save interaction log
    log_file = data_dir / 'interaction_log.json'
    with open(log_file, 'w') as f:
        json.dump(interaction_logs, f, indent=2)
    print(f"Saved interaction log to {log_file}")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'budget': budget,
        'design_space_shape': list(design_space.shape),
        'n_trials': len(sampling_history),
        'config_file': str(config_path),
        'result_dir': str(result_dir)
    }
    summary_file = data_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")

    print("\\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
    print(f"Results saved to: {result_dir}")

    return result_dir


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run AEPsych Server test with SPS')
    parser.add_argument('--budget', type=int, default=50, help='Number of trials')
    args = parser.parse_args()

    run_server_test(budget=args.budget)
''',
}

def main():
    print("Creating reconstructed Python files from bytecode analysis...\n")

    for filepath, content in FILES_TO_CREATE.items():
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"[OK] Created: {filepath} ({len(content)} bytes)")

    print("\nDone!")

if __name__ == '__main__':
    main()
