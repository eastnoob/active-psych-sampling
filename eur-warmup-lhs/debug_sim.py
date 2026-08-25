import pandas as pd
import numpy as np
from pathlib import Path

def check_simulation_integrity(folder_path):
    files = list(Path(folder_path).glob("*.csv"))
    all_data = pd.concat([pd.read_csv(f) for f in files])
    
    # Calculate interaction terms
    all_data['inter_0_1'] = all_data.iloc[:, 0] * all_data.iloc[:, 1]
    all_data['inter_2_3'] = all_data.iloc[:, 2] * all_data.iloc[:, 3]
    
    # Check correlations between factors
    print("\nFactor Correlation Matrix:")
    print(all_data.iloc[:, :6].corr())
    
    # Calculate interaction terms
    all_data['inter_0_1'] = all_data.iloc[:, 0] * all_data.iloc[:, 1]
    all_data['inter_2_3'] = all_data.iloc[:, 2] * all_data.iloc[:, 3]
    all_data['inter_1_3'] = all_data.iloc[:, 1] * all_data.iloc[:, 3]
    
    # Calculate correlations
    corr_0_1 = all_data['y'].corr(all_data['inter_0_1'])
    corr_2_3 = all_data['y'].corr(all_data['inter_2_3'])
    corr_1_3 = all_data['y'].corr(all_data['inter_1_3'])
    
    print(f"\nCorrelation y vs (x1*x2) [Target]: {corr_0_1:.4f}")
    print(f"Correlation y vs (x3*x4) [Target]: {corr_2_3:.4f}")
    print(f"Correlation y vs (x2*x4) [Ghost]: {corr_1_3:.4f}")

if __name__ == "__main__":
    check_simulation_integrity(r"d:\ENVS\active-psych-sampling\eur-warmup\output\20251229_100435\step1_5")
