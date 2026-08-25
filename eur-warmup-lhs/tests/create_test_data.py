import pandas as pd
import numpy as np
from pathlib import Path

def create_test_design():
    data = {
        'x1': np.linspace(0, 1, 10),
        'x2': np.linspace(0, 1, 10),
        'cat1': ['A', 'B'] * 5
    }
    df = pd.DataFrame(data)
    # Create a grid
    x1, x2 = np.meshgrid(data['x1'], data['x2'])
    grid_df = pd.DataFrame({
        'x1': x1.flatten(),
        'x2': x2.flatten(),
        'cat1': (['A'] * 50) + (['B'] * 50)
    })
    
    Path("data").mkdir(exist_ok=True)
    grid_df.to_csv("data/test_design.csv", index=False)
    print("Created data/test_design.csv")

if __name__ == "__main__":
    create_test_design()
