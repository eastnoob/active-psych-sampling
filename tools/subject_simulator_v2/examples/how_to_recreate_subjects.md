# How to Recreate Subjects in Other Places

After running Step 1.5, you have generated 5 subjects with a `fixed_weights_auto.json` file. This guide shows you how to recreate similar subjects elsewhere.

## File Structure

After Step 1.5, you'll have these files in `result/`:

```
result/
├── subject_1.csv ... subject_5.csv         # Response data
├── combined_results.csv                     # All subjects combined
├── subject_1_spec.json ... subject_5_spec.json  # Full subject parameters (V2 format)
├── fixed_weights_auto.json                  # Population parameters ⭐ USE THIS
├── cluster_summary.json                     # Cluster metadata
└── MODEL_SUMMARY.txt/md                     # Human-readable summary
```

### `fixed_weights_auto.json` Format (V2 Extended)

```json
{
  "global": [[w1, w2, ..., w6]],      // Population weights (main effects)
  "interactions": {                    // Interaction weights
    "3,4": 0.189,                      // Weight for x3*x4
    "0,1": 0.206                       // Weight for x0*x1
  },
  "bias": -7.65                        // Bias (shared by all subjects)
}
```

## Method 1: Using warmup_adapter (Recommended)

**Use Case:** Recreate the same 5-subject cluster on a new sampling plan

```python
from subject_simulator_v2.adapters.warmup_adapter import run

# Recreate subjects using fixed_weights_auto.json
run(
    input_dir="path/to/new/sampling/plan",  # New sampling plan directory
    seed=99,  # Same seed as original
    output_mode="combined",
    clean=True,
    interaction_pairs=[(3, 4), (0, 1)],  # Must match original
    interaction_scale=0.25,               # Must match original
    output_type="likert",
    likert_levels=5,
    likert_mode="tanh",
    likert_sensitivity=2.0,
    population_mean=0.0,
    population_std=0.3,
    individual_std_percent=0.3,
    ensure_normality=False,  # Disable for recreation
    bias=0.0,  # Will auto-load from fixed_weights
    noise_std=0.0,
    fixed_weights_file="path/to/fixed_weights_auto.json",  # ⭐ Specify this
    design_space_csv="path/to/full_design_space.csv",
)
```

**Result:** Exactly the same 5 subjects, answering on new sampling points.

## Method 2: Using LinearSubject (Flexible)

**Use Case:** Create a single subject for real-time answering (e.g., Oracle simulation)

```python
import json
import numpy as np
from subject_simulator_v2 import LinearSubject

# Load fixed_weights_auto.json
with open("fixed_weights_auto.json", 'r') as f:
    fixed_data = json.load(f)

# Extract parameters
weights = np.array(fixed_data['global'][0])

# Convert interactions format
interaction_weights = {}
for key, value in fixed_data['interactions'].items():
    i, j = map(int, key.split(','))
    interaction_weights[(i, j)] = value

bias = fixed_data['bias']

# Create subject
subject = LinearSubject(
    weights=weights,
    interaction_weights=interaction_weights,
    bias=bias,
    noise_std=0.0,
    likert_levels=5,
    likert_sensitivity=2.0,
    seed=42  # Can use different seed
)

# Use the subject
x = np.array([2.8, 6.5, 2.0, 0.0, 1.0, 0.0])  # Numeric input
response = subject(x)  # Returns Likert 1-5
print(f"Response: {response}")
```

**Result:** A single subject you can call anytime, anywhere.

## Method 3: Generate Similar Subjects (Extended)

**Use Case:** Create more subjects "similar" to the original 5

```python
import json
import numpy as np
from subject_simulator_v2 import LinearSubject

# Load population weights
with open("fixed_weights_auto.json", 'r') as f:
    fixed_data = json.load(f)

population_weights = np.array(fixed_data['global'][0])
interaction_weights = {
    (int(k.split(',')[0]), int(k.split(',')[1])): v
    for k, v in fixed_data['interactions'].items()
}
bias = fixed_data['bias']

# Create 3 new subjects with individual variations
individual_std = 0.3 * 0.3  # population_std * individual_std_percent

for subject_id in range(1, 4):
    np.random.seed(100 + subject_id)

    # Individual weights = population weights + random deviation
    deviation = np.random.normal(0, individual_std, size=len(population_weights))
    individual_weights = population_weights + deviation

    new_subject = LinearSubject(
        weights=individual_weights,
        interaction_weights=interaction_weights,
        bias=bias,
        noise_std=0.0,
        likert_levels=5,
        likert_sensitivity=2.0,
        seed=100 + subject_id
    )

    print(f"New subject {subject_id}: {individual_weights}")
```

**Result:** New subjects from the same population, but with individual differences.

## Comparison

| Method | Use Case | Advantages | Example |
|--------|----------|-----------|---------|
| Method 1 (warmup_adapter) | Recreate full 5-subject cluster on new sampling plan | Simple, automated, fully compatible | New experiment with same subject model |
| Method 2 (LinearSubject) | Single subject for real-time answering | Flexible, integrates anywhere | Oracle simulation, interactive experiments |
| Method 3 (Individual variations) | Generate more "similar" subjects | Keeps population traits, adds individual variation | Expand subject pool, Monte Carlo simulation |

## Important Notes

### 1. Parameter Compatibility

When using Method 1, ensure these parameters match the original:
- `interaction_pairs` (MUST match)
- `interaction_scale` (MUST match)
- `likert_levels`, `likert_sensitivity`, `likert_mode` (should match)
- `seed` (use same seed for exact reproduction, different seed for variation)

### 2. Categorical Variable Mapping

If your data has categorical variables (like 'Strict', 'Rotated', 'Chaos'), they are automatically mapped to numbers (0, 1, 2). The mapping is consistent and alphabetically sorted.

Example:
```
'Chaos': 0
'Rotated': 1
'Strict': 2
```

### 3. Design Space

- **For bias calculation:** Use `design_space_csv` to specify the full design space (e.g., 324 points)
- **For normality check:** The full design space is also used
- **For response generation:** The subject CSVs (e.g., 25 points) are used

## Quick Reference

**To recreate exactly the same subjects:**
```python
run(..., fixed_weights_file="fixed_weights_auto.json", seed=99, ensure_normality=False)
```

**To create a single subject:**
```python
subject = LinearSubject.from_dict(json.load(open("subject_1_spec.json")))
```

**To generate similar subjects:**
```python
# Add random deviation to population_weights
individual_weights = population_weights + np.random.normal(0, individual_std, size=n_features)
```
