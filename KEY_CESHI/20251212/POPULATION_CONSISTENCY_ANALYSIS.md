# Population Consistency Analysis

## Question
Are the current simulated subject parameters (loaded by PopulationConsistentOracle) sampled from the same population distribution as the BaseGP training subjects?

## Answer: NO - But with Nuance

### Evidence 1: Fixed Weights Do Not Match Population Distributions

**Fixed weights from fixed_weights_auto.json:**
- Main effects: [-0.059 to 0.381]
- Bias: -0.218

**Documented population distributions:**
- β₀: N(2.93, 1.18²) → range [0.59, 5.27]
- β₁: N(-0.08, 0.40²) → range [-0.88, 0.72]
- β₂: N(0.74, 0.37²) → range [-0.00, 1.48]
- β₄: N(-1.40, 0.44²) → range [-2.28, -0.52]

**Magnitude mismatch:** Population distributions have values up to ±2.28, while fixed weights are in [-0.381, 0.381]

### Evidence 2: How fixed_weights_auto.json Is Actually Generated

Source: `extensions/warmup_budget_check/core/simulation_runner.py`

```python
default_fw = (
    np.random.RandomState(seed)
    .uniform(
        -base.weight_range,
        base.weight_range,
        size=(base.num_observed_vars or 1, base.num_features),
    )
    .tolist()
)
```

**This means:**
- Fixed weights are sampled from **Uniform(-weight_range, weight_range)**
- NOT from the documented population distributions N(μ, σ²)
- Random generation seeded for reproducibility

### Evidence 3: What "Population-Consistent" Actually Means

Based on the code and documentation:

| Interpretation | Reality |
|---|---|
| **Sampled from population distributions** | ❌ NO |
| **Uses BaseGP training data as reference** | ✅ YES |
| **Represents one subject from same cohort** | ⚠️ MAYBE |
| **Maintains population-level statistical properties** | ❌ NOT NECESSARILY |

The term "PopulationConsistentOracle" seems to mean:
- Uses weights from (or compatible with) the BaseGP phase1_analysis_output
- Ensures experiments are conducted with a **single consistent Oracle** across EUR vs Random comparisons
- Does NOT necessarily sample from documented population distributions

### Evidence 4: Actual Weight Generation Path

1. **202511301011 (original)**: Sample population, 5 subjects studied
2. **Simulated_Subject_Parameters.md**: Population distributions derived from 5 subjects
3. **202512081445 (BaseGP)**: BaseGP trained on new subjects (not same as original 5)
4. **fixed_weights_auto.json**: Randomly generated uniform weights for simulation

The population distributions and fixed_weights are from **different phases**:
- Population distributions ← from warmup_budget_check/sample/202511271517
- Fixed weights ← from warmup_budget_check/phase1_analysis_output/202512081445

### Conclusion

**The current simulated subject is NOT sampled from the documented population distributions.**

Rather:
1. PopulationConsistentOracle loads **pre-generated random weights** from fixed_weights_auto.json
2. These weights are **uniform random**, not normally distributed
3. The term "population-consistent" refers to using a **fixed Oracle** for paired EUR vs Random comparisons
4. This ensures **fair comparison** (same Oracle state) but NOT population-statistical consistency

### Recommendations

**If true population consistency is needed:**

1. Modify PopulationConsistentOracle to sample weights from documented distributions:
   ```python
   def sample_population_consistent_oracle():
       weights = np.array([
           np.random.normal(2.93, 1.18),   # beta_0
           np.random.normal(-0.08, 0.40),  # beta_1
           np.random.normal(0.74, 0.37),   # beta_2
           np.random.normal(0.11, 0.17),   # beta_3
           np.random.normal(-1.40, 0.44),  # beta_4
           np.random.normal(0.09, 0.19),   # beta_5
       ])
       return PopulationConsistentOracle(weights)
   ```

2. Or, update fixed_weights_auto.json to contain samples from these distributions

**Current behavior is acceptable if:**
- Goal is to test **EUR vs Random strategy efficiency** with a fixed, consistent Oracle
- Not to validate **population-level statistical properties**

