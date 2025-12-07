"""
Test script for INI configuration file integration with SCOUT Warm-up Generator
"""

import sys
import os
import pandas as pd
import numpy as np
import configparser

# Add the parent directory to the path so we can import the generator
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scout_warmup_generator import WarmupAEPsychGenerator


def create_sample_ini_config():
    """Create a sample INI configuration file for testing."""
    config = configparser.ConfigParser()

    # Add generator section
    config["generator"] = {
        "name": "SCOUT_warmup_generator.WarmupAEPsychGenerator",
        "n_subjects": "10",
        "total_budget": "350",
        "n_batches": "3",
        "seed": "42",
    }

    # Add experiment section
    config["experiment"] = {
        "design_file": "sample_design.csv",
        "output_file": "trial_schedule.csv",
    }

    # Write to file
    with open("sample_config.ini", "w") as configfile:
        config.write(configfile)

    print("Sample INI configuration file created: sample_config.ini")


def create_sample_design_file():
    """Create a sample design CSV file for testing."""
    # Create sample design data
    np.random.seed(42)
    n_stimuli = 100
    design_data = {
        "f1": np.random.rand(n_stimuli),
        "f2": np.random.rand(n_stimuli),
        "f3": np.random.rand(n_stimuli),
        "f4": np.random.rand(n_stimuli),
        "stimulus_id": range(n_stimuli),
    }

    design_df = pd.DataFrame(design_data)
    design_df.to_csv("sample_design.csv", index=False)
    print("Sample design file created: sample_design.csv")


def test_ini_config_loading():
    """Test loading configuration from INI file."""
    print("Testing INI configuration loading...")

    # Create sample files
    create_sample_ini_config()
    create_sample_design_file()

    # Load configuration
    config = configparser.ConfigParser()
    config.read("sample_config.ini")

    # Extract generator parameters
    generator_config = config["generator"]
    n_subjects = int(generator_config.get("n_subjects", 10))
    total_budget = int(generator_config.get("total_budget", 350))
    n_batches = int(generator_config.get("n_batches", 3))
    seed = (
        int(generator_config.get("seed", None))
        if generator_config.get("seed")
        else None
    )

    print(f"  Loaded generator parameters:")
    print(f"    n_subjects: {n_subjects}")
    print(f"    total_budget: {total_budget}")
    print(f"    n_batches: {n_batches}")
    print(f"    seed: {seed}")

    # Load design data
    design_file = config.get("experiment", "design_file", fallback="sample_design.csv")
    design_df = pd.read_csv(design_file)

    print(
        f"  Loaded design data with {len(design_df)} stimuli and {len(design_df.columns)} columns"
    )

    # Create generator with loaded parameters
    gen = WarmupAEPsychGenerator(
        design_df=design_df,
        n_subjects=n_subjects,
        total_budget=total_budget,
        n_batches=n_batches,
        seed=seed,
    )

    # Run the generator
    gen.fit_planning()
    trials = gen.generate_trials()
    summary = gen.summarize()

    print(f"  Generated {len(trials)} trials")
    print(f"  Summary contains {len(summary)} top-level keys")

    # Save trial schedule
    output_file = config.get("experiment", "output_file", fallback="trial_schedule.csv")
    trials.to_csv(output_file, index=False)
    print(f"  Trial schedule saved to: {output_file}")

    print("INI configuration loading test completed.\n")


def demonstrate_aepsych_integration():
    """Demonstrate how the generator integrates with AEPsych."""
    print("Demonstrating AEPsych integration...")

    # This is a conceptual demonstration of how the generator would be used in AEPsych
    print(
        """
In AEPsych, the generator would be used as follows:

1. Configuration in INI file:
   [generator]
   name = SCOUT_warmup_generator.WarmupAEPsychGenerator
   n_subjects = 10
   total_budget = 350
   n_batches = 3
   seed = 42

2. The AEPsych server would load the generator dynamically:
   from aepsych.generators import initialize_generators
   generators = initialize_generators(config)
   
3. The generator would be called to produce trials:
   trials = generator.generate_trials()
   
4. Trials would be fed to the AEPsych scheduler or trial runner
   """
    )

    print("AEPsych integration demonstration completed.\n")


def main():
    """Run all INI integration tests."""
    print("Running INI Configuration Integration Tests\n")

    try:
        test_ini_config_loading()
        demonstrate_aepsych_integration()

        print("All INI integration tests completed successfully!")
        print("\nFiles created:")
        print("  - sample_config.ini: Sample INI configuration")
        print("  - sample_design.csv: Sample design data")
        print("  - trial_schedule.csv: Generated trial schedule")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
