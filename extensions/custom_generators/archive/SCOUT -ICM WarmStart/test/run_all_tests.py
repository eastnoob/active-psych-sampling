"""
Run all tests for SCOUT Warm-up Generator
"""

import sys
import os
import subprocess


def run_test_script(script_name):
    """Run a test script and return the result."""
    print(f"Running {script_name}...")
    try:
        # Change to the test directory
        test_dir = os.path.dirname(os.path.abspath(__file__))
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=test_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            print(f"  {script_name} completed successfully")
            return True
        else:
            print(f"  {script_name} failed with return code {result.returncode}")
            print(f"  stdout: {result.stdout}")
            print(f"  stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  {script_name} timed out")
        return False
    except Exception as e:
        print(f"  {script_name} failed with exception: {e}")
        return False


def main():
    """Run all test scripts."""
    print("Running all SCOUT Warm-up Generator tests\n")

    # List of test scripts to run
    test_scripts = [
        "test_scout_warmup.py",
        "test_ini_integration.py",
        "test_toy_example.py",
    ]

    # Run each test script
    results = []
    for script in test_scripts:
        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(script_path):
            success = run_test_script(script)
            results.append((script, success))
        else:
            print(f"Test script not found: {script}")
            results.append((script, False))

    # Print summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print("=" * 50)

    all_passed = True
    for script, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"  {script}: {status}")
        if not success:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
