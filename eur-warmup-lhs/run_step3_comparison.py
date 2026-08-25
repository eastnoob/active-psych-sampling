import sys
import os
from pathlib import Path
import toml
from loguru import logger

# Add project root and parent to sys.path
ROOT_DIR = Path(__file__).parent.absolute()
PARENT_DIR = ROOT_DIR.parent.absolute()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from core.context import Context
from modules.step3 import Step3Module

def run_default_step3():
    # 1. Paths
    config_path = ROOT_DIR / "config" / "full_workflow_template.toml"
    output_root = ROOT_DIR / "output"
    timestamp = "20251229_110543 add"
    step1_5_dir = output_root / timestamp / "step1_5"
    step2_dir = output_root / timestamp / "step2"
    
    # 2. Load Config and modify to default
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
    
    config["step3"]["kernel_type"] = "default"
    
    # 3. Setup Context
    context = Context()
    context.output_root = output_root
    context.timestamp = timestamp + "_default" # Use a different timestamp for comparison
    
    # Find subject files
    subject_files = list(step1_5_dir.glob("subject_*.csv"))
    context.subject_files = [str(f) for f in subject_files]
    
    # Set model spec path from step 2
    context.model_spec_path = str(step2_dir / "model_spec.json")
    
    # Set design space path from config
    context.design_space_path = config.get("step1", {}).get("design_csv")
    
    # 4. Run Step 3
    step3 = Step3Module()
    try:
        context = step3.run(config, context)
        logger.success("Step 3 (Default) execution finished successfully.")
    except Exception as e:
        logger.exception(f"Step 3 (Default) failed: {e}")

if __name__ == "__main__":
    run_default_step3()
