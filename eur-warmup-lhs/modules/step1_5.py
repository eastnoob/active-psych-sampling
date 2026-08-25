import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from core.base_module import BaseModule
from core.context import Context
from loguru import logger
from rich.console import Console
from tools.subject_simulator_v2.adapters.warmup_adapter import run as simulate_responses

console = Console()

class Step1_5Module(BaseModule):
    """Step 1.5: Simulation Module."""
    name = "Simulation"
    description = "Simulate responses (for testing)"

    def get_default_config(self) -> str:
        return """# Step 1.5: Simulation Configuration
seed = 42 # 随机种子
population_mean = 0.0 # 群体权重均值
population_std = 0.25 # 群体权重标准差
individual_std_percent = 0.5 # 个体差异比例
noise_std = 0.1 # 响应噪声标准差 (0.1 约为信号强度的 40%)
output_type = "likert" # 输出类型: continuous (连续), likert (李克特量表)
likert_levels = 5 # Likert量表等级数
likert_mode = "tanh" # Likert模式: tanh (拟真), percentile (均匀)
likert_sensitivity = 2.0 # Likert灵敏度
interaction_scale = 0.25 # 交互项权重尺度
interaction_pairs = [] # 模拟中包含的交互对索引列表，例如 [[0, 1], [2, 3]]
response_col = "y" # 响应列名
"""

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        s15 = config.get("step1_5", {})
        
        if not s15:
            errors.append("Missing [step1_5] section in configuration")
            return False, errors

        output_type = s15.get("output_type", "continuous")
        if output_type == "likert":
            likert_levels = s15.get("likert_levels")
            if likert_levels is None:
                errors.append("likert_levels is required when output_type is 'likert'")
            elif not isinstance(likert_levels, int) or likert_levels < 2:
                errors.append(f"likert_levels must be an integer >= 2 (found: {likert_levels})")

        # 如果使用非线性模型，确保存在 func_configs（可以在 step1_5 或 oracle.nonlinear_tanh 中）
        model_type = s15.get("model_type", "linear")
        if model_type == "nonlinear_tanh":
            func_cfg = s15.get("func_configs") or config.get("oracle", {}).get("nonlinear_tanh", {}).get("func_configs")
            if not func_cfg:
                errors.append("model_type='nonlinear_tanh' requires 'func_configs' defined in [step1_5] or [oracle.nonlinear_tanh]")

        return len(errors) == 0, errors

    def run(self, config: Dict[str, Any], context: Context) -> Context:
        s15 = config.get("step1_5")
        logger.info("Running Step 1.5: Simulation...")
        
        if not context.subject_files:
            logger.error("No subject files found in context. Did you run Step 1?")
            raise RuntimeError("Missing input files for simulation.")
            
        # Get input directory (where Step 1 files are)
        # Robust logic: Step 1.5 should always use Step 1's output as input
        first_file = Path(context.subject_files[0])
        run_dir = first_file.parent.parent # output/TIMESTAMP/
        s1_dir = run_dir / "step1"
        
        if s1_dir.exists() and any(s1_dir.glob("subject_*.csv")):
            input_dir = s1_dir
            logger.info(f"Using Step 1 output as simulation input: {input_dir}")
        else:
            input_dir = first_file.parent
            logger.info(f"Using current subject files directory as simulation input: {input_dir}")
            
        output_dir = context.get_step_output_dir("step1_5")
        
        # Process interaction weights if present
        raw_weights = s15.get("interaction_weights")
        interaction_weights = None
        if raw_weights:
            interaction_weights = {}
            for k, v in raw_weights.items():
                # Convert "1,3" string key to (1, 3) tuple key
                if isinstance(k, str) and "," in k:
                    pair = tuple(map(int, k.split(",")))
                    interaction_weights[pair] = float(v)
                else:
                    # Handle other potential formats if necessary
                    pass

        # 如果模型为nonlinear_tanh，优先查找 func_configs；如果在 step1_5 中缺失，则回退到 [oracle.nonlinear_tanh]
        model_type = s15.get("model_type", "linear")
        func_configs = s15.get("func_configs")
        if model_type == "nonlinear_tanh" and not func_configs:
            oracle_section = config.get("oracle", {}) or {}
            nl_section = oracle_section.get("nonlinear_tanh", {}) if isinstance(oracle_section, dict) else {}
            func_configs = nl_section.get("func_configs")
            if func_configs:
                logger.info("Loaded 'func_configs' from [oracle.nonlinear_tanh] section")
            else:
                # 明确报错，方便用户修复配置
                raise RuntimeError("model_type='nonlinear_tanh' requires 'func_configs' in [step1_5] or [oracle.nonlinear_tanh]")

        # Call the adapter from tools/subject_simulator_v2
        simulate_responses(
            input_dir=input_dir,
            seed=s15.get("seed", 42),
            output_mode="individual",
            clean=True,
            interaction_pairs=[tuple(p) for p in s15.get("interaction_pairs", [])],
            interaction_scale=s15.get("interaction_scale", 0.25),
            interaction_weights=interaction_weights,
            output_type=s15.get("output_type", "likert"),
            likert_levels=s15.get("likert_levels", 5),
            likert_mode=s15.get("likert_mode", "tanh"),
            likert_sensitivity=s15.get("likert_sensitivity", 2.0),
            population_mean=s15.get("population_mean", 0.0),
            population_std=s15.get("population_std", 0.25),
            individual_std_percent=s15.get("individual_std_percent", 0.5),
            noise_std=s15.get("noise_std", 0.1),
            design_space_csv=context.design_space_path or config.get("step1", {}).get("design_csv"),
            interaction_as_features=s15.get("interaction_as_features", False), # 是否使用V3方法
            model_type=model_type, # 支持 nonlinear_tanh
            func_configs=func_configs, # 仅用于非线性
            save_model_summary=True,
            print_model=True
        )
        
        # Move files from input_dir/result to output_dir
        result_dir = input_dir / "result"
        import shutil
        for f in result_dir.glob("*"):
            dest = output_dir / f.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(f), str(dest))
        
        # Clean up result dir
        if result_dir.exists():
            result_dir.rmdir()
        
        # Update context
        new_subject_files = sorted(list(output_dir.glob("subject_*.csv")))
        context.subject_files = [str(f) for f in new_subject_files]
        
        logger.success(f"Step 1.5 completed using tools/subject_simulator_v2. Simulated responses for {len(new_subject_files)} subjects.")
        
        return context
