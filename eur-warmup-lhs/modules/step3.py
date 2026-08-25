import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from core.base_module import BaseModule
from core.context import Context
from loguru import logger
from core.internal.phase1_step3_base_gp import process_step3

class Step3Module(BaseModule):
    """Step 3: Base GP Module."""
    name = "Base GP"
    description = "Train prior model & scan space"

    def get_default_config(self) -> str:
        return """# Step 3: Base GP Configuration
design_csv = "data/design.csv" # 设计空间CSV文件路径 (若从Step 1连续运行则可省略)
max_iters = 100 # GP训练最大迭代次数
learning_rate = 0.05 # 学习率
use_cuda = false # 是否使用CUDA
ensure_diversity = true # 是否确保采样多样性
subject_col = "subject_id" # 被试编号列名
response_col = "y" # 响应变量列名

# 数据处理与噪声控制
y_normalization_type = "zscore" # 标准化类型: zscore (减均值除标准差) 或 mean_centering (仅减均值)
noise_lower_bound = 0.01 # 噪声下界 (likelihood noise constraint)，防止过拟合

# --- 高级核函数配置 ---
kernel_type = "default" # 可选: "default" (Matern), "custom_anova"
# 如果使用 custom_anova，请取消下面注释并配置
# interaction_mode = "all" # "all", "list", "none"
# dim_specs = [
#     { name = "x1", type = "continuous" },
#     { name = "x2", type = "continuous" },
#     { name = "cat1", type = "discrete", n_categories = 2 }
# ]
"""

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        s3 = config.get("step3", {})
        
        if s3.get("max_iters", 0) <= 0:
            errors.append("max_iters must be > 0")
            
        if s3.get("kernel_type") == "custom_anova":
            if "dim_specs" not in s3:
                errors.append("custom_anova requires 'dim_specs' configuration")
            
        return len(errors) == 0, errors

    def run(self, config: Dict[str, Any], context: Context) -> Context:
        s3 = config.get("step3")
        logger.info("Running Step 3: Base GP Training...")
        
        if not context.subject_files:
            logger.error("No subject data files found in context.")
            raise RuntimeError("Missing input files for GP training.")
            
        # Resolve design space path
        design_space_csv = context.design_space_path
        if not design_space_csv:
            # Try step3 config
            design_space_csv = s3.get("design_csv")
        if not design_space_csv:
            # Try step1 config as fallback
            design_space_csv = config.get("step1", {}).get("design_csv")
            
        if not design_space_csv:
            logger.error("Design space CSV path not found in context or config.")
            raise RuntimeError("Step 3 requires 'design_csv' to scan the design space. Please specify it in [step3] or [step1] config.")

        output_dir = context.get_step_output_dir("step3")
        
        # 尝试加载 Step 2 生成s的 model_spec.json
        model_spec = None
        if hasattr(context, "model_spec_path") and Path(context.model_spec_path).exists():
            import json
            with open(context.model_spec_path, "r", encoding="utf-8") as f:
                model_spec = json.load(f)
            logger.info(f"Loaded model spec from {context.model_spec_path}")
        
        # 确保 model_spec 包含当前运行的核函数信息，以便报告正确显示
        if model_spec is None:
            model_spec = {}

        # 重要：Step3 配置优先级高于 Step2 的建议
        # 如果用户在 Step3 配置中明确指定了 kernel_type，则覆盖 Step2 的建议
        if "kernel_type" in s3:
            model_spec["kernel_type"] = s3["kernel_type"]
        elif "kernel_type" not in model_spec:
            model_spec["kernel_type"] = "default"

        if "kernel_combination" in s3:
            model_spec["kernel_combination"] = s3["kernel_combination"]
        elif "kernel_combination" not in model_spec:
            model_spec["kernel_combination"] = "mult"
        
        # Handle Factory instantiation
        factory = None
        kernel_type = model_spec["kernel_type"]
        
        if kernel_type == "custom_anova":
            logger.info("Using Custom ANOVA Kernel Factory")
            try:
                # 动态导入
                from extensions.custom_factory.custom_anova_kernel_factory import CustomAnovaKernelFactory
                
                # 优先顺序: config.toml (如果明确指定) > model_spec.json > context.analysis_results
                interaction_mode = s3.get("interaction_mode")
                interaction_pairs = s3.get("interaction_pairs")
                
                if interaction_mode is None:
                    # 如果 TOML 没写，再看 model_spec.json
                    interaction_mode = model_spec.get("interaction_mode")
                    interaction_pairs = model_spec.get("interaction_pairs")
                
                if interaction_mode is None:
                    # 如果还是没写，看 context.analysis_results
                    if context.analysis_results and "model_spec" in context.analysis_results:
                        spec = context.analysis_results["model_spec"]
                        interaction_pairs = spec.get("interaction_pairs")
                        interaction_mode = spec.get("interaction_mode", "list")
                        logger.info("Recovered interaction settings from analysis_results")
                    else:
                        # 最后的默认值
                        interaction_mode = "all"
                        interaction_pairs = None
                
                # 转换 interaction_mode 以兼容 CustomAnovaKernelFactory
                if interaction_mode == "specified_only":
                    interaction_mode = "list"
                elif interaction_mode in ["free", "hybrid", "auto"]:
                    interaction_mode = "all"
                
                # 转换 interaction_pairs 为 tuple 列表 (JSON 加载后是 list of lists)
                if interaction_pairs:
                    interaction_pairs = [tuple(p) for p in interaction_pairs]
                
                # 最后的防御：如果模式是 list 但没有对子
                if interaction_mode == "list" and not interaction_pairs:
                    if context.analysis_results and "selected_pairs" in context.analysis_results:
                        interaction_pairs = [tuple(p) for p in context.analysis_results["selected_pairs"]]
                        logger.info(f"Auto-filled interaction_pairs from analysis results: {interaction_pairs}")
                    
                    # 如果还是没有，且模式是 list，强制降级为 none 避免崩溃
                    if not interaction_pairs:
                        logger.warning("interaction_mode is 'list' but no pairs found. Falling back to 'none'.")
                        interaction_mode = "none"
                
                dim_specs = s3["dim_specs"]
                
                # 同步 model_spec 以确保报告准确
                model_spec["interaction_mode"] = interaction_mode
                model_spec["interaction_pairs"] = interaction_pairs
                
                factory = CustomAnovaKernelFactory(
                    dim=len(dim_specs),
                    dim_specs=dim_specs,
                    interaction_mode=interaction_mode,
                    interaction_pairs=interaction_pairs,
                    interaction_shrink_loc=s3.get("interaction_shrink_loc", -2.0),
                    interaction_shrink_scale=s3.get("interaction_shrink_scale", 0.5),
                    interaction_initial_scale=s3.get("interaction_initial_scale", 0.1)
                )
            except ImportError as e:
                logger.error(f"Failed to import CustomAnovaKernelFactory: {e}")
                logger.warning("Falling back to default Matern kernel.")
            except Exception as e:
                logger.error(f"Error initializing factory: {e}")
                raise
        
        elif kernel_type == "custom_basegp_residual_mixed":
            logger.info("Using Custom BaseGP Residual Mixed Factory (Mixed Mult/Add)")
            try:
                from extensions.custom_factory.custom_basegp_residual_mixed_factory import CustomBaseGPResidualMixedFactory
                
                dim_specs = s3.get("dim_specs", [])
                if not dim_specs:
                    raise ValueError("custom_basegp_residual_mixed requires 'dim_specs' in config")
                
                continuous_params = [d["name"] for d in dim_specs if d["type"] == "continuous"]
                discrete_params = {d["name"]: d["n_categories"] for d in dim_specs if d["type"] == "discrete"}
                
                factory = CustomBaseGPResidualMixedFactory(
                    dim=len(dim_specs),
                    continuous_params=continuous_params,
                    discrete_params=discrete_params,
                    kernel_combination=s3.get("kernel_combination", "mult"),
                    mean_type="pure_residual" # Step 3 默认使用纯残差模式
                )
            except ImportError as e:
                logger.error(f"Failed to import CustomBaseGPResidualMixedFactory: {e}")
                logger.warning("Falling back to default Matern kernel.")
            except Exception as e:
                logger.error(f"Error initializing factory: {e}")
                raise

        first_file = Path(context.subject_files[0])
        data_dir = first_file.parent
        
        # Call the internal process_step3
        results = process_step3(
            data_csv_path=str(data_dir),
            design_space_csv=design_space_csv,
            subject_col=s3["subject_col"],
            response_col=s3["response_col"],
            output_dir=str(output_dir),
            max_iters=s3["max_iters"],
            lr=s3["learning_rate"],
            use_cuda=s3["use_cuda"],
            ensure_diversity=s3["ensure_diversity"],
            factory=factory,
            model_spec=model_spec,
            y_normalization_type=s3.get("y_normalization_type", "zscore"),
            noise_lower_bound=s3.get("noise_lower_bound", 0.01),
        )
        
        context.model_path = str(output_dir / "base_gp_state.pth")
        logger.success(f"Step 3 completed. Model saved to {context.model_path}")
        
        return context
