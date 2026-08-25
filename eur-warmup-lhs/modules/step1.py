import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any

from core.base_module import BaseModule
from core.context import Context
from core.internal.lhs_sampler import LhsSampler
from loguru import logger

class Step1Module(BaseModule):
    """Step 1: Warmup Sampling Module."""
    name = "Sampling"
    description = "Generate subject sampling plans"

    def get_default_config(self) -> str:
        return """# Step 1: Warmup Sampling Configuration
design_csv = "data/design.csv" # 设计空间CSV文件路径（只包含自变量）
n_subjects = 5 # 被试数量
trials_per_subject = 10 # 每个被试的测试次数
interaction_mode = "hybrid" # 交互设置保留给后续步骤与元信息
interaction_pairs = [] # 指定的交互对索引列表，例如 [[0, 1], [2, 3]]
output_root = "output" # 整个工作流输出根目录；实际步骤目录固定为 output_root/timestamp/step1
allocation_mode = "disjoint" # 分配模式: disjoint (默认每人不同) / shared (所有人同一套)
shared_points = 0 # 所有被试共享的 LHS 点数，默认 0 表示纯覆盖
lhs_oversample_factor = 5 # LHS 候选放大倍数，越大越容易映射到更分散的离散点
merge = false # 是否合并为一个CSV文件
auto_confirm = false # 是否自动确认预算评估 (如果为 false，则会暂停等待用户确认)
repeat_one_point = false # 是否在被试内增加一个重复点 (用于估计纯误差/噪声)
random_seed = 42 # 随机种子
"""

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        s1 = config.get("step1", {})
        
        design_csv = s1.get("design_csv")
        if not design_csv or not Path(design_csv).exists():
            errors.append(f"Design CSV not found: {design_csv}")
            
        if s1.get("n_subjects", 0) <= 0:
            errors.append("n_subjects must be > 0")
            
        if s1.get("trials_per_subject", 0) <= 0:
            errors.append("trials_per_subject must be > 0")
            
        mode = s1.get("interaction_mode", "hybrid")
        if mode not in ["free", "specified_only", "hybrid"]:
            errors.append(f"Invalid interaction_mode: {mode}")

        allocation_mode = s1.get("allocation_mode", "disjoint")
        if allocation_mode not in ["disjoint", "shared"]:
            errors.append(f"Invalid allocation_mode: {allocation_mode}")

        shared_points = s1.get("shared_points", 0)
        if shared_points < 0:
            errors.append("shared_points must be >= 0")
        elif shared_points > s1.get("trials_per_subject", 0):
            errors.append("shared_points cannot exceed trials_per_subject")

        if "output_dir" in s1 and s1.get("output_dir") != "step1":
            errors.append("output_dir is fixed to 'step1' in the LHS version to preserve the original step1/step1_5/step2/step3 structure")
            
        return len(errors) == 0, errors

    def run(self, config: Dict[str, Any], context: Context) -> Context:
        from rich.prompt import Confirm

        s1 = config.get("step1")
        logger.info("Running Step 1: LHS Warmup Sampling...")
        
        design_csv = s1["design_csv"]
        n_subjects = s1["n_subjects"]
        trials = s1["trials_per_subject"]

        interaction_pairs = [tuple(p) for p in s1.get("interaction_pairs", [])]

        sampler = LhsSampler(design_csv)
        sampler.describe_design_space()
        plan = sampler.evaluate_plan(
            n_subjects=n_subjects,
            trials_per_subject=trials,
            shared_points=s1.get("shared_points", 0),
            allocation_mode=s1.get("allocation_mode", "disjoint"),
        )
        sampler.print_plan_report(plan, interaction_pairs=interaction_pairs)
        
        if not s1.get("auto_confirm", False):
            if not Confirm.ask("\n[bold yellow]LHS sampling plan ready. Proceed with generating sampling plans?[/bold yellow]"):
                logger.warning("User cancelled execution after budget estimation.")
                raise InterruptedError("Execution cancelled by user.")

        output_root = s1.get("output_root")
        if output_root:
            context.output_root = Path(output_root)

        output_dir = context.get_step_output_dir("step1")

        subject_files = sampler.generate_samples(
            n_subjects=n_subjects,
            trials_per_subject=trials,
            output_dir=str(output_dir),
            merge=s1.get("merge", False),
            subject_col_name="subject_id",
            shared_points=s1.get("shared_points", 0),
            allocation_mode=s1.get("allocation_mode", "disjoint"),
            lhs_oversample_factor=s1.get("lhs_oversample_factor", 5),
            interaction_pairs_to_explore=interaction_pairs,
            repeat_one_point=s1.get("repeat_one_point", False),
            random_seed=s1.get("random_seed", 42)
        )
        
        context.design_space_path = design_csv
        context.design_df = pd.read_csv(design_csv)
        context.subject_files = [str(f) for f in subject_files if f.endswith(".csv")]
        
        logger.success(f"Step 1 completed. Generated {len(context.subject_files)} subject files.")
        return context
