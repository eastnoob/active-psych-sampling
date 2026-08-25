import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any
from core.base_module import BaseModule
from core.context import Context
from loguru import logger
from core.internal.phase1_analyzer import analyze_phase1_data

class Step2Module(BaseModule):
    """Step 2: Analysis Module."""
    name = "Analysis"
    description = "Extract Phase 2 parameters"

    def get_default_config(self) -> str:
        return """# Step 2: Analysis Configuration
design_csv = "data/design.csv" # 设计空间CSV文件路径 (可选，用于持久化路径信息)
subject_col = "subject_id" # 被试编号列名
response_col = "y" # 响应变量列名
max_pairs = 5 # 最多选择的交互对数量
min_pairs = 0 # 最少选择的交互对数量 (设为0允许算法判断无交互)
selection_method = "elbow" # 交互对选择方法: elbow (小样本推荐,用全部数据), stability (更保守), bic, top_k
suspected_pairs = [] # 值得怀疑的交互对索引列表，例如 [[0, 1], [2, 3]] (Phase 2 强制保留)
# 注意: 在小样本场景下(如20次/人)，自动检测准确率有限，建议使用 suspected_pairs 指定先验知识
# 原理: elbow使用全部数据,覆盖度优于stability的子采样;但可能有误报,需结合suspected_pairs验证
"""

    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        s2 = config.get("step2", {})
        
        if s2.get("max_pairs", 0) < s2.get("min_pairs", 0):
            errors.append("max_pairs cannot be less than min_pairs")
            
        return len(errors) == 0, errors

    def run(self, config: Dict[str, Any], context: Context) -> Context:
        s2 = config.get("step2")
        logger.info("Running Step 2: Analysis...")
        
        # Ensure design_space_path is in context if provided in config
        if not context.design_space_path and s2.get("design_csv"):
            context.design_space_path = s2.get("design_csv")
            
        if not context.subject_files:
            logger.error("No subject data files found in context.")
            raise RuntimeError("Missing input files for analysis.")
            
        # Load all data
        all_data = []
        for f in context.subject_files:
            df = pd.read_csv(f)
            # Ensure subject_id exists
            if s2["subject_col"] not in df.columns:
                df[s2["subject_col"]] = Path(f).stem
            all_data.append(df)
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Prepare data for analyzer
        # We need to separate factors and response
        response_col = s2["response_col"]
        subject_col = s2["subject_col"]
        
        if response_col not in combined_df.columns:
            logger.error(f"Response column '{response_col}' not found in data.")
            raise ValueError(f"Missing response column: {response_col}")
            
        # Factors are all columns except subject and response
        factor_cols = [c for c in combined_df.columns if c not in [subject_col, response_col]]
        
        X = combined_df[factor_cols].copy()
        # Encode categorical factors
        for col in X.columns:
            if X[col].dtype == object or X[col].dtype.name == 'category':
                X[col] = X[col].astype('category').cat.codes
        
        y = combined_df[response_col].values
        subjects = combined_df[subject_col].values
        
        # Call the analyzer
        results = analyze_phase1_data(
            X_warmup=X.values,
            y_warmup=y,
            subject_ids=subjects,
            factor_names=factor_cols,
            max_pairs=s2["max_pairs"],
            min_pairs=s2["min_pairs"],
            selection_method=s2["selection_method"],
            suspected_pairs=[tuple(p) for p in s2.get("suspected_pairs", [])],
            verbose=True
        )
        
        context.analysis_results = results
        
        # Save results to JSON
        import json
        output_dir = context.get_step_output_dir("step2")
        
        # Helper to convert numpy types for JSON
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            if isinstance(obj, dict):
                # Convert keys to string if they are tuples (common for interaction pairs)
                return {str(k) if isinstance(k, tuple) else k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, tuple):
                return [convert_numpy(i) for i in obj]
            return obj

        # 1. 保存详细分析结果
        analysis_path = output_dir / "analysis_results.json"
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(convert_numpy(results), f, indent=4, ensure_ascii=False)
        
        # 2. Save Model Specification (Recipe) - for Step 3
        spec_path = output_dir / "model_spec.json"
        
        # Add Chinese hints to the spec for user friendliness
        spec_with_hints = convert_numpy(results["model_spec"])
        spec_with_hints["_hints"] = {
            "kernel_type": "建议的核函数类型 (default 或 custom_anova)",
            "interaction_mode": "交互模式 (list: 仅使用识别到的对子)",
            "interaction_pairs": "识别到的关键交互项索引列表",
            "noise_floor": "估计的噪声底噪 (用于 GP 初始化)",
            "lengthscale_priors": "估计的主效应长度尺度先验 (用于 GP 初始化)"
        }
        
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(spec_with_hints, f, indent=4, ensure_ascii=False)
            
        # 3. Generate Human-readable Markdown Summary
        summary_path = output_dir / "model_spec_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Phase 2 Model Configuration Proposal\n\n")
            f.write(f"**Analysis Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            spec = results["model_spec"]
            f.write("## 1. Core Decisions\n")
            f.write(f"- **Recommended Kernel**: `{spec['kernel_type']}`\n")
            f.write(f"- **Interaction Mode**: `{spec['interaction_mode']}`\n\n")
            
            if spec['interaction_pairs']:
                f.write("## 2. Identified Key Interactions\n")
                f.write("The analyzer suggests monitoring these pairs in Phase 2:\n\n")
                for pair in spec['interaction_pairs']:
                    names = [factor_cols[i] for i in pair]
                    f.write(f"- {names[0]} × {names[1]} (Index: {pair})\n")
                f.write("\n")
            else:
                f.write("## 2. Interaction Conclusion\n")
                f.write("No significant interactions found. Main-effects-only modeling is recommended.\n\n")
                
            f.write("## 3. Prior Parameter Suggestions\n")
            f.write(f"- **Noise Floor**: {spec.get('noise_floor', 'N/A')}\n")
            f.write(f"- **Main Effect Lengthscales**: {spec.get('lengthscales', 'N/A')}\n\n")
            
            f.write("---\n")
            f.write("*Tip: You can manually edit `model_spec.json` in the same directory to override these settings.*\n")

        context.model_spec_path = str(spec_path)
        context.model_summary_path = str(summary_path)
        logger.success(f"Step 2 completed. Model spec saved to {spec_path}")
        
        return context
