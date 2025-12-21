#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEPsych实验模块包
"""

# Oracle模块
from .oracle import (
    load_fixed_weights_from_json,
    create_oracle,
    print_oracle_spec,
    save_oracle_spec,
)

# 评估模块
from .evaluation_effect_recovery import (
    evaluate_effect_recovery_v2,
    evaluate_variance_components,
)
from .evaluation_effect_capture import (
    evaluate_effect_capture,
)
from .evaluation_model_discovery import (
    evaluate_effect_capture_v4,
)

# 采样循环模块
from .sampling_loop import (
    run_sampling_loop,
)

# Server管理模块
from .server_manager import (
    initialize_server,
    verify_server_components,
)

# 被试模型
# from .single_output_subject import (
#     SingleOutputLatentSubject,
# )

# 参数验证模块（251206新增）
from .param_validator import (
    AEPsychParamValidator,
    integrate_param_validation,
)

# 设计空间模块（251206新增）
from .design_space import (
    load_design_space,
    transform_to_numeric,
    generate_fallback_design_space,
    create_x4_4level_mapping,
    load_and_transform_design_space,
)

# 数据保存模块（251206新增）
from .data_saver import (
    save_sampling_data,
    update_summary_with_results,
    save_experiment_summary,
    load_sampling_history,
    load_interaction_log,
    load_eur_diagnostics,
)

__all__ = [
    # Oracle
    "load_fixed_weights_from_json",
    "create_oracle",
    "print_oracle_spec",
    "save_oracle_spec",
    # Evaluation
    "evaluate_effect_recovery_v2",
    "evaluate_variance_components",
    "evaluate_effect_capture",
    "evaluate_effect_capture_v4",
    # Sampling
    "run_sampling_loop",
    # Server
    "initialize_server",
    "verify_server_components",
    # Subject
    # "SingleOutputLatentSubject",
    # Parameter Validation (251206)
    "AEPsychParamValidator",
    "integrate_param_validation",
    # Design Space (251206)
    "load_design_space",
    "transform_to_numeric",
    "generate_fallback_design_space",
    "create_x4_4level_mapping",
    "load_and_transform_design_space",
    # Data Saver (251206)
    "save_sampling_data",
    "update_summary_with_results",
    "save_experiment_summary",
    "load_sampling_history",
    "load_interaction_log",
    "load_eur_diagnostics",
]
