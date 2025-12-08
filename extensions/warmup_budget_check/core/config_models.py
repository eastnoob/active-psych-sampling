"""
配置数据类 - 提供类型安全的配置管理
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
import time
import os
from pathlib import Path


@dataclass
class Step1Config:
    """
    Step1 配置：生成预热采样方案

    Attributes:
        design_csv_path: 设计空间CSV文件路径（只包含自变量列）
        n_subjects: 被试数量
        trials_per_subject: 每个被试的测试次数
        skip_interaction: 是否跳过交互效应探索
        output_dir: 输出目录路径
        merge: 是否合并为单个CSV文件
        subject_col_name: 被试编号列名（仅在merge=True时使用）
        auto_confirm: 是否自动确认（跳过交互确认）
    """

    # 必需参数
    design_csv_path: str
    n_subjects: int
    trials_per_subject: int

    # 可选参数（带默认值）
    skip_interaction: bool = True
    output_dir: Optional[str] = None  # 如果为None，使用默认格式
    merge: bool = False
    subject_col_name: str = "subject_id"
    auto_confirm: bool = True

    def __post_init__(self):
        """初始化后处理"""
        # 设置默认输出目录
        if self.output_dir is None:
            timestamp = time.strftime("%Y%m%d%H%M")
            self.output_dir = f"sample/{timestamp}"

    def validate(self) -> tuple[bool, List[str]]:
        """
        验证配置有效性

        Returns:
            tuple: (是否有效, 错误信息列表)
        """
        errors = []

        # 检查必需参数
        if not self.design_csv_path:
            errors.append("design_csv_path 不能为空")
        elif not Path(self.design_csv_path).exists():
            errors.append(f"设计文件不存在: {self.design_csv_path}")

        if self.n_subjects <= 0:
            errors.append("n_subjects 必须大于 0")
        if self.n_subjects > 1000:
            errors.append("n_subjects 不应超过 1000")

        if self.trials_per_subject <= 0:
            errors.append("trials_per_subject 必须大于 0")
        if self.trials_per_subject > 500:
            errors.append("trials_per_subject 不应超过 500")

        # 检查输出目录
        if self.output_dir and not isinstance(self.output_dir, str):
            errors.append("output_dir 必须是字符串")

        # 检查被试列名
        if not self.subject_col_name:
            errors.append("subject_col_name 不能为空")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "design_csv_path": self.design_csv_path,
            "n_subjects": self.n_subjects,
            "trials_per_subject": self.trials_per_subject,
            "skip_interaction": self.skip_interaction,
            "output_dir": self.output_dir,
            "merge": self.merge,
            "subject_col_name": self.subject_col_name,
            "auto_confirm": self.auto_confirm,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Step1Config":
        """从字典创建配置对象"""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: str) -> "Step1Config":
        """从JSON文件加载配置"""
        import json

        with open(json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_json(self, json_path: str) -> None:
        """保存配置到JSON文件"""
        import json

        config_dict = self.to_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


@dataclass
class Step2Config:
    """
    Step2 配置：分析 Phase 1 数据

    Attributes:
        data_csv_path: 实验数据CSV路径（包含响应列）
        subject_col: 被试编号列名
        response_col: 响应变量列名
        max_pairs: 最多选择的交互对数量
        min_pairs: 最少选择的交互对数量
        selection_method: 选择方法（elbow/bic_threshold/top_k）
        phase2_n_subjects: Phase 2 被试数
        phase2_trials_per_subject: Phase 2 每人测试次数
        lambda_adjustment: λ调整系数
        output_dir: 输出目录
        prefix: 文件前缀
        report_format: 报告格式（md/txt）
    """

    # 必需参数
    data_csv_path: str
    subject_col: str = "subject"
    response_col: str = "y"

    # 分析参数
    max_pairs: int = 5
    min_pairs: int = 5
    selection_method: str = "elbow"  # "elbow", "bic_threshold", "top_k"

    # Phase 2 参数
    phase2_n_subjects: int = 20
    phase2_trials_per_subject: int = 25
    lambda_adjustment: float = 1.2

    # 输出配置
    output_dir: str = "analysis_output"
    prefix: str = "phase1"
    report_format: str = "md"  # "md", "txt"

    def validate(self) -> tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []

        if not self.data_csv_path:
            errors.append("data_csv_path 不能为空")
        elif not Path(self.data_csv_path).exists():
            errors.append(f"数据文件不存在: {self.data_csv_path}")

        if not self.subject_col:
            errors.append("subject_col 不能为空")
        if not self.response_col:
            errors.append("response_col 不能为空")

        if self.max_pairs < self.min_pairs:
            errors.append("max_pairs 不能小于 min_pairs")
        if self.max_pairs < 1:
            errors.append("max_pairs 必须大于等于 1")
        if self.min_pairs < 1:
            errors.append("min_pairs 必须大于等于 1")

        if self.selection_method not in ["elbow", "bic_threshold", "top_k"]:
            errors.append(
                "selection_method 必须是 'elbow', 'bic_threshold', 或 'top_k'"
            )

        if self.phase2_n_subjects <= 0:
            errors.append("phase2_n_subjects 必须大于 0")
        if self.phase2_trials_per_subject <= 0:
            errors.append("phase2_trials_per_subject 必须大于 0")

        if self.lambda_adjustment <= 0:
            errors.append("lambda_adjustment 必须大于 0")

        if self.report_format not in ["md", "txt"]:
            errors.append("report_format 必须是 'md' 或 'txt'")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "data_csv_path": self.data_csv_path,
            "subject_col": self.subject_col,
            "response_col": self.response_col,
            "max_pairs": self.max_pairs,
            "min_pairs": self.min_pairs,
            "selection_method": self.selection_method,
            "phase2_n_subjects": self.phase2_n_subjects,
            "phase2_trials_per_subject": self.phase2_trials_per_subject,
            "lambda_adjustment": self.lambda_adjustment,
            "output_dir": self.output_dir,
            "prefix": self.prefix,
            "report_format": self.report_format,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Step2Config":
        """从字典创建配置对象"""
        return cls(**config_dict)


@dataclass
class Step3Config:
    """
    Step3 配置：Base GP 训练与设计空间扫描

    Attributes:
        data_csv_path: Phase1 数据路径（含响应列）
        design_space_csv: 设计空间CSV路径
        subject_col: 被试列名
        response_col: 响应列名
        max_iters: 训练最大迭代次数
        learning_rate: 学习率
        use_cuda: 是否使用CUDA
        ensure_diversity: 是否确保采样多样性
        output_dir: 输出目录
    """

    # 数据路径
    data_csv_path: str
    design_space_csv: str
    subject_col: str = "subject"
    response_col: str = "y"

    # 训练参数
    max_iters: int = 60
    learning_rate: float = 0.05
    use_cuda: bool = False

    # 采样参数
    ensure_diversity: bool = True

    # 输出配置
    output_dir: str = "base_gp_output"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "data_csv_path": self.data_csv_path,
            "design_space_csv": self.design_space_csv,
            "subject_col": self.subject_col,
            "response_col": self.response_col,
            "max_iters": self.max_iters,
            "learning_rate": self.learning_rate,
            "use_cuda": self.use_cuda,
            "ensure_diversity": self.ensure_diversity,
            "output_dir": self.output_dir,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Step3Config":
        """从字典创建配置对象"""
        return cls(**config_dict)

    def validate(self) -> tuple[bool, List[str]]:
        """验证配置有效性"""
        errors = []

        if not self.data_csv_path:
            errors.append("data_csv_path 不能为空")
        elif not Path(self.data_csv_path).exists():
            errors.append(f"数据文件不存在: {self.data_csv_path}")

        if not self.design_space_csv:
            errors.append("design_space_csv 不能为空")
        elif not Path(self.design_space_csv).exists():
            errors.append(f"设计空间文件不存在: {self.design_space_csv}")

        if not self.subject_col:
            errors.append("subject_col 不能为空")
        if not self.response_col:
            errors.append("response_col 不能为空")

        if self.max_iters <= 0:
            errors.append("max_iters 必须大于 0")
        if self.learning_rate <= 0 or self.learning_rate > 1.0:
            errors.append("learning_rate 必须在 (0, 1.0] 范围内")

        return len(errors) == 0, errors


@dataclass
class WarmupPipelineConfig:
    """
    完整流程配置：包含所有步骤的配置

    Attributes:
        step1: Step1 配置
        step2: Step2 配置（可选）
        step3: Step3 配置（可选）
    """

    step1: Step1Config
    step2: Optional[Step2Config] = None
    step3: Optional[Step3Config] = None

    def validate_all(self) -> tuple[bool, Dict[str, List[str]]]:
        """
        验证所有配置的有效性

        Returns:
            tuple: (是否全部有效, 各步骤的错误信息字典)
        """
        results = {}
        all_valid = True

        # 验证 Step1
        valid1, errors1 = self.step1.validate()
        results["step1"] = errors1
        if not valid1:
            all_valid = False

        # 验证 Step2（如果存在）
        if self.step2:
            valid2, errors2 = self.step2.validate()
            results["step2"] = errors2
            if not valid2:
                all_valid = False

        # 验证 Step3（如果存在）
        if self.step3:
            valid3, errors3 = self.step3.validate()
            results["step3"] = errors3
            if not valid3:
                all_valid = False

        return all_valid, results

    @classmethod
    def create_minimal(
        cls, design_csv: str, n_subjects: int, trials_per_subject: int
    ) -> "WarmupPipelineConfig":
        """创建最小配置（只包含 Step1）"""
        step1_config = Step1Config(
            design_csv_path=design_csv,
            n_subjects=n_subjects,
            trials_per_subject=trials_per_subject,
        )
        return cls(step1=step1_config)
