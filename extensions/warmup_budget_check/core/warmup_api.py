"""
外部 API 模块 - 提供易于使用的编程接口

功能：
1. 函数式 API：直接调用函数
2. 类式 API：流程管理器
3. 便捷函数：快速使用
4. 完整的结果返回和错误处理
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict

# 导入内部模块
try:
    from warmup_sampler import WarmupSampler
    from analyze_phase1 import Phase1DataAnalyzer
    from phase1_step3_base_gp import process_step3
except ImportError:
    # 如果直接运行，添加当前目录到路径
    sys.path.append(str(Path(__file__).parent))
    from warmup_sampler import WarmupSampler
    from analyze_phase1 import Phase1DataAnalyzer
    from phase1_step3_base_gp import process_step3

from config_models import Step1Config, Step2Config, Step3Config, WarmupPipelineConfig


# ==================== 常量定义 ====================

# 返回值状态常量
SUCCESS = "success"
ERROR = "error"
WARNING = "warning"

# 预算评估结果
ADEQUACY_EXCELLENT = "充分"
ADEQUACY_GOOD = "刚好"
ADEQUACY_ADEQUATE = "基本满足"
ADEQUACY_INSUFFICIENT = "不足"
ADEQUACY_SEVERE = "严重不足"
ADEQUACY_EXCESSIVE = "过度充足（可优化）"
ADEQUACY_MARGINAL = "勉强"


# ==================== 工具函数 ====================


def _ensure_output_dir(output_dir: str) -> str:
    """确保输出目录存在"""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _format_duration(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def _safe_call(func: Callable, *args, **kwargs) -> Dict[str, Any]:
    """
    安全调用函数，捕获异常并返回标准格式

    Args:
        func: 要调用的函数
        *args: 函数参数
        **kwargs: 函数关键字参数

    Returns:
        标准格式的结果字典
    """
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        return {
            "success": True,
            "result": result,
            "warnings": [],
            "errors": [],
            "execution_time": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    except Exception as e:
        duration = time.time() - start_time

        return {
            "success": False,
            "result": None,
            "warnings": [],
            "errors": [str(e), traceback.format_exc()],
            "execution_time": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


# ==================== Step1 API ====================


def run_step1(
    config: Union[Step1Config, Dict[str, Any]], strict_mode: bool = False
) -> Dict[str, Any]:
    """
    运行 Step1：生成预热采样方案

    Args:
        config: Step1Config 对象或配置字典
        strict_mode: 是否严格模式（预算不足时抛出异常）

    Returns:
        Dict: 包含执行结果的字典

        {
            "success": bool,
            "adequacy": str,
            "budget": dict,
            "files": list,
            "output_dir": str,
            "warnings": list,
            "errors": list,
            "execution_time": float,
            "timestamp": str
        }
    """

    # 配置标准化
    if isinstance(config, dict):
        config = Step1Config.from_dict(config)

    # 验证配置
    is_valid, validation_errors = config.validate()
    if not is_valid:
        return {
            "success": False,
            "adequacy": None,
            "budget": None,
            "files": [],
            "output_dir": None,
            "warnings": [],
            "errors": validation_errors,
            "execution_time": 0.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _run_step1_internal():
        """内部执行函数"""
        # 确保输出目录存在
        output_dir = _ensure_output_dir(config.output_dir)

        # 创建采样器
        sampler = WarmupSampler(config.design_csv_path)

        # 评估预算
        adequacy, budget = sampler.evaluate_budget(
            n_subjects=config.n_subjects,
            trials_per_subject=config.trials_per_subject,
            skip_interaction=config.skip_interaction,
        )

        # 检查预算是否充足（严格模式）
        if strict_mode and adequacy in [ADEQUACY_INSUFFICIENT, ADEQUACY_SEVERE]:
            raise ValueError(f"预算评估为【{adequacy}】，在严格模式下不允许继续")

        # 生成采样文件
        exported_files = sampler.generate_samples(
            budget=budget,
            output_dir=output_dir,
            merge=config.merge,
            subject_col_name=config.subject_col_name,
        )

        return {
            "adequacy": adequacy,
            "budget": budget,
            "files": exported_files,
            "output_dir": output_dir,
            "config": config.to_dict(),
        }

    # 执行并返回结果
    result = _safe_call(_run_step1_internal)

    # 如果执行失败，直接返回
    if not result["success"]:
        return result

    # 处理成功的情况
    internal_result = result["result"]

    return {
        "success": True,
        "adequacy": internal_result["adequacy"],
        "budget": internal_result["budget"],
        "files": internal_result["files"],
        "output_dir": internal_result["output_dir"],
        "warnings": [],
        "errors": [],
        "execution_time": result["execution_time"],
        "timestamp": result["timestamp"],
        "metadata": {
            "config": internal_result["config"],
            "duration_formatted": _format_duration(result["execution_time"]),
        },
    }


def quick_step1(
    design_csv: str, n_subjects: int, trials_per_subject: int, **kwargs
) -> Dict[str, Any]:
    """
    快速运行 Step1（最少参数）

    Args:
        design_csv: 设计空间CSV路径
        n_subjects: 被试数量
        trials_per_subject: 每人测试次数
        **kwargs: 其他可选参数

    Returns:
        Dict: 执行结果
    """
    config = Step1Config(
        design_csv_path=design_csv,
        n_subjects=n_subjects,
        trials_per_subject=trials_per_subject,
        **kwargs,
    )
    return run_step1(config)


# ==================== Step2 API ====================


def run_step2(
    config: Union[Step2Config, Dict[str, Any]], strict_mode: bool = False
) -> Dict[str, Any]:
    """
    运行 Step2：分析 Phase 1 数据

    Args:
        config: Step2Config 对象或配置字典
        strict_mode: 是否严格模式

    Returns:
        Dict: 包含分析结果的字典
    """

    # 配置标准化
    if isinstance(config, dict):
        config = Step2Config.from_dict(config)

    # 验证配置
    is_valid, validation_errors = config.validate()
    if not is_valid:
        return {
            "success": False,
            "errors": validation_errors,
            "warnings": [],
            "execution_time": 0.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _run_step2_internal():
        """内部执行函数"""
        # 创建分析器
        analyzer = Phase1DataAnalyzer(
            data_csv_path=config.data_csv_path,
            subject_col=config.subject_col,
            response_col=config.response_col,
        )

        # 执行分析
        analysis = analyzer.analyze(
            max_pairs=config.max_pairs,
            min_pairs=config.min_pairs,
            selection_method=config.selection_method,
            verbose=False,  # API 模式下不输出详细信息
        )

        # 生成 Phase 2 配置
        phase2_config = analyzer.generate_phase2_config(
            n_subjects=config.phase2_n_subjects,
            trials_per_subject=config.phase2_trials_per_subject,
            lambda_adjustment=config.lambda_adjustment,
        )

        # 导出报告
        exported_files = analyzer.export_report(
            phase2_config=phase2_config,
            output_dir=config.output_dir,
            prefix=config.prefix,
            report_format=config.report_format,
        )

        return {
            "analysis": analysis,
            "phase2_config": phase2_config,
            "files": exported_files,
            "config": config.to_dict(),
        }

    # 执行并返回结果
    result = _safe_call(_run_step2_internal)

    if not result["success"]:
        return result

    # 处理成功的情况
    internal_result = result["result"]

    return {
        "success": True,
        "analysis": internal_result["analysis"],
        "phase2_config": internal_result["phase2_config"],
        "files": internal_result["files"],
        "warnings": [],
        "errors": [],
        "execution_time": result["execution_time"],
        "timestamp": result["timestamp"],
        "metadata": {
            "config": internal_result["config"],
            "duration_formatted": _format_duration(result["execution_time"]),
        },
    }


# ==================== Step3 API ====================


def run_step3(
    config: Union[Step3Config, Dict[str, Any]], strict_mode: bool = False
) -> Dict[str, Any]:
    """
    运行 Step3：训练 Base GP & 扫描设计空间

    Args:
        config: Step3Config 对象或配置字典
        strict_mode: 是否严格模式

    Returns:
        Dict: 包含训练结果的字典
    """

    # 配置标准化
    if isinstance(config, dict):
        config = Step3Config.from_dict(config)

    # 验证配置
    is_valid, validation_errors = config.validate()
    if not is_valid:
        return {
            "success": False,
            "errors": validation_errors,
            "warnings": [],
            "execution_time": 0.0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _run_step3_internal():
        """内部执行函数"""
        result = process_step3(
            data_csv_path=config.data_csv_path,
            design_space_csv=config.design_space_csv,
            subject_col=config.subject_col,
            response_col=config.response_col,
            output_dir=config.output_dir,
            max_iters=config.max_iters,
            lr=config.learning_rate,
            use_cuda=config.use_cuda,
            ensure_diversity=config.ensure_diversity,
        )

        return {"result": result, "config": config.to_dict()}

    # 执行并返回结果
    result = _safe_call(_run_step3_internal)

    if not result["success"]:
        return result

    # 处理成功的情况
    internal_result = result["result"]

    return {
        "success": True,
        "result": internal_result["result"],
        "warnings": [],
        "errors": [],
        "execution_time": result["execution_time"],
        "timestamp": result["timestamp"],
        "metadata": {
            "config": internal_result["config"],
            "duration_formatted": _format_duration(result["execution_time"]),
        },
    }


# ==================== 流程管理器 API ====================


class WarmupPipeline:
    """
    预热采样流程管理器

    提供链式调用和完整的流程管理
    """

    def __init__(self, config: Optional[WarmupPipelineConfig] = None):
        """
        初始化流程管理器

        Args:
            config: 流程配置对象
        """
        self.config = config
        self.results = {}
        self.execution_history = []

    def configure_step1(self, **kwargs) -> "WarmupPipeline":
        """
        配置 Step1 参数（链式调用）

        Returns:
            WarmupPipeline: 返回自身以支持链式调用
        """
        if self.config is None:
            raise ValueError("请先使用 create_minimal() 创建基础配置")

        # 更新 Step1 配置
        step1_dict = self.config.step1.to_dict()
        step1_dict.update(kwargs)
        self.config.step1 = Step1Config.from_dict(step1_dict)
        return self

    def configure_step2(self, **kwargs) -> "WarmupPipeline":
        """
        配置 Step2 参数（链式调用）

        Returns:
            WarmupPipeline: 返回自身以支持链式调用
        """
        if self.config is None:
            raise ValueError("请先使用 create_minimal() 创建基础配置")

        if self.config.step2 is None:
            self.config.step2 = Step2Config(
                data_csv_path="placeholder.csv"
            )  # 会被后续参数覆盖

        # 更新 Step2 配置
        step2_dict = self.config.step2.to_dict()
        step2_dict.update(kwargs)
        self.config.step2 = Step2Config.from_dict(step2_dict)
        return self

    def configure_step3(self, **kwargs) -> "WarmupPipeline":
        """
        配置 Step3 参数（链式调用）

        Returns:
            WarmupPipeline: 返回自身以支持链式调用
        """
        if self.config is None:
            raise ValueError("请先使用 create_minimal() 创建基础配置")

        if self.config.step3 is None:
            self.config.step3 = Step3Config(
                data_csv_path="placeholder.csv", design_space_csv="placeholder.csv"
            )

        # 更新 Step3 配置
        step3_dict = self.config.step3.to_dict()
        step3_dict.update(kwargs)
        self.config.step3 = Step3Config.from_dict(step3_dict)
        return self

    def run_step1(self, strict_mode: bool = False) -> Dict[str, Any]:
        """
        执行 Step1

        Returns:
            Dict: 执行结果
        """
        if self.config is None or self.config.step1 is None:
            raise ValueError("Step1 配置未设置")

        result = run_step1(self.config.step1, strict_mode)
        self.results["step1"] = result
        self.execution_history.append(("step1", result))
        return result

    def run_step2(self, strict_mode: bool = False) -> Dict[str, Any]:
        """
        执行 Step2

        Returns:
            Dict: 执行结果
        """
        if self.config is None or self.config.step2 is None:
            raise ValueError("Step2 配置未设置")

        result = run_step2(self.config.step2, strict_mode)
        self.results["step2"] = result
        self.execution_history.append(("step2", result))
        return result

    def run_step3(self, strict_mode: bool = False) -> Dict[str, Any]:
        """
        执行 Step3

        Returns:
            Dict: 执行结果
        """
        if self.config is None or self.config.step3 is None:
            raise ValueError("Step3 配置未设置")

        result = run_step3(self.config.step3, strict_mode)
        self.results["step3"] = result
        self.execution_history.append(("step3", result))
        return result

    def run_all(self, strict_mode: bool = False) -> Dict[str, Any]:
        """
        执行完整流程

        Returns:
            Dict: 完整流程结果
        """
        overall_result = {
            "success": True,
            "steps": {},
            "execution_summary": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 执行 Step1
        if self.config and self.config.step1:
            step1_result = self.run_step1(strict_mode)
            overall_result["steps"]["step1"] = step1_result
            if not step1_result["success"] and strict_mode:
                overall_result["success"] = False
                return overall_result

        # 执行 Step2
        if self.config and self.config.step2:
            step2_result = self.run_step2(strict_mode)
            overall_result["steps"]["step2"] = step2_result
            if not step2_result["success"] and strict_mode:
                overall_result["success"] = False
                return overall_result

        # 执行 Step3
        if self.config and self.config.step3:
            step3_result = self.run_step3(strict_mode)
            overall_result["steps"]["step3"] = step3_result
            if not step3_result["success"] and strict_mode:
                overall_result["success"] = False
                return overall_result

        # 计算总执行时间
        total_time = sum(
            result.get("execution_time", 0)
            for result in overall_result["steps"].values()
        )
        overall_result["execution_summary"] = {
            "total_execution_time": total_time,
            "total_steps": len(overall_result["steps"]),
            "successful_steps": sum(
                1 for result in overall_result["steps"].values() if result["success"]
            ),
            "duration_formatted": _format_duration(total_time),
        }

        return overall_result

    def get_result(self, step_name: str) -> Optional[Dict[str, Any]]:
        """
        获取指定步骤的结果

        Args:
            step_name: 步骤名称（"step1", "step2", "step3"）

        Returns:
            Dict: 步骤结果，如果不存在则返回 None
        """
        return self.results.get(step_name)

    def get_all_results(self) -> Dict[str, Any]:
        """
        获取所有结果

        Returns:
            Dict: 所有步骤的结果
        """
        return self.results.copy()

    def save_results(self, output_path: str) -> None:
        """
        保存所有结果到 JSON 文件

        Args:
            output_path: 输出文件路径
        """
        import json

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)


# ==================== 便捷函数 ====================


def create_pipeline(
    design_csv: str, n_subjects: int, trials_per_subject: int
) -> WarmupPipeline:
    """
    创建流程管理器

    Args:
        design_csv: 设计空间CSV路径
        n_subjects: 被试数量
        trials_per_subject: 每人测试次数

    Returns:
        WarmupPipeline: 流程管理器实例
    """
    config = WarmupPipelineConfig.create_minimal(
        design_csv, n_subjects, trials_per_subject
    )
    return WarmupPipeline(config)


def batch_step1(
    configs: List[Union[Step1Config, Dict[str, Any]]], output_dir: str = "batch_results"
) -> Dict[str, Any]:
    """
    批量运行 Step1

    Args:
        configs: 配置列表
        output_dir: 输出目录

    Returns:
        Dict: 批量执行结果
    """
    results = {
        "success": True,
        "total_configs": len(configs),
        "successful": 0,
        "failed": 0,
        "results": [],
        "summary": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    for i, config in enumerate(configs):
        if isinstance(config, dict):
            config = Step1Config.from_dict(config)

        # 为每个配置创建独立的输出目录
        config_output_dir = f"{output_dir}/config_{i+1}"
        config.output_dir = config_output_dir

        result = run_step1(config)
        results["results"].append(
            {"config_index": i, "config": config.to_dict(), "result": result}
        )

        if result["success"]:
            results["successful"] += 1
        else:
            results["failed"] += 1
            results["success"] = False

    results["summary"] = {
        "success_rate": (
            results["successful"] / results["total_configs"]
            if results["total_configs"] > 0
            else 0
        ),
        "average_adequacy": (
            sum(r["result"].get("adequacy", "") for r in results["results"])
            / len(results["results"])
            if results["results"]
            else 0
        ),
    }

    return results


# ==================== 结果类 ====================


@dataclass
class Step1Result:
    """Step1 执行结果"""

    success: bool
    output_dir: str
    exported_files: List[str]
    budget_adequacy: str
    budget_details: Dict[str, Any]
    execution_time: float
    timestamp: str


@dataclass
class Step2Result:
    """Step2 执行结果"""

    success: bool
    selected_pairs: List[Dict[str, Any]]
    phase2_config: Dict[str, Any]
    exported_files: Dict[str, str]
    execution_time: float
    timestamp: str


@dataclass
class Step3Result:
    """Step3 执行结果"""

    success: bool
    output_dir: str
    n_design_points: int
    key_points: List[Dict[str, Any]]
    lengthscales: Dict[str, float]
    execution_time: float
    timestamp: str


@dataclass
class ChainResult:
    """链式执行结果"""

    step1_result: Optional[Step1Result] = None
    step2_result: Optional[Step2Result] = None
    step3_result: Optional[Step3Result] = None
    success: bool = False
    execution_time: float = 0.0
    timestamp: str = ""


# ==================== 简化流程管理器 ====================


class Step1Step2Chain:
    """步骤1 -> 步骤2 链式执行管理器"""

    def __init__(self, step1_config: Step1Config, step2_config: Step2Config):
        self.step1_config = step1_config
        self.step2_config = step2_config

    def execute(self) -> ChainResult:
        """执行完整的链式流程"""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 执行步骤1
            step1_api_result = run_step1(self.step1_config)

            if not step1_api_result["success"]:
                return ChainResult(
                    success=False,
                    execution_time=time.time() - start_time,
                    timestamp=timestamp,
                )

            # 自动设置步骤2的数据文件路径：当用户配置的路径为空或不存在时，尝试使用 Step1 的输出
            try:
                configured_path = (
                    Path(self.step2_config.data_csv_path)
                    if getattr(self.step2_config, "data_csv_path", None)
                    else None
                )
            except Exception:
                configured_path = None

            if configured_path is None or not configured_path.exists():
                step2_data_path = (
                    Path(step1_api_result["output_dir"])
                    / "result"
                    / "combined_results.csv"
                )
                if step2_data_path.exists():
                    self.step2_config.data_csv_path = str(step2_data_path)

            # 执行步骤2
            step2_api_result = run_step2(self.step2_config)

            if not step2_api_result["success"]:
                return ChainResult(
                    step1_result=Step1Result(
                        success=True,
                        output_dir=step1_api_result["output_dir"],
                        exported_files=step1_api_result["files"],
                        budget_adequacy=step1_api_result["adequacy"],
                        budget_details=step1_api_result["budget"],
                        execution_time=step1_api_result["execution_time"],
                        timestamp=step1_api_result["timestamp"],
                    ),
                    success=False,
                    execution_time=time.time() - start_time,
                    timestamp=timestamp,
                )

            return ChainResult(
                step1_result=Step1Result(
                    success=True,
                    output_dir=step1_api_result["output_dir"],
                    exported_files=step1_api_result["files"],
                    budget_adequacy=step1_api_result["adequacy"],
                    budget_details=step1_api_result["budget"],
                    execution_time=step1_api_result["execution_time"],
                    timestamp=step1_api_result["timestamp"],
                ),
                step2_result=Step2Result(
                    success=True,
                    selected_pairs=step2_api_result["analysis"]["selected_pairs"],
                    phase2_config=step2_api_result["phase2_config"],
                    exported_files=step2_api_result["files"],
                    execution_time=step2_api_result["execution_time"],
                    timestamp=step2_api_result["timestamp"],
                ),
                success=True,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
            )

        except Exception as e:
            return ChainResult(
                success=False,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
            )


class Step1Step2Step3Chain:
    """步骤1 -> 步骤2 -> 步骤3 链式执行管理器"""

    def __init__(
        self,
        step1_config: Step1Config,
        step2_config: Step2Config,
        step3_config: Step3Config,
    ):
        self.step1_config = step1_config
        self.step2_config = step2_config
        self.step3_config = step3_config

    def execute(self) -> ChainResult:
        """执行完整的链式流程"""
        start_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 执行步骤1
            step1_api_result = run_step1(self.step1_config)

            if not step1_api_result["success"]:
                return ChainResult(
                    success=False,
                    execution_time=time.time() - start_time,
                    timestamp=timestamp,
                )

            # 自动设置步骤2的数据文件路径：当用户配置的路径为空或不存在时，尝试使用 Step1 的输出
            try:
                configured_path = (
                    Path(self.step2_config.data_csv_path)
                    if getattr(self.step2_config, "data_csv_path", None)
                    else None
                )
            except Exception:
                configured_path = None

            if configured_path is None or not configured_path.exists():
                step2_data_path = (
                    Path(step1_api_result["output_dir"])
                    / "result"
                    / "combined_results.csv"
                )
                if step2_data_path.exists():
                    self.step2_config.data_csv_path = str(step2_data_path)

            # 执行步骤2
            step2_api_result = run_step2(self.step2_config)

            if not step2_api_result["success"]:
                return ChainResult(
                    step1_result=Step1Result(
                        success=True,
                        output_dir=step1_api_result["output_dir"],
                        exported_files=step1_api_result["files"],
                        budget_adequacy=step1_api_result["adequacy"],
                        budget_details=step1_api_result["budget"],
                        execution_time=step1_api_result["execution_time"],
                        timestamp=step1_api_result["timestamp"],
                    ),
                    success=False,
                    execution_time=time.time() - start_time,
                    timestamp=timestamp,
                )

            # 自动设置步骤3的数据文件路径：当用户配置的路径为空或不存在时，尝试使用 Step1 的输出
            try:
                configured_path3 = (
                    Path(self.step3_config.data_csv_path)
                    if getattr(self.step3_config, "data_csv_path", None)
                    else None
                )
            except Exception:
                configured_path3 = None

            if configured_path3 is None or not configured_path3.exists():
                step3_data_path = (
                    Path(step1_api_result["output_dir"])
                    / "result"
                    / "combined_results.csv"
                )
                if step3_data_path.exists():
                    self.step3_config.data_csv_path = str(step3_data_path)

            # 自动设置步骤3的设计空间文件（若未指定或不存在，则使用 Step1 的输入设计文件）
            try:
                configured_design = (
                    Path(self.step3_config.design_space_csv)
                    if getattr(self.step3_config, "design_space_csv", None)
                    else None
                )
            except Exception:
                configured_design = None

            if configured_design is None or not configured_design.exists():
                self.step3_config.design_space_csv = self.step1_config.design_csv_path

            # 执行步骤3
            step3_api_result = run_step3(self.step3_config)

            if not step3_api_result["success"]:
                return ChainResult(
                    step1_result=Step1Result(
                        success=True,
                        output_dir=step1_api_result["output_dir"],
                        exported_files=step1_api_result["files"],
                        budget_adequacy=step1_api_result["adequacy"],
                        budget_details=step1_api_result["budget"],
                        execution_time=step1_api_result["execution_time"],
                        timestamp=step1_api_result["timestamp"],
                    ),
                    step2_result=Step2Result(
                        success=True,
                        selected_pairs=step2_api_result["analysis"]["selected_pairs"],
                        phase2_config=step2_api_result["phase2_config"],
                        exported_files=step2_api_result["files"],
                        execution_time=step2_api_result["execution_time"],
                        timestamp=step2_api_result["timestamp"],
                    ),
                    success=False,
                    execution_time=time.time() - start_time,
                    timestamp=timestamp,
                )

            return ChainResult(
                step1_result=Step1Result(
                    success=True,
                    output_dir=step1_api_result["output_dir"],
                    exported_files=step1_api_result["files"],
                    budget_adequacy=step1_api_result["adequacy"],
                    budget_details=step1_api_result["budget"],
                    execution_time=step1_api_result["execution_time"],
                    timestamp=step1_api_result["timestamp"],
                ),
                step2_result=Step2Result(
                    success=True,
                    selected_pairs=step2_api_result["analysis"]["selected_pairs"],
                    phase2_config=step2_api_result["phase2_config"],
                    exported_files=step2_api_result["files"],
                    execution_time=step2_api_result["execution_time"],
                    timestamp=step2_api_result["timestamp"],
                ),
                step3_result=Step3Result(
                    success=True,
                    output_dir=step3_api_result["result"]["output_dir"],
                    n_design_points=step3_api_result["result"]["n_design_points"],
                    key_points=step3_api_result["result"].get("key_points", []),
                    lengthscales=step3_api_result["result"].get("lengthscales", {}),
                    execution_time=step3_api_result["execution_time"],
                    timestamp=step3_api_result["timestamp"],
                ),
                success=True,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
            )

        except Exception as e:
            return ChainResult(
                success=False,
                execution_time=time.time() - start_time,
                timestamp=timestamp,
            )
