"""
AEPsych 配置文件构建和验证工具

这个模块提供 AEPsychConfigBuilder 类，用于辅助创建和验证 AEPsych 实验配置。
支持以下功能：
1. 新建配置：从默认模板开始，快速构建新配置
2. 加载现有配置：从 INI 文件加载现有配置进行编辑
3. 配置验证：验证配置的完整性和正确性
4. 实时预览：使用 print_configuration() 查看当前配置状态
5. 配置输出：保存为 INI 格式文件
"""

import configparser
import os
import re
from typing import Any, Dict, List, Optional, Tuple


# ANSI 颜色代码
class ColorCode:
    """ANSI 颜色代码常数"""

    # 前景色
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # 背景色
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"

    # 样式
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"

    # 重置
    RESET = "\033[0m"


class AEPsychConfigBuilder:
    """
    AEPsych 配置文件构建和验证工具

    提供便利的方法来创建和验证 AEPsych 实验配置。支持以下功能：
    1. 新建配置时，自动加载默认模板
    2. 加载现有 INI 文件进行编辑
    3. 验证配置的完整性和正确性
    4. 防止遗漏必需字段
    5. 输出为 INI 格式
    6. 实时预览配置状态（彩色高亮【】占位符）
    """

    # 参数类型及其必需字段的映射
    PARAMETER_TYPE_REQUIRED_FIELDS = {
        "continuous": ["par_type", "lower_bound", "upper_bound"],
        "integer": ["par_type", "lower_bound", "upper_bound"],
        "binary": ["par_type"],
        "fixed": ["par_type", "value"],
        "categorical": ["par_type", "choices"],
    }

    # 策略的必需字段
    STRATEGY_TERMINATION_CRITERIA = ["min_asks", "min_total_tells", "max_asks"]

    # 生成器及其依赖
    GENERATOR_DEPENDENCIES = {
        "OptimizeAcqfGenerator": ["model"],  # 必需 model
        "UCBGenerator": ["model"],  # 通常需要 model
    }

    def __init__(self, auto_load_template: bool = True):
        """
        初始化配置构建器

        Args:
            auto_load_template: 新建时是否自动加载默认配置模板（默认 True）
        """
        self.config_dict: Dict[str, Dict[str, Any]] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []

        if auto_load_template:
            self._load_default_template()

    def add_common(
        self,
        parnames: List[str],
        stimuli_per_trial: int,
        outcome_types: List[str],
        strategy_names: List[str],
        **kwargs,
    ):
        """
        添加 [common] 部分

        Args:
            parnames: 参数名称列表，如 ['par1', 'par2']
            stimuli_per_trial: 刺激数量（1 或 2）
            outcome_types: 结果类型列表，如 ['binary'], ['continuous']
            strategy_names: 策略名称列表，如 ['init_strat', 'opt_strat']
            **kwargs: 其他可选字段，如 pregen_asks, extensions, outcome_names
        """
        if "common" not in self.config_dict:
            self.config_dict["common"] = {}

        # 添加必需字段
        self.config_dict["common"]["parnames"] = self._format_list(parnames)
        self.config_dict["common"]["stimuli_per_trial"] = str(stimuli_per_trial)
        self.config_dict["common"]["outcome_types"] = self._format_list(outcome_types)
        self.config_dict["common"]["strategy_names"] = self._format_list(strategy_names)

        # 添加可选字段
        if "pregen_asks" in kwargs:
            self.config_dict["common"]["pregen_asks"] = str(
                kwargs["pregen_asks"]
            ).lower()

        if "extensions" in kwargs:
            self.config_dict["common"]["extensions"] = self._format_list(
                kwargs["extensions"]
            )

        if "outcome_names" in kwargs:
            self.config_dict["common"]["outcome_names"] = self._format_list(
                kwargs["outcome_names"]
            )

        # 添加其他 kwargs
        for key, value in kwargs.items():
            if key not in ["pregen_asks", "extensions", "outcome_names"]:
                if isinstance(value, list):
                    self.config_dict["common"][key] = self._format_list(value)
                else:
                    self.config_dict["common"][key] = str(value)

    def add_parameter(self, name: str, par_type: str, **kwargs) -> bool:
        """
        添加参数定义，自动验证必需字段

        Args:
            name: 参数名称
            par_type: 参数类型（continuous, integer, binary, fixed, categorical）
            **kwargs: 参数配置，如 lower_bound, upper_bound, value, choices 等

        Returns:
            bool: 是否成功添加（验证通过）
        """
        if name not in self.config_dict:
            self.config_dict[name] = {}

        self.config_dict[name]["par_type"] = par_type

        # 添加参数配置
        for key, value in kwargs.items():
            if isinstance(value, list):
                self.config_dict[name][key] = self._format_list(value)
            else:
                self.config_dict[name][key] = str(value)

        # 验证该参数配置
        errors = self._validate_parameter(name, self.config_dict[name])
        if errors:
            self.errors.extend(errors)
            return False

        return True

    def add_strategy(self, name: str, generator: str, **kwargs) -> bool:
        """
        添加策略配置，自动验证依赖关系

        Args:
            name: 策略名称
            generator: 生成器类名（SobolGenerator, OptimizeAcqfGenerator 等）
            **kwargs: 策略配置，如 model, acqf, min_asks, max_asks 等

        Returns:
            bool: 是否成功添加（验证通过）
        """
        if name not in self.config_dict:
            self.config_dict[name] = {}

        self.config_dict[name]["generator"] = generator

        # 添加策略配置
        for key, value in kwargs.items():
            if isinstance(value, list):
                self.config_dict[name][key] = self._format_list(value)
            else:
                self.config_dict[name][key] = str(value)

        # 验证该策略配置
        errors = self._validate_strategy(name, self.config_dict[name])
        if errors:
            self.errors.extend(errors)
            return False

        return True

    def add_component_config(self, component_name: str, **kwargs):
        """
        添加组件配置（模型、生成器、采集函数等）

        这些配置是可选的，用于为特定组件提供详细配置。

        Args:
            component_name: 组件名称，如 'GPClassificationModel', 'OptimizeAcqfGenerator'
            **kwargs: 组件配置参数
        """
        if component_name not in self.config_dict:
            self.config_dict[component_name] = {}

        for key, value in kwargs.items():
            if isinstance(value, list):
                self.config_dict[component_name][key] = self._format_list(value)
            else:
                self.config_dict[component_name][key] = str(value)

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        """
        验证配置完整性，返回是否有效、错误列表和警告列表

        验证按以下优先级进行：
        1. 结构完整性：检查 [common] 部分及其必需字段
        2. 参数存在性：检查 parnames 中的每个参数是否有对应配置节
        3. 参数完整性：验证每个参数配置节的必需字段
        4. 策略存在性：检查 strategy_names 中的每个策略是否有对应配置节
        5. 策略完整性：验证每个策略配置节的必需字段和依赖
        6. 类型正确性：验证字段类型

        Returns:
            Tuple[bool, List[str], List[str]]: (是否有效, 错误列表, 警告列表)
        """
        self.errors = []
        self.warnings = []

        # 1. 检查 [common] 部分
        if "common" not in self.config_dict:
            self.errors.append("Missing [common] section")
            return False, self.errors, self.warnings

        common = self.config_dict["common"]

        # 检查 [common] 必需字段
        required_common = [
            "parnames",
            "stimuli_per_trial",
            "outcome_types",
            "strategy_names",
        ]
        for field in required_common:
            if field not in common:
                self.errors.append(f"[common]: missing required field '{field}'")

        if self.errors:
            return False, self.errors, self.warnings

        # 解析 parnames 和 strategy_names
        parnames = self._parse_list(common.get("parnames", "[]"))
        strategy_names = self._parse_list(common.get("strategy_names", "[]"))

        # 2. 检查参数存在性和完整性
        for parname in parnames:
            if parname not in self.config_dict:
                self.errors.append(
                    f"Parameter '{parname}' defined in parnames but has no config section"
                )
            else:
                errors = self._validate_parameter(parname, self.config_dict[parname])
                self.errors.extend(errors)

        # 3. 检查策略存在性和完整性
        for strategy_name in strategy_names:
            if strategy_name not in self.config_dict:
                self.errors.append(
                    f"Strategy '{strategy_name}' defined in strategy_names but has no config section"
                )
            else:
                errors = self._validate_strategy(
                    strategy_name, self.config_dict[strategy_name]
                )
                self.errors.extend(errors)

        # 4. 检查 stimuli_per_trial 的有效性
        try:
            stimuli = int(common.get("stimuli_per_trial", -1))
            if stimuli not in [1, 2]:
                self.errors.append(
                    f"[common]: stimuli_per_trial must be 1 or 2, got {stimuli}"
                )
        except ValueError:
            self.errors.append(
                f"[common]: stimuli_per_trial must be an integer, got '{common.get('stimuli_per_trial')}'"
            )

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings

    def to_ini(self, filepath: str, force: bool = False) -> None:
        """
        保存为 INI 文件

        Args:
            filepath: 输出文件路径
            force: 是否强制覆盖默认模板文件（默认 False）

        Raises:
            ValueError: 如果尝试覆盖默认模板文件且 force=False
        """
        # 检查是否尝试覆盖默认模板文件
        if not force and self._is_default_template_file(filepath):
            raise ValueError(
                f"无法覆盖默认模板文件: {filepath}\n"
                f"为了保护原始模板，请使用其他文件名保存。\n"
                f"如果确实要覆盖，请使用: to_ini(filepath, force=True)"
            )

        config = configparser.ConfigParser()

        for section, options in self.config_dict.items():
            config[section] = {}
            for key, value in options.items():
                config[section][key] = str(value)

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            config.write(f)

    @classmethod
    def from_ini(cls, filepath: str) -> "AEPsychConfigBuilder":
        """
        从现有 INI 文件加载（不加载默认模板）

        Args:
            filepath: INI 文件路径

        Returns:
            AEPsychConfigBuilder: 加载后的构建器实例
        """
        config = configparser.ConfigParser()
        config.read(filepath)

        # auto_load_template=False，不加载默认模板
        builder = cls(auto_load_template=False)
        builder.config_dict = {
            section: dict(config[section]) for section in config.sections()
        }

        return builder

    def get_missing_fields(self) -> Dict[str, List[str]]:
        """
        返回每个部分缺失的必需字段

        Returns:
            Dict[str, List[str]]: 字典，键为部分名称，值为缺失字段列表
        """
        missing = {}

        # 检查 [common]
        if "common" in self.config_dict:
            required = [
                "parnames",
                "stimuli_per_trial",
                "outcome_types",
                "strategy_names",
            ]
            common_missing = [
                f for f in required if f not in self.config_dict["common"]
            ]
            if common_missing:
                missing["[common]"] = common_missing
        else:
            missing["[common]"] = ["entire section missing"]

        # 检查参数
        if "common" in self.config_dict:
            parnames = self._parse_list(
                self.config_dict["common"].get("parnames", "[]")
            )
            for parname in parnames:
                if parname in self.config_dict:
                    par_type = self.config_dict[parname].get("par_type")
                    if par_type and par_type in self.PARAMETER_TYPE_REQUIRED_FIELDS:
                        required = self.PARAMETER_TYPE_REQUIRED_FIELDS[par_type]
                        param_missing = [
                            f for f in required if f not in self.config_dict[parname]
                        ]
                        if param_missing:
                            missing[f"[{parname}]"] = param_missing
                else:
                    missing[f"[{parname}]"] = ["entire section missing"]

        # 检查策略
        if "common" in self.config_dict:
            strategy_names = self._parse_list(
                self.config_dict["common"].get("strategy_names", "[]")
            )
            for strategy_name in strategy_names:
                if strategy_name in self.config_dict:
                    strategy_missing = []
                    # 检查是否有终止条件
                    has_termination = any(
                        c in self.config_dict[strategy_name]
                        for c in self.STRATEGY_TERMINATION_CRITERIA
                    )
                    if not has_termination:
                        strategy_missing.append(
                            "at least one of: min_asks, min_total_tells, max_asks"
                        )

                    # 检查生成器
                    if "generator" not in self.config_dict[strategy_name]:
                        strategy_missing.append("generator")

                    if strategy_missing:
                        missing[f"[{strategy_name}]"] = strategy_missing
                else:
                    missing[f"[{strategy_name}]"] = ["entire section missing"]

        return missing

    def print_validation_report(self) -> None:
        """打印验证报告"""
        is_valid, errors, warnings = self.validate()

        print("=" * 60)
        print("AEPsych 配置验证报告")
        print("=" * 60)

        if is_valid:
            print("✓ 配置有效！")
        else:
            print("✗ 配置无效")
            print("\n错误:")
            for error in errors:
                print(f"  - {error}")

        if warnings:
            print("\n警告:")
            for warning in warnings:
                print(f"  - {warning}")

        missing = self.get_missing_fields()
        if missing:
            print("\n缺失的必需字段:")
            for section, fields in missing.items():
                print(f"  {section}:")
                for field in fields:
                    print(f"    - {field}")
        else:
            print("\n所有必需字段都已配置")

        print("=" * 60)

    def get_summary(self) -> str:
        """获取配置摘要"""
        summary = []

        if "common" in self.config_dict:
            common = self.config_dict["common"]
            parnames = self._parse_list(common.get("parnames", "[]"))
            strategy_names = self._parse_list(common.get("strategy_names", "[]"))
            outcome_types = self._parse_list(common.get("outcome_types", "[]"))

            summary.append("配置摘要:")
            summary.append(f"  参数数量: {len(parnames)} - {parnames}")
            summary.append(f"  策略数量: {len(strategy_names)} - {strategy_names}")
            summary.append(f"  结果类型: {outcome_types}")
            summary.append(f"  刺激数/试验: {common.get('stimuli_per_trial', 'N/A')}")
        else:
            summary.append("配置摘要: 未初始化")

        return "\n".join(summary)

    # ==================== 私有辅助方法 ====================

    @staticmethod
    def _format_list(lst: List) -> str:
        """将 Python 列表格式化为 INI 列表字符串"""
        if isinstance(lst, str):
            return lst
        return str(lst)

    @staticmethod
    def _parse_list(list_str: str) -> List[str]:
        """解析 INI 列表字符串为 Python 列表"""
        if not list_str or list_str.strip() == "":
            return []

        # 移除外层括号和空格
        list_str = list_str.strip()
        if list_str.startswith("["):
            list_str = list_str[1:]
        if list_str.endswith("]"):
            list_str = list_str[:-1]

        # 分割并清理
        items = [item.strip().strip("'\"") for item in list_str.split(",")]
        return [item for item in items if item]

    def _validate_parameter(
        self, param_name: str, param_config: Dict[str, str]
    ) -> List[str]:
        """
        验证单个参数配置

        Args:
            param_name: 参数名称
            param_config: 参数配置字典

        Returns:
            List[str]: 错误列表
        """
        errors = []

        if "par_type" not in param_config:
            errors.append(f"Parameter [{param_name}]: missing 'par_type'")
            return errors

        par_type = param_config["par_type"].strip()

        if par_type == "continuous":
            if "lower_bound" not in param_config:
                errors.append(
                    f"Parameter [{param_name}]: 'continuous' type requires 'lower_bound'"
                )
            if "upper_bound" not in param_config:
                errors.append(
                    f"Parameter [{param_name}]: 'continuous' type requires 'upper_bound'"
                )

            # 验证边界是数字
            if "lower_bound" in param_config:
                try:
                    float(param_config["lower_bound"])
                except ValueError:
                    errors.append(
                        f"Parameter [{param_name}]: 'lower_bound' must be numeric"
                    )

            if "upper_bound" in param_config:
                try:
                    float(param_config["upper_bound"])
                except ValueError:
                    errors.append(
                        f"Parameter [{param_name}]: 'upper_bound' must be numeric"
                    )

        elif par_type == "integer":
            if "lower_bound" not in param_config:
                errors.append(
                    f"Parameter [{param_name}]: 'integer' type requires 'lower_bound'"
                )
            if "upper_bound" not in param_config:
                errors.append(
                    f"Parameter [{param_name}]: 'integer' type requires 'upper_bound'"
                )

            # 验证边界是整数
            if "lower_bound" in param_config:
                try:
                    int(param_config["lower_bound"])
                except ValueError:
                    errors.append(
                        f"Parameter [{param_name}]: 'lower_bound' must be integer"
                    )

            if "upper_bound" in param_config:
                try:
                    int(param_config["upper_bound"])
                except ValueError:
                    errors.append(
                        f"Parameter [{param_name}]: 'upper_bound' must be integer"
                    )

        elif par_type == "binary":
            if "lower_bound" in param_config or "upper_bound" in param_config:
                errors.append(
                    f"Parameter [{param_name}]: 'binary' type should not have bounds"
                )

        elif par_type == "fixed":
            if "value" not in param_config:
                errors.append(
                    f"Parameter [{param_name}]: 'fixed' type requires 'value'"
                )

        elif par_type == "categorical":
            if "choices" not in param_config:
                errors.append(
                    f"Parameter [{param_name}]: 'categorical' type requires 'choices'"
                )

        else:
            errors.append(
                f"Parameter [{param_name}]: unsupported par_type '{par_type}'"
            )

        return errors

    def _validate_strategy(
        self, strategy_name: str, strategy_config: Dict[str, str]
    ) -> List[str]:
        """
        验证策略配置

        Args:
            strategy_name: 策略名称
            strategy_config: 策略配置字典

        Returns:
            List[str]: 错误列表
        """
        errors = []

        # 检查至少有一个终止条件
        termination_criteria = self.STRATEGY_TERMINATION_CRITERIA
        has_termination = any(crit in strategy_config for crit in termination_criteria)
        if not has_termination:
            errors.append(
                f"Strategy [{strategy_name}]: must have at least one termination criterion "
                f"({', '.join(termination_criteria)})"
            )

        # 检查生成器
        if "generator" not in strategy_config:
            errors.append(f"Strategy [{strategy_name}]: missing 'generator'")
            return errors

        generator = strategy_config["generator"].strip()

        # 检查生成器依赖
        if generator in self.GENERATOR_DEPENDENCIES:
            required_deps = self.GENERATOR_DEPENDENCIES[generator]
            for dep in required_deps:
                if dep not in strategy_config:
                    errors.append(
                        f"Strategy [{strategy_name}]: generator '{generator}' requires '{dep}'"
                    )

        return errors

    # ==================== 辅助方法 ====================

    def _is_default_template_file(self, filepath: str) -> bool:
        """
        检查给定的文件路径是否是默认模板文件

        Args:
            filepath: 要检查的文件路径

        Returns:
            bool: 如果是默认模板文件返回 True，否则返回 False
        """
        try:
            # 获取要检查的文件的绝对路径（规范化）
            check_path = os.path.abspath(filepath)

            # 尝试找到默认模板文件的路径
            import importlib.resources

            if hasattr(importlib.resources, "files"):
                # Python 3.9+
                template_file = importlib.resources.files(
                    "extensions.config_builder"
                ).joinpath("default_template.ini")
                template_path = str(template_file)
            else:
                # Python 3.7-3.8 fallback
                import pkg_resources

                template_path = pkg_resources.resource_filename(
                    "extensions.config_builder", "default_template.ini"
                )

            # 规范化模板路径并比较
            default_path = os.path.abspath(template_path)

            # 检查是否相同（考虑大小写敏感性）
            return check_path.lower() == default_path.lower()

        except Exception:
            # 如果无法确定默认模板路径，返回 False（不予限制）
            return False

    # ==================== 配置操作方法 ====================

    def _load_default_template(self) -> None:
        """从 default_template.ini 加载默认配置"""
        import importlib.resources

        try:
            # 尝试使用 importlib.resources 加载包内文件
            if hasattr(importlib.resources, "files"):
                # Python 3.9+
                template_file = importlib.resources.files(
                    "extensions.config_builder"
                ).joinpath("default_template.ini")
                config = configparser.ConfigParser()
                config.read_string(template_file.read_text())
            else:
                # Python 3.7-3.8 fallback
                import pkg_resources

                template_path = pkg_resources.resource_filename(
                    "extensions.config_builder", "default_template.ini"
                )
                config = configparser.ConfigParser()
                config.read(template_path)

            # 加载模板配置
            self.config_dict = {
                section: dict(config[section]) for section in config.sections()
            }
        except Exception as e:
            # 如果文件加载失败，创建最小配置
            self._create_minimal_configuration()

    def _create_minimal_configuration(self) -> None:
        """创建最小实现配置"""
        # 创建最小的 [common] 部分
        self.config_dict["common"] = {
            "parnames": "['【parameter_1】']",
            "stimuli_per_trial": "1",
            "outcome_types": "['binary']",
            "strategy_names": "['【strategy_1】']",
        }

    def preview_configuration(self, highlight: bool = True, color: bool = False) -> str:
        """
        生成配置的预览字符串，用【】标记缺失/可选字段

        Args:
            highlight: 是否使用【】标记标签（默认 True）
            color: 是否使用 ANSI 颜色高亮【】标记（默认 False）

        Returns:
            str: 格式化的预览字符串
        """
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("  配置预览 (【】表示需要填充的字段)")
        lines.append("=" * 70 + "\n")

        for section, options in self.config_dict.items():
            lines.append(f"[{section}]")
            for key, value in options.items():
                # 检查是否包含 【】 标记
                value_str = str(value)
                has_bracket = "【" in value_str and "】" in value_str

                if color and has_bracket:
                    # 使用颜色高亮 【】 标记
                    colored_value = self._colorize_brackets(value_str)
                    lines.append(f"{key} = {colored_value}")
                else:
                    lines.append(f"{key} = {value_str}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def print_configuration(self, color: bool = True) -> None:
        """
        打印配置的预览

        Args:
            color: 是否使用颜色高亮【】标记（默认 True）
        """
        print(self.preview_configuration(color=color))

    # 向后兼容别名
    def preview_template(self, highlight: bool = True, color: bool = False) -> str:
        """已弃用：请使用 preview_configuration()"""
        return self.preview_configuration(highlight=highlight, color=color)

    def print_template(self, color: bool = True) -> None:
        """已弃用：请使用 print_configuration()"""
        self.print_configuration(color=color)

    def _colorize_brackets(
        self, text: str, fg_color: str = None, bg_color: str = None
    ) -> str:
        """
        将【】标记的内容用颜色高亮

        Args:
            text: 输入文本
            fg_color: 前景色代码（默认使用高亮黄色）
            bg_color: 背景色代码（默认 None）

        Returns:
            str: 带颜色代码的文本
        """
        if fg_color is None:
            fg_color = ColorCode.BOLD + ColorCode.YELLOW

        # 使用正则表达式找到【】中的内容并高亮
        pattern = r"(【[^】]+】)"
        replacement = fg_color + r"\1" + ColorCode.RESET

        result = re.sub(pattern, replacement, text)
        return result

    def get_configuration_string(self) -> str:
        """
        获取配置的完整字符串表示

        Returns:
            str: INI 格式的字符串
        """
        lines = []
        for section, options in self.config_dict.items():
            lines.append(f"[{section}]")
            for key, value in options.items():
                lines.append(f"{key} = {value}")
            lines.append("")
        return "\n".join(lines)

    # 向后兼容别名
    def get_template_string(self) -> str:
        """已弃用：请使用 get_configuration_string()"""
        return self.get_configuration_string()

    def show_configuration_guide(self) -> None:
        """
        显示配置编辑指南并提供提示

        示例:
            配置中用【parameter_1】表示需要你填写的参数名
            用【】标记表示必需字段
        """
        print("\n" + "=" * 70)
        print("  配置编辑指南")
        print("=" * 70)
        print(
            """
使用说明:
1. 【】中的内容表示需要你手动填写的字段
2. 【】可能表示:
   - 参数名称 (如 【parameter_1】)
   - 必需的值 (如 【lower_bound】)
   - 可选项 (如 【outcome_name】)

编辑步骤:
1. 查看生成的配置预览
2. 将 【】 内的占位符替换为实际值
3. 删除不需要的字段
4. 使用 print_configuration() 检查修改
5. 使用 to_ini() 保存配置

示例替换:
  before: parnames = ['【parameter_1】']
  after:  parnames = ['intensity', 'frequency']
  
  before: [【parameter_1】]
  after:  [intensity]
        """
        )
        print("=" * 70)
        print("\n当前配置:\n")
        self.print_configuration()

    # 向后兼容别名
    def show_template_with_hints(self) -> None:
        """已弃用：请使用 show_configuration_guide()"""
        self.show_configuration_guide()
