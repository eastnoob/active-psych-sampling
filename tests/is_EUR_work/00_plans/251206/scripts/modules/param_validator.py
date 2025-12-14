"""
AEPsych参数验证辅助模块

处理AEPsych在EUR阶段发送Condition_ID而非实际参数值的问题.
Condition_ID是设计空间中的行索引,需要转换为实际参数值.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import tempfile
from pathlib import Path


class AEPsychParamValidator:
    """AEPsych参数验证器."""

    design_space_path: str
    validation_log_path: str

    def __init__(self, design_space_path: str, validation_log_path: str = None):
        """
        初始化参数验证器

        Args:
            design_space_path: 设计空间CSV文件路径
            validation_log_path: 验证日志路径(可选)
        """
        self.design_space = pd.read_csv(design_space_path)
        self.valid_ranges = self._extract_valid_ranges()

        if validation_log_path is None:
            validation_log_path = Path(tempfile.gettempdir()) / 'aepsych_validation.log'
        self.validation_log_path = validation_log_path

        # 初始化日志文件
        with open(self.validation_log_path, 'w', encoding='utf-8') as f:
            f.write("AEPsych Parameter Validation Log\n")

        logging.info(f"[Validator] 参数验证日志: {self.validation_log_path}")
        logging.info("=" * 80)

    def _extract_valid_ranges(self) -> Dict[str, List[float]]:
        """提取每个参数的有效值范围."""
        valid_ranges = {}
        for col in self.design_space.columns:
            valid_ranges[col] = sorted(self.design_space[col].unique().tolist())
        return valid_ranges

    def validate_and_correct_params(self, x_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """
        验证并修正参数

        Args:
            x_dict: 参数字典

        Returns:
            (修正后的参数字典, 是否进行了修正)
        """
        corrected_dict = x_dict.copy()
        was_corrected = False

        for param, value in x_dict.items():
            if param in self.valid_ranges:
                valid_vals = self.valid_ranges[param]
                if value not in valid_vals:
                    # 找到最近的有效值
                    closest = min(valid_vals, key=lambda v: abs(v - value))
                    corrected_dict[param] = closest
                    was_corrected = True
                    self._log_validation_entry(param, value, closest)

        return corrected_dict, was_corrected

    def _log_validation_entry(self, param: str, original: Any, corrected: Any) -> None:
        """记录验证条目."""
        with open(self.validation_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[CORRECTED] {param}: {original} -> {corrected}\n")

    def log_validation_summary(self) -> None:
        """记录验证摘要."""
        logging.info(f"Validation log saved to: {self.validation_log_path}")


def integrate_param_validation(server, design_space_path: str):
    """
    将参数验证集成到AEPsych server中

    Args:
        server: AEPsych server实例
        design_space_path: 设计空间路径

    Returns:
        validator实例
    """
    validator = AEPsychParamValidator(design_space_path)
    # TODO: 集成到server的ask/tell流程中
    return validator
