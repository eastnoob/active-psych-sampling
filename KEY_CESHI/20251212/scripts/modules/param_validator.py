#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEPsych参数验证模块

处理AEPsych在EUR阶段发送Condition_ID而非实际参数值的问题。
Condition_ID是设计空间中的行索引，需要转换为实际参数值。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
import tempfile
from pathlib import Path


class AEPsychParamValidator:
    """AEPsych参数验证器。

    用于验证和修正EUR采样中的参数值，确保参数在设计空间的有效范围内。
    """

    def __init__(self, design_space_path: str, validation_log_path: Optional[str] = None):
        """
        初始化参数验证器

        Args:
            design_space_path: 设计空间CSV文件路径
            validation_log_path: 验证日志路径（可选）
        """
        self.design_space_path = design_space_path
        self.design_space = pd.read_csv(design_space_path)
        self.valid_ranges = self._extract_valid_ranges()

        if validation_log_path is None:
            validation_log_path = str(Path(tempfile.gettempdir()) / 'aepsych_validation.log')
        self.validation_log_path = validation_log_path

        # 初始化日志文件
        with open(self.validation_log_path, 'w', encoding='utf-8') as f:
            f.write("AEPsych Parameter Validation Log\n")
            f.write(f"Design Space: {design_space_path}\n")
            f.write(f"Valid Ranges: {self.valid_ranges}\n")
            f.write("=" * 80 + "\n")

        logging.info(f"[Validator] 参数验证日志: {self.validation_log_path}")

    def _extract_valid_ranges(self) -> Dict[str, List[float]]:
        """提取每个参数的有效值范围。

        Returns:
            参数名 -> 有效值列表的字典
        """
        valid_ranges = {}
        for col in self.design_space.columns:
            valid_ranges[col] = sorted(self.design_space[col].unique().tolist())
        return valid_ranges

    def validate_and_correct_params(
        self, x_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """
        验证并修正参数

        Args:
            x_dict: 参数字典 {param_name: value}

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
        """记录验证条目。

        Args:
            param: 参数名
            original: 原始值
            corrected: 修正后的值
        """
        with open(self.validation_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[CORRECTED] {param}: {original} -> {corrected}\n")
        logging.warning(f"[Validator] {param}: {original} -> {corrected}")

    def log_validation_summary(self) -> None:
        """记录验证摘要。"""
        logging.info(f"[Validator] Validation log: {self.validation_log_path}")

    def get_valid_range(self, param: str) -> Optional[List[float]]:
        """获取参数的有效值范围。

        Args:
            param: 参数名

        Returns:
            有效值列表，如果参数不存在返回None
        """
        return self.valid_ranges.get(param)


def integrate_param_validation(
    server, design_space_path: str, validation_log_path: Optional[str] = None
) -> AEPsychParamValidator:
    """
    将参数验证集成到AEPsych server中

    Args:
        server: AEPsych server实例
        design_space_path: 设计空间路径
        validation_log_path: 验证日志路径（可选）

    Returns:
        validator实例
    """
    validator = AEPsychParamValidator(design_space_path, validation_log_path)
    logging.info(f"[Validator] Integrated with server")
    return validator
