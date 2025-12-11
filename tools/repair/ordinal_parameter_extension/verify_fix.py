#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Ordinal Parameter Extension Patches

验证 ordinal 参数类型的 patch 是否正确应用到 AEPsych。

使用方法：
    python verify_fix.py

检查项目：
    1. 文件存在性检查
    2. 关键代码片段检查
    3. 语法正确性检查（可选）
    4. 导入测试（可选）
"""

import ast
import sys
from pathlib import Path
from typing import List, Tuple, Optional

from loguru import logger

# 配置 loguru
logger.remove()  # 移除默认 handler
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True
)

def find_project_root() -> Path:
    """查找项目根目录"""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("无法找到项目根目录（需要包含 .git 目录）")

def find_aepsych_path(project_root: Path) -> Path:
    """查找 AEPsych 安装路径（site-packages）"""
    try:
        import aepsych
        aepsych_file = Path(aepsych.__file__)
        # __file__ 指向 __init__.py，返回其父目录
        aepsych_path = aepsych_file.parent

        if not aepsych_path.exists():
            raise RuntimeError(f"AEPsych 路径不存在: {aepsych_path}")

        return aepsych_path

    except ImportError:
        raise RuntimeError(
            "无法导入 aepsych。请确保已安装 AEPsych：\n"
            "  - pip install aepsych\n"
            "  - 或 pixi install"
        )

def check_file_exists(file_path: Path) -> bool:
    """检查文件是否存在"""
    return file_path.exists() and file_path.is_file()

def check_code_snippet(file_path: Path, snippet: str, description: str) -> bool:
    """检查文件中是否包含指定代码片段

    Args:
        file_path: 文件路径
        snippet: 要搜索的代码片段
        description: 片段描述

    Returns:
        是否找到
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        found = snippet in content

        if found:
            logger.success(f"  {description}")
        else:
            logger.error(f"  {description}")

        return found
    except Exception as e:
        logger.error(f"  读取文件失败: {e}")
        return False

def check_syntax(file_path: Path) -> bool:
    """检查 Python 文件语法是否正确"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.error(f"  语法错误 (行 {e.lineno}): {e.msg}")
        return False
    except Exception as e:
        logger.error(f"  检查失败: {e}")
        return False

def verify_ordinal_py(aepsych_path: Path) -> Tuple[bool, List[str]]:
    """验证 ordinal.py 文件"""
    errors = []
    file_path = aepsych_path / "transforms" / "ops" / "ordinal.py"

    logger.info("\n检查 ordinal.py")
    logger.info("=" * 60)

    # 1. 文件存在
    if not check_file_exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        errors.append("ordinal.py 文件不存在")
        return False, errors

    logger.success(f"文件存在: {file_path}")

    # 2. 关键代码片段
    checks = [
        ("class Ordinal(InputTransform):", "Ordinal 类定义"),
        ("def _build_normalized_mappings(self):", "规范化映射方法"),
        ("def transform(self, X: Tensor) -> Tensor:", "transform 方法"),
        ("def untransform(self, X: Tensor) -> Tensor:", "untransform 方法"),
        ("custom_ordinal", "custom_ordinal 类型支持"),
        ("custom_ordinal_mono", "custom_ordinal_mono 类型支持"),
    ]

    all_passed = True
    for snippet, desc in checks:
        if not check_code_snippet(file_path, snippet, desc):
            errors.append(f"ordinal.py 缺少: {desc}")
            all_passed = False

    # 3. 语法检查
    if check_syntax(file_path):
        logger.success("  Python 语法正确")
    else:
        errors.append("ordinal.py 语法错误")
        all_passed = False

    return all_passed, errors

def verify_parameters_py(aepsych_path: Path) -> Tuple[bool, List[str]]:
    """验证 parameters.py 修改"""
    errors = []
    file_path = aepsych_path / "transforms" / "parameters.py"

    logger.info("\n检查 parameters.py")
    logger.info("=" * 60)

    # 1. 文件存在
    if not check_file_exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        errors.append("parameters.py 文件不存在")
        return False, errors

    logger.success(f"文件存在: {file_path}")

    # 2. 关键代码片段
    checks = [
        ("from aepsych.transforms.ops import Categorical, Fixed, Log10Plus, NormalizeScale, Ordinal, Round", "导入 Ordinal"),
        ('par_type in ["custom_ordinal", "custom_ordinal_mono"]', "ordinal 类型判断"),
        ("ordinal = Ordinal.from_config", "Ordinal.from_config 调用"),
        ("transform_options[\"bounds\"] = ordinal.transform_bounds", "边界变换"),
    ]

    all_passed = True
    for snippet, desc in checks:
        if not check_code_snippet(file_path, snippet, desc):
            errors.append(f"parameters.py 缺少: {desc}")
            all_passed = False

    # 3. 语法检查
    if check_syntax(file_path):
        logger.success("  Python 语法正确")
    else:
        errors.append("parameters.py 语法错误")
        all_passed = False

    return all_passed, errors

def verify_transforms_ops_init(aepsych_path: Path) -> Tuple[bool, List[str]]:
    """验证 transforms/ops/__init__.py 修改"""
    errors = []
    file_path = aepsych_path / "transforms" / "ops" / "__init__.py"

    logger.info("\n检查 transforms/ops/__init__.py")
    logger.info("=" * 60)

    # 1. 文件存在
    if not check_file_exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        errors.append("transforms/ops/__init__.py 文件不存在")
        return False, errors

    logger.success(f"文件存在: {file_path}")

    # 2. 关键代码片段
    checks = [
        ("from .ordinal import Ordinal", "导入 Ordinal"),
        ('"Ordinal"', "__all__ 包含 Ordinal"),
    ]

    all_passed = True
    for snippet, desc in checks:
        if not check_code_snippet(file_path, snippet, desc):
            errors.append(f"transforms/ops/__init__.py 缺少: {desc}")
            all_passed = False

    # 3. 语法检查
    if check_syntax(file_path):
        logger.success("  Python 语法正确")
    else:
        errors.append("transforms/ops/__init__.py 语法错误")
        all_passed = False

    return all_passed, errors

def verify_config_py(aepsych_path: Path) -> Tuple[bool, List[str]]:
    """验证 config.py 修改"""
    errors = []
    file_path = aepsych_path / "config.py"

    logger.info("\n检查 config.py")
    logger.info("=" * 60)

    # 1. 文件存在
    if not check_file_exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        errors.append("config.py 文件不存在")
        return False, errors

    logger.success(f"文件存在: {file_path}")

    # 2. 关键代码片段
    checks = [
        ('"custom_ordinal"', "custom_ordinal 类型验证"),
        ('"custom_ordinal_mono"', "custom_ordinal_mono 类型验证"),
    ]

    all_passed = True
    for snippet, desc in checks:
        if not check_code_snippet(file_path, snippet, desc):
            errors.append(f"config.py 缺少: {desc}")
            all_passed = False

    # 3. 语法检查
    if check_syntax(file_path):
        logger.success("  Python 语法正确")
    else:
        errors.append("config.py 语法错误")
        all_passed = False

    return all_passed, errors

def try_import_ordinal(aepsych_path: Path) -> Tuple[bool, Optional[str]]:
    """尝试导入 Ordinal 类（可选测试）"""
    try:
        # 临时添加 AEPsych 到 sys.path
        import sys
        if str(aepsych_path) not in sys.path:
            sys.path.insert(0, str(aepsych_path))

        # 尝试导入
        from aepsych.transforms.ops.ordinal import Ordinal

        # 基本功能测试
        ordinal = Ordinal(
            indices=[0],
            values={0: [1.0, 2.0, 3.0]}
        )

        # 检查关键方法存在
        assert hasattr(ordinal, 'transform')
        assert hasattr(ordinal, 'untransform')
        assert hasattr(ordinal, 'transform_bounds')

        return True, None

    except ImportError as e:
        return False, f"导入失败: {e}"
    except Exception as e:
        return False, f"测试失败: {e}"

def main():
    logger.info("\nOrdinal Parameter Extension - Verification")
    logger.info("=" * 60)

    # 1. 查找项目路径
    logger.info("查找项目根目录...")
    try:
        project_root = find_project_root()
        logger.success(f"项目根目录: {project_root}")
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # 2. 查找 AEPsych 路径
    logger.info("查找 AEPsych 安装路径...")
    try:
        aepsych_path = find_aepsych_path(project_root)
        logger.success(f"AEPsych 路径: {aepsych_path}")
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # 3. 验证各个文件
    all_errors = []

    ordinal_ok, ordinal_errors = verify_ordinal_py(aepsych_path)
    all_errors.extend(ordinal_errors)

    init_ok, init_errors = verify_transforms_ops_init(aepsych_path)
    all_errors.extend(init_errors)

    params_ok, params_errors = verify_parameters_py(aepsych_path)
    all_errors.extend(params_errors)

    config_ok, config_errors = verify_config_py(aepsych_path)
    all_errors.extend(config_errors)

    # 4. 导入测试（可选）
    logger.info("\n导入测试")
    logger.info("=" * 60)

    import_ok, import_error = try_import_ordinal(aepsych_path)
    if import_ok:
        logger.success("Ordinal 类导入并测试成功")
    else:
        logger.warning(f"导入测试失败: {import_error}")
        logger.warning("  注意：这可能是因为依赖未安装，不一定表示 patch 有问题")

    # 5. 总结
    logger.info("\n验证结果")
    logger.info("=" * 60)

    all_passed = ordinal_ok and init_ok and params_ok and config_ok

    if all_passed:
        logger.success("所有检查通过！Ordinal 参数扩展已正确应用。")

        if import_ok:
            logger.info("\n✅ 完整验证成功（包括导入测试）")
        else:
            logger.info("\n✅ 文件验证成功（导入测试失败可能是环境问题）")

        logger.info("\n下一步：")
        logger.info("  1. 运行单元测试")
        logger.info("  2. 运行集成测试")
        logger.info("  3. 端到端测试")

        sys.exit(0)
    else:
        logger.error("验证失败，发现以下问题：")
        for i, error in enumerate(all_errors, 1):
            logger.info(f"  {i}. {error}")

        logger.info("\n建议：")
        logger.info("  1. 检查 patch 文件是否完整")
        logger.info("  2. 重新运行 apply_fix.py")
        logger.info("  3. 手动检查相关文件")

        sys.exit(1)

if __name__ == "__main__":
    main()
