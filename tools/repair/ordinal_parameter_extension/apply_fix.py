#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply Ordinal Parameter Extension Patches to AEPsych

自动应用 ordinal 参数类型的所有 patch 文件到 AEPsych。

使用方法：
    python apply_fix.py [--dry-run]

选项：
    --dry-run: 只检查 patch 是否可以应用，不实际修改文件

Patch 列表：
    1. aepsych_ordinal_transforms.patch - 新增 ordinal.py 文件
    2. aepsych_transforms_ops_init.patch - 修改 transforms/ops/__init__.py
    3. aepsych_transforms_parameters.patch - 修改 parameters.py
    4. aepsych_config.patch - 修改 config.py
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

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
    # 向上查找直到找到包含 .git 的目录
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("无法找到项目根目录（需要包含 .git 目录）")

def find_aepsych_path() -> Path:
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

def get_patch_files(repair_dir: Path) -> List[Tuple[str, Path, str]]:
    """获取所有 patch 文件及其目标路径

    Returns:
        List of (patch_name, patch_file, target_file_relative)
    """
    patches = [
        (
            "Ordinal Transforms",
            repair_dir / "aepsych_ordinal_transforms.patch",
            "aepsych/transforms/ops/ordinal.py"
        ),
        (
            "Transforms Ops Init",
            repair_dir / "aepsych_transforms_ops_init.patch",
            "aepsych/transforms/ops/__init__.py"
        ),
        (
            "Parameters",
            repair_dir / "aepsych_transforms_parameters.patch",
            "aepsych/transforms/parameters.py"
        ),
        (
            "Config",
            repair_dir / "aepsych_config.patch",
            "aepsych/config/config.py"
        ),
    ]

    # 验证 patch 文件存在
    for name, patch_file, _ in patches:
        if not patch_file.exists():
            raise FileNotFoundError(f"Patch 文件不存在: {patch_file}")

    return patches

def apply_patch(
    patch_file: Path,
    target_dir: Path,
    dry_run: bool = False
) -> Tuple[bool, str]:
    """应用单个 patch 文件

    Args:
        patch_file: patch 文件路径
        target_dir: 目标目录（AEPsych 根目录）
        dry_run: 是否只检查不实际应用

    Returns:
        (success, message)
    """
    # 使用 -p2 因为 patch 路径包含 aepsych/ 前缀
    # 例如: a/aepsych/transforms/ops/ordinal.py
    # -p2 strips "a/aepsych/" → transforms/ops/ordinal.py
    cmd = ["patch", "-p2", "-d", str(target_dir), "-i", str(patch_file)]
    if dry_run:
        cmd.append("--dry-run")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr

    except FileNotFoundError:
        return False, (
            "未找到 'patch' 命令。请安装：\n"
            "  - Windows: choco install patch 或使用 Git Bash\n"
            "  - Linux: sudo apt install patch\n"
            "  - macOS: brew install gpatch"
        )
    except Exception as e:
        return False, str(e)

def check_git_status(aepsych_path: Path) -> None:
    """检查 AEPsych 是否有未提交的修改"""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=aepsych_path,
            capture_output=True,
            text=True,
            check=True
        )

        if result.stdout.strip():
            logger.warning("AEPsych 有未提交的修改。建议先提交或备份。")
            logger.info(f"未提交的文件:\n{result.stdout}")

            response = input("\n是否继续？(y/N): ")
            if response.lower() != 'y':
                logger.warning("操作已取消")
                sys.exit(0)

    except subprocess.CalledProcessError:
        logger.warning("无法检查 git 状态（可能不是 git 仓库）")

def main():
    parser = argparse.ArgumentParser(
        description="Apply ordinal parameter extension patches to AEPsych"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查 patch 是否可以应用，不实际修改文件"
    )
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("Ordinal Parameter Extension - Patch Application")
    logger.info("="*60)

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
        aepsych_path = find_aepsych_path()
        logger.success(f"AEPsych 路径: {aepsych_path}")
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    # 3. 检查 git 状态
    if not args.dry_run:
        logger.info("检查 git 状态...")
        check_git_status(aepsych_path)

    # 4. 获取 patch 文件
    repair_dir = Path(__file__).parent
    logger.info("加载 patch 文件...")
    try:
        patches = get_patch_files(repair_dir)
        logger.success(f"找到 {len(patches)} 个 patch 文件")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # 5. 应用 patch
    logger.info("="*60)
    logger.info("应用 Patches")
    logger.info("="*60)

    if args.dry_run:
        logger.warning("【DRY RUN 模式】只检查，不修改文件")

    success_count = 0
    failed_patches = []

    for name, patch_file, target_rel in patches:
        logger.info(f"\n[{name}]")
        logger.info(f"Patch: {patch_file.name}")
        logger.info(f"目标: {target_rel}")

        success, message = apply_patch(patch_file, aepsych_path, args.dry_run)

        if success:
            logger.success(f"应用{'检查' if args.dry_run else ''}成功")
            success_count += 1
        else:
            logger.error(f"应用{'检查' if args.dry_run else ''}失败")
            logger.error(f"错误信息:\n{message}")
            failed_patches.append((name, message))

    # 6. 总结
    logger.info("="*60)
    logger.info("总结")
    logger.info("="*60)

    if success_count == len(patches):
        logger.success(
            f"所有 {len(patches)} 个 patch {'检查通过' if args.dry_run else '应用成功'}！"
        )

        if not args.dry_run:
            logger.info("\n下一步：")
            logger.info("  1. 运行 verify_fix.py 验证修改")
            logger.info("  2. 测试 AEPsych 功能")
            logger.info("  3. 提交更改到 git")
        else:
            logger.info("\n运行以下命令实际应用 patch：")
            logger.info(f"  python {Path(__file__).name}")

        sys.exit(0)
    else:
        log_func = logger.warning if success_count > 0 else logger.error
        log_func(
            f"{success_count}/{len(patches)} 个 patch {'检查通过' if args.dry_run else '应用成功'}"
        )

        if failed_patches:
            logger.error("\n失败的 patch：")
            for name, msg in failed_patches:
                logger.error(f"  - {name}: {msg.split(chr(10))[0][:80]}...")

        sys.exit(1)

if __name__ == "__main__":
    main()
