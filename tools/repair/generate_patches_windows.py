#!/usr/bin/env python3
"""为 Windows 环境生成准确的 AEPsych patch 文件

此脚本使用 Python difflib 模块生成 unified diff 格式的 patch 文件，
不依赖于 Unix 的 diff 命令。

工作流程：
1. 创建临时的干净 aepsych 环境
2. 对比原始文件与修改后的文件
3. 生成 unified diff 格式的 patch
4. 保存到 tools/repair/ordinal_parameter_extension/
"""

import difflib
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
REPAIR_DIR = PROJECT_ROOT / "tools" / "repair" / "ordinal_parameter_extension"
AEPSYCH_MODIFIED = PROJECT_ROOT / ".pixi" / "envs" / "default" / "Lib" / "site-packages" / "aepsych"

# 需要生成 patch 的文件
FILES_TO_PATCH = [
    {
        'rel_path': 'transforms/ops/__init__.py',
        'patch_name': 'aepsych_transforms_ops_init.patch',
        'is_new': False
    },
    {
        'rel_path': 'transforms/parameters.py',
        'patch_name': 'aepsych_transforms_parameters.patch',
        'is_new': False
    },
    {
        'rel_path': 'config.py',
        'patch_name': 'aepsych_config.patch',
        'is_new': False
    },
    {
        'rel_path': 'transforms/ops/ordinal.py',
        'patch_name': 'aepsych_ordinal_transforms.patch',
        'is_new': True  # 新文件
    }
]

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

def create_clean_aepsych_env():
    """创建临时的干净 aepsych 环境"""
    print_section("步骤 1: 创建干净的 AEPsych 环境")

    temp_dir = Path(tempfile.mkdtemp(prefix="aepsych-clean-"))
    print(f"临时目录: {temp_dir}")

    # 创建最小的 pixi.toml
    pixi_toml = temp_dir / "pixi.toml"
    pixi_toml_content = """[project]
name = "aepsych-clean"
channels = ["conda-forge"]
platforms = ["win-64"]

[dependencies]
aepsych = "*"
"""
    pixi_toml.write_text(pixi_toml_content, encoding='utf-8')
    print_success("创建 pixi.toml")

    # 运行 pixi install
    print("\n正在安装干净的 aepsych...（这可能需要几分钟）")
    try:
        result = subprocess.run(
            ["pixi", "install"],
            cwd=temp_dir,
            check=True,
            capture_output=True,
            text=True
        )
        print_success("AEPsych 安装成功")
    except subprocess.CalledProcessError as e:
        print_error(f"安装失败: {e}")
        print(e.stderr)
        sys.exit(1)

    aepsych_dir = temp_dir / ".pixi" / "envs" / "default" / "Lib" / "site-packages" / "aepsych"
    if not aepsych_dir.exists():
        print_error(f"AEPsych 目录不存在: {aepsych_dir}")
        sys.exit(1)

    print_success(f"干净的 AEPsych 位于: {aepsych_dir}")
    return temp_dir, aepsych_dir

def read_file_lines(filepath):
    """读取文件的所有行，保留换行符"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        return None
    except Exception as e:
        print_error(f"读取文件失败 {filepath}: {e}")
        return None

def generate_unified_diff(original_lines, modified_lines, rel_path):
    """生成 unified diff 格式的 patch

    Args:
        original_lines: 原始文件的行列表（包含换行符）
        modified_lines: 修改后文件的行列表（包含换行符）
        rel_path: 文件的相对路径

    Returns:
        patch 内容的字符串
    """
    if original_lines is None:
        original_lines = []

    if modified_lines is None:
        print_error(f"无法读取修改后的文件: {rel_path}")
        return None

    # 生成 unified diff
    diff_lines = list(difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        lineterm=''
    ))

    if not diff_lines:
        print_warning(f"文件没有差异: {rel_path}")
        return None

    # 添加换行符
    patch_content = '\n'.join(diff_lines) + '\n'
    return patch_content

def generate_all_patches(original_aepsych_dir, modified_aepsych_dir):
    """为所有文件生成 patch"""
    print_section("步骤 2: 生成 Patch 文件")

    output_dir = REPAIR_DIR / "new_patches"
    output_dir.mkdir(parents=True, exist_ok=True)

    patches_generated = []

    for file_info in FILES_TO_PATCH:
        rel_path = file_info['rel_path']
        patch_name = file_info['patch_name']
        is_new = file_info['is_new']

        print(f"\n处理: {rel_path}")

        # 读取原始文件
        if is_new:
            print("  → 这是新文件")
            original_lines = []
        else:
            original_file = original_aepsych_dir / rel_path
            original_lines = read_file_lines(original_file)
            if original_lines is None:
                print_error(f"  原始文件不存在: {original_file}")
                continue

        # 读取修改后的文件
        modified_file = modified_aepsych_dir / rel_path
        modified_lines = read_file_lines(modified_file)
        if modified_lines is None:
            print_error(f"  修改后的文件不存在: {modified_file}")
            continue

        # 生成 patch
        patch_content = generate_unified_diff(original_lines, modified_lines, rel_path)
        if patch_content is None:
            continue

        # 保存 patch 文件
        patch_file = output_dir / patch_name
        patch_file.write_text(patch_content, encoding='utf-8')
        print_success(f"  生成: {patch_file.name} ({len(patch_content.splitlines())} 行)")

        patches_generated.append({
            'name': patch_name,
            'path': patch_file,
            'lines': len(patch_content.splitlines())
        })

    return patches_generated

def compare_with_existing_patches(new_patches_dir):
    """对比新旧 patch 文件"""
    print_section("步骤 3: 对比新旧 Patch")

    differences_found = []

    for file_info in FILES_TO_PATCH:
        patch_name = file_info['patch_name']
        old_patch = REPAIR_DIR / patch_name
        new_patch = new_patches_dir / patch_name

        if not new_patch.exists():
            print_warning(f"新 patch 不存在: {patch_name}")
            continue

        if not old_patch.exists():
            print_warning(f"旧 patch 不存在: {patch_name}（这是新添加的）")
            differences_found.append(patch_name)
            continue

        # 读取文件内容
        old_content = old_patch.read_text(encoding='utf-8').splitlines()
        new_content = new_patch.read_text(encoding='utf-8').splitlines()

        # 对比
        if old_content == new_content:
            print_success(f"{patch_name}: 完全一致")
        else:
            print_error(f"{patch_name}: 发现差异")
            differences_found.append(patch_name)

            # 显示差异统计
            diff = list(difflib.unified_diff(
                old_content,
                new_content,
                fromfile=f"旧/{patch_name}",
                tofile=f"新/{patch_name}",
                lineterm=''
            ))

            added_lines = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
            removed_lines = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))

            print(f"  旧版本: {len(old_content)} 行")
            print(f"  新版本: {len(new_content)} 行")
            print(f"  差异: +{added_lines} -{removed_lines}")

    return differences_found

def backup_existing_patches():
    """备份现有的 patch 文件"""
    print_section("备份现有 Patch 文件")

    backup_dir = REPAIR_DIR / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for file_info in FILES_TO_PATCH:
        patch_name = file_info['patch_name']
        old_patch = REPAIR_DIR / patch_name

        if old_patch.exists():
            shutil.copy2(old_patch, backup_dir / patch_name)
            print_success(f"备份: {patch_name}")

    print(f"\n备份目录: {backup_dir}")
    return backup_dir

def replace_patches(new_patches_dir):
    """用新的 patch 文件替换旧的"""
    print_section("更新 Patch 文件")

    for file_info in FILES_TO_PATCH:
        patch_name = file_info['patch_name']
        new_patch = new_patches_dir / patch_name
        target_patch = REPAIR_DIR / patch_name

        if new_patch.exists():
            shutil.copy2(new_patch, target_patch)
            print_success(f"更新: {patch_name}")

def main():
    print_section("AEPsych Patch 文件自动同步工具")

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Repair 目录: {REPAIR_DIR}")
    print(f"修改后的 AEPsych: {AEPSYCH_MODIFIED}")

    if not AEPSYCH_MODIFIED.exists():
        print_error(f"修改后的 AEPsych 目录不存在: {AEPSYCH_MODIFIED}")
        sys.exit(1)

    # 询问用户确认
    print("\n此脚本将：")
    print("1. 创建临时的干净 AEPsych 环境")
    print("2. 生成准确的 patch 文件")
    print("3. 对比新旧 patch 的差异")
    print("4. （可选）备份并替换旧的 patch 文件")

    response = input("\n是否继续？ (y/n): ")
    if response.lower() != 'y':
        print("操作已取消")
        sys.exit(0)

    # 步骤 1: 创建干净环境
    temp_dir, clean_aepsych = create_clean_aepsych_env()

    try:
        # 步骤 2: 生成 patch
        patches = generate_all_patches(clean_aepsych, AEPSYCH_MODIFIED)

        if not patches:
            print_error("没有生成任何 patch 文件")
            sys.exit(1)

        print(f"\n成功生成 {len(patches)} 个 patch 文件")

        # 步骤 3: 对比
        new_patches_dir = REPAIR_DIR / "new_patches"
        differences = compare_with_existing_patches(new_patches_dir)

        # 步骤 4: 询问是否替换
        if differences:
            print(f"\n发现 {len(differences)} 个 patch 文件有差异：")
            for diff in differences:
                print(f"  - {diff}")

            response = input("\n是否备份旧文件并使用新的 patch？ (y/n): ")
            if response.lower() == 'y':
                backup_dir = backup_existing_patches()
                replace_patches(new_patches_dir)
                print_success("\nPatch 文件已更新！")
                print(f"旧文件备份在: {backup_dir}")
            else:
                print("\n保持现有 patch 文件不变")
                print(f"新的 patch 文件保存在: {new_patches_dir}")
        else:
            print_success("\n所有 patch 文件都是最新的，无需更新")

    finally:
        # 清理临时目录
        print("\n清理临时文件...")
        try:
            shutil.rmtree(temp_dir)
            print_success(f"已删除临时目录: {temp_dir}")
        except Exception as e:
            print_warning(f"无法删除临时目录: {e}")

    print_section("完成")
    print("\n后续步骤：")
    print("1. 验证新的 patch 文件: cd tools/repair/ordinal_parameter_extension && python verify_fix.py")
    print("2. 在干净的环境中测试应用 patch")
    print("3. 更新文档（如有必要）")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
