#!/usr/bin/env python3
"""同步 tools/repair 补丁文件与实际 aepsych 核心文件的修改

此脚本用于确保 tools/repair 中的补丁文件准确反映了对 aepsych 核心文件的所有修改。

工作流程：
1. 检查实际的 aepsych 核心文件
2. 与原始版本对比，生成实际的 diff
3. 对比 repair 目录中现有的 patch 文件
4. 报告差异并提供更新建议
"""

import os
import subprocess
import sys
from pathlib import Path
import difflib
import re

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
REPAIR_DIR = PROJECT_ROOT / "tools" / "repair"
AEPSYCH_DIR = PROJECT_ROOT / ".pixi" / "envs" / "default" / "Lib" / "site-packages" / "aepsych"

# 需要检查的文件列表
FILES_TO_CHECK = [
    "transforms/ops/__init__.py",
    "transforms/parameters.py",
    "config.py",
    "transforms/ops/ordinal.py",  # 新增文件
]

# 颜色输出
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.RESET}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠ {msg}{Colors.RESET}")

def print_error(msg):
    print(f"{Colors.RED}✗ {msg}{Colors.RESET}")

def read_file_safe(filepath):
    """安全地读取文件内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()
    except Exception as e:
        print_error(f"无法读取文件 {filepath}: {e}")
        return None

def get_original_aepsych_content(rel_path):
    """获取原始 aepsych 文件内容

    策略：
    1. 尝试从 git 历史中获取（如果文件在某个commit之前存在）
    2. 尝试从 pixi cache 获取
    3. 从网络获取官方版本
    """
    # 先尝试从最近的一个已知的纯净版本获取
    # 这里我们假设 fcc7275 是修改之前的commit

    # 策略1: 检查是否为新增文件
    if "ordinal.py" in rel_path:
        return []  # 新增文件，原始内容为空

    # 策略2: 从 patch 文件中反推原始内容
    # 暂时返回 None，后续实现
    return None

def check_file_modifications(rel_path):
    """检查文件的实际修改内容"""
    actual_file = AEPSYCH_DIR / rel_path

    if not actual_file.exists():
        print_error(f"文件不存在: {actual_file}")
        return None

    actual_content = read_file_safe(actual_file)
    if actual_content is None:
        return None

    # 检查文件是否被修改过（通过检查特定标记）
    has_ordinal_import = any("from .ordinal import Ordinal" in line or "Ordinal" in line for line in actual_content)
    has_custom_ordinal = any("custom_ordinal" in line for line in actual_content)

    return {
        'path': rel_path,
        'actual_file': actual_file,
        'actual_content': actual_content,
        'has_ordinal_import': has_ordinal_import,
        'has_custom_ordinal': has_custom_ordinal,
        'line_count': len(actual_content)
    }

def check_patch_files(repair_subdir):
    """检查指定 repair 子目录下的补丁文件"""
    patch_dir = REPAIR_DIR / repair_subdir
    if not patch_dir.exists():
        print_warning(f"目录不存在: {patch_dir}")
        return {}

    patch_files = list(patch_dir.glob("*.patch"))
    patches = {}

    for patch_file in patch_files:
        content = read_file_safe(patch_file)
        if content:
            patches[patch_file.name] = {
                'file': patch_file,
                'content': content,
                'line_count': len(content)
            }

    return patches

def generate_diff_report(file_info, patch_info):
    """生成文件修改与补丁的对比报告"""
    print_section(f"文件: {file_info['path']}")

    print(f"实际文件路径: {file_info['actual_file']}")
    print(f"实际文件行数: {file_info['line_count']}")
    print(f"包含 Ordinal 导入: {file_info['has_ordinal_import']}")
    print(f"包含 custom_ordinal: {file_info['has_custom_ordinal']}")

    if patch_info:
        print(f"\n关联的 patch 文件数: {len(patch_info)}")
        for patch_name, info in patch_info.items():
            print(f"  - {patch_name} ({info['line_count']} 行)")
    else:
        print_warning("没有找到关联的 patch 文件")

def extract_modifications_from_file(file_content, rel_path):
    """从实际文件中提取关键修改点

    这个函数分析文件内容，找出与 ordinal 相关的修改。
    """
    modifications = []

    for i, line in enumerate(file_content, 1):
        line_lower = line.lower()

        # 检测关键修改
        if 'ordinal' in line_lower:
            context_start = max(0, i-3)
            context_end = min(len(file_content), i+3)
            modifications.append({
                'line_number': i,
                'content': line.strip(),
                'context': file_content[context_start:context_end],
                'type': 'ordinal_related'
            })

        if 'custom_ordinal' in line_lower:
            context_start = max(0, i-3)
            context_end = min(len(file_content), i+3)
            modifications.append({
                'line_number': i,
                'content': line.strip(),
                'context': file_content[context_start:context_end],
                'type': 'custom_ordinal_related'
            })

    return modifications

def main():
    print_section("AEPsych 核心文件修改同步检查")

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Repair 目录: {REPAIR_DIR}")
    print(f"AEPsych 目录: {AEPSYCH_DIR}")

    if not AEPSYCH_DIR.exists():
        print_error(f"AEPsych 目录不存在: {AEPSYCH_DIR}")
        print("请先运行 'pixi install' 安装依赖")
        sys.exit(1)

    # 检查所有文件
    all_results = {}

    for rel_path in FILES_TO_CHECK:
        print_section(f"检查文件: {rel_path}")

        file_info = check_file_modifications(rel_path)
        if file_info is None:
            continue

        # 提取修改内容
        modifications = extract_modifications_from_file(
            file_info['actual_content'],
            rel_path
        )

        print(f"\n找到 {len(modifications)} 处与 ordinal 相关的修改：")
        for mod in modifications:
            print(f"  行 {mod['line_number']}: {mod['content'][:80]}")

        all_results[rel_path] = {
            'file_info': file_info,
            'modifications': modifications
        }

    # 检查 ordinal_parameter_extension 目录下的 patch 文件
    print_section("检查现有 Patch 文件")
    ordinal_patches = check_patch_files("ordinal_parameter_extension")

    print(f"找到 {len(ordinal_patches)} 个 patch 文件：")
    for patch_name, info in ordinal_patches.items():
        print(f"  - {patch_name}")

    # 生成报告
    print_section("修改摘要")

    total_modifications = sum(len(r['modifications']) for r in all_results.values())
    print(f"\n总共检查了 {len(FILES_TO_CHECK)} 个文件")
    print(f"找到 {total_modifications} 处与 ordinal 相关的修改")
    print(f"现有 {len(ordinal_patches)} 个 patch 文件")

    # 建议
    print_section("建议")

    print("\n为确保 repair 目录与实际修改完全一致，建议：")
    print("\n1. 使用以下命令为每个修改的文件生成准确的 diff：")
    print("   需要先获取原始版本的 aepsych 文件")
    print("\n2. 手动检查每个 patch 文件的上下文是否与实际代码匹配")
    print("\n3. 运行 verify_fix.py 验证 patch 可以正确应用")

    print("\n" + "="*60)
    print("\n下一步：生成准确的 diff 并更新 patch 文件")
    print("\n推荐方法：")
    print("1. 创建一个临时的干净 aepsych 安装")
    print("2. 对比当前版本与干净版本")
    print("3. 生成准确的 unified diff patch")
    print("4. 更新 tools/repair/ordinal_parameter_extension/ 中的 patch 文件")

    return 0

if __name__ == "__main__":
    sys.exit(main())
