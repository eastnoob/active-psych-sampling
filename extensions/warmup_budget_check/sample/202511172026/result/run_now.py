#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【一键脚本 - 改进版】支持单个 CSV 或文件夹，灵活选择输出格式

特性：
  • 路径支持 \\ 形式（Windows）
  • 接受单个 CSV 文件或文件夹（自动查找 subject_*.csv）
  • 可选合并输出或单个输出
  • 自动 archive 历史文件与无用文件
  • 自动生成模型参数与误差报告
"""

from pathlib import Path
import shutil
from datetime import datetime
from run_simulation_minimal import run
from generate_report import generate_model_report

# ============================================================================
# 【改这里】- 用户配置
# ============================================================================

# 【输入路径】支持两种形式：
#   1. 单个 CSV 文件：r"d:\data\subject_1.csv"
#   2. 文件夹（自动查找 subject_*.csv）：r"d:\data"
INPUT_PATH = r"d:\WORKSPACE\python\aepsych-source\extensions\warmup_budget_check\sample\202511172026\result\test_sample"

# 【输出目录】如果为 None，默认输出到 INPUT_PATH/result（或同目录）
OUTPUT_DIR = None  # 例如 r"d:\output\exp1"

# 【输出模式】
#   "merged"  = 输出合并文件（combined_results.csv + 各个 subject_*_result.csv）
#   "single"  = 仅输出单个 CSV（不生成合并文件）
OUTPUT_MODE = "merged"  # "merged" 或 "single"

# 【交互项配置】
INTERACTION_PAIRS = [(3, 4), (0, 1), (1, 3)]  # [(x4*x5), (x1*x2), (x2*x4)]

# 【交互项权重尺度】
INTERACTION_SCALE = 0.4

# 【随机种子】
SEED = 42

# 【是否 archive 历史文件】（备份旧结果到 archive 文件夹）
AUTO_ARCHIVE = True

# 【报告格式】
#   "md"  = Markdown 格式（默认）
#   "txt" = 纯文本格式
REPORT_FORMAT = "md"  # "md" 或 "txt"

# ============================================================================
# 实现
# ============================================================================


def normalize_path(path_str):
    """
    【标准化路径】
    支持 \\ 形式和 / 形式，返回 Path 对象
    """
    # 替换反斜杠为正斜杠（兼容 \\ 形式）
    normalized = path_str.replace("\\\\", "\\").replace("\\", "/")
    return Path(normalized).resolve()


def archive_old_results(output_dir):
    """
    【Archive 历史文件】
    如果输出目录已存在，将旧文件备份到 archive 子目录
    """
    if not output_dir.exists():
        return

    # 检查是否有旧的结果文件
    old_files = (
        list(output_dir.glob("subject_*_result.csv"))
        + list(output_dir.glob("combined_results.csv"))
        + list(output_dir.glob("model_spec.txt"))
    )

    if not old_files:
        return

    # 创建 archive 目录
    archive_dir = output_dir / "archive"
    archive_subdir = archive_dir / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_subdir.mkdir(parents=True, exist_ok=True)

    print(f"\n【Archive 旧文件】")
    for fpath in old_files:
        try:
            shutil.move(str(fpath), str(archive_subdir / fpath.name))
            print(f"  OK - {fpath.name} -> archive")
        except Exception as e:
            print(f"  NG - {fpath.name}: {e}")


def find_csv_files(input_path):
    """
    【查找 CSV 文件】

    如果是文件：返回 [file]（单个 CSV）
    如果是文件夹：返回所有 subject_*.csv 文件列表
    """
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return [input_path]

    if input_path.is_dir():
        csv_files = sorted(list(input_path.glob("subject_*.csv")))
        return csv_files

    return []


def determine_output_dir(input_path, csv_files, user_output_dir=None):
    """
    【确定输出目录】

    • 如果用户指定了 OUTPUT_DIR，用那个
    • 如果输入是文件夹，输出到 input_path/result
    • 如果输入是单个 CSV，输出到同目录/result
    """
    if user_output_dir:
        return normalize_path(user_output_dir)

    if input_path.is_dir():
        return input_path / "result"
    else:
        return input_path.parent / "result"


def run_single_csv(
    csv_path, output_dir, seed, interaction_pairs, interaction_scale, output_mode
):
    """
    【处理单个 CSV 文件】
    为了支持单个 CSV，我们需要创建一个临时目录
    """
    import tempfile
    import pandas as pd

    # 创建临时目录放单个 CSV
    temp_dir = Path(tempfile.mkdtemp())

    # 读原 CSV
    df = pd.read_csv(csv_path)

    # 改名为 subject_1.csv（符合模拟脚本期望）
    temp_csv = temp_dir / "subject_1.csv"
    df.to_csv(temp_csv, index=False)

    # 运行模拟
    output_dir.mkdir(parents=True, exist_ok=True)
    run(temp_dir, output_dir, seed, interaction_pairs, interaction_scale)

    # 处理输出：如果是单个 CSV，改名为原文件名
    result_file = output_dir / "subject_1_result.csv"
    if result_file.exists() and output_mode == "single":
        # 改名为原文件名 + _result
        new_name = csv_path.stem + "_result.csv"
        result_file.rename(output_dir / new_name)

    # 清理临时目录
    shutil.rmtree(temp_dir)


def cleanup_unwanted_files(output_dir, output_mode):
    """
    【清理不需要的文件】

    • 如果 output_mode="single"，删除 combined_results.csv
    • 如果 output_mode="merged"，保留所有
    """
    if output_mode == "single":
        combined_file = output_dir / "combined_results.csv"
        if combined_file.exists():
            combined_file.unlink()
            print(f"  OK - 删除 combined_results.csv（单个模式）")


if __name__ == "__main__":
    # 【标准化输入路径】
    input_path = normalize_path(INPUT_PATH)

    print("=" * 80)
    print(f"输入路径：{input_path}")
    print(f"输入类型：{'单个 CSV' if input_path.is_file() else '文件夹'}")
    print("=" * 80)

    # 【查找 CSV 文件】
    csv_files = find_csv_files(input_path)

    if not csv_files:
        print(f"ERROR: 找不到 CSV 文件")
        print(f"  期望：")
        print(f"    • 单个文件：*.csv")
        print(f"    • 文件夹模式：subject_*.csv")
        exit(1)

    print(f"找到 {len(csv_files)} 个 CSV 文件")
    for f in csv_files:
        print(f"  - {f.name}")

    # 【确定输出目录】
    output_dir = determine_output_dir(input_path, csv_files, OUTPUT_DIR)

    print(f"\n输出目录：{output_dir}")
    print(f"输出模式：{OUTPUT_MODE}")
    print(f"交互项：{INTERACTION_PAIRS}")
    print(f"交互项权重尺度：{INTERACTION_SCALE}")
    print(f"随机种子：{SEED}")
    print("=" * 80)

    # 【Archive 旧文件】
    if AUTO_ARCHIVE:
        archive_old_results(output_dir)

    # 【运行模拟】
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects_info = None  # 用来保存被试信息

    if len(csv_files) == 1 and csv_files[0].is_file():
        # 单个 CSV 文件
        print("\n【处理单个 CSV 文件】")
        run_single_csv(
            csv_files[0],
            output_dir,
            SEED,
            INTERACTION_PAIRS,
            INTERACTION_SCALE,
            OUTPUT_MODE,
        )
    else:
        # 文件夹模式（多个 CSV）
        print("\n【处理文件夹中的多个 CSV】")
        subjects_info = run(
            input_path, output_dir, SEED, INTERACTION_PAIRS, INTERACTION_SCALE
        )

    # 【清理不需要的文件】
    print("\n【输出处理】")
    cleanup_unwanted_files(output_dir, OUTPUT_MODE)

    # 【生成模型报告】
    if subjects_info:
        generate_model_report(subjects_info, output_dir, report_format=REPORT_FORMAT)

    # 【验证输出】
    print("\n【输出文件】")
    result_files = (
        list(output_dir.glob("subject_*_result.csv"))
        + list(output_dir.glob("combined_results.csv"))
        + list(output_dir.glob("model_spec.txt"))
        + list(output_dir.glob("model_report.*"))
        + list(output_dir.glob("group_statistics.txt"))
    )

    if result_files:
        for fpath in sorted(result_files):
            size_kb = fpath.stat().st_size / 1024
            print(f"  OK - {fpath.name} ({size_kb:.1f} KB)")
    else:
        print(f"  WARNING: 没有输出文件")

    print("\n" + "=" * 80)
    print("✓ 完成！")
    print("=" * 80)
