#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【清理脚本】Archive 当前目录中的所有无用文件和历史脚本
"""

from pathlib import Path
import shutil
from datetime import datetime

# 【需要 Archive 的文件类型】
FILES_TO_ARCHIVE = [
    # 测试脚本（已弃用）
    "check_realism.py",
    "check_within_subject.py",
    "validate_realism.py",
    "validate_realism_final.py",
    "test_scout_warmup_fixes.py",
    "analyze_within.py",
    "all_metrics.py",
    "metrics_corrected.py",
    # 数据文件
    "*.md",
    "*.txt",  # markdown 和文本输出
    "test_*.txt",
    "test_*.csv",
    "*_log.md",
    "*.log",
]

DIRS_TO_ARCHIVE = [
    "test_sample",
    "phase1_outputs",
    "test_phase1_analysis_output",
    "_trash can",
]

if __name__ == "__main__":
    current_dir = Path(__file__).parent

    # 创建 archive 目录
    archive_dir = current_dir / "archive"
    archive_subdir = archive_dir / f"cleanup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_subdir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"【Archive 清理】")
    print(f"当前目录：{current_dir}")
    print(f"Archive 目录：{archive_subdir}")
    print("=" * 80)

    # 归档文件
    print("\n【文件】")
    for pattern in FILES_TO_ARCHIVE:
        for fpath in current_dir.glob(pattern):
            if fpath.is_file() and fpath.name != "run_now.py":
                try:
                    shutil.move(str(fpath), str(archive_subdir / fpath.name))
                    print(f"  OK - {fpath.name}")
                except Exception as e:
                    print(f"  NG - {fpath.name}: {e}")

    # 归档目录
    print("\n【目录】")
    for dir_pattern in DIRS_TO_ARCHIVE:
        for dpath in current_dir.glob(dir_pattern):
            if dpath.is_dir():
                try:
                    shutil.move(str(dpath), str(archive_subdir / dpath.name))
                    print(f"  OK - {dpath.name}/")
                except Exception as e:
                    print(f"  NG - {dpath.name}: {e}")

    print("\n" + "=" * 80)
    print(f"✓ 清理完成！所有文件已 Archive 到 {archive_subdir.name}/")
    print("=" * 80)
