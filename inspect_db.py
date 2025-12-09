#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查现有AEPsych实验数据库的结构
"""

import sys
import sqlite3
from pathlib import Path


def inspect_database(db_path):
    """检查数据库结构"""
    print(f"检查数据库: {db_path}")

    if not Path(db_path).exists():
        print("数据库文件不存在")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"数据库表: {[t[0] for t in tables]}")

        # 检查每个表的结构
        for (table_name,) in tables:
            print(f"\n=== 表 {table_name} 的结构 ===")
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            for col in columns:
                print(f"  {col[1]} ({col[2]})")

            # 显示前几行数据
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cursor.fetchall()
            if rows:
                print(f"前5行数据:")
                for row in rows:
                    print(f"  {row}")
            else:
                print("  (表为空)")

        # 特别检查参数相关的表
        for table_name in ["param_data", "param_history", "raw_data"]:
            if any(t[0] == table_name for t in tables):
                print(f"\n=== 详细检查 {table_name} ===")
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                print(f"总行数: {len(rows)}")
                if rows:
                    print("所有数据:")
                    for i, row in enumerate(rows):
                        print(f"  {i+1}: {row}")

        conn.close()

    except Exception as e:
        print(f"数据库检查失败: {e}")


if __name__ == "__main__":
    # 检查最新的实验数据库
    db_paths = [
        "tests/is_EUR_work/00_plans/251206/scripts/results/20251209_180110_db_fixed_test/debug/experiment.db",
        "tests/is_EUR_work/00_plans/251206/scripts/results/20251209_175010/debug/experiment.db",
    ]

    for db_path in db_paths:
        print("=" * 80)
        inspect_database(db_path)
        print()
