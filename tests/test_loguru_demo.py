#!/usr/bin/env python3
"""演示 loguru 日志等级和配置"""

from loguru import logger
import sys
import os

# 从环境变量读取日志级别，默认为 INFO
log_level = os.getenv("LOG_LEVEL", "INFO")

logger.remove()  # 移除默认处理器
logger.add(
    sys.stderr, format="{time:HH:mm:ss} | {level: <8} | {message}", level=log_level
)

print("=" * 70)
print(f"loguru 日志演示 (当前日志级别: {log_level})")
print("=" * 70)

logger.trace("📍 TRACE: 最详细的跟踪信息（只在 LOG_LEVEL=TRACE 时显示）")
logger.debug("🔍 DEBUG: 调试信息（在 LOG_LEVEL=DEBUG 或 TRACE 时显示）")
logger.info("ℹ️  INFO: 重要信息（在 LOG_LEVEL=INFO 及以上时显示）")
logger.warning("⚠️  WARNING: 警告（始终显示）")
logger.error("❌ ERROR: 错误（始终显示）")

print("\n" + "=" * 70)
print("如何控制日志级别：")
print("\n方式 1 - 环境变量（推荐）:")
print("  PowerShell: $env:LOG_LEVEL = 'DEBUG'")
print("  PowerShell: & python.exe test_loguru_demo.py")
print("\n方式 2 - 直接运行（使用默认的 INFO 级别）:")
print("  & python.exe test_loguru_demo.py")
print("\n可用的级别（从高到低）:")
print("  TRACE < DEBUG < INFO < SUCCESS < WARNING < ERROR < CRITICAL")
print("=" * 70)
