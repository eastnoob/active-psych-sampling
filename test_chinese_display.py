#!/usr/bin/env python3
"""Test Chinese character display in Windows GBK environment."""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from extensions.dynamic_eur_acquisition.logging_setup import configure_logger
from loguru import logger

# Configure logger with our UTF-8 wrapper
configure_logger(level="INFO")

print("=" * 70)
print("Testing Chinese character display with loguru")
print("=" * 70)
print()

# Test various Chinese messages
logger.info("【测试开始】")
logger.info("主效应贡献：均值=0.5, 标准差=0.1")
logger.info("二阶交互效应：均值=0.3, 标准差=0.05")
logger.info("三阶交互效应：均值=0.1, 标准差=0.02")
logger.info("信息项测试")
logger.info("覆盖项测试")
logger.warning("⚠️  效应贡献数据不可用")
logger.success("✓ 测试完成")

print()
print("=" * 70)
print("If you see Chinese characters above (not ???), it works!")
print("如果上面显示中文字符（不是 ???），说明成功了！")
print("=" * 70)
