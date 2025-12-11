#!/usr/bin/env python3
"""快速验证 get_config_options 修复"""

import sys
import os
from pathlib import Path

# 设置路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "extensions" / "custom_generators"))
sys.path.insert(0, str(PROJECT_ROOT / "temp_aepsych"))

try:
    from custom_pool_based_generator import CustomPoolBasedGenerator
    from aepsych.config import Config

    print("✅ CustomPoolBasedGenerator 导入成功")

    # 检查 get_config_options 是否是类方法
    import inspect

    method = getattr(CustomPoolBasedGenerator, "get_config_options")
    is_classmethod = isinstance(
        inspect.getattr_static(CustomPoolBasedGenerator, "get_config_options"),
        classmethod,
    )

    print(f"✅ get_config_options 是否为 classmethod: {is_classmethod}")

    if is_classmethod:
        print("✅ 修复成功！get_config_options 现在是 @classmethod")
    else:
        print("❌ 修复失败！get_config_options 仍然不是 classmethod")

except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback

    traceback.print_exc()
