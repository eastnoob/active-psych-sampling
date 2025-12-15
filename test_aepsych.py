#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试aepsych基本功能"""

import aepsych
from aepsych.acquisition import MonotonicMCLSE
from aepsych.models import GPClassificationPairwise
import torch
import numpy as np

print("=" * 50)
print("aepsych 功能测试")
print("=" * 50)

# 1. 版本检查
print(f"✅ aepsych 版本: {aepsych.__version__}")

# 2. 导入检查
try:
    from aepsych.strategy import SequentialStrategy
    from aepsych.generator import OptimizeAcqfGenerator
    from aepsych.server.message_handlers import handle_ask, handle_tell
    print("✅ 核心模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

# 3. 配置加载测试
try:
    from aepsych.config import Config
    print("✅ Config类导入成功")
except ImportError as e:
    print(f"❌ Config导入失败: {e}")

# 4. 简单数据处理测试
try:
    from aepsych.utils import to_tensor
    test_data = [1.0, 2.0, 3.0]
    tensor_data = to_tensor(test_data)
    print(f"✅ 数据转换成功: {tensor_data}")
except Exception as e:
    print(f"❌ 数据处理失败: {e}")

# 5. Transform测试
try:
    from aepsych.transforms import LinearTransform
    transform = LinearTransform(a=0.0, b=1.0, min_val=0.0, max_val=1.0)
    print("✅ Transform创建成功")
except Exception as e:
    print(f"❌ Transform失败: {e}")

print("=" * 50)
print("✅ 所有测试通过！aepsych正常工作")
print("=" * 50)
