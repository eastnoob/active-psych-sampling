#!/usr/bin/env python3
"""验证 get_available_indices 调用修复"""

import sys
from pathlib import Path
import inspect

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "extensions" / "custom_generators" / "models"))

from pool_utils import get_available_indices

# 检查函数签名
sig = inspect.signature(get_available_indices)
print("get_available_indices 函数签名:")
print(f"  {get_available_indices.__name__}{sig}")
print()

# 列出参数
params = list(sig.parameters.keys())
print("参数顺序（必须按此顺序调用）:")
for i, param in enumerate(params, 1):
    param_obj = sig.parameters[param]
    print(
        f"  {i}. {param}: {param_obj.annotation if param_obj.annotation != inspect.Parameter.empty else '(无类型提示)'}"
    )

print()
print("✅ 现在代码调用时应使用:")
print("  pool_utils.get_available_indices(")
print("      self._used_indices,           # 1. 当前运行中已使用的索引")
print("      len(self.pool_points),        # 2. 池的总大小")
print("      self.pool_points,              # 3. 所有池点")
print("      self._historical_points        # 4. 历史点集合")
print("  )")
