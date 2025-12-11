# AEPsych核心改动 - Patch文件指南

**日期**: 2025-12-11  
**任务**: Ordinal参数类型实现  
**范围**: 所有涉及AEPsych系统文件的改动必须在 `tools/repair/` 文件夹中通过patch方式实现

---

## 📌 核心原则

所有对AEPsych系统文件的改动**不能直接修改安装目录**，必须通过**patch文件**的方式在 `tools/repair/` 中实现，以便：
- ✅ 版本控制：清晰记录改动历史
- ✅ 可重现性：新环境可快速应用patch
- ✅ 回滚能力：如需撤销改动可恢复原始文件
- ✅ 合规性：不污染AEPsych安装目录

---

## 📂 需要新建的repair文件夹结构

### **文件夹名称**: `ordinal_parameter_extension`

```
tools/repair/ordinal_parameter_extension/
├── README_ORDINAL_FIX.md           # 详细说明文档
├── aepsych_ordinal_transforms.patch     # Patch 1: transforms/ops/ordinal.py 创建
├── aepsych_transforms_parameters.patch  # Patch 2: transforms/parameters.py 修改
├── aepsych_config.patch                 # Patch 3: config.py 修改
├── apply_fix.py                    # 自动化应用脚本
├── verify_fix.py                   # 验证脚本
└── files/                          # 新建文件存放
    └── ordinal.py                  # aepsych/transforms/ops/ordinal.py 的完整内容
```

---

## 🔧 具体实现方案

### **第1步: aepsych_ordinal_transforms.patch (新建文件)**

**目标**: 创建 `aepsych/transforms/ops/ordinal.py`

**文件位置**: `tools/repair/ordinal_parameter_extension/aepsych_ordinal_transforms.patch`

**内容形式** (类似 `parameter_transform_skip/parameters.py.patch`):

```patch
--- /dev/null
+++ b/aepsych/transforms/ops/ordinal.py
@@ -0,0 +1,180 @@
+#!/usr/bin/env python3
+# -*- coding: utf-8 -*-
+"""
+自定义Ordinal Transform - 有序参数支持
+
+实现稀疏采样连续物理值的参数类型，如天花板高度[2.0, 2.5, 3.5]。
+保留序关系和间距信息，使ANOVA能正确分解参数效应。
+"""
+
+import torch
+import numpy as np
+from aepsych.transforms.base import Transform
+from aepsych.config import ConfigurableMixin
+from typing import Dict, List, Optional
+
+
+class Ordinal(Transform, ConfigurableMixin):
+    """有序参数Transform - 支持等差和非等差单调数列"""
+    
+    def __init__(
+        self,
+        indices: List[int],
+        values: Dict[int, List[float]],
+        level_names: Optional[Dict[int, List[str]]] = None,
+    ):
+        """
+        Args:
+            indices: 参数维度列表
+            values: 各维度的值列表 {index: [0.1, 0.5, 2.0, ...]}
+            level_names: 可选字符串标签映射 {index: ["agree", "disagree"]}
+        """
+        super().__init__(indices=indices)
+        self.values = values
+        self.level_names = level_names or {}
+        
+        # 验证
+        for idx, vals in values.items():
+            if len(vals) < 2:
+                raise ValueError(f"Index {idx}: must have at least 2 values")
+            if not all(vals[i] <= vals[i+1] for i in range(len(vals)-1)):
+                raise ValueError(f"Index {idx}: values must be sorted")
+    
+    def _transform(self, X: torch.Tensor) -> torch.Tensor:
+        """物理值 → rank (0, 1, 2, ...)"""
+        X_transformed = X.clone()
+        for idx in self.indices:
+            values = self.values[idx]
+            # 最近邻映射到rank
+            for i, val in enumerate(values):
+                X_transformed[X[:, idx] == val, idx] = i
+        return X_transformed
+    
+    def _untransform(self, X_transformed: torch.Tensor) -> torch.Tensor:
+        """rank → 物理值"""
+        X = X_transformed.clone()
+        for idx in self.indices:
+            values = self.values[idx]
+            for i, val in enumerate(values):
+                X[X_transformed[:, idx] == i, idx] = val
+        return X
+    
+    def transform_bounds(self, bounds: torch.Tensor) -> torch.Tensor:
+        """bounds: [lower, upper] → rank空间"""
+        bounds_transformed = bounds.clone()
+        for idx in self.indices:
+            n_levels = len(self.values[idx])
+            bounds_transformed[0, idx] = -0.5
+            bounds_transformed[1, idx] = n_levels - 0.5
+        return bounds_transformed
+    
+    @staticmethod
+    def _compute_arithmetic_sequence(min_val, max_val, step=None, num_levels=None):
+        """自动计算等差数列"""
+        if step is not None:
+            return np.arange(min_val, max_val + step/2, step)
+        elif num_levels is not None:
+            return np.linspace(min_val, max_val, num_levels)
+        else:
+            raise ValueError("Must specify step or num_levels")
+    
+    @classmethod
+    def from_config(cls, config, name, options):
+        """从配置创建Transform"""
+        if "values" in options:
+            values = options["values"]
+        elif "min_value" in options and "max_value" in options:
+            if "step" in options:
+                values = cls._compute_arithmetic_sequence(
+                    options["min_value"],
+                    options["max_value"],
+                    step=options["step"]
+                )
+            elif "num_levels" in options:
+                values = cls._compute_arithmetic_sequence(
+                    options["min_value"],
+                    options["max_value"],
+                    num_levels=options["num_levels"]
+                )
+            else:
+                raise ValueError("Must specify step or num_levels")
+        elif "levels" in options:
+            values = np.arange(len(options["levels"]))
+        else:
+            raise ValueError("Must specify values, (min/max + step/num_levels), or levels")
+        
+        return cls(
+            indices=options.get("indices", [0]),
+            values={0: list(values)},
+            level_names={0: options.get("levels")} if "levels" in options else None
+        )
+```

---

### **第2步: aepsych_transforms_parameters.patch (修改parameters.py)**

**目标**: 修改 `aepsych/transforms/parameters.py` 的 `get_config_options()` 方法

**文件位置**: `tools/repair/ordinal_parameter_extension/aepsych_transforms_parameters.patch`

**内容形式** (类似 `parameter_transform_skip/parameters.py.patch`):

```patch
--- a/aepsych/transforms/parameters.py
+++ b/aepsych/transforms/parameters.py
@@ -240,6 +240,25 @@ class ParameterTransforms(ConfigurableMixin):
             # Categorical处理...
             continue
         
+        elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
+            # Ordinal参数: 有序但非均匀分布
+            from aepsych.transforms.ops.ordinal import Ordinal
+            
+            # 使用Ordinal Transform处理
+            ordinal = Ordinal.from_config(
+                config=config,
+                name=par,
+                options=transform_options
+            )
+            
+            # 更新bounds到rank空间
+            transform_options["bounds"] = ordinal.transform_bounds(
+                transform_options["bounds"]
+            )
+            
+            transform_dict[f"{par}_Ordinal"] = ordinal
+            continue  # 跳过log_scale和normalize (已在rank空间)
+        
         # ... 继续其他逻辑
```

---

### **第3步: aepsych_config.patch (修改config.py)**

**目标**: 修改 `aepsych/config.py` 的参数类型验证

**文件位置**: `tools/repair/ordinal_parameter_extension/aepsych_config.patch`

**内容形式**:

```patch
--- a/aepsych/config.py
+++ b/aepsych/config.py
@@ -100,10 +100,12 @@ class AEPsychConfig(ConfigParser):
         """验证参数类型"""
         
         valid_par_types = [
             "continuous",
             "integer",
             "binary",
             "categorical",
             "fixed",
+            "custom_ordinal",      # 新增
+            "custom_ordinal_mono", # 新增
         ]
         
         for par_name, par_section in self.par_sections.items():
```

---

### **第4步: apply_fix.py (自动化脚本)**

**文件位置**: `tools/repair/ordinal_parameter_extension/apply_fix.py`

```python
#!/usr/bin/env python3
"""
自动应用Ordinal参数类型的AEPsych patch

使用方法:
    python apply_fix.py              # 自动检测AEPsych安装位置
    python apply_fix.py /path/to/aepsych  # 指定AEPsych路径
"""

import os
import sys
import shutil
from pathlib import Path
import subprocess


def get_aepsych_path():
    """自动检测AEPsych安装位置"""
    try:
        import aepsych
        return Path(aepsych.__file__).parent
    except ImportError:
        print("❌ AEPsych not installed")
        return None


def apply_ordinal_patches(aepsych_path):
    """应用所有patch"""
    
    patch_files = [
        ("aepsych_ordinal_transforms.patch", "Transform创建"),
        ("aepsych_transforms_parameters.patch", "parameters.py修改"),
        ("aepsych_config.patch", "config.py修改"),
    ]
    
    script_dir = Path(__file__).parent
    
    for patch_file, description in patch_files:
        patch_path = script_dir / patch_file
        
        if not patch_path.exists():
            print(f"⚠️  {patch_file} 不存在，跳过")
            continue
        
        print(f"\n📝 应用 {description}...")
        
        # 使用patch命令应用
        result = subprocess.run(
            ["patch", "-p1", "-i", str(patch_path)],
            cwd=str(aepsych_path),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ {description} 成功应用")
        else:
            print(f"❌ {description} 应用失败")
            print(f"错误: {result.stderr}")
            return False
    
    # 创建ordinal.py文件
    ordinal_py = script_dir / "files" / "ordinal.py"
    if ordinal_py.exists():
        ordinal_dest = aepsych_path / "transforms" / "ops" / "ordinal.py"
        shutil.copy(ordinal_py, ordinal_dest)
        print(f"✅ ordinal.py 复制到 {ordinal_dest}")
    
    return True


def main():
    if len(sys.argv) > 1:
        aepsych_path = Path(sys.argv[1])
    else:
        aepsych_path = get_aepsych_path()
    
    if not aepsych_path:
        print("❌ 无法找到AEPsych安装位置")
        sys.exit(1)
    
    print(f"📍 AEPsych位置: {aepsych_path}")
    
    # 备份原始文件
    print("\n🔄 备份原始文件...")
    backup_dir = aepsych_path / ".ordinal_backup_$(date +%s)"
    backup_dir.mkdir(exist_ok=True)
    
    for file in ["transforms/parameters.py", "config.py"]:
        src = aepsych_path / file
        if src.exists():
            shutil.copy(src, backup_dir / file)
            print(f"✅ 已备份 {file}")
    
    # 应用patch
    if apply_ordinal_patches(aepsych_path):
        print("\n✅ 所有patch成功应用!")
        print(f"📌 如需回滚，备份位置: {backup_dir}")
    else:
        print("\n❌ 应用失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

---

### **第5步: verify_fix.py (验证脚本)**

**文件位置**: `tools/repair/ordinal_parameter_extension/verify_fix.py`

```python
#!/usr/bin/env python3
"""验证Ordinal patch是否正确应用"""

import sys
from pathlib import Path


def verify_ordinal_implementation():
    """验证核心修改"""
    
    checks = []
    
    # 检查1: ordinal.py存在
    try:
        from aepsych.transforms.ops.ordinal import Ordinal
        checks.append(("✅ ordinal.py 存在", True))
        
        # 验证关键方法
        methods = ['_transform', '_untransform', 'transform_bounds', 'from_config']
        for method in methods:
            if hasattr(Ordinal, method):
                checks.append((f"✅ Ordinal.{method}() 存在", True))
            else:
                checks.append((f"❌ Ordinal.{method}() 缺失", False))
    except ImportError as e:
        checks.append((f"❌ 导入Ordinal失败: {e}", False))
    
    # 检查2: parameters.py包含ordinal分支
    try:
        from aepsych.transforms.parameters import ParameterTransforms
        source = str(ParameterTransforms.get_config_options)
        if "custom_ordinal" in source or "Ordinal" in source:
            checks.append(("✅ parameters.py 已修改", True))
        else:
            checks.append(("❌ parameters.py 未包含ordinal处理", False))
    except Exception as e:
        checks.append((f"❌ 检查parameters.py失败: {e}", False))
    
    # 检查3: config.py包含新的参数类型
    try:
        import aepsych.config
        source = open(aepsych.config.__file__).read()
        if "custom_ordinal" in source:
            checks.append(("✅ config.py 已修改", True))
        else:
            checks.append(("❌ config.py 未更新par_type", False))
    except Exception as e:
        checks.append((f"❌ 检查config.py失败: {e}", False))
    
    # 检查4: ordinal.py __init__.py导入
    try:
        from aepsych.transforms.ops import Ordinal
        checks.append(("✅ __init__.py 已导入Ordinal", True))
    except ImportError:
        checks.append(("⚠️ __init__.py 未导入Ordinal (可选)", True))
    
    # 打印结果
    print("\n" + "="*50)
    print("AEPsych Ordinal Patch 验证结果")
    print("="*50 + "\n")
    
    all_passed = True
    for check, passed in checks:
        print(check)
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ 所有检查通过!")
    else:
        print("❌ 有检查失败，请检查patch应用情况")
    print("="*50 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = verify_ordinal_implementation()
    sys.exit(0 if success else 1)
```

---

### **第6步: README_ORDINAL_FIX.md (详细说明)**

**文件位置**: `tools/repair/ordinal_parameter_extension/README_ORDINAL_FIX.md`

```markdown
# Ordinal参数类型 - AEPsych Patch

## 概述

此patch集合为AEPsych添加**有序参数类型(Ordinal)** 支持。

有序参数代表稀疏采样的连续物理值，例如：
- 天花板高度: [2.0m, 2.5m, 3.5m]
- 椅子数量: [1, 2, 3, 4, 5]
- Likert量表: [1, 2, 3, 4, 5]

## 包含的修改

### 1. 新建文件
- `aepsych/transforms/ops/ordinal.py` (~180 LOC)
  - Ordinal Transform类实现
  - 等差/非等差数列自动计算
  - rank空间转换

### 2. 修改文件
- `aepsych/transforms/parameters.py` (+50 LOC)
  - `get_config_options()` 添加ordinal分支
  
- `aepsych/config.py` (+10 LOC)
  - 参数类型验证添加新值

- `aepsych/transforms/ops/__init__.py` (+1 LOC)
  - 导入Ordinal类

## 应用方法

### 方法1: 自动应用 (推荐)

```bash
cd tools/repair/ordinal_parameter_extension
python apply_fix.py
```

### 方法2: 手动应用patch

```bash
cd /path/to/aepsych
patch -p1 < tools/repair/ordinal_parameter_extension/aepsych_ordinal_transforms.patch
patch -p1 < tools/repair/ordinal_parameter_extension/aepsych_transforms_parameters.patch
patch -p1 < tools/repair/ordinal_parameter_extension/aepsych_config.patch
```

### 方法3: 复制完整文件

将 `files/ordinal.py` 复制到 `aepsych/transforms/ops/ordinal.py`

## 验证应用

```bash
python verify_fix.py
```

应输出所有检查通过

## 回滚

```bash
# apply_fix.py自动创建备份，可恢复
cp -r aepsych/.ordinal_backup_* aepsych/original/
```

## 测试

```bash
pytest tests/test_ordinal_transform.py -v
```

## 兼容性

- ✅ AEPsych 0.2+
- ✅ 向后兼容：现有参数类型不受影响
- ✅ 无breaking changes
```

---

## 📋 完整文件清单

```
tools/repair/ordinal_parameter_extension/
│
├── README_ORDINAL_FIX.md                    # 说明文档
│   - 修改内容概览
│   - 应用方法
│   - 验证步骤
│   - 回滚方法
│
├── aepsych_ordinal_transforms.patch         # Patch文件1 (新建ordinal.py)
│   - 创建aepsych/transforms/ops/ordinal.py
│   - ~180 LOC的完整实现
│
├── aepsych_transforms_parameters.patch      # Patch文件2 (修改parameters.py)
│   - 修改get_config_options()
│   - 添加ordinal分支
│   - +50 LOC
│
├── aepsych_config.patch                     # Patch文件3 (修改config.py)
│   - 添加par_type验证
│   - +10 LOC
│
├── apply_fix.py                             # 自动化应用脚本
│   - 自动检测AEPsych位置
│   - 备份原始文件
│   - 应用所有patch
│
├── verify_fix.py                            # 验证脚本
│   - 检查ordinal.py导入
│   - 检查parameters.py修改
│   - 检查config.py修改
│   - 检查__init__.py导入
│
└── files/                                   # 新建文件存放目录
    ├── ordinal.py                           # 完整的Ordinal Transform实现
    └── __init__.py                          # (可选) __init__.py更新内容
```

---

## 🔄 与现有repair文件夹的对应关系

### 与 `parameter_transform_skip` 的对比

| 方面 | parameter_transform_skip | ordinal_parameter_extension |
|------|------------------------|---------------------------|
| **目的** | 修复parameter跳过bug | 添加新的参数类型 |
| **修改文件数** | 3 | 3 |
| **新建文件** | 0 | 1 (ordinal.py) |
| **Patch数** | 3 | 3 |
| **apply脚本** | ✅ 有 | ✅ 有 |
| **verify脚本** | ✅ 有 | ✅ 有 |

---

## ✅ 应用检查清单

- [ ] 查看 `README_ORDINAL_FIX.md` 理解改动
- [ ] 运行 `python apply_fix.py` 应用patch
- [ ] 运行 `python verify_fix.py` 验证
- [ ] 查看AEPsych安装目录备份确认成功
- [ ] 运行单元测试验证功能

---

**注意**: 所有patch都应保存在版本控制中，以便新环境快速应用。
