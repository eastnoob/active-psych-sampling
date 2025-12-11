# Ordinal参数类型实现 - tools/repair文件夹新建指引

**关键信息**: 所有AEPsych核心系统文件的改动都必须在 `tools/repair/` 文件夹中通过**patch方式**实现。

---

## 📂 需要新建的文件夹

```
tools/repair/ordinal_parameter_extension/
```

---

## 📝 完整文件列表 (复制粘贴清单)

### 1. **README_ORDINAL_FIX.md**
- 位置: `tools/repair/ordinal_parameter_extension/README_ORDINAL_FIX.md`
- 内容: 详细的patch说明、应用方法、验证步骤
- 大小: ~2KB

### 2. **aepsych_ordinal_transforms.patch**
- 位置: `tools/repair/ordinal_parameter_extension/aepsych_ordinal_transforms.patch`
- 目标: 创建 `aepsych/transforms/ops/ordinal.py`
- 内容: ~180 LOC的Ordinal类完整实现
- 类型: 统一diff format

### 3. **aepsych_transforms_parameters.patch**
- 位置: `tools/repair/ordinal_parameter_extension/aepsych_transforms_parameters.patch`
- 目标: 修改 `aepsych/transforms/parameters.py`
- 修改: `get_config_options()` 方法 (+50 LOC)
- 内容: 添加ordinal参数处理分支

### 4. **aepsych_config.patch**
- 位置: `tools/repair/ordinal_parameter_extension/aepsych_config.patch`
- 目标: 修改 `aepsych/config.py`
- 修改: 参数类型验证 (+10 LOC)
- 内容: 添加"custom_ordinal"和"custom_ordinal_mono"到有效类型列表

### 5. **apply_fix.py**
- 位置: `tools/repair/ordinal_parameter_extension/apply_fix.py`
- 功能: 自动化应用所有patch文件
- 特性:
  - 自动检测AEPsych安装位置
  - 自动备份原始文件
  - 依次应用3个patch
  - 错误处理和报告

### 6. **verify_fix.py**
- 位置: `tools/repair/ordinal_parameter_extension/verify_fix.py`
- 功能: 验证patch是否正确应用
- 检查项:
  - ordinal.py是否可导入
  - parameters.py是否包含修改
  - config.py是否包含新参数类型
  - __init__.py是否导入Ordinal

### 7. **files/ordinal.py**
- 位置: `tools/repair/ordinal_parameter_extension/files/ordinal.py`
- 内容: `aepsych/transforms/ops/ordinal.py`的完整源代码
- 大小: ~180 LOC
- 用途: 备份和直接复制应用

---

## 🔧 如何快速新建

### 步骤1: 创建文件夹结构
```bash
mkdir -p tools/repair/ordinal_parameter_extension/files
```

### 步骤2: 从handoff文件夹复制关键内容
- 从 `handoff/AEPSYCH_MODIFICATIONS_PATCH_GUIDE.md` 中提取patch内容
- 从 `handoff/20251211_ordinal_monotonic_parameter_extension.md` 中提取ordinal.py代码

### 步骤3: 创建各文件

#### **创建 files/ordinal.py**
```bash
# 从文档中提取的Ordinal类完整代码
# 行数: ~180
# 包含: __init__, _transform, _untransform, transform_bounds, from_config等方法
```

#### **创建 aepsych_ordinal_transforms.patch**
```
--- /dev/null
+++ b/aepsych/transforms/ops/ordinal.py
@@ -0,0 +1,180 @@
+#!/usr/bin/env python3
+(文件内容)
```

#### **创建 aepsych_transforms_parameters.patch**
```
--- a/aepsych/transforms/parameters.py
+++ b/aepsych/transforms/parameters.py
@@ -240,6 +240,25 @@ class ParameterTransforms(ConfigurableMixin):
         # 添加elif分支
         elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
             ...
```

#### **创建 aepsych_config.patch**
```
--- a/aepsych/config.py
+++ b/aepsych/config.py
@@ -100,10 +100,12 @@ class AEPsychConfig(ConfigParser):
         valid_par_types = [
             ...
             "fixed",
+            "custom_ordinal",
+            "custom_ordinal_mono",
         ]
```

#### **创建 apply_fix.py**
- 基于 `parameter_transform_skip/apply_fix.py` 的模式
- 修改为应用3个patch文件
- 添加ordinal_parameter_extension特定的逻辑

#### **创建 verify_fix.py**
- 基于 `parameter_transform_skip/verify_fix.py` 的模式
- 添加验证Ordinal类导入
- 检查parameters.py中的ordinal分支
- 检查config.py中的新参数类型

#### **创建 README_ORDINAL_FIX.md**
- 包含修改概述
- 应用方法 (自动/手动)
- 验证步骤
- 回滚方法
- 兼容性说明

---

## 📌 关键要点

### **与现有repair文件夹保持一致**

对比 `parameter_transform_skip` 文件夹:

```
parameter_transform_skip/
├── README_FIX.md                         ← 对应 README_ORDINAL_FIX.md
├── apply_fix.py                          ← 对应 apply_fix.py
├── verify_fix.py                         ← 对应 verify_fix.py
├── parameters.py.patch                   ← 对应 aepsych_*.patch
├── custom_pool_based_generator.py.patch
├── manual_generator.py.patch
└── ISSUE_DESCRIPTION.md

ordinal_parameter_extension/
├── README_ORDINAL_FIX.md                 ← 类似结构
├── apply_fix.py                          ← 类似结构
├── verify_fix.py                         ← 类似结构
├── aepsych_ordinal_transforms.patch      ← 新建文件patch
├── aepsych_transforms_parameters.patch   ← 修改参数处理patch
├── aepsych_config.patch                  ← 修改配置patch
└── files/
    └── ordinal.py                        ← 新建文件的完整源代码
```

### **新建文件的patch格式**

对于新建的 `ordinal.py`，patch格式为:
```patch
--- /dev/null
+++ b/aepsych/transforms/ops/ordinal.py
@@ -0,0 +1,180 @@
+#!/usr/bin/env python3
+# ... 完整文件内容
```

### **修改文件的patch格式**

对于修改的 `parameters.py` 和 `config.py`，patch格式为:
```patch
--- a/aepsych/transforms/parameters.py
+++ b/aepsych/transforms/parameters.py
@@ -240,6 +240,25 @@ class ParameterTransforms(ConfigurableMixin):
     # ... 上下文行
     elif par_type in ["custom_ordinal", "custom_ordinal_mono"]:
+        # ... 新增的代码
```

---

## ✅ 完成检查清单

- [ ] 创建文件夹 `tools/repair/ordinal_parameter_extension/`
- [ ] 创建 `files/` 子文件夹
- [ ] 创建 `README_ORDINAL_FIX.md` (说明文档)
- [ ] 创建 `aepsych_ordinal_transforms.patch` (新建ordinal.py)
- [ ] 创建 `aepsych_transforms_parameters.patch` (修改parameters.py)
- [ ] 创建 `aepsych_config.patch` (修改config.py)
- [ ] 创建 `apply_fix.py` (自动化脚本)
- [ ] 创建 `verify_fix.py` (验证脚本)
- [ ] 创建 `files/ordinal.py` (完整源代码)
- [ ] 测试 `python apply_fix.py` 是否成功
- [ ] 测试 `python verify_fix.py` 是否验证通过

---

## 📚 参考资源

### handoff文件夹中的参考文档:
- `AEPSYCH_MODIFICATIONS_PATCH_GUIDE.md` - 详细的patch实现指南
- `20251211_ordinal_monotonic_parameter_extension.md` - Ordinal类完整代码
- `FINAL_CHECKLIST.md` - 最终验证清单

### tools/repair中的参考模板:
- `parameter_transform_skip/` - apply_fix.py和verify_fix.py的参考
- `categorical_numeric_fix/` - 另一个patch实现的参考
- `train_inputs_shadowing_fix/` - 备份机制的参考

---

## 🚀 应用方式

### 自动应用:
```bash
cd tools/repair/ordinal_parameter_extension
python apply_fix.py
```

### 验证:
```bash
python verify_fix.py
```

### 回滚:
```bash
# apply_fix.py会自动创建备份，恢复原始文件
```

---

**注意**: 此方式确保所有AEPsych系统文件的改动都有清晰的版本控制记录，便于维护和在新环境快速应用。
