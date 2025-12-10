# AEPsych 修复补丁集合

本目录包含针对 AEPsych 及相关组件的已知bug修复补丁。

---

## 📦 可用修复

### 1. [parameter_transform_skip](./parameter_transform_skip/) ⭐ RECOMMENDED

**问题**: ParameterTransformedGenerator unconditionally untransforms generator outputs

**影响**: Categorical numeric parameters double/triple transformed (2.8 → 5.6 → 17.0)

**状态**: ✅ 已修复并验证

**快速修复**:
```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/parameter_transform_skip/apply_fix.py
```

**文件**:
- `README_FIX.md` - 修复说明
- `apply_fix.py` - 自动修复脚本
- `verify_fix.py` - 验证脚本
- `parameters.py.patch` - ParameterTransformedGenerator 补丁
- `manual_generator.py.patch` - ManualGenerator 补丁
- `custom_pool_based_generator.py.patch` - CustomPoolBasedGenerator 补丁
- `ISSUE_DESCRIPTION.md` - 问题描述

**特色**: 🎯 Root cause fix - 彻底解决 transform 架构不匹配问题

---

### 2. [train_inputs_shadowing_fix](./train_inputs_shadowing_fix/)

**问题**: `ParameterTransformedModel.train_inputs` 属性遮蔽导致返回陈旧数据

**影响**: EUR实验中的动态权重更新(`lambda_t`, `gamma_t`)失效

**状态**: ✅ 已修复并验证

**快速修复**:
```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/train_inputs_shadowing_fix/apply_fix.py
```

**文件**:
- `README_FIX.md` - 修复说明
- `apply_fix.py` - 自动修复脚本
- `verify_issue_reproduction.py` - 验证脚本
- `parameters.py.patch` - 补丁代码
- `ISSUE_ParameterTransformedModel_train_inputs_shadowing.md` - 问题描述
- `TRAIN_INPUTS_SHADOWING_BUG_FIX.md` - 修复报告

---

### 3. [categorical_numeric_fix](./categorical_numeric_fix/)

**问题**: AEPsych Categorical transform 无法正确处理数值型categorical参数

**影响**: Server返回indices而非actual values，导致Oracle接收错误参数

**状态**: ⚠️ 已被 parameter_transform_skip 替代（更彻底的修复）

**快速修复**:
```bash
cd d:\ENVS\active-psych-sampling
pixi run python tools/repair/categorical_numeric_fix/verify_fix.py
```

**文件**:
- `README_FIX.md` - 修复说明
- `categorical.py.patch` - 方案A补丁 (AEPsych修复)
- `generator_fallback_integrated.md` - 方案B说明 (已集成)
- `verify_fix.py` - 验证脚本
- `ISSUE_DESCRIPTION.md` - 问题描述

**特色**: 🛡️ 双保险架构（已被 parameter_transform_skip 替代，但仍可用作 fallback）
- **方案A (外层)**: 修复AEPsych源码
- **方案B (内层)**: Generator fallback (已自动集成)

---

## 🚀 使用流程

### 1. 验证是否需要修复

每个修复目录都包含验证脚本，先运行验证：

```bash
# 验证 train_inputs 修复
pixi run python tools/repair/train_inputs_shadowing_fix/verify_issue_reproduction.py

# 验证 categorical 修复
pixi run python tools/repair/categorical_numeric_fix/verify_fix.py
```

### 2. 查看修复说明

每个目录的 `README_FIX.md` 包含详细的修复指南。

### 3. 应用修复

根据验证结果和README说明，选择自动或手动修复。

---

## 📁 目录结构规范

每个修复目录应包含：

- ✅ `README_FIX.md` - 快速修复指南
- ✅ `ISSUE_*.md` - 问题详细描述
- ✅ `*.patch` - 补丁代码
- ✅ `verify_*.py` - 验证脚本
- ✅ `apply_*.py` (可选) - 自动修复脚本

---

## 🔍 修复优先级

1. **高优先级** 🔴: 影响实验数据正确性
   - `parameter_transform_skip` - 根本性修复 transform 架构问题 ⭐ RECOMMENDED
   - `train_inputs_shadowing_fix` - 动态权重失效
   - `categorical_numeric_fix` - 参数值错误（已被 parameter_transform_skip 替代）

2. **中优先级** 🟡: 影响性能或稳定性
   - (待添加)

3. **低优先级** 🟢: 优化或便利性改进
   - (待添加)

---

## 📝 添加新修复

创建新修复时，请遵循以下规范：

1. **目录命名**: `<功能描述>_fix` (如 `categorical_numeric_fix`)

2. **必需文件**:
   ```
   <fix_name>/
   ├── README_FIX.md            # 快速指南
   ├── ISSUE_*.md               # 问题描述
   ├── *.patch                  # 补丁代码
   └── verify_*.py              # 验证脚本
   ```

3. **可选文件**:
   - `apply_*.py` - 自动修复脚本
   - `*_REPORT.md` - 详细修复报告
   - 其他辅助文件

4. **更新本文件**: 在"可用修复"部分添加新条目

---

## 🤝 贡献

发现新的bug或改进建议？欢迎：
1. 在相应issue中报告问题
2. 按照上述规范创建修复
3. 提交PR

---

## 📚 相关文档

- [AEPsych 官方文档](https://aepsych.org/)
- [BoTorch 文档](https://botorch.org/)
- 项目诊断报告: `tests/is_EUR_work/tests/`

---

**最后更新**: 2025-12-10
**维护者**: Active Psych Sampling Team
