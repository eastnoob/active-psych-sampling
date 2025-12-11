README - AEPsych Categorical Transform 查询结果
================================================

CONTENTS - 文档内容
===================

已为你生成以下 4 个完整文档（共约 1800+ 行详细内容）:

1. QUERY_SUMMARY.md
   └─ 查询完成总结，包含快速导航和索引
   
2. AEPsych_Categorical_Transform_Analysis.md
   └─ 详细的源代码分析，包含所有 5 个问题需求
   
3. AEPsych_Categorical_Complete_Source.py
   └─ 完整的 Categorical 类源代码，每个方法都有详细中文注释
   
4. AEPsych_Categorical_QuickRef.md
   └─ 快速参考表和查询索引
   
5. AEPsych_Categorical_Problems_and_Fixes.md
   └─ 三个核心问题的详细演示和修复方案


QUICK START - 快速开始
====================

如果你只有 5 分钟: 
→ 打开 QUERY_SUMMARY.md 的「核心发现」部分

如果你只有 15 分钟:
→ 按顺序读: QuickRef.md → Transform_Analysis.md (前半部分)

如果你只有 1 小时:
→ 读完所有文档，优先看 Complete_Source.py 的代码注释

如果需要实施修复:
→ 重点阅读 Problems_and_Fixes.md 的「修复优先级」和「测试用例」


KEY FINDINGS - 核心发现
======================

源文件: .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py

三个核心问题:

1. element_type=str (第 97 行)
   问题: 数值分类被强制转换为字符串
   影响: 所有数值型分类参数 (如 [2.8, 4.0, 8.5])
   优先级: 🔴 MUST FIX

2. _untransform 假设输入是 indices (第 60-68 行)
   问题: 没有检查输入是否已是实际值
   影响: 双重转换导致值错误 (2.8 → 5.6 → 17.0)
   优先级: 🟠 RECOMMENDED

3. ParameterTransformedGenerator 无条件 untransform
   问题: 无法判断 generator 是否已处理转换
   影响: 某些 generator 的输出被重复处理
   优先级: 🟡 OPTIONAL


WHAT YOU ASKED FOR - 你的 5 个需求
==================================

1. Categorical 类的完整 __init__ 和主要方法
   ✓ Analysis.md 第 1 部分 (第 1-37 行)
   ✓ Complete_Source.py 第 26-42 行 (完整代码 + 详细注释)
   
2. _transform 和 _untransform 的实现
   ✓ Analysis.md 第 2 部分 (第 42-116 行)
   ✓ Complete_Source.py 第 44-73 行
   ✓ QuickRef.md 第 2-3 部分

3. get_config_options 的实现
   ✓ Analysis.md 第 3 部分 (第 118-194 行) 【包含问题分析】
   ✓ Complete_Source.py 第 75-143 行 【最详细的注释】
   ✓ Problems_and_Fixes.md 问题 1 演示

4. bounds 的设置方式
   ✓ Analysis.md 第 4 部分 (第 196-272 行)
   ✓ Complete_Source.py 第 145-229 行
   ✓ QuickRef.md 第 4-5 部分

5. 特殊的配置逻辑
   ✓ Analysis.md 第 5 部分 (第 274-349 行)
   ✓ Complete_Source.py 第 231-307 行
   ✓ 包含 StringParameterMixin.indices_to_str() 的完整实现


DOCUMENT READING GUIDE - 文档阅读指南
====================================

按级别选择:

L1 快速浏览 (5-10 分钟):
  └─ QUERY_SUMMARY.md: 核心发现 + 快速查询索引
  
L2 快速参考 (15-20 分钟):
  └─ QuickRef.md: 所有表格和代码片段
  
L3 深入理解 (30-45 分钟):
  └─ Transform_Analysis.md: 完整的逐方法分析
  
L4 完全掌握 (1-2 小时):
  └─ Complete_Source.py: 完整源代码 + 所有注释
  
L5 实施修复 (2-3 小时):
  └─ Problems_and_Fixes.md: 问题演示 + 修复方案 + 测试


NAVIGATION - 快速导航
====================

快速查找...

"__init__ 做了什么?"
  → Complete_Source.py 第 26-42 行
  → Analysis.md 第 1 部分
  → QuickRef.md 第 1 部分

"_transform 和 _untransform 有什么区别?"
  → Complete_Source.py 第 44-73 行 (注释最详细)
  → Analysis.md 第 2 部分
  → 答: 都只是四舍五入，无实际映射

"choices 怎么被解析的?"
  → Complete_Source.py 第 91-143 行 (最详细)
  → Analysis.md 第 3 部分
  → Problems_and_Fixes.md 问题 1

"为什么数值分类出现错误?"
  → Problems_and_Fixes.md 问题 1 + 问题 2 演示
  → Complete_Source.py 的 double transform 部分

"怎么修复?"
  → Problems_and_Fixes.md: 完整修复方案 (3 个方案对比)
  → Complete_Source.py 最后的"修复建议"部分
  → QuickRef.md 最后的"推荐修复方案"

"Bounds 怎么工作的?"
  → Analysis.md 第 4 部分 (有详细表格)
  → Complete_Source.py 第 145-229 行 (完整源代码)
  → QuickRef.md Bounds 部分


FILE LOCATIONS - 文件位置
=========================

所有文档都在工作区根目录:

d:\ENVS\active-psych-sampling\
├── QUERY_SUMMARY.md  ⭐ 从这里开始
├── AEPsych_Categorical_Transform_Analysis.md
├── AEPsych_Categorical_Complete_Source.py
├── AEPsych_Categorical_QuickRef.md
├── AEPsych_Categorical_Problems_and_Fixes.md
└── README (本文件)

原始源文件位置:
.pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/categorical.py (165 行)


RELATED DOCUMENTATION - 相关文档参考
===================================

本工作区的相关修复文件:

tools/repair/categorical_numeric_fix/
  ├── README_FIX.md - 修复说明 (方案 A)
  ├── generator_fallback_integrated.md - 方案 B 说明
  └── verify_fix.py - 验证脚本

tools/repair/parameter_transform_skip/
  ├── README_FIX.md - 参数转换跳过修复
  └── apply_fix.py - 自动修复脚本

extensions/handoff/
  ├── 20251210_categorical_transform_root_issue.md - 根本问题分析
  └── README_ORDINAL_HANDOFF.md - 序数参数替代方案

extensions/custom_generators/custom_pool_based_generator.py
  └─ 包含已集成的 Fallback 机制 (方案 B)


TIPS & NOTES - 提示与注意
=========================

✓ 所有代码片段都可以直接复制使用

✓ 每个文档都是独立的，可按任意顺序阅读

✓ Complete_Source.py 中的注释是最详细的，建议重点学习

✓ Problems_and_Fixes.md 提供了完整的测试用例，可直接运行

✓ 修复方案按优先级排序 (🔴 > 🟠 > 🟡)

✓ QuickRef.md 适合作为速查表收藏


CONTACT & QUESTIONS - 问题咨询
=============================

如果需要:

- 实施具体的代码修复
  → 参考 Problems_and_Fixes.md 的「修复优先级」部分

- 理解具体的算法原理
  → 参考 Complete_Source.py 中对应方法的注释

- 进行单元测试
  → 参考 Problems_and_Fixes.md 的「测试用例」部分

- 性能优化
  → 参考 Analysis.md 中关于 Bounds 的部分


=====================================
生成时间: 2025-12-11
查询内容: AEPsych Categorical Transform 完整实现分析
生成文档数: 5 份 (1800+ 行)
=====================================
