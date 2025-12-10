"""
自动集成：CustomPoolBasedGenerator Fallback Mapping (方案B)

这是一个已自动集成到代码中的fallback机制，无需手动应用补丁。

【功能描述】

在 CustomPoolBasedGenerator 中实现了自动的 indices → actual values 映射，
作为 AEPsych Categorical transform 失效时的备用方案。

【工作原理】

1. **在 pool generation 时**：
   - 自动提取并存储 categorical mappings
   - 格式：{param_idx: {0: 2.8, 1: 4.0, 2: 8.5}}

2. **在 gen() 返回点时**：
   - 自动检测返回的点是否为 indices
   - 如果是 indices，自动应用映射转换为 actual values
   - 如果已经是 actual values，保持不变

3. **触发条件**：
   - 当 AEPsych 的 Categorical transform 失效时
   - Generator 会记录 warning 日志表明 fallback 被触发

【验证方法】

运行测试验证 fallback 机制：

    pixi run python tests/is_EUR_work/tests/20251210_100524_pool_constraint_diagnosis/test_generator_fallback_mapping.py

预期输出：
    [SUCCESS] Mapping applied correctly!
    方案B (Fallback mapping) is working correctly!

【双保险架构】

┌─────────────────────────────────────────────┐
│  外层防护 (方案A: AEPsych transform)         │
│  Pool → GP → Transform → 2.8 → Oracle       │
└─────────────────────────────────────────────┘
                ↓ (如果失效)
┌─────────────────────────────────────────────┐
│  内层防护 (方案B: Generator fallback) ✅已集成│
│  Pool → GP → gen() → 2.8 → Oracle           │
└─────────────────────────────────────────────┘

【修改的文件】

extensions/custom_generators/custom_pool_based_generator.py

关键修改：
1. 第 201-203 行：添加 _categorical_mappings 属性
2. 第 719-729 行：_generate_pool_from_config 返回 mappings
3. 第 502-550 行：新增 _ensure_actual_values() 方法
4. 第 494-498 行：gen() 中调用 fallback mapping

【优势】

- ✅ 自动生效，无需手动配置
- ✅ 智能检测，仅在需要时触发
- ✅ 向后兼容，不影响现有功能
- ✅ 持久有效，不受 AEPsych 版本影响

【日志监控】

如果看到以下 warning 日志，说明 fallback 被触发：

    [PoolGen FALLBACK] Applied N categorical mappings (AEPsych transform failed)

这表明方案A可能失效，但方案B成功兜底，系统仍正常运行。

【注意事项】

- 此方案已自动集成，无需额外操作
- 与方案A互补，构成双保险机制
- 即使 AEPsych 升级，此方案仍然有效
"""
