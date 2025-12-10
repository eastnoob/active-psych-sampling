# 2025-12-10 EUR categorical transform handoff

## 状态

- 仍在发生参数修正: AEPsych ask 返回索引 (x1=0.0, x2=1.0) 被 param_validator 修正为实际值 (2.8, 6.5).
- design_space 已改为为 x1/x2 提供索引 pool_points；pool_points 里仍存在归一化/越界值 (如 x4=3.0) 待查。
- AEPsych `_tensor_to_config` 会调用 `indices_to_str`, 但返回仍是索引；怀疑 CustomPoolBasedGenerator 或 pool_points 编码导致 transform 未正确映射。

## 已知关键点

- 配置 `eur_config_residual.ini` 使用 categorical choices+lb/ub (索引范围)。
- param_validator 是 workaround，EUR 阶段启用并会改写 x_array。
- pool_points 注入自 `server_manager.transform_to_numeric` 输出；当前映射: x1 2.8→0,4.0→1,8.5→2；x2 6.5→0,8.0→1；x3/4/5/6 映射为 0/1/2。

## 下一步建议

1) 检查 pool_points 生成链路，定位为何出现非整数索引(如 x4=3.0、0.75 等归一化值)。
2) 确认 CustomPoolBasedGenerator 返回张量是否被 transforms 处理；必要时在 AEPsych `_tensor_to_config` 前后打印 `next_x` 与 `parnames`。
3) 若 transform 仍不生效，可临时保留 param_validator，待 pool_points 编码修正后再移除。

## 补充计划

- 在 `server_manager.transform_to_numeric` 后直接 assert/日志各列是否为合法整数索引，阻断越界池点进入 config。
- 在 AEPsych `_tensor_to_config` 前后加日志，验证 indices_to_str 输入/输出；如仍返回索引，最小复现单元测试 indices_to_str。
- 定位 pool_points 归一化来源（是否存在额外 scaling）；必要时在生成阶段强制 round/clip 到合法索引集合。
