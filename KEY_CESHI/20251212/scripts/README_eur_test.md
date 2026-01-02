# EUR采集器残差测试工具

## 功能

测试EUR采集函数，使用BaseGP关键点作为warmup数据。

## 资源文件

- **配置文件**: [configs/eur_residual_test.ini](configs/eur_residual_test.ini)
- **设计空间**: `data/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv` (6个参数)
- **BaseGP关键点**: `extensions/warmup_budget_check/phase1_analysis_output/202512081445/step3/base_gp_key_points.json`

## 参数空间

| 参数 | 类型 | 范围/值 |
|------|------|--------|
| x1_CeilingHeight | 连续 | [2.8, 8.5] |
| x2_GridModule | 连续 | [6.5, 8.0] |
| x3_OuterFurniture | 序数 | Chaos < Rotated < Strict |
| x4_VisualBoundary | 序数 | Color < Solid < Translucent |
| x5_PhysicalBoundary | 序数 | Closed < Open |
| x6_InnerFurniture | 序数 | Chaos < Rotated < Strict |

## BaseGP关键点

使用3个关键点进行warmup：

1. **x_best**: 最高预测均值的点
   - [2.8, 6.5, 2.0, 2.0, 0.0, 0.0]
   - 预测均值: 1.37, 标准差: 0.79

2. **x_worst**: 最低预测均值的点
   - [4.0, 6.5, 0.0, 0.0, 1.0, 2.0]
   - 预测均值: -1.59, 标准差: 0.79

3. **x_max_std**: 最大不确定性的点
   - [8.5, 8.0, 2.0, 2.0, 1.0, 0.0]
   - 预测均值: 0.75, 标准差: 0.80

## 使用方法

### 基本运行

```bash
cd KEY_CESHI/20251212
python run_eur_residual.py
```

### 自定义参数

```bash
python run_eur_residual.py --budget 100 --seed 123
```

**参数说明**：
- `--budget`: EUR采样预算（默认50）
- `--seed`: 随机种子（默认42）

## 测试流程

1. **加载资源**
   - 设计空间CSV
   - BaseGP关键点
   - EUR配置文件

2. **创建Strategy**
   - 初始化策略（3次询问）
   - EUR采样策略（50次询问）

3. **Warmup阶段**
   - 使用3个BaseGP关键点
   - 查询SimpleOracle获取响应
   - 添加到Strategy

4. **EUR采样**
   - 运行50次EUR采集
   - 每10次打印进度
   - 记录所有采样历史

5. **保存结果**
   - `results/<timestamp>/data_files/`
     - `sampling_history.npy`: 采样历史
     - `interaction_log.json`: 交互日志
     - `warmup_logs.json`: Warmup日志
   - `results/<timestamp>/experiment_summary.json`: 实验摘要

## 输出示例

```
================================================================================
EUR Residual Sampling Test
================================================================================
Budget: 50, Seed: 42
Results directory: KEY_CESHI/20251212/results/20251213_142536
Loading resources...
  Config: KEY_CESHI/20251212/configs/eur_residual_test.ini
  Design space: data/i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv
  BaseGP keypoints: extensions/.../base_gp_key_points.json
  Design space shape: (324, 7)
Loaded 3 BaseGP keypoints
Oracle weights: [ 0.24  -0.13   0.35  -0.08   0.19  -0.27]
Oracle bias: 3.0
Creating Strategy...
Parameters: ['x1_CeilingHeight', 'x2_GridModule', ...]
Strategy created with 2 sub-strategies
Warmup with BaseGP keypoints...
  Warmup x_best: y=1
  Warmup x_worst: y=1
  Warmup x_max_std: y=1
Warmup completed: 3 keypoints added
Running EUR sampling...
Starting EUR sampling: budget=50
  Trial 10/50 completed
  Trial 20/50 completed
  Trial 30/50 completed
  Trial 40/50 completed
  Trial 50/50 completed
Sampling completed: 50 samples
Saving results...
================================================================================
Done!
Results saved to: KEY_CESHI/20251212/results/20251213_142536
================================================================================
```

## 依赖模块

使用[KEY_CESHI/20251212/scripts/modules](scripts/modules/)中的模块：

- `design_space.py`: 设计空间加载
- `data_saver.py`: 数据保存工具

## 注意事项

1. **Oracle模型**: 当前使用SimpleOracle（随机权重），实际应用中应替换为真实的被试模型
2. **DynamicEURGenerator**: 需要extensions/dynamic_eur_acquisition可用
3. **序数参数**: 配置文件中使用ordinal类型，需要AEPsych支持

## 验证测试

运行测试验证功能：

```bash
python run_eur_residual.py --budget 10 --seed 42
```

应该看到：
- ✅ 成功加载3个BaseGP关键点
- ✅ Warmup完成3次查询
- ✅ EUR采样完成10次查询
- ✅ 结果保存到`results/<timestamp>/`
