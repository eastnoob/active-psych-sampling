# AEPsychConfigBuilder - 快速参考

## 快速开始

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

# 1. 创建构建器
builder = AEPsychConfigBuilder()

# 2. 添加基本信息
builder.add_common(
    parnames=['x', 'y'],
    stimuli_per_trial=1,
    outcome_types=['binary'],
    strategy_names=['init', 'opt']
)

# 3. 定义参数
builder.add_parameter('x', par_type='continuous', lower_bound=0, upper_bound=1)
builder.add_parameter('y', par_type='integer', lower_bound=1, upper_bound=10)

# 4. 配置策略
builder.add_strategy('init', generator='SobolGenerator', min_asks=10)
builder.add_strategy('opt', generator='OptimizeAcqfGenerator', 
                    model='GPClassificationModel', max_asks=30)

# 5. 验证
is_valid, errors, warnings = builder.validate()

# 6. 保存
if is_valid:
    builder.to_ini('config.ini')
```

## 参数类型快速表

| 类型 | 字段 | 示例 |
|------|------|------|
| `continuous` | `par_type`, `lower_bound`, `upper_bound` | `lower_bound=0, upper_bound=1` |
| `integer` | `par_type`, `lower_bound`, `upper_bound` | `lower_bound=1, upper_bound=100` |
| `binary` | `par_type` | 仅需 `par_type='binary'` |
| `fixed` | `par_type`, `value` | `value=0.5` |
| `categorical` | `par_type`, `choices` | `choices=['a', 'b', 'c']` |

## 生成器和模型

### SobolGenerator

```python
builder.add_strategy('name', generator='SobolGenerator', min_asks=10)
```

### OptimizeAcqfGenerator（需要模型）

```python
builder.add_strategy('name', 
                    generator='OptimizeAcqfGenerator',
                    model='GPClassificationModel',
                    max_asks=30)
```

## 常见模式

### 基础配置

```python
builder = AEPsychConfigBuilder()
builder.add_common(['x'], 1, ['binary'], ['strat1'])
builder.add_parameter('x', 'continuous', 0, 1)
builder.add_strategy('strat1', 'SobolGenerator', min_asks=5)
builder.validate()
builder.to_ini('config.ini')
```

### 多参数配置

```python
builder.add_common(['x', 'y', 'z'], 1, ['binary'], ['init', 'opt'])
builder.add_parameter('x', 'continuous', 0, 1)
builder.add_parameter('y', 'integer', 10, 100)
builder.add_parameter('z', 'categorical', choices=['A', 'B', 'C'])
```

### 多策略配置

```python
builder.add_common(
    parnames=['x'],
    stimuli_per_trial=1,
    outcome_types=['binary'],
    strategy_names=['s1', 's2', 's3']
)
builder.add_parameter('x', 'continuous', 0, 1)
builder.add_strategy('s1', 'SobolGenerator', min_asks=10)
builder.add_strategy('s2', 'SobolGenerator', min_asks=20)
builder.add_strategy('s3', 'OptimizeAcqfGenerator', model='GPClassificationModel', max_asks=100)
```

## 文件操作

### 保存

```python
builder.to_ini('my_config.ini')
```

### 加载

```python
builder = AEPsychConfigBuilder.from_ini('my_config.ini')
```

### 加载后编辑

```python
builder = AEPsychConfigBuilder.from_ini('old_config.ini')
builder.add_parameter('new_param', 'binary')
builder.to_ini('new_config.ini')
```

## 验证和调试

### 获取验证结果

```python
is_valid, errors, warnings = builder.validate()
```

### 查看缺失字段

```python
missing = builder.get_missing_fields()
for section, fields in missing.items():
    print(f"{section}: {fields}")
```

### 打印完整报告

```python
builder.print_validation_report()
```

### 获取摘要

```python
summary = builder.get_summary()
print(summary)
```

## 故障排除

### 问题: 参数在 parnames 但没有配置

```python
# 错误信息: Parameter 'x' defined in parnames but has no config section
# 解决: 添加参数配置
builder.add_parameter('x', 'continuous', 0, 1)
```

### 问题: 缺少必需的参数字段

```python
# 错误信息: Parameter [x]: 'continuous' type requires 'upper_bound'
# 解决: 添加缺失的 upper_bound
builder.add_parameter('x', 'continuous', lower_bound=0, upper_bound=1)
```

### 问题: 策略缺少终止条件

```python
# 错误信息: Strategy [s1]: needs at least one termination criterion
# 解决: 添加 min_asks, min_total_tells, 或 max_asks 中的一个
builder.add_strategy('s1', 'SobolGenerator', min_asks=10)
```

### 问题: OptimizeAcqfGenerator 缺少模型

```python
# 错误信息: Strategy [s1]: OptimizeAcqfGenerator requires model
# 解决: 指定模型
builder.add_strategy('s1', 'OptimizeAcqfGenerator', model='GPClassificationModel', max_asks=30)
```

## 导入和集成

```python
# 导入 ConfigBuilder
from extensions.config_builder.builder import AEPsychConfigBuilder

# 在 AEPsych 中使用
from aepsych.config import Config

builder = AEPsychConfigBuilder()
# ... 构建配置 ...
builder.to_ini('config.ini')

# 加载到 AEPsych
config = Config(config_fnames=['config.ini'])
```

## 完整工作流示例

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

# 创建配置
builder = AEPsychConfigBuilder()

# 第1步: 基本信息
builder.add_common(
    parnames=['stimulus_level', 'frequency'],
    stimuli_per_trial=1,
    outcome_types=['binary'],
    outcome_names=['detection'],
    strategy_names=['exploration', 'exploitation']
)

# 第2步: 参数
builder.add_parameter('stimulus_level', 'continuous', 0, 100, log_scale=True)
builder.add_parameter('frequency', 'integer', 20, 2000)

# 第3步: 策略
builder.add_strategy('exploration', 'SobolGenerator', min_asks=20)
builder.add_strategy('exploitation', 'OptimizeAcqfGenerator', 
                    model='GPClassificationModel', max_asks=100)

# 第4步: 组件配置
builder.add_component_config('GPClassificationModel', inducing_size=100)
builder.add_component_config('OptimizeAcqfGenerator', restarts=10, samps=1000)

# 第5步: 验证
is_valid, errors, warnings = builder.validate()
if not is_valid:
    print("配置错误:")
    for error in errors:
        print(f"  - {error}")
    exit(1)

# 第6步: 保存
builder.to_ini('stimulus_experiment.ini')
print("✓ 配置已保存到 stimulus_experiment.ini")

# 第7步: 在 AEPsych 中使用
from aepsych.config import Config
config = Config(config_fnames=['stimulus_experiment.ini'])
print("✓ 配置已加载到 AEPsych")
```

## 文件位置

- **实现**: `extensions/config_builder/builder.py`
- **初始化**: `extensions/config_builder/__init__.py`
- **文档**: `extensions/config_builder/README.md`
- **测试**: `test/AEPsychConfigBuilder_test/`
  - `test_config_builder.py` - 基础功能测试
  - `test_integration.py` - 与 AEPsych 的集成测试
