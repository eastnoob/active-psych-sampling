# Warmup Budget Check 外部 API 文档

## 📖 概述

本 API 为 `warmup_budget_check` 扩展提供了易于使用的编程接口，让你可以在任何地方调用 Step1 功能，就像使用 `quick_start.py` 一样灵活调整参数。

## 🚀 快速开始

### 最简单的使用方式

```python
from extensions.warmup_budget_check.warmup_api import quick_step1

# 只需3个参数
result = quick_step1(
    design_csv="path/to/design_space.csv",
    n_subjects=5,
    trials_per_subject=25
)

if result["success"]:
    print(f"生成文件: {result['files']}")
    print(f"预算评估: {result['adequacy']}")
```

### 使用配置对象（推荐）

```python
from extensions.warmup_budget_check.warmup_api import run_step1
from extensions.warmup_budget_check.config_models import Step1Config

# 创建配置对象（IDE自动补全支持）
config = Step1Config(
    design_csv_path="path/to/design_space.csv",
    n_subjects=5,
    trials_per_subject=25,
    skip_interaction=False,
    output_dir="my_output"
)

# 验证配置
is_valid, errors = config.validate()
if not is_valid:
    for error in errors:
        print(f"❌ {error}")
    exit()

# 运行
result = run_step1(config)
```

### 使用流程管理器（链式调用）

```python
from extensions.warmup_budget_check.warmup_api import create_pipeline

# 创建流程管理器
pipeline = create_pipeline(
    design_csv="path/to/design_space.csv",
    n_subjects=5,
    trials_per_subject=25
)

# 链式配置和执行
result = (pipeline
    .configure_step1(skip_interaction=False, output_dir="pipeline_output")
    .run_step1())
```

## 📚 API 参考

### 配置类

#### `Step1Config`

Step1 配置类，用于生成预热采样方案。

**参数：**

- `design_csv_path` (str): 设计空间CSV文件路径（必需）
- `n_subjects` (int): 被试数量（必需）
- `trials_per_subject` (int): 每个被试的测试次数（必需）
- `skip_interaction` (bool): 是否跳过交互效应探索，默认 `True`
- `output_dir` (str): 输出目录，默认自动生成时间戳
- `merge` (bool): 是否合并为单个CSV文件，默认 `False`
- `subject_col_name` (str): 被试编号列名，默认 `"subject_id"`
- `auto_confirm` (bool): 是否自动确认，默认 `True`

**方法：**

- `validate() -> tuple[bool, List[str]]`: 验证配置有效性
- `to_dict() -> Dict[str, Any]`: 转换为字典
- `from_dict(config_dict) -> Step1Config`: 从字典创建
- `to_json(json_path) -> None`: 保存到JSON文件
- `from_json(json_path) -> Step1Config`: 从JSON文件加载

#### `Step2Config`

Step2 配置类，用于分析 Phase 1 数据。

#### `Step3Config`

Step3 配置类，用于训练 Base GP。

### 函数式 API

#### `run_step1(config, strict_mode=False) -> Dict[str, Any]`

运行 Step1：生成预热采样方案。

**参数：**

- `config` (Step1Config | Dict): 配置对象或字典
- `strict_mode` (bool): 严格模式，预算不足时抛出异常

**返回值：**

```python
{
    "success": bool,           # 是否成功
    "adequacy": str,          # 预算评估结果
    "budget": dict,           # 预算详情
    "files": list,            # 生成的文件列表
    "output_dir": str,        # 输出目录
    "warnings": list,         # 警告信息
    "errors": list,           # 错误信息
    "execution_time": float,  # 执行时间（秒）
    "timestamp": str,         # 时间戳
    "metadata": {             # 元数据
        "config": dict,
        "duration_formatted": str
    }
}
```

#### `quick_step1(design_csv, n_subjects, trials_per_subject, **kwargs) -> Dict[str, Any]`

快速运行 Step1，最少参数。

#### `batch_step1(configs, output_dir) -> Dict[str, Any]`

批量运行 Step1。

### 类式 API

#### `WarmupPipeline`

流程管理器，提供链式调用。

**方法：**

- `configure_step1(**kwargs) -> WarmupPipeline`: 配置 Step1 参数
- `configure_step2(**kwargs) -> WarmupPipeline`: 配置 Step2 参数
- `configure_step3(**kwargs) -> WarmupPipeline`: 配置 Step3 参数
- `run_step1(strict_mode=False) -> Dict`: 执行 Step1
- `run_step2(strict_mode=False) -> Dict`: 执行 Step2
- `run_step3(strict_mode=False) -> Dict`: 执行 Step3
- `run_all(strict_mode=False) -> Dict`: 执行完整流程
- `get_result(step_name) -> Dict`: 获取指定步骤结果
- `get_all_results() -> Dict`: 获取所有结果
- `save_results(output_path) -> None`: 保存结果到JSON

## 🔧 高级用法

### 错误处理

```python
result = run_step1(config, strict_mode=False)

if result["success"]:
    print("✅ 执行成功")
    print(f"预算评估: {result['adequacy']}")
else:
    print("❌ 执行失败")
    for error in result["errors"]:
        print(f"错误: {error}")
    for warning in result["warnings"]:
        print(f"警告: {warning}")
```

### 配置验证

```python
config = Step1Config(...)
is_valid, errors = config.validate()

if not is_valid:
    print("配置错误:")
    for error in errors:
        print(f"  ❌ {error}")
else:
    print("配置验证通过")
```

### 批量处理

```python
from extensions.warmup_budget_check.warmup_api import batch_step1

configs = [config1, config2, config3, ...]
batch_result = batch_step1(configs, "batch_output")

print(f"成功: {batch_result['successful']}/{batch_result['total_configs']}")
```

### 配置序列化

```python
# 保存配置
config.to_json("my_config.json")

# 加载配置
config = Step1Config.from_json("my_config.json")
```

## 📊 返回值说明

### 预算评估结果

- `"充分"`: 预算充足，覆盖性好
- `"刚好"`: 预算刚好满足需求
- `"基本满足"`: 预算基本满足，有少量不足
- `"勉强"`: 预算勉强可用
- `"不足"`: 预算不足
- `"严重不足"`: 预算严重不足
- `"过度充足（可优化）"`: 预算过多，可优化

### 预算详情

```python
"budget": {
    "core1_configs": 8,         # Core-1 配置数
    "core1_samples": 40,        # Core-1 采样次数
    "core2a_configs": 50,       # Core-2a 配置数
    "core2b_configs": 0,        # Core-2b 配置数
    "boundary_configs": 30,     # 边界配置数
    "lhs_configs": 20,          # LHS 配置数
    "total_samples": 90,        # 总采样次数
    "unique_configs": 108       # 独特配置总数
}
```

## 🎯 最佳实践

### 1. 配置管理

```python
# ✅ 推荐：使用配置对象
config = Step1Config(
    design_csv_path="path/to/file.csv",
    n_subjects=5,
    trials_per_subject=25,
    skip_interaction=False
)

# ❌ 不推荐：直接使用字典（容易出错）
config = {
    "design_csv_path": "path/to/file.csv",
    "n_subjects": 5,
    "trials_per_subject": 25,
    "skip_interaction": False
}
```

### 2. 错误处理

```python
# ✅ 推荐：检查配置有效性
is_valid, errors = config.validate()
if not is_valid:
    for error in errors:
        print(f"配置错误: {error}")
    return

# ✅ 推荐：检查执行结果
result = run_step1(config)
if not result["success"]:
    for error in result["errors"]:
        print(f"执行错误: {error}")
    return
```

### 3. 参数选择

```python
# 根据实验规模选择参数
if n_subjects <= 5:
    trials_per_subject = 25  # 小规模实验
elif n_subjects <= 10:
    trials_per_subject = 20  # 中等规模
else:
    trials_per_subject = 15  # 大规模实验
```

## 🔄 与 quick_start.py 的兼容性

新的 API 完全保持与 `quick_start.py` 的兼容性：

1. **功能不变**：`quick_start.py` 的所有功能保持不变
2. **配置不变**：原有的配置变量名和格式不变
3. **输出不变**：输出格式和交互流程不变
4. **内部重构**：`quick_start.py` 内部使用新 API，但对外接口不变

## 📝 示例代码

查看 `examples/` 目录中的示例：

- `example_basic.py`: 基础使用示例
- `example_advanced.py`: 高级功能示例
- `example_batch.py`: 批量处理示例

## 🐛 故障排除

### 常见问题

1. **文件路径问题**

   ```python
   # ✅ 使用原始字符串或双反斜杠
   config = Step1Config(
       design_csv_path=r"D:\path\to\file.csv"
   )
   ```

2. **配置验证失败**

   ```python
   # 检查配置
   is_valid, errors = config.validate()
   print("验证错误:", errors)
   ```

3. **导入错误**

   ```python
   # 确保路径正确
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent / "extensions" / "warmup_budget_check"))
   ```

## 📞 支持

如有问题，请查看：

1. 示例代码 (`examples/`)
2. 配置验证错误信息
3. 执行结果的错误和警告信息
