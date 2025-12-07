# 配置构建工作流

## 概述

`AEPsychConfigBuilder` 现在提供清晰直观的配置管理工作流：

- **新建配置**：自动加载默认模板，快速开始
- **编辑配置**：逐步添加参数、策略等配置项
- **预览配置**：实时查看当前配置状态（彩色高亮占位符）
- **验证配置**：确保配置完整有效
- **保存配置**：导出为 INI 格式文件

---

## 工作流 1: 从零开始创建新配置

### 步骤 1: 初始化构建器

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

# 创建新构建器，自动加载默认模板
builder = AEPsychConfigBuilder()

# 查看初始配置
builder.print_configuration()
```

**输出示例:**

```
======================================================================
  配置预览 (【】表示需要填充的字段)
======================================================================

[common]
parnames = ['【parameter_1】']
stimuli_per_trial = 1
outcome_types = ['binary']
strategy_names = ['【strategy_1】']

[【strategy_1】]
generator = 【SobolGenerator】
min_asks = 【10】

======================================================================
```

【】标记表示需要你填写的部分。

### 步骤 2: 添加参数

```python
# 添加一个连续参数 "intensity"
builder.add_parameter(
    'intensity',
    'continuous',
    lower_bound=0,
    upper_bound=100
)

# 再添加一个参数
builder.add_parameter(
    'frequency',
    'continuous',
    lower_bound=1,
    upper_bound=10
)

# 查看更新后的配置
builder.print_configuration()
```

### 步骤 3: 更新 common 部分

```python
# 更新 [common] 部分，指定实际的参数名和策略名
builder.add_common(
    parnames=['intensity', 'frequency'],
    stimuli_per_trial=1,
    outcome_types=['binary'],
    strategy_names=['init_strategy', 'opt_strategy']
)

# 预览：【】被替换为实际值
builder.print_configuration()
```

### 步骤 4: 添加策略

```python
# 添加初始策略
builder.add_strategy(
    'init_strategy',
    'SobolGenerator',
    min_asks=10
)

# 添加优化策略
builder.add_strategy(
    'opt_strategy',
    'OptimizeAcqfGenerator',
    model='GPClassificationModel',
    acqf='qUCB',
    min_asks=50
)

# 预览最终配置
builder.print_configuration()
```

### 步骤 5: 验证配置

```python
# 验证配置是否有效
is_valid, errors, warnings = builder.validate()

if is_valid:
    print("✅ 配置有效！")
else:
    print("❌ 配置有错误:")
    for error in errors:
        print(f"  - {error}")
```

### 步骤 6: 保存配置

```python
# 保存为 INI 文件
builder.to_ini('path/to/my_config.ini')

print("✅ 配置已保存到 my_config.ini")
```

---

## 工作流 2: 加载并编辑现有配置

### 步骤 1: 加载现有配置文件

```python
# 从现有 INI 文件加载（不加载默认模板）
builder = AEPsychConfigBuilder.from_ini('path/to/existing_config.ini')

# 查看当前配置
builder.print_configuration()
```

**关键点**：`from_ini()` **不会** 加载默认模板，直接使用文件中的配置。

### 步骤 2: 修改配置

```python
# 修改参数
builder.add_parameter('intensity', 'continuous', lower_bound=10, upper_bound=200)

# 添加新的参数
builder.add_parameter('phase', 'continuous', lower_bound=0, upper_bound=360)

# 查看修改后的配置
builder.print_configuration()
```

### 步骤 3: 验证并保存

```python
# 验证
is_valid, errors, warnings = builder.validate()

if is_valid:
    # 保存修改
    builder.to_ini('path/to/existing_config.ini')
    print("✅ 配置已更新")
else:
    print("❌ 配置有错误，请修复后再保存")
```

---

## 工作流 3: 交互式构建配置

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

def interactive_config_builder():
    """交互式配置构建器"""
    builder = AEPsychConfigBuilder()
    
    print("\n🔧 AEPsych 配置交互式构建器\n")
    
    # 第一步：查看模板
    print("第一步：查看默认配置模板")
    builder.print_configuration()
    input("按 Enter 继续...")
    
    # 第二步：添加参数
    print("\n第二步：添加参数")
    num_params = int(input("输入参数个数: "))
    params = []
    
    for i in range(num_params):
        name = input(f"  参数 {i+1} 名称: ")
        param_type = input(f"  {name} 类型 (continuous/binary/categorical): ")
        
        if param_type == 'continuous':
            lower = float(input(f"    下界: "))
            upper = float(input(f"    上界: "))
            builder.add_parameter(name, param_type, lower_bound=lower, upper_bound=upper)
        elif param_type == 'binary':
            builder.add_parameter(name, param_type)
        elif param_type == 'categorical':
            choices = input(f"    选项 (逗号分隔): ").split(',')
            builder.add_parameter(name, param_type, choices=choices)
        
        params.append(name)
    
    print("\n✅ 参数已添加：")
    builder.print_configuration()
    
    # 第三步：配置策略
    print("\n第三步：配置策略")
    strategy_name = input("策略名称: ")
    generator = input("生成器 (SobolGenerator/OptimizeAcqfGenerator): ")
    min_asks = int(input("最小查询次数: "))
    
    builder.add_common(
        parnames=params,
        stimuli_per_trial=1,
        outcome_types=['binary'],
        strategy_names=[strategy_name]
    )
    
    builder.add_strategy(strategy_name, generator, min_asks=min_asks)
    
    print("\n✅ 配置已完成：")
    builder.print_configuration()
    
    # 第四步：验证
    print("\n第四步：验证配置")
    is_valid, errors, warnings = builder.validate()
    
    if is_valid:
        print("✅ 配置有效！")
        
        # 保存
        save_path = input("\n保存路径 (留空则不保存): ")
        if save_path:
            builder.to_ini(save_path)
            print(f"✅ 配置已保存到 {save_path}")
    else:
        print("❌ 配置有错误：")
        for error in errors:
            print(f"  - {error}")

# 运行
if __name__ == '__main__':
    interactive_config_builder()
```

---

## 关键方法参考

### 创建和加载

| 方法 | 说明 |
|------|------|
| `AEPsychConfigBuilder()` | 创建新构建器，自动加载默认模板 |
| `AEPsychConfigBuilder.from_ini(filepath)` | 加载现有 INI 文件（**不加载**默认模板） |

### 配置操作

| 方法 | 说明 |
|------|------|
| `add_common(...)` | 添加/更新 [common] 部分 |
| `add_parameter(name, type, ...)` | 添加参数配置 |
| `add_strategy(name, generator, ...)` | 添加策略配置 |
| `add_component_config(name, ...)` | 添加组件配置（可选） |

### 预览和输出

| 方法 | 说明 |
|------|------|
| `print_configuration(color=True)` | 打印配置预览（彩色高亮） |
| `preview_configuration(highlight=True, color=False)` | 获取配置字符串 |
| `show_configuration_guide()` | 显示配置编辑指南 |
| `get_configuration_string()` | 获取 INI 格式字符串 |

### 验证和保存

| 方法 | 说明 |
|------|------|
| `validate()` | 验证配置完整性 |
| `to_ini(filepath)` | 保存为 INI 文件 |
| `get_missing_fields()` | 获取缺失字段列表 |

---

## 【】标记说明

### 含义

在 `print_configuration()` 的输出中，【】标记表示：

1. **占位符**：需要你填写的实际值
2. **参数名称**：如 `【parameter_1】` → 替换为实际参数名
3. **必需字段**：必须填写的值
4. **策略名称**：如 `【strategy_1】` → 替换为实际策略名

### 彩色显示

- 默认情况下，`print_configuration()` 使用**粗体黄色**高亮【】标记
- 可以禁用颜色：`print_configuration(color=False)`

---

## 使用 default_template.ini

### 文件位置

```
extensions/config_builder/default_template.ini
```

### 文件内容

```ini
[common]
parnames = ['【parameter_1】']
stimuli_per_trial = 1
outcome_types = ['binary']
strategy_names = ['【strategy_1】']

[【strategy_1】]
generator = 【SobolGenerator】
min_asks = 【10】
```

### 加载流程

1. **创建新构建器时**：自动从 `default_template.ini` 加载

   ```python
   builder = AEPsychConfigBuilder()  # ← 加载模板
   ```

2. **从现有文件加载时**：不加载默认模板

   ```python
   builder = AEPsychConfigBuilder.from_ini('existing.ini')  # ← 不加载
   ```

3. **禁用自动加载**：

   ```python
   builder = AEPsychConfigBuilder(auto_load_template=False)  # ← 空配置
   ```

---

## 常见场景

### 场景 1: 快速创建简单配置

```python
builder = AEPsychConfigBuilder()

builder.add_parameter('x', 'continuous', lower_bound=0, upper_bound=1)
builder.add_common(['x'], 1, ['binary'], ['sobol'])
builder.add_strategy('sobol', 'SobolGenerator', min_asks=20)

builder.print_configuration()
builder.to_ini('simple_config.ini')
```

### 场景 2: 从模板修改

```python
# 加载现有配置
builder = AEPsychConfigBuilder.from_ini('base_config.ini')

# 修改参数范围
builder.add_parameter('intensity', 'continuous', lower_bound=50, upper_bound=150)

# 查看修改
builder.print_configuration()

# 保存新版本
builder.to_ini('modified_config.ini')
```

### 场景 3: 验证配置有效性

```python
builder = AEPsychConfigBuilder.from_ini('config_to_validate.ini')

is_valid, errors, warnings = builder.validate()

if not is_valid:
    print("❌ 配置有以下错误：")
    for error in errors:
        print(f"  • {error}")
    
    # 获取具体缺失字段
    missing = builder.get_missing_fields()
    print("\n缺失字段：")
    for section, fields in missing.items():
        print(f"  {section}: {fields}")
```

---

## 向后兼容

旧方法仍然可用（已弃用但仍有效）：

| 旧方法 | 新方法 | 备注 |
|------|------|------|
| `preview_template()` | `preview_configuration()` | 已弃用 |
| `print_template()` | `print_configuration()` | 已弃用 |
| `show_template_with_hints()` | `show_configuration_guide()` | 已弃用 |
| `get_template_string()` | `get_configuration_string()` | 已弃用 |

```python
# 旧代码仍然可以工作
builder.print_template()  # ✅ 仍然有效（调用新方法）
```

---

## 总结

新的工作流提供了清晰直观的配置管理体验：

✅ **新建配置**：从默认模板快速开始  
✅ **加载配置**：不覆盖现有配置  
✅ **预览配置**：实时查看状态（彩色高亮）  
✅ **编辑配置**：逐步完善配置内容  
✅ **验证配置**：确保完整性  
✅ **保存配置**：导出为标准 INI 格式  
