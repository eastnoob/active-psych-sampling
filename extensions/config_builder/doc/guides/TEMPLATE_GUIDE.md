# AEPsychConfigBuilder - 模板功能指南

## 新功能概览

AEPsychConfigBuilder 现在支持自动模板生成功能，让创建配置变得更加简单！

## 功能特性

### 1. 自动模板生成

在初始化时自动创建最小实现模板，包含必需字段的占位符。

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

# 自动生成模板（默认）
builder = AEPsychConfigBuilder()

# 或禁用自动生成
builder = AEPsychConfigBuilder(auto_template=False)
```

### 2. 预览和打印模板

用【】标记标注缺失和可选字段，方便快速检查。

```python
# 打印预览
builder.print_template()

# 获取预览字符串
preview = builder.preview_template()
print(preview)
```

### 3. 获取模板字符串

导出模板为字符串，用于进一步处理和编辑。

```python
# 获取完整的 INI 格式字符串
template_str = builder.get_template_string()

# 可以用于字符串替换或其他处理
template_str = template_str.replace('【parameter_1】', 'my_parameter')
```

### 4. 交互式指南

显示编辑指南和当前模板，帮助用户快速上手。

```python
# 显示详细的使用说明和当前模板
builder.show_template_with_hints()
```

## 使用场景

### 场景 1: 快速创建新配置

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

# 步骤 1: 创建构建器（自动生成模板）
builder = AEPsychConfigBuilder()

# 步骤 2: 查看模板
builder.print_template()

# 步骤 3: 添加实际配置
builder.add_common(
    parnames=['intensity', 'frequency'],
    stimuli_per_trial=1,
    outcome_types=['binary'],
    strategy_names=['init', 'opt']
)

builder.add_parameter('intensity', par_type='continuous', 0, 100)
builder.add_parameter('frequency', par_type='integer', 10, 1000)

builder.add_strategy('init', generator='SobolGenerator', min_asks=10)
builder.add_strategy('opt', generator='OptimizeAcqfGenerator',
                    model='GPClassificationModel', max_asks=50)

# 步骤 4: 验证和保存
if builder.validate()[0]:
    builder.to_ini('config.ini')
```

### 场景 2: 字符串替换工作流

```python
builder = AEPsychConfigBuilder()

# 获取模板字符串
template = builder.get_template_string()

# 进行替换
template = template.replace('【parameter_1】', 'parameter_A')
template = template.replace('【strategy_1】', 'strategy_1')

# 保存到文件
with open('config.ini', 'w') as f:
    f.write(template)
```

### 场景 3: 配置检查清单

```python
builder = AEPsychConfigBuilder()

# 显示带有提示的模板
builder.show_template_with_hints()

# 手动编辑配置...

# 定期检查
while True:
    builder.print_template()
    response = input("\nContinue editing? (y/n): ")
    if response.lower() != 'y':
        break
    
    # 用户编辑代码...
```

## API 参考

### 构造函数

```python
AEPsychConfigBuilder(auto_template: bool = True)
```

**参数**:

- `auto_template` (bool): 是否自动生成最小实现模板，默认为 True

**示例**:

```python
# 使用自动模板
builder1 = AEPsychConfigBuilder()

# 不使用自动模板
builder2 = AEPsychConfigBuilder(auto_template=False)
```

### print_template()

打印当前配置的预览，用【】标记标注占位符。

```python
builder.print_template()
```

**输出示例**:

```
======================================================================
  配置预览 (【】表示需要填充的字段)
======================================================================

[common]
parnames = ['【parameter_1】']
stimuli_per_trial = 1
outcome_types = ['binary']
strategy_names = ['strategy_1']

======================================================================
```

### preview_template(highlight: bool = True) -> str

生成配置预览字符串。

```python
preview = builder.preview_template()
print(preview)

# 不使用高亮
preview_plain = builder.preview_template(highlight=False)
```

**返回值**:

- str: 格式化的预览字符串

### get_template_string() -> str

获取配置的 INI 格式字符串表示。

```python
template_str = builder.get_template_string()

# 用于进一步处理
processed = template_str.replace('【parameter_1】', 'my_param')

# 保存到文件
with open('config.ini', 'w') as f:
    f.write(template_str)
```

**返回值**:

- str: INI 格式的配置字符串

### show_template_with_hints()

显示详细的使用指南和当前模板。

```python
builder.show_template_with_hints()
```

**输出内容**:

- 使用说明
- 【】标记的含义
- 编辑步骤
- 示例替换
- 当前模板预览

## 【】标记说明

### 标记含义

- **【parameter_X】**: 表示需要填入参数名称的占位符
- **【strategy_X】**: 表示需要填入策略名称的占位符
- **【value】**: 表示需要填入参数值的占位符
- **【lower_bound】**: 表示需要填入下界的占位符
- **【upper_bound】**: 表示需要填入上界的占位符

### 替换方式

```python
# 方式 1: 直接字符串替换
template = builder.get_template_string()
template = template.replace('【parameter_1】', 'intensity')

# 方式 2: 正则表达式替换
import re
template = re.sub(r'【parameter_(\d+)】', r'param_\1', template)

# 方式 3: 使用 add_* 方法直接编辑
builder.add_parameter('intensity', par_type='continuous', 0, 100)
```

## 完整工作流示例

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

def create_config_interactively():
    """交互式配置创建"""
    
    # 1. 初始化（自动生成模板）
    builder = AEPsychConfigBuilder()
    
    # 2. 显示指南
    builder.show_template_with_hints()
    
    # 3. 用户输入
    param_name = input("\nEnter parameter name: ")
    strategy_name = input("Enter strategy name: ")
    
    # 4. 编辑配置
    builder.add_common(
        parnames=[param_name],
        stimuli_per_trial=1,
        outcome_types=['binary'],
        strategy_names=[strategy_name]
    )
    
    builder.add_parameter(param_name, par_type='continuous', 0, 1)
    builder.add_strategy(strategy_name, generator='SobolGenerator', min_asks=10)
    
    # 5. 检查
    builder.print_template()
    
    # 6. 验证
    is_valid, errors, _ = builder.validate()
    if is_valid:
        builder.to_ini('config.ini')
        print("Config saved!")
    else:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")

# 运行
if __name__ == '__main__':
    create_config_interactively()
```

## 常见问题

### Q1: 如何禁用自动模板生成？

```python
builder = AEPsychConfigBuilder(auto_template=False)
```

### Q2: 如何获取不带【】标记的字符串？

当你使用 `add_*` 方法添加配置后，新的部分不会包含【】标记：

```python
builder = AEPsychConfigBuilder()  # 有【】标记
builder.add_common(['x'], 1, ['binary'], ['strat'])  # 这个部分不再有【】
template_str = builder.get_template_string()  # 混合模板和实际配置
```

### Q3: 如何提取所有需要填充的字段？

```python
template_str = builder.get_template_string()
lines = template_str.split('\n')
placeholders = set()

for line in lines:
    if '【' in line and '】' in line:
        start = line.find('【')
        end = line.find('】')
        if start != -1 and end != -1:
            placeholder = line[start:end+1]
            placeholders.add(placeholder)

print("Placeholders to fill:")
for ph in sorted(placeholders):
    print(f"  - {ph}")
```

### Q4: 可以在初始化时就设置初始值吗？

目前不行。auto_template 只生成最小模板。建议的做法是：

```python
builder = AEPsychConfigBuilder(auto_template=True)
# 立即添加你的配置
builder.add_common(['x'], 1, ['binary'], ['strat'])
```

## 新增功能的内部实现

### `_create_minimal_template()`

在 `__init__` 中调用，创建包含占位符的最小配置：

```python
def _create_minimal_template(self) -> None:
    """创建最小实现模板"""
    self.config_dict['common'] = {
        'parnames': "['【parameter_1】']",
        'stimuli_per_trial': '1',
        'outcome_types': "['binary']",
        'strategy_names': "['strategy_1']"
    }
```

### 关键特点

1. **占位符使用 【】**: 便于识别和替换
2. **最小实现**: 只包含必需字段
3. **易于扩展**: 可以添加更多占位符
4. **向后兼容**: 不影响现有的使用方式

## 总结

新的模板功能提供了：

✅ 自动生成最小实现  
✅ 可视化预览和编辑  
✅ 灵活的字符串处理  
✅ 交互式编辑指南  
✅ 完全向后兼容  

让 AEPsychConfigBuilder 更加易用！
