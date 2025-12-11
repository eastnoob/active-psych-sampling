# 默认模板保护机制

## 📋 功能说明

AEPsychConfigBuilder 现在包含一个保护机制，防止原始 `default_template.ini` 文件被意外覆盖。

## 🛡️ 保护规则

| 操作 | 是否被阻止 | 说明 |
|------|----------|------|
| 保存到新文件 | ❌ 否 | `to_ini('new_config.ini')` ✅ 正常 |
| 保存到原始模板 | ✅ 是 | `to_ini('default_template.ini')` ❌ 被阻止 |
| 强制覆盖模板 | ❌ 否 | `to_ini('default_template.ini', force=True)` ⚠️ 需谨慎 |

## 📝 使用示例

### ✅ 正确用法 - 保存到新文件

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

builder = AEPsychConfigBuilder()
builder.add_parameter('intensity', 'continuous', lower_bound=0, upper_bound=100)

# 保存到新文件（正常，推荐）
builder.to_ini('my_config.ini')
print("配置已保存到 my_config.ini")
```

### ❌ 被阻止的操作 - 覆盖默认模板

```python
builder = AEPsychConfigBuilder()
builder.add_parameter('intensity', 'continuous', lower_bound=0, upper_bound=100)

# 这会抛出 ValueError
try:
    builder.to_ini('extensions/config_builder/default_template.ini')
except ValueError as e:
    print("被阻止!")
    print(e)
    # 输出:
    # 无法覆盖默认模板文件: ...default_template.ini
    # 为了保护原始模板，请使用其他文件名保存。
    # 如果确实要覆盖，请使用: to_ini(filepath, force=True)
```

### ⚠️ 强制覆盖 - 不推荐

```python
builder = AEPsychConfigBuilder()
builder.add_parameter('intensity', 'continuous', lower_bound=0, upper_bound=100)

# 仅在确实需要时使用
builder.to_ini('extensions/config_builder/default_template.ini', force=True)
print("模板已更新（不推荐）")
```

## 🔍 工作原理

### 检测机制

```python
def _is_default_template_file(self, filepath: str) -> bool:
    """
    检查给定的文件路径是否是默认模板文件
    
    通过以下方式识别：
    1. 获取要检查的文件的绝对路径
    2. 定位默认模板文件的实际路径
    3. 比较两个路径是否相同（不区分大小写）
    """
```

### 保护流程

```
调用 to_ini(filepath)
    ↓
检查 force 参数
    ↓
force=False (默认)
    ↓
检查 filepath 是否是默认模板
    ↓
是 → 抛出 ValueError ❌
否 → 正常保存 ✅
    ↓
force=True
    ↓
跳过检查，直接保存 ⚠️
```

## 📚 API 参考

### to_ini() 方法

```python
def to_ini(self, filepath: str, force: bool = False) -> None:
    """
    保存为 INI 文件

    Args:
        filepath (str): 输出文件路径
        force (bool): 是否强制覆盖默认模板文件（默认 False）
                     仅在 filepath 指向默认模板且需要覆盖时使用

    Raises:
        ValueError: 如果尝试覆盖默认模板文件且 force=False

    示例:
        builder.to_ini('config.ini')  # 正常保存
        builder.to_ini('config.ini', force=False)  # 同上
        builder.to_ini('default_template.ini')  # 抛出 ValueError
        builder.to_ini('default_template.ini', force=True)  # 强制覆盖
    """
```

### _is_default_template_file() 方法（内部）

```python
def _is_default_template_file(self, filepath: str) -> bool:
    """
    检查给定的文件路径是否是默认模板文件

    Args:
        filepath (str): 要检查的文件路径

    Returns:
        bool: 如果是默认模板文件返回 True，否则返回 False

    说明:
        - 支持相对路径和绝对路径
        - 不区分大小写（Windows）
        - 如果无法定位默认模板，返回 False（不予限制）
    """
```

## 🧪 测试验证

所有保护机制都已测试并验证：

✅ **测试 1**: 保存到新文件 - PASS  
✅ **测试 2**: 检测默认模板文件 - PASS  
✅ **测试 3**: 阻止覆盖默认模板 - PASS  
✅ **测试 4**: 强制覆盖功能 - PASS  
✅ **测试 5**: 保存到不同路径 - PASS  
✅ **测试 6**: 检测普通文件 - PASS  

## 💡 最佳实践

### 1. 始终使用新文件名

```python
builder = AEPsychConfigBuilder()
# ... 进行修改 ...

# 推荐：保存到新文件
builder.to_ini('my_experiments/exp_1_config.ini')
```

### 2. 版本管理

```python
from datetime import datetime

builder = AEPsychConfigBuilder()
# ... 进行修改 ...

# 版本化文件名
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
builder.to_ini(f'configs/config_{timestamp}.ini')
```

### 3. 分离配置

```python
# 不要修改原始模板
builder_original = AEPsychConfigBuilder()  # 使用默认模板

# 为特定实验创建新配置
builder_exp1 = AEPsychConfigBuilder()
builder_exp1.add_parameter('param_1', 'continuous', lower_bound=0, upper_bound=1)
builder_exp1.to_ini('experiments/exp_1/config.ini')

# 为另一个实验创建新配置
builder_exp2 = AEPsychConfigBuilder()
builder_exp2.add_parameter('param_2', 'continuous', lower_bound=0, upper_bound=1)
builder_exp2.to_ini('experiments/exp_2/config.ini')
```

## ⚙️ 配置

### 环境变量（未来扩展）

目前没有环境变量可配置，但保护机制可以通过以下方式禁用：

```python
# 如果需要禁用保护（不推荐）
builder.to_ini('default_template.ini', force=True)
```

## 🔒 安全性说明

### 保护范围

✅ 保护 `extensions/config_builder/default_template.ini`  
✅ 支持相对路径和绝对路径  
✅ 不区分大小写（Windows 系统）  
✅ 支持符号链接识别（通过绝对路径）  

### 不保护的情况

❌ 其他文件（即使名为 `default_template.ini`）  
❌ 系统权限允许时直接文件操作  
❌ 使用 `force=True` 参数时  

## 📋 FAQ

### Q: 如何恢复被覆盖的默认模板？

A: 默认模板现在受保护，不应该被覆盖。如果被 `force=True` 覆盖：

```bash
# 恢复源代码中的模板
git checkout extensions/config_builder/default_template.ini
```

### Q: 为什么需要这个保护？

A: 防止用户意外覆盖原始模板，导致：

- 丢失原始配置参考
- 所有新配置都基于错误的模板
- 难以调试配置问题

### Q: 可以删除这个保护吗？

A: 可以，但不推荐。如果确实需要：

```python
builder.to_ini('default_template.ini', force=True)
```

### Q: 如何在脚本中处理 ValueError？

A: 捕获异常并使用备用文件名：

```python
try:
    builder.to_ini(filepath)
except ValueError as e:
    # 使用时间戳作为备用文件名
    import time
    backup_file = f"{filepath}.{int(time.time())}"
    builder.to_ini(backup_file)
    print(f"已保存到备用文件: {backup_file}")
```

## 📊 版本信息

- **实现版本**: 1.0
- **引入时间**: 2025年10月18日
- **兼容性**: 完全向后兼容（新参数可选）
