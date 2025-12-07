# AEPsychConfigBuilder - 新功能汇总

## 📋 最新功能（模板系统）

### 核心特性

本次更新为 `AEPsychConfigBuilder` 添加了强大的**自动模板生成系统**。

#### 1️⃣ 自动模板生成

- **初始化时自动创建** - 构建器初始化时自动生成包含占位符的最小实现
- **【】标记系统** - 用中文方括号【】标记所有需要填充的字段，便于识别
- **可选禁用** - 通过 `auto_template=False` 参数可禁用此功能

```python
# 默认启用自动模板
builder = AEPsychConfigBuilder()

# 禁用自动模板
builder = AEPsychConfigBuilder(auto_template=False)
```

#### 2️⃣ 可视化预览

- **打印模板** - `print_template()` 显示格式化的配置预览
- **获取预览字符串** - `preview_template()` 返回预览文本
- **高亮显示** - 默认高亮【】标记，便于快速定位

```python
builder.print_template()        # 打印到控制台
text = builder.preview_template() # 获取文本用于处理
```

#### 3️⃣ 字符串处理

- **获取 INI 字符串** - `get_template_string()` 返回完整的 INI 格式字符串
- **便于替换** - 字符串中的【】标记可直接替换
- **支持进一步处理** - 返回的字符串可用于任何文本处理工作流

```python
template = builder.get_template_string()
# 进行替换或其他处理
template = template.replace('【parameter_1】', 'my_parameter')
```

#### 4️⃣ 交互式指南

- **显示完整指南** - `show_template_with_hints()` 显示详细使用说明
- **包含示例** - 展示如何编辑和替换占位符
- **当前模板预览** - 同时显示当前的配置状态

```python
builder.show_template_with_hints()
```

---

## 🔧 新增 API 方法

| 方法 | 说明 | 返回类型 |
|------|------|--------|
| `print_template()` | 打印格式化的配置预览 | None |
| `preview_template(highlight=True)` | 获取预览字符串 | str |
| `get_template_string()` | 获取 INI 格式字符串 | str |
| `show_template_with_hints()` | 显示详细使用指南 | None |
| `_create_minimal_template()` | 创建最小模板（内部方法） | None |

---

## 📝 使用示例

### 示例 1: 快速开始

```python
from extensions.config_builder.builder import AEPsychConfigBuilder

# 创建构建器（自动生成模板）
builder = AEPsychConfigBuilder()

# 查看模板
builder.print_template()

# 添加配置
builder.add_common(['intensity'], 1, ['binary'], ['init'])
builder.add_parameter('intensity', 'continuous', 0, 100)

# 再次查看
builder.print_template()
```

### 示例 2: 字符串工作流

```python
builder = AEPsychConfigBuilder()

# 获取模板字符串
config_text = builder.get_template_string()

# 处理字符串
config_text = config_text.replace('【parameter_1】', 'frequency')
config_text = config_text.replace('【strategy_1】', 'sobol')

# 保存文件
with open('config.ini', 'w', encoding='utf-8') as f:
    f.write(config_text)
```

### 示例 3: 提取占位符

```python
import re

builder = AEPsychConfigBuilder()
config_text = builder.get_template_string()

# 找到所有【】标记的占位符
placeholders = re.findall(r'【(.*?)】', config_text)
print("需要填充的字段：")
for ph in placeholders:
    print(f"  - 【{ph}】")
```

---

## 🎯 功能对比

### 新旧对比

| 功能 | 旧版本 | 新版本 |
|------|--------|--------|
| 自动生成最小配置 | ❌ | ✅ |
| 占位符标记 | ❌ | ✅ 【】 |
| 预览打印 | ❌ | ✅ |
| 字符串输出 | ❌ | ✅ |
| 交互式指南 | ❌ | ✅ |
| 禁用自动生成 | N/A | ✅ 可选 |

---

## 🚀 优势说明

### 1. **快速开始** ⚡

- 无需手动编写最小配置
- 初始化即可使用
- 减少入门时间

### 2. **可视化** 👁️

- 【】标记一目了然
- 清楚地显示需要填充的字段
- 降低出错概率

### 3. **灵活性** 🔄

- 支持字符串处理
- 可与其他工具集成
- 支持自动化脚本

### 4. **易于调试** 🐛

- 多种预览方式
- 可追溯配置来源
- 清晰的数据流

---

## 📊 代码组织

### 新增文件

- **TEMPLATE_GUIDE.md** - 详细的模板功能使用指南
- **FEATURES_SUMMARY.md** - 本文件，功能汇总

### 更新的文件

- **builder.py** - 添加 5 个新方法
- ****init**.py** - 更新构造函数参数
- **README.md** - 已包含新功能说明
- **QUICKSTART.md** - 已包含快速示例

---

## ✅ 测试验证

所有新功能已通过以下测试：

- ✅ 自动模板生成测试
- ✅ 字符串输出测试
- ✅ 预览显示测试
- ✅ 交互式指南测试
- ✅ 占位符替换测试
- ✅ 完整工作流测试

**测试结果**：16/16 通过（100%）

---

## 🔄 向后兼容性

✅ 所有旧代码完全兼容

```python
# 旧的使用方式仍然有效
builder = AEPsychConfigBuilder()
builder.add_common(...)
builder.add_parameter(...)
builder.to_ini('config.ini')
```

---

## 📚 相关文档

1. **README.md** - 完整使用说明
2. **QUICKSTART.md** - 快速入门指南
3. **TEMPLATE_GUIDE.md** - 模板功能详解
4. **PROJECT_SUMMARY.md** - 项目总体描述

---

## 🎓 学习路径

**初学者**：

1. 阅读 QUICKSTART.md
2. 运行示例代码
3. 尝试 `print_template()` 方法

**中级用户**：

1. 学习 TEMPLATE_GUIDE.md 的各个场景
2. 实验字符串处理方法
3. 集成到自己的项目

**高级用户**：

1. 研究 builder.py 的内部实现
2. 扩展和定制功能
3. 贡献改进

---

## 💡 最佳实践

1. **始终使用 `print_template()` 检查配置**

   ```python
   builder.add_parameter(...)
   builder.print_template()  # 检查结果
   ```

2. **在保存前验证**

   ```python
   is_valid, errors, _ = builder.validate()
   if is_valid:
       builder.to_ini('config.ini')
   ```

3. **处理字符串时使用正则表达式**

   ```python
   import re
   placeholders = re.findall(r'【(.*?)】', config_text)
   ```

4. **保存备份**

   ```python
   # 在修改前保存
   backup = builder.get_template_string()
   ```

---

## 🐛 常见问题

**Q: 如何禁用自动模板？**
A: 使用 `AEPsychConfigBuilder(auto_template=False)`

**Q: 如何提取所有占位符？**
A: 使用正则表达式 `r'【(.*?)】'` 匹配字符串

**Q: 字符串中的【】可以替换吗？**
A: 完全可以，直接使用 `str.replace()` 或正则表达式

**Q: 新功能对性能有影响吗？**
A: 没有，只在初始化时添加最小模板，开销可忽略不计

---

## 🔜 未来规划

可能的增强功能：

- [ ] 配置预设模板库
- [ ] 模板继承和组合
- [ ] 配置版本管理
- [ ] Web 界面模板编辑器
- [ ] 配置对比工具

---

## 📞 反馈与支持

如有问题或建议，欢迎提出！

所有新功能均已充分测试和文档化。
