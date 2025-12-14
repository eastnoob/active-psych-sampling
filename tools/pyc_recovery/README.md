# Python字节码恢复工具集

本工具集用于从Python .pyc字节码文件中恢复源代码。当源文件被删除但.pyc缓存仍存在时,可以使用这些工具提取元数据并重建功能骨架。

## 工具列表

### 1. extract_pyc_metadata.py
**用途**: 从.pyc文件中提取详细的元数据信息

**功能**:
- 提取函数名、类名、变量名
- 提取导入语句和引用
- 提取字符串常量和docstring
- 反汇编字节码以理解程序结构
- 递归处理嵌套的代码对象(内部函数/类)

**使用方法**:
```bash
python extract_pyc_metadata.py <pyc_file1> [<pyc_file2> ...] > output.txt
```

**示例**:
```bash
python extract_pyc_metadata.py "tests/is_EUR_work/__pycache__/run_server.cpython-314.pyc" > analysis.txt
```

### 2. recreate_run_scripts.py
**用途**: 重建单个run_server_sps_test.py脚本

**功能**:
- 基于字节码分析重建run_server_sps_test.py
- 包含完整的函数结构和导入语句

**使用方法**:
```bash
python recreate_run_scripts.py
```

### 3. create_remaining_files.py
**用途**: 批量创建多个run*.py脚本文件

**功能**:
- 创建run_eur_residual.py(两个位置)
- 创建run_sobol_test.py

**使用方法**:
```bash
python create_remaining_files.py
```

### 4. create_modules.py
**用途**: 重建完整的modules包(8个模块文件)

**功能**:
- 创建modules/__init__.py
- 创建design_space.py, oracle.py, evaluation.py
- 创建server_manager.py, sampling_loop.py
- 创建data_saver.py, param_validator.py

**使用方法**:
```bash
python create_modules.py
```

## 恢复原理

### Python .pyc文件结构
Python将源文件编译为字节码并缓存在`__pycache__/`目录。.pyc文件包含:
- 魔术数字(Python版本)
- 时间戳和文件大小
- **编译后的代码对象(Code Object)**

### Code Object包含的信息
通过`marshal.load()`可以提取:
- `co_name`: 函数/类名
- `co_varnames`: 局部变量名列表
- `co_names`: 全局引用(imports, 函数调用等)
- `co_consts`: 常量(字符串, 数字, docstring, 嵌套代码对象)
- `co_firstlineno`: 起始行号
- 字节码指令序列

### 恢复步骤

1. **读取.pyc文件**:
   ```python
   with open(pyc_file, 'rb') as f:
       f.read(16)  # 跳过头部
       code = marshal.load(f)  # 加载代码对象
   ```

2. **提取元数据**:
   ```python
   imports = code.co_names  # 所有导入和引用
   variables = code.co_varnames  # 变量名
   constants = code.co_consts  # 常量和嵌套函数
   ```

3. **递归处理嵌套代码**:
   ```python
   for const in code.co_consts:
       if isinstance(const, CodeType):
           # 处理嵌套函数/类
           extract_info(const)
   ```

4. **重建源代码**:
   - 从`co_names`重建import语句
   - 从嵌套代码对象重建函数定义
   - 从`co_consts`中的docstring恢复文档
   - 根据字节码disassembly推断控制流

### 为什么uncompyle6失败?
- uncompyle6版本3.9.3不支持Python 3.14字节码
- Python 3.14引入了新的字节码指令
- 错误: `Unsupported Python version, 3.14, for decompilation`

### 替代方案
使用**元数据重建**而非完整反编译:
- ✅ 完整恢复: imports, 函数签名, 类定义, docstrings
- ⚠️ 需补充: 具体实现逻辑, 注释, 复杂控制流
- 优势: 保证API接口完全一致

## 恢复质量

| 项目 | 恢复程度 |
|------|----------|
| Import语句 | ✅ 100% |
| 函数名和签名 | ✅ 100% |
| 类定义 | ✅ 100% |
| Docstrings | ✅ 100% |
| 常量值 | ✅ 100% |
| 变量名 | ✅ 100% |
| 具体实现逻辑 | ⚠️ 需推断 |
| 代码注释 | ❌ 不可恢复 |

## 使用场景

1. **误删除源文件**: 当.py文件被删除但.pyc仍存在
2. **git未跟踪文件**: 文件从未提交到版本控制
3. **清理操作后**: 代码清理时意外删除了工作文件
4. **Python版本不匹配**: 反编译工具不支持当前Python版本

## 限制

- **必须有.pyc文件**: 没有.pyc文件则无法恢复
- **Python 3.7+**: 工具设计用于Python 3.7+的.pyc格式
- **实现细节丢失**: 只能恢复结构,具体逻辑需推断
- **不含注释**: .pyc不保存注释

## 本次恢复案例

### 恢复的文件(2025-12-12)

**脚本文件(4个)**:
1. tests/is_EUR_work/run_server_sps_test.py (8.5KB)
2. tests/is_EUR_work/00_plans/251206/scripts/run_eur_residual.py (3.5KB)
3. tests/is_EUR_work/00_plans/251212/run_eur_residual.py (3.5KB)
4. tests/is_EUR_work/archive/scripts/run_sobol_test.py (2.1KB)

**模块文件(8个)**:
1. modules/__init__.py (846 bytes)
2. modules/design_space.py (1.5KB)
3. modules/oracle.py (1.5KB)
4. modules/evaluation.py (2.3KB)
5. modules/server_manager.py (1.5KB)
6. modules/sampling_loop.py (1.6KB)
7. modules/data_saver.py (1.5KB)
8. modules/param_validator.py (3.4KB)

### 关键发现
- 源文件**从未提交到git** (git log返回空)
- 编译时间: 2025-12-08 19:18:57
- 原始文件大小: run_server_sps_test.py为14,477字节

## 技术细节

### .pyc文件格式 (Python 3.7+)
```
[0-4]   魔术数字 (Python版本标识)
[4-8]   Bit field (Python 3.7+新增)
[8-12]  时间戳
[12-16] 源文件大小
[16+]   marshal序列化的代码对象
```

### Code Object属性
```python
code.co_name          # 函数/模块名
code.co_argcount      # 参数数量
code.co_varnames      # 局部变量名
code.co_names         # 全局名称(imports等)
code.co_consts        # 常量和嵌套代码
code.co_filename      # 原始文件路径
code.co_firstlineno   # 起始行号
code.co_code          # 字节码
```

## 参考资料

- [PEP 552](https://www.python.org/dev/peps/pep-0552/) - 确定性.pyc文件
- [Python marshal模块文档](https://docs.python.org/3/library/marshal.html)
- [Python dis模块文档](https://docs.python.org/3/library/dis.html)
- [uncompyle6项目](https://github.com/rocky/python-uncompyle6/)

## 作者

工具集由Claude Code (Anthropic)于2025-12-12创建,用于恢复意外删除的Python源文件。
