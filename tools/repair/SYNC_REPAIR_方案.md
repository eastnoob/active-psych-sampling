# AEPsych 核心文件修改同步方案

## 问题描述

你对 aepsych 核心文件做了修改（添加 ordinal 参数支持），但 `tools/repair` 目录中的补丁文件可能与实际修改不完全一致。需要确保两者完全同步。

## 当前状态

经过自动检查，发现：

### 实际修改的文件（4个）

1. **transforms/ops/__init__.py** - 添加 Ordinal 导入（2处修改）
2. **transforms/parameters.py** - 添加 custom_ordinal 处理逻辑（8处修改）
3. **config.py** - 添加 custom_ordinal 参数类型验证（4处修改）
4. **transforms/ops/ordinal.py** - 新增文件（完整实现，14+处代码）

### 现有补丁文件（4个）

- `aepsych_transforms_ops_init.patch`
- `aepsych_transforms_parameters.patch`
- `aepsych_config.patch`
- `aepsych_ordinal_transforms.patch`

## 完整同步方案

### 方案 A：手动验证和更新（推荐）

**优点**: 精确可控，不依赖外部工具
**缺点**: 需要手动操作

#### 步骤 1: 创建原始 aepsych 参考副本

```bash
# 在临时目录创建干净的 pixi 环境
cd /tmp
mkdir aepsych-clean
cd aepsych-clean

# 创建最小的 pixi.toml
cat > pixi.toml << 'EOF'
[project]
name = "aepsych-clean"
channels = ["conda-forge"]
platforms = ["win-64", "linux-64", "osx-64", "osx-arm64"]

[dependencies]
aepsych = "*"
EOF

# 安装
pixi install

# 复制原始文件到参考目录
mkdir -p ~/aepsych-original
cp -r .pixi/envs/default/Lib/site-packages/aepsych ~/aepsych-original/
```

#### 步骤 2: 生成准确的 diff

```bash
cd d:/ENVS/active-psych-sampling

# 为每个修改的文件生成 unified diff

# 1. transforms/ops/__init__.py
diff -u ~/aepsych-original/aepsych/transforms/ops/__init__.py \
        .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/__init__.py \
        > tools/repair/ordinal_parameter_extension/aepsych_transforms_ops_init.patch.new

# 2. transforms/parameters.py
diff -u ~/aepsych-original/aepsych/transforms/parameters.py \
        .pixi/envs/default/Lib/site-packages/aepsych/transforms/parameters.py \
        > tools/repair/ordinal_parameter_extension/aepsych_transforms_parameters.patch.new

# 3. config.py
diff -u ~/aepsych-original/aepsych/config.py \
        .pixi/envs/default/Lib/site-packages/aepsych/config.py \
        > tools/repair/ordinal_parameter_extension/aepsych_config.patch.new

# 4. transforms/ops/ordinal.py (新文件)
diff -u /dev/null \
        .pixi/envs/default/Lib/site-packages/aepsych/transforms/ops/ordinal.py \
        > tools/repair/ordinal_parameter_extension/aepsych_ordinal_transforms.patch.new
```

#### 步骤 3: 对比新旧 patch 文件

```bash
# 对比每个 patch 文件
for file in tools/repair/ordinal_parameter_extension/*.patch; do
    echo "\n=== Comparing $file ==="
    if [ -f "${file}.new" ]; then
        diff "$file" "${file}.new" || echo "Differences found!"
    fi
done
```

#### 步骤 4: 验证新 patch 可以正确应用

```bash
# 在干净的副本上测试应用
cd /tmp/aepsych-clean

# 应用每个新的 patch
for patch in ~/active-psych-sampling/tools/repair/ordinal_parameter_extension/*.patch.new; do
    echo "Testing patch: $patch"
    patch -p0 --dry-run < "$patch"
    if [ $? -eq 0 ]; then
        echo "✓ Patch applies successfully"
    else
        echo "✗ Patch failed to apply"
    fi
done
```

#### 步骤 5: 更新 repair 目录

```bash
cd d:/ENVS/active-psych-sampling/tools/repair/ordinal_parameter_extension

# 备份旧的 patch 文件
mkdir -p backup_$(date +%Y%m%d)
mv *.patch backup_$(date +%Y%m%d)/

# 使用新的 patch 文件
for file in *.patch.new; do
    mv "$file" "${file%.new}"
done
```

### 方案 B：使用自动化脚本（快速但风险较高）

创建一个自动化脚本来完成整个过程：

```python
#!/usr/bin/env python3
# tools/repair/generate_accurate_patches.py

import subprocess
import tempfile
import shutil
from pathlib import Path

def create_clean_aepsych():
    """创建干净的 aepsych 环境"""
    temp_dir = Path(tempfile.mkdtemp(prefix="aepsych-clean-"))

    # 创建 pixi.toml
    pixi_toml = temp_dir / "pixi.toml"
    pixi_toml.write_text("""
[project]
name = "aepsych-clean"
channels = ["conda-forge"]
platforms = ["win-64"]

[dependencies]
aepsych = "*"
    """)

    # 安装
    subprocess.run(["pixi", "install"], cwd=temp_dir, check=True)

    return temp_dir / ".pixi" / "envs" / "default" / "Lib" / "site-packages" / "aepsych"

def generate_diffs(original_dir, modified_dir, output_dir):
    """生成所有 diff 文件"""
    files_to_diff = [
        ("transforms/ops/__init__.py", "aepsych_transforms_ops_init.patch"),
        ("transforms/parameters.py", "aepsych_transforms_parameters.patch"),
        ("config.py", "aepsych_config.patch"),
        ("transforms/ops/ordinal.py", "aepsych_ordinal_transforms.patch"),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, patch_name in files_to_diff:
        original_file = original_dir / rel_path
        modified_file = modified_dir / rel_path
        patch_file = output_dir / patch_name

        # 使用 diff -u 生成 unified diff
        if not original_file.exists():
            original_file = Path("/dev/null")  # 新文件

        result = subprocess.run(
            ["diff", "-u", str(original_file), str(modified_file)],
            capture_output=True,
            text=True
        )

        patch_file.write_text(result.stdout)
        print(f"✓ Generated: {patch_file}")

if __name__ == "__main__":
    print("Creating clean AEPsych environment...")
    original_aepsych = create_clean_aepsych()

    print(f"Original AEPsych: {original_aepsych}")

    modified_aepsych = Path(".pixi/envs/default/Lib/site-packages/aepsych")
    output_dir = Path("tools/repair/ordinal_parameter_extension/new_patches")

    print("Generating diffs...")
    generate_diffs(original_aepsych, modified_aepsych, output_dir)

    print("\nDone! New patches are in:", output_dir)
```

### 方案 C：直接从实际文件提取关键修改（最简单）

如果你不需要完美的 unified diff 格式，可以：

1. **记录所有修改点**: 使用已经创建的 `sync_repair_with_actual.py` 脚本
2. **手动创建 patch 文件**: 基于实际代码编写 patch
3. **重点验证**: 确保 patch 可以在干净的 aepsych 上应用

```bash
# 简化版：只验证现有 patch 是否正确
cd tools/repair/ordinal_parameter_extension
python verify_fix.py --verbose
```

## 推荐执行流程

### Windows 环境（你的情况）

由于你在 Windows 上，`diff` 命令可能不可用。推荐：

#### 选项 1: 使用 Git Bash

```bash
# 在 Git Bash 中执行方案 A 的命令
```

#### 选项 2: 使用 Python difflib

创建一个 Python 脚本来生成 patch：

```python
# tools/repair/generate_patches_windows.py
import difflib
from pathlib import Path

def generate_unified_diff(original_lines, modified_lines, original_file, modified_file):
    """生成 unified diff 格式"""
    return list(difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile=f"a/{original_file}",
        tofile=f"b/{modified_file}",
        lineterm=''
    ))

# 使用示例
original = open("original_file.py").readlines()
modified = open("modified_file.py").readlines()
diff = generate_unified_diff(original, modified, "file.py", "file.py")
print('\n'.join(diff))
```

## 验证清单

完成后，验证以下项目：

- [ ] 所有 4 个 patch 文件都已更新
- [ ] patch 文件的上下文与当前 aepsych 版本匹配
- [ ] 在干净的 aepsych 上运行 `patch --dry-run` 成功
- [ ] `verify_fix.py` 脚本通过所有检查
- [ ] 文档（README_ORDINAL_FIX.md）中的示例与实际 patch 一致

## 注意事项

1. **版本兼容性**: 确保原始 aepsych 版本与你修改的版本一致
2. **行结尾**: Windows/Unix 行结尾差异可能导致 patch 失败
3. **空格和制表符**: 确保使用正确的缩进
4. **文件路径**: patch 文件中的路径应该是相对于 aepsych 包根目录

## 快速检查命令

```bash
# 检查当前 aepsych 版本
pixi run python -c "import aepsych; print(aepsych.__version__)"

# 检查实际文件是否包含 ordinal 修改
grep -r "custom_ordinal" .pixi/envs/default/Lib/site-packages/aepsych/

# 验证现有 patch 文件
cd tools/repair/ordinal_parameter_extension
python verify_fix.py
```

## 后续步骤

1. 选择一个方案（推荐方案 A + Python difflib）
2. 创建原始 aepsych 参考副本
3. 生成准确的 diff
4. 验证并更新 patch 文件
5. 运行完整测试套件确认修改正确

## 需要帮助？

如果遇到问题，请提供：
- 当前 aepsych 版本
- 错误消息
- 你选择的方案编号
