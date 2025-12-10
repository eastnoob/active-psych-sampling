#!/usr/bin/env python3
"""
自动应用 ParameterTransformedModel train_inputs 修复

Usage:
    cd d:\\ENVS\\active-psych-sampling
    pixi run python tools/repair/apply_fix.py
"""

import sys
from pathlib import Path

# 修复代码
FIX_CODE = """
    # ========== Fix for _train_inputs shadowing bug ==========
    @property
    def train_inputs(self) -> tuple[torch.Tensor, ...] | None:
        \"\"\"Delegate train_inputs to the underlying model.\"\"\"
        return self._base_obj.train_inputs

    @train_inputs.setter
    def train_inputs(self, value: tuple[torch.Tensor, ...] | None) -> None:
        \"\"\"Delegate train_inputs setting to the underlying model.\"\"\"
        self._base_obj.train_inputs = value
    # ========== End of fix ==========
"""

def find_aepsych_path():
    """查找 aepsych 安装路径"""
    try:
        import aepsych
        aepsych_dir = Path(aepsych.__file__).parent
        return aepsych_dir / "transforms" / "parameters.py"
    except ImportError:
        print("[错误] 无法导入 aepsych，请确保已通过 pixi 安装")
        return None

def check_if_already_fixed(file_path):
    """检查是否已经修复"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return "Fix for _train_inputs shadowing bug" in content

def apply_fix(file_path):
    """应用修复"""
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 查找插入位置：return self._base_obj.posterior(X=X, **kwargs) 之后
    insert_idx = None
    for i, line in enumerate(lines):
        if "return self._base_obj.posterior(X=X, **kwargs)" in line:
            # 找到下一个空行
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == "":
                    insert_idx = j + 1
                    break
            break

    if insert_idx is None:
        print("[错误] 无法找到插入位置")
        return False

    # 插入修复代码
    lines.insert(insert_idx, FIX_CODE)

    # 写回文件
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return True

def main():
    print("=" * 70)
    print("ParameterTransformedModel train_inputs Bug 自动修复工具")
    print("=" * 70)

    # 查找文件
    print("\n[步骤 1] 查找 aepsych 安装路径...")
    file_path = find_aepsych_path()

    if file_path is None:
        sys.exit(1)

    print(f"  找到: {file_path}")

    if not file_path.exists():
        print(f"[错误] 文件不存在: {file_path}")
        sys.exit(1)

    # 检查是否已修复
    print("\n[步骤 2] 检查是否已修复...")
    if check_if_already_fixed(file_path):
        print("  [ :) ] 已经修复过了，无需重复操作")
        sys.exit(0)

    print("  [ ! ] 未修复，准备应用修复...")

    # 备份原文件
    print("\n[步骤 3] 备份原文件...")
    backup_path = file_path.with_suffix(".py.backup")
    import shutil
    shutil.copy2(file_path, backup_path)
    print(f"  备份至: {backup_path}")

    # 应用修复
    print("\n[步骤 4] 应用修复...")
    if apply_fix(file_path):
        print("  [ :) ] 修复成功！")
    else:
        print("  [ :( ] 修复失败")
        # 恢复备份
        shutil.copy2(backup_path, file_path)
        print(f"  已从备份恢复: {file_path}")
        sys.exit(1)

    # 验证修复
    print("\n[步骤 5] 验证修复...")
    print("  请运行以下命令验证：")
    print("    pixi run python tools/repair/verify_issue_reproduction.py")
    print()
    print("  如果显示 '[ :) ] Fix effective' 则修复成功！")

if __name__ == "__main__":
    main()
