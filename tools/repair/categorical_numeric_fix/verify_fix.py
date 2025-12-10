#!/usr/bin/env python3
"""
验证脚本：检查 AEPsych Categorical Numeric Parameters Bug 是否已修复

Usage:
    cd d:\\ENVS\\active-psych-sampling
    pixi run python tools/repair/categorical_numeric_fix/verify_fix.py
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "temp_aepsych"))


def verify_aepsych_fix():
    """验证方案A (AEPsych fix) 是否生效"""
    print("=" * 80)
    print("验证方案A: AEPsych Categorical Transform Fix")
    print("=" * 80)

    try:
        from aepsych.transforms.ops import Categorical
        import numpy as np

        # 测试 numeric categorical
        cat = Categorical(
            indices=[0],
            categories={0: [2.8, 4.0, 8.5]}  # float list
        )

        test_input = np.array([[0.0, 0.0, 0.0]], dtype=object)
        result = cat.indices_to_str(test_input)

        print(f"\n[TEST] Numeric categorical transform")
        print(f"  Input:  index 0")
        print(f"  Output: {result[0, 0]} (type: {type(result[0, 0]).__name__})")

        if isinstance(result[0, 0], float) and abs(result[0, 0] - 2.8) < 1e-5:
            print("  [SUCCESS] AEPsych correctly maps index 0 -> 2.8 (float)")
            return True
        else:
            print(f"  [FAIL] Expected 2.8 (float), got {result[0, 0]} ({type(result[0, 0]).__name__})")
            return False

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_generator_fallback():
    """验证方案B (Generator fallback) 是否集成"""
    print("\n" + "=" * 80)
    print("验证方案B: Generator Fallback Mapping")
    print("=" * 80)

    try:
        sys.path.insert(0, str(project_root / "extensions" / "custom_generators"))
        from custom_pool_based_generator import CustomPoolBasedGenerator
        import torch
        from botorch.acquisition import qUpperConfidenceBound

        # 创建带 mapping 的 generator
        pool_points = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        categorical_mappings = {0: {0: 2.8, 1: 4.0}}

        generator = CustomPoolBasedGenerator(
            lb=torch.tensor([0.0, 0.0]),
            ub=torch.tensor([1.0, 1.0]),
            pool_points=pool_points,
            acqf=qUpperConfidenceBound,
            shuffle=False,
            _categorical_mappings=categorical_mappings,
        )

        print(f"\n[TEST] Generator fallback mechanism")
        print(f"  Has _categorical_mappings: {hasattr(generator, '_categorical_mappings')}")
        print(f"  Has _ensure_actual_values: {hasattr(generator, '_ensure_actual_values')}")

        if hasattr(generator, '_categorical_mappings') and hasattr(generator, '_ensure_actual_values'):
            # 测试 mapping
            test_input = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
            result = generator._ensure_actual_values(test_input)

            print(f"  Test mapping: 0.0 -> {result[0, 0].item()}")

            if abs(result[0, 0].item() - 2.8) < 1e-5:
                print("  [SUCCESS] Generator fallback correctly maps 0 -> 2.8")
                return True
            else:
                print(f"  [FAIL] Expected 2.8, got {result[0, 0].item()}")
                return False
        else:
            print("  [FAIL] Generator missing fallback methods")
            return False

    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " " * 20 + "修复验证：Categorical Numeric Bug" + " " * 24 + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")

    # 验证方案A
    aepsych_ok = verify_aepsych_fix()

    # 验证方案B
    generator_ok = verify_generator_fallback()

    # 总结
    print("\n" + "=" * 80)
    print("验证结果总结")
    print("=" * 80)
    print(f"  方案A (AEPsych fix):      {'[OK] WORKING' if aepsych_ok else '[FAIL] NOT WORKING'}")
    print(f"  方案B (Generator fallback): {'[OK] WORKING' if generator_ok else '[FAIL] NOT WORKING'}")
    print()

    if aepsych_ok and generator_ok:
        print("[SUCCESS] 双保险机制完全正常!")
        print()
        print("说明：")
        print("  - 方案A 正常工作，AEPsych 正确处理 numeric categorical")
        print("  - 方案B 作为备用，在方案A失效时自动生效")
        print("  - 系统具有完整的容错能力")
        return 0
    elif generator_ok:
        print("[PARTIAL] 方案B 正常，方案A 未生效")
        print()
        print("说明：")
        print("  - AEPsych 修复未应用或已回滚")
        print("  - 但方案B (Generator fallback) 仍会自动工作")
        print("  - 系统功能正常，建议应用方案A以获得最佳性能")
        print()
        print("修复方案A：")
        print("  pixi run python tools/repair/categorical_numeric_fix/apply_fix.py")
        return 1
    else:
        print("[FAIL] 双保险均未生效")
        print()
        print("说明：")
        print("  - 方案A 和方案B 都未正确应用")
        print("  - 请运行自动修复脚本")
        print()
        print("修复命令：")
        print("  pixi run python tools/repair/categorical_numeric_fix/apply_fix.py")
        return 2


if __name__ == "__main__":
    sys.exit(main())
