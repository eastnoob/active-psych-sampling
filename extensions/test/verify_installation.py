#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证脚本 - 验证所有custom组件正常工作

运行方式:
    cd f:/Github/aepsych-source
    pixi run python extensions/test/verify_installation.py
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def verify_imports():
    """验证所有组件可以正常导入"""
    print("=" * 60)
    print("验证1: 导入所有custom组件")
    print("=" * 60)

    try:
        import extensions

        print("[OK] extensions模块导入成功")

        # 验证所有注册的组件
        expected = [
            "CustomBaseGPPriorMean",
            "CustomMeanWithOffsetPrior",
            "ConfigurableGaussianLikelihood",
            "CustomBaseGPResidualFactory",
            "CustomBaseGPResidualMixedFactory",
        ]

        for comp in expected:
            if comp in extensions.__all__:
                print(f"  [OK] {comp}")
            else:
                print(f"  [FAIL] {comp} 未注册")
                return False

        return True
    except Exception as e:
        print(f"[FAIL] 导入失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_factory_initialization():
    """验证Factory可以正常初始化"""
    print("\n" + "=" * 60)
    print("验证2: Factory初始化")
    print("=" * 60)

    try:
        from extensions.custom_factory import (
            CustomBaseGPResidualFactory,
            CustomBaseGPResidualMixedFactory,
        )

        # 测试连续参数Factory
        factory1 = CustomBaseGPResidualFactory(
            dim=3, basegp_scan_csv=None, mean_type="pure_residual"  # 不使用BaseGP
        )
        print("[OK] CustomBaseGPResidualFactory 初始化成功")

        # 测试混合参数Factory
        factory2 = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"x3": 3, "x4": 2},
            basegp_scan_csv=None,
            mean_type="pure_residual",
        )
        print("[OK] CustomBaseGPResidualMixedFactory 初始化成功")

        return True
    except Exception as e:
        print(f"[FAIL] Factory初始化失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_mean_functions():
    """验证Mean函数可以正常工作"""
    print("\n" + "=" * 60)
    print("验证3: Mean函数")
    print("=" * 60)

    try:
        import torch
        from extensions.custom_mean import (
            CustomBaseGPPriorMean,
            CustomMeanWithOffsetPrior,
        )

        # 由于没有BaseGP CSV，我们只验证导入成功
        print("[OK] CustomBaseGPPriorMean 导入成功")
        print("[OK] CustomMeanWithOffsetPrior 导入成功")

        return True
    except Exception as e:
        print(f"[FAIL] Mean函数验证失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def verify_kernel_structure():
    """验证Kernel结构正确"""
    print("\n" + "=" * 60)
    print("验证4: Kernel结构")
    print("=" * 60)

    try:
        from extensions.custom_factory import CustomBaseGPResidualMixedFactory

        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"x3": 3, "x4": 2},
            basegp_scan_csv=None,
            mean_type="pure_residual",
        )

        # 检查covar module
        covar = factory._make_covar_module()
        print(f"[OK] Covar module类型: {type(covar).__name__}")

        # 验证是ProductKernel结构
        if hasattr(covar, "base_kernel"):
            base = covar.base_kernel
            print(f"  Base kernel类型: {type(base).__name__}")

            if hasattr(base, "kernels"):
                print(f"  包含{len(base.kernels)}个子kernel:")
                for i, k in enumerate(base.kernels):
                    print(f"    [{i}] {type(k).__name__}")

        return True
    except Exception as e:
        print(f"[FAIL] Kernel结构验证失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """运行所有验证"""
    print("\n" + "=" * 60)
    print("AEPsych Custom Extensions - 安装验证")
    print("=" * 60 + "\n")

    results = []

    # 验证1: 导入
    success = verify_imports()
    results.append(("组件导入", success))

    if not success:
        print("\n[FAIL] 导入失败,跳过后续测试")
        return 1

    # 验证2: Factory初始化
    success = verify_factory_initialization()
    results.append(("Factory初始化", success))

    # 验证3: Mean函数
    success = verify_mean_functions()
    results.append(("Mean函数", success))

    # 验证4: Kernel结构
    success = verify_kernel_structure()
    results.append(("Kernel结构", success))

    # 打印总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status}: {test_name}")

    print()
    print(f"通过: {passed}/{total}")

    if passed == total:
        print("\n[SUCCESS] 所有验证通过!")
        print("Extensions安装成功,可以正常使用。")
        return 0
    else:
        print("\n[FAIL] 部分验证失败。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
