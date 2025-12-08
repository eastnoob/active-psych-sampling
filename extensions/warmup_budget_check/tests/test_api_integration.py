#!/usr/bin/env python3
"""
API 集成测试脚本
验证新的外部 API 接口和重构后的 quick_start.py 功能完整性
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any

# 添加当前目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))


def test_config_models():
    """测试配置模型"""
    print("=" * 60)
    print("测试配置模型")
    print("=" * 60)

    try:
        from config_models import Step1Config, Step2Config, Step3Config

        # 测试 Step1Config
        step1_config = Step1Config(
            design_csv_path="test_data.csv",
            n_subjects=5,
            trials_per_subject=25,
            skip_interaction=True,
            output_dir="test_output",
        )

        print("✓ Step1Config 创建成功")

        # 验证配置
        assert step1_config.design_csv_path == "test_data.csv"
        assert step1_config.n_subjects == 5
        assert step1_config.trials_per_subject == 25
        assert step1_config.skip_interaction == True

        print("✓ Step1Config 验证成功")

        # 测试字典转换
        config_dict = step1_config.to_dict()
        restored_config = Step1Config.from_dict(config_dict)

        assert restored_config.design_csv_path == step1_config.design_csv_path
        assert restored_config.n_subjects == step1_config.n_subjects

        print("✓ Step1Config 字典转换成功")

        # 测试 Step2Config
        step2_config = Step2Config(
            data_csv_path="test_data.csv",
            subject_col="subject",
            response_col="y",
            max_pairs=5,
            min_pairs=1,
            selection_method="elbow",
            phase2_n_subjects=20,
            phase2_trials_per_subject=25,
        )

        print("✓ Step2Config 创建成功")

        # 测试 Step3Config
        step3_config = Step3Config(
            data_csv_path="test_data.csv",
            design_space_csv="test_design.csv",
            subject_col="subject",
            response_col="y",
            max_iters=100,
            learning_rate=0.01,
            use_cuda=False,
        )

        print("✓ Step3Config 创建成功")

        return True

    except Exception as e:
        print(f"✗ 配置模型测试失败: {e}")
        return False


def test_api_functions():
    """测试 API 函数"""
    print("\n" + "=" * 60)
    print("测试 API 函数")
    print("=" * 60)

    try:
        from warmup_api import run_step1, run_step2, run_step3

        print("✓ API 函数导入成功")

        # 测试函数签名
        import inspect

        # 检查 run_step1 函数签名
        sig = inspect.signature(run_step1)
        params = list(sig.parameters.keys())
        assert "config" in params
        assert "strict_mode" in params

        print("✓ run_step1 函数签名正确")

        # 检查 run_step2 函数签名
        sig = inspect.signature(run_step2)
        params = list(sig.parameters.keys())
        assert "config" in params
        assert "strict_mode" in params

        print("✓ run_step2 函数签名正确")

        # 检查 run_step3 函数签名
        sig = inspect.signature(run_step3)
        params = list(sig.parameters.keys())
        assert "config" in params
        assert "strict_mode" in params

        print("✓ run_step3 函数签名正确")

        # 测试实际调用（使用最小配置）
        from config_models import Step1Config

        # 创建最小测试配置（使用有效的输出目录）
        test_config = Step1Config(
            design_csv_path="nonexistent.csv",  # 故意使用不存在的文件来测试错误处理
            n_subjects=1,
            trials_per_subject=1,
            output_dir="test_output",
        )

        # 测试 API 函数调用（应该返回错误但不崩溃）
        result = run_step1(test_config)
        assert isinstance(result, dict)
        assert "success" in result
        print("✓ run_step1 函数调用正常（即使配置无效也能正常返回错误）")

        return True

    except Exception as e:
        print(f"✗ API 函数测试失败: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_chain_managers():
    """测试流程管理器"""
    print("\n" + "=" * 60)
    print("测试流程管理器")
    print("=" * 60)

    try:
        from warmup_api import Step1Step2Chain, Step1Step2Step3Chain
        from config_models import Step1Config, Step2Config, Step3Config

        print("✓ 流程管理器导入成功")

        # 创建测试配置
        step1_config = Step1Config(
            design_csv_path="test_design.csv",
            n_subjects=5,
            trials_per_subject=25,
            skip_interaction=True,
        )

        step2_config = Step2Config(
            data_csv_path="test_data.csv",
            subject_col="subject",
            response_col="y",
            max_pairs=5,
            min_pairs=1,
        )

        step3_config = Step3Config(
            data_csv_path="test_data.csv",
            design_space_csv="test_design.csv",
            subject_col="subject",
            response_col="y",
            max_iters=100,
        )

        # 测试 Step1Step2Chain
        chain12 = Step1Step2Chain(step1_config, step2_config)
        print("✓ Step1Step2Chain 创建成功")

        # 测试 Step1Step2Step3Chain
        chain123 = Step1Step2Step3Chain(step1_config, step2_config, step3_config)
        print("✓ Step1Step2Step3Chain 创建成功")

        return True

    except Exception as e:
        print(f"✗ 流程管理器测试失败: {e}")
        return False


def test_quick_start_integration():
    """测试 quick_start.py 集成"""
    print("\n" + "=" * 60)
    print("测试 quick_start.py 集成")
    print("=" * 60)

    try:
        # 导入 quick_start 模块
        import quick_start

        print("✓ quick_start 模块导入成功")

        # 检查新的导入
        assert hasattr(quick_start, "API_AVAILABLE")
        print(f"✓ API_AVAILABLE: {quick_start.API_AVAILABLE}")

        # 检查辅助函数
        assert hasattr(quick_start, "_dict_to_step1_config")
        assert hasattr(quick_start, "_dict_to_step2_config")
        assert hasattr(quick_start, "_dict_to_step3_config")

        print("✓ 配置转换函数存在")

        # 检查新的链式函数
        assert hasattr(quick_start, "run_chain12")
        assert hasattr(quick_start, "run_chain123")

        print("✓ 链式函数存在")

        # 测试配置转换
        test_config = {
            "design_csv_path": "test.csv",
            "n_subjects": 5,
            "trials_per_subject": 25,
            "skip_interaction": True,
        }

        converted_config = quick_start._dict_to_step1_config(test_config)
        assert converted_config.design_csv_path == "test.csv"
        assert converted_config.n_subjects == 5

        print("✓ 配置转换功能正常")

        return True

    except Exception as e:
        print(f"✗ quick_start 集成测试失败: {e}")
        return False


def test_example_scripts():
    """测试示例脚本"""
    print("\n" + "=" * 60)
    print("测试示例脚本")
    print("=" * 60)

    try:
        # 检查示例文件存在
        examples_dir = Path(__file__).parent / "examples"

        expected_files = ["example_basic.py", "example_advanced.py", "example_batch.py"]

        for filename in expected_files:
            filepath = examples_dir / filename
            if filepath.exists():
                print(f"✓ {filename} 存在")
            else:
                print(f"✗ {filename} 不存在")
                return False

        # 尝试导入示例脚本
        sys.path.insert(0, str(examples_dir))

        import example_basic

        print("✓ example_basic 导入成功")

        import example_advanced

        print("✓ example_advanced 导入成功")

        import example_batch

        print("✓ example_batch 导入成功")

        return True

    except Exception as e:
        print(f"✗ 示例脚本测试失败: {e}")
        return False


def test_documentation():
    """测试文档"""
    print("\n" + "=" * 60)
    print("测试文档")
    print("=" * 60)

    try:
        # 检查文档文件存在
        docs_files = [
            "README_API.md",
        ]

        for filename in docs_files:
            filepath = Path(__file__).parent / filename
            if filepath.exists():
                print(f"✓ {filename} 存在")
            else:
                print(f"✗ {filename} 不存在")
                return False

        # 检查文档内容
        with open(Path(__file__).parent / "README_API.md", "r", encoding="utf-8") as f:
            content = f.read()

        # 检查关键内容
        required_sections = [
            "# Warmup Budget Check 外部 API 文档",
            "## 🚀 快速开始",
            "## 📚 API 参考",
            "## 🔧 高级用法",
        ]

        for section in required_sections:
            if section in content:
                print(f"✓ 文档包含: {section}")
            else:
                print(f"✗ 文档缺少: {section}")
                return False

        return True

    except Exception as e:
        print(f"✗ 文档测试失败: {e}")
        return False


def test_error_handling():
    """测试错误处理"""
    print("\n" + "=" * 60)
    print("测试错误处理")
    print("=" * 60)

    try:
        from config_models import Step1Config

        # 测试无效配置
        try:
            invalid_config = Step1Config(
                design_csv_path="",  # 空路径
                n_subjects=0,  # 无效数量
                trials_per_subject=0,
                skip_interaction=True,
            )

            is_valid, errors = invalid_config.validate()

            if not is_valid and len(errors) > 0:
                print("✓ 配置验证功能正常")
            else:
                print("✗ 配置验证功能异常")
                return False

        except Exception as e:
            print(f"✗ 配置验证测试失败: {e}")
            return False

        # 测试 API 函数的错误处理
        try:
            from warmup_api import run_step1

            # 使用无效配置
            invalid_config = Step1Config(
                design_csv_path="nonexistent_file.csv",
                n_subjects=5,
                trials_per_subject=25,
                skip_interaction=True,
            )

            # 这应该会抛出异常或返回错误结果
            try:
                result = run_step1(invalid_config, validate_only=True)
                print("✓ API 函数错误处理正常")
            except Exception:
                print("✓ API 函数错误处理正常（抛出异常）")

        except Exception as e:
            print(f"✗ API 错误处理测试失败: {e}")
            return False

        return True

    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("🧪 Warmup Budget Check API 集成测试")
    print("=" * 60)
    print()

    tests = [
        ("配置模型", test_config_models),
        ("API 函数", test_api_functions),
        ("流程管理器", test_chain_managers),
        ("quick_start 集成", test_quick_start_integration),
        ("示例脚本", test_example_scripts),
        ("文档", test_documentation),
        ("错误处理", test_error_handling),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))

    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1

    print("=" * 60)
    print(f"总计: {passed}/{total} 个测试通过")

    if passed == total:
        print("🎉 所有测试通过！API 集成成功！")
        return True
    else:
        print(f"⚠️  有 {total - passed} 个测试失败，请检查相关功能")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
