#!/usr/bin/env python3
"""
基础使用示例 - 展示最简单的 API 调用方式
"""

import sys
import os
from pathlib import Path

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from warmup_api import quick_step1, run_step1, create_pipeline
from config_models import Step1Config


def example_1_quick_step1():
    """示例1：使用 quick_step1（最简单）"""
    print("=" * 60)
    print("示例1：使用 quick_step1（最简单）")
    print("=" * 60)

    # 只需3个必需参数
    result = quick_step1(
        design_csv="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        n_subjects=5,
        trials_per_subject=25,
    )

    if result["success"]:
        print("✅ 执行成功！")
        print(f"   预算评估: {result['adequacy']}")
        print(f"   生成文件: {len(result['files'])} 个")
        print(f"   输出目录: {result['output_dir']}")
        print(f"   执行时间: {result['metadata']['duration_formatted']}")
    else:
        print("❌ 执行失败！")
        print(f"   错误: {result['errors']}")


def example_2_config_object():
    """示例2：使用配置对象（类型安全）"""
    print("\n" + "=" * 60)
    print("示例2：使用配置对象（类型安全）")
    print("=" * 60)

    # 创建配置对象（IDE会提供自动补全）
    config = Step1Config(
        design_csv_path="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        n_subjects=5,
        trials_per_subject=25,
        skip_interaction=False,
        output_dir="example_output_config",
        merge=False,
    )

    # 验证配置
    is_valid, errors = config.validate()
    if not is_valid:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"   {error}")
        return

    print("✅ 配置验证通过")
    print(f"   设计文件: {config.design_csv_path}")
    print(f"   被试数量: {config.n_subjects}")
    print(f"   每人试验: {config.trials_per_subject}")
    print(f"   跳过交互: {config.skip_interaction}")

    # 运行
    result = run_step1(config)

    if result["success"]:
        print("✅ 执行成功！")
        print(f"   预算评估: {result['adequacy']}")
        print(f"   总采样数: {result['budget']['total_samples']}")
        print(f"   独特配置: {result['budget']['unique_configs']}")
    else:
        print("❌ 执行失败！")
        for error in result["errors"]:
            print(f"   {error}")


def example_3_pipeline():
    """示例3：使用流程管理器（链式调用）"""
    print("\n" + "=" * 60)
    print("示例3：使用流程管理器（链式调用）")
    print("=" * 60)

    # 创建流程管理器
    pipeline = create_pipeline(
        design_csv="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        n_subjects=5,
        trials_per_subject=25,
    )

    # 链式配置
    pipeline.configure_step1(
        skip_interaction=False, output_dir="example_output_pipeline", merge=False
    )

    print("✅ 流程管理器配置完成")
    print(f"   被试数量: {pipeline.config.step1.n_subjects}")
    print(f"   每人试验: {pipeline.config.step1.trials_per_subject}")
    print(f"   跳过交互: {pipeline.config.step1.skip_interaction}")

    # 执行
    result = pipeline.run_step1()

    if result["success"]:
        print("✅ 执行成功！")
        print(f"   预算评估: {result['adequacy']}")
        print(f"   生成文件: {result['files']}")

        # 获取结果
        step1_result = pipeline.get_result("step1")
        print(f"   从管理器获取结果: {step1_result['success']}")
    else:
        print("❌ 执行失败！")
        for error in result["errors"]:
            print(f"   {error}")


def example_4_dict_config():
    """示例4：使用字典配置（灵活）"""
    print("\n" + "=" * 60)
    print("示例4：使用字典配置（灵活）")
    print("=" * 60)

    # 使用字典配置
    config_dict = {
        "design_csv_path": "D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        "n_subjects": 5,
        "trials_per_subject": 25,
        "skip_interaction": True,
        "output_dir": "example_output_dict",
        "merge": False,
    }

    # 运行
    result = run_step1(config_dict)

    if result["success"]:
        print("✅ 执行成功！")
        print(f"   预算评估: {result['adequacy']}")
        print(f"   配置来源: 字典")
        print(f"   输出目录: {result['output_dir']}")
    else:
        print("❌ 执行失败！")
        for error in result["errors"]:
            print(f"   {error}")


def example_5_error_handling():
    """示例5：错误处理演示"""
    print("\n" + "=" * 60)
    print("示例5：错误处理演示")
    print("=" * 60)

    # 测试无效配置
    invalid_config = Step1Config(
        design_csv_path="nonexistent_file.csv",  # 不存在的文件
        n_subjects=0,  # 无效的被试数
        trials_per_subject=-5,  # 无效的试验数
        output_dir="invalid_output",
    )

    # 验证配置
    is_valid, errors = invalid_config.validate()
    if not is_valid:
        print("✅ 配置验证正确识别错误:")
        for error in errors:
            print(f"   ❌ {error}")

    # 尝试运行（会失败）
    result = run_step1(invalid_config)

    if not result["success"]:
        print("✅ API 正确处理错误:")
        for error in result["errors"]:
            print(f"   ❌ {error}")
    else:
        print("❌ 应该失败但没有失败")


def main():
    """主函数：运行所有示例"""
    print("Warmup Budget Check API 基础使用示例")
    print("=====================================")

    try:
        example_1_quick_step1()
        example_2_config_object()
        example_3_pipeline()
        example_4_dict_config()
        example_5_error_handling()

        print("\n" + "=" * 60)
        print("🎉 所有示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
