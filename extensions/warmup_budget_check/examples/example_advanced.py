#!/usr/bin/env python3
"""
高级使用示例 - 展示复杂场景和批量处理
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from warmup_api import run_step1, run_step2, run_step3, batch_step1, create_pipeline
from config_models import Step1Config, Step2Config, Step3Config, WarmupPipelineConfig


def example_1_batch_processing():
    """示例1：批量处理不同参数组合"""
    print("=" * 60)
    print("示例1：批量处理不同参数组合")
    print("=" * 60)

    # 创建多个配置
    configs: List[Step1Config] = []

    # 测试不同的被试数量和试验次数
    test_params = [
        (3, 20, True),
        (5, 25, True),
        (7, 30, False),
        (10, 20, False),
        (5, 50, True),
    ]

    design_csv = "D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"

    for i, (n_subjects, trials, skip_inter) in enumerate(test_params):
        config = Step1Config(
            design_csv_path=design_csv,
            n_subjects=n_subjects,
            trials_per_subject=trials,
            skip_interaction=skip_inter,
            output_dir=f"batch_output/config_{i+1}",
            merge=False,
        )
        configs.append(config)

    print(f"创建了 {len(configs)} 个配置")
    for i, config in enumerate(configs):
        print(
            f"  配置{i+1}: N={config.n_subjects}, trials={config.trials_per_subject}, skip_inter={config.skip_interaction}"
        )

    # 批量执行
    batch_result = batch_step1(configs, "batch_results")

    print(f"\n批量执行结果:")
    print(f"  总配置数: {batch_result['total_configs']}")
    print(f"  成功: {batch_result['successful']}")
    print(f"  失败: {batch_result['failed']}")
    print(f"  成功率: {batch_result['summary']['success_rate']:.1%}")

    # 显示每个配置的结果
    for i, result_info in enumerate(batch_result["results"]):
        config = result_info["config"]
        result = result_info["result"]

        status = "✅" if result["success"] else "❌"
        adequacy = result.get("adequacy", "N/A")

        print(
            f"  配置{i+1}: {status} {adequacy} (N={config['n_subjects']}, trials={config['trials_per_subject']})"
        )


def example_2_full_pipeline():
    """示例2：完整的三步流程"""
    print("\n" + "=" * 60)
    print("示例2：完整的三步流程")
    print("=" * 60)

    # 创建完整流程配置
    pipeline_config = WarmupPipelineConfig(
        step1=Step1Config(
            design_csv_path="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
            n_subjects=5,
            trials_per_subject=25,
            output_dir="full_pipeline/step1_output",
        ),
        step2=Step2Config(
            data_csv_path="full_pipeline/step1_output/result/combined_results.csv",
            subject_col="subject",
            response_col="y",
            max_pairs=5,
            min_pairs=3,
            selection_method="elbow",
            phase2_n_subjects=20,
            phase2_trials_per_subject=25,
            output_dir="full_pipeline/step2_output",
        ),
        step3=Step3Config(
            data_csv_path="full_pipeline/step1_output/result/combined_results.csv",
            design_space_csv="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
            subject_col="subject",
            response_col="y",
            max_iters=50,
            learning_rate=0.05,
            output_dir="full_pipeline/step3_output",
        ),
    )

    # 验证所有配置
    all_valid, errors = pipeline_config.validate_all()
    if not all_valid:
        print("❌ 配置验证失败:")
        for step, step_errors in errors.items():
            if step_errors:
                print(f"  {step}:")
                for error in step_errors:
                    print(f"    ❌ {error}")
        return

    print("✅ 配置验证通过")

    # 创建流程管理器
    pipeline = WarmupPipeline(pipeline_config)

    # 执行完整流程
    print("\n执行完整流程...")
    full_result = pipeline.run_all(strict_mode=False)

    print(f"流程执行结果:")
    print(f"  总体成功: {'✅' if full_result['success'] else '❌'}")
    print(f"  执行步骤: {full_result['execution_summary']['total_steps']}")
    print(f"  成功步骤: {full_result['execution_summary']['successful_steps']}")
    print(f"  总时间: {full_result['execution_summary']['duration_formatted']}")

    # 显示各步骤结果
    for step_name, step_result in full_result["steps"].items():
        status = "✅" if step_result["success"] else "❌"
        print(f"  {step_name}: {status}")


def example_3_config_serialization():
    """示例3：配置序列化和保存"""
    print("\n" + "=" * 60)
    print("示例3：配置序列化和保存")
    print("=" * 60)

    # 创建配置
    config = Step1Config(
        design_csv_path="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        n_subjects=5,
        trials_per_subject=25,
        skip_interaction=False,
        output_dir="serialization_test",
        merge=True,
        subject_col_name="participant_id",
    )

    # 验证配置
    is_valid, errors = config.validate()
    if not is_valid:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  {error}")
        return

    print("✅ 配置验证通过")

    # 转换为字典
    config_dict = config.to_dict()
    print(f"✅ 配置转换为字典:")
    for key, value in config_dict.items():
        print(f"  {key}: {value}")

    # 保存为JSON
    json_path = "example_config.json"
    config.to_json(json_path)
    print(f"✅ 配置已保存到: {json_path}")

    # 从JSON加载
    loaded_config = Step1Config.from_json(json_path)
    print(f"✅ 从JSON加载配置:")
    print(f"  设计文件: {loaded_config.design_csv_path}")
    print(f"  被试数量: {loaded_config.n_subjects}")
    print(f"  每人试验: {loaded_config.trials_per_subject}")

    # 验证加载的配置
    is_valid, errors = loaded_config.validate()
    if is_valid:
        print("✅ 加载的配置验证通过")
    else:
        print("❌ 加载的配置验证失败:")
        for error in errors:
            print(f"  {error}")


def example_4_error_handling():
    """示例4：详细的错误处理和恢复"""
    print("\n" + "=" * 60)
    print("示例4：详细的错误处理和恢复")
    print("=" * 60)

    # 测试各种错误情况
    error_cases = [
        {
            "name": "文件不存在",
            "config": Step1Config(
                design_csv_path="nonexistent.csv",
                n_subjects=5,
                trials_per_subject=25,
                output_dir="error_test_1",
            ),
        },
        {
            "name": "被试数为0",
            "config": Step1Config(
                design_csv_path="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
                n_subjects=0,
                trials_per_subject=25,
                output_dir="error_test_2",
            ),
        },
        {
            "name": "输出目录无效",
            "config": Step1Config(
                design_csv_path="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
                n_subjects=5,
                trials_per_subject=25,
                output_dir="",  # 空字符串
                merge=True,
                subject_col_name="",  # 空字符串
            ),
        },
    ]

    for i, error_case in enumerate(error_cases):
        print(f"\n测试错误情况 {i+1}: {error_case['name']}")
        print("-" * 40)

        config = error_case["config"]

        # 配置验证
        is_valid, validation_errors = config.validate()
        if not is_valid:
            print("✅ 配置验证正确识别错误:")
            for error in validation_errors:
                print(f"  ❌ {error}")

        # 尝试执行
        result = run_step1(config, strict_mode=False)

        if not result["success"]:
            print("✅ API 正确处理错误:")
            for error in result["errors"]:
                print(f"  ❌ {error}")
        else:
            print("❌ 应该失败但没有失败")

    # 测试严格模式
    print(f"\n测试严格模式:")
    print("-" * 40)

    # 使用一个会预算不足的配置
    insufficient_config = Step1Config(
        design_csv_path="D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        n_subjects=1,  # 很少的被试
        trials_per_subject=5,  # 很少的试验次数
        output_dir="strict_test",
    )

    print("非严格模式（应该成功但有警告）:")
    result_normal = run_step1(insufficient_config, strict_mode=False)
    print(f"  成功: {result_normal['success']}")
    print(f"  预算评估: {result_normal.get('adequacy', 'N/A')}")

    print("严格模式（应该失败）:")
    try:
        result_strict = run_step1(insufficient_config, strict_mode=True)
        print(f"  成功: {result_strict['success']}")
    except Exception as e:
        print(f"  ✅ 正确抛出异常: {e}")


def example_5_custom_workflow():
    """示例5：自定义工作流程"""
    print("\n" + "=" * 60)
    print("示例5：自定义工作流程")
    print("=" * 60)

    # 创建多个相似的配置进行比较
    base_config = {
        "design_csv_path": "D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        "n_subjects": 5,
        "trials_per_subject": 25,
        "output_dir": "custom_workflow",
    }

    variations = [
        {"skip_interaction": True, "merge": False, "suffix": "_no_interaction"},
        {"skip_interaction": False, "merge": False, "suffix": "_with_interaction"},
        {"skip_interaction": True, "merge": True, "suffix": "_merged"},
        {
            "skip_interaction": False,
            "merge": True,
            "suffix": "_merged_with_interaction",
        },
    ]

    results = []

    for i, variation in enumerate(vocations):
        # 创建变体配置
        config_dict = base_config.copy()
        config_dict.update(
            {
                "skip_interaction": variation["skip_interaction"],
                "merge": variation["merge"],
                "output_dir": f"{base_config['output_dir']}{variation['suffix']}",
            }
        )

        config = Step1Config.from_dict(config_dict)

        print(f"执行变体 {i+1}: {variation['suffix']}")
        print(f"  跳过交互: {config.skip_interaction}")
        print(f"  合并文件: {config.merge}")

        # 执行
        result = run_step1(config)

        if result["success"]:
            print(f"  ✅ 成功 - 预算评估: {result['adequacy']}")
            print(f"  📁 输出: {result['output_dir']}")

            # 收集结果用于比较
            results.append(
                {
                    "variation": variation["suffix"],
                    "adequacy": result["adequacy"],
                    "total_samples": result["budget"]["total_samples"],
                    "unique_configs": result["budget"]["unique_configs"],
                    "core1_samples": result["budget"]["core1_samples"],
                    "success": True,
                }
            )
        else:
            print(f"  ❌ 失败 - {result['errors']}")
            results.append(
                {
                    "variation": variation["suffix"],
                    "success": False,
                    "error": result["errors"],
                }
            )

        print()

    # 比较结果
    print("变体比较结果:")
    print("-" * 60)
    for result in results:
        if result["success"]:
            print(
                f"{result['variation']:25} | {result['adequacy']:8} | "
                f"总样本: {result['total_samples']:3} | 独特配置: {result['unique_configs']:3}"
            )

    # 保存比较结果
    comparison_file = "workflow_comparison.json"
    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n比较结果已保存到: {comparison_file}")


def main():
    """主函数：运行所有高级示例"""
    print("Warmup Budget Check API 高级使用示例")
    print("=====================================")

    try:
        example_1_batch_processing()
        example_2_full_pipeline()
        example_3_config_serialization()
        example_4_error_handling()
        example_5_custom_workflow()

        print("\n" + "=" * 60)
        print("🎉 所有高级示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
