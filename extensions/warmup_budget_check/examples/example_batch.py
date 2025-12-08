#!/usr/bin/env python3
"""
批量处理示例 - 展示如何批量处理多个实验设计
"""

import sys
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# 添加模块路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from warmup_api import batch_step1, run_step1
from config_models import Step1Config


def example_1_parameter_sweep():
    """示例1：参数扫描 - 测试不同的被试数量和试验次数"""
    print("=" * 60)
    print("示例1：参数扫描 - 测试不同的被试数量和试验次数")
    print("=" * 60)

    # 设计空间文件
    design_csv = "D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"

    # 参数网格
    n_subjects_list = [3, 5, 7, 10, 15]
    trials_list = [15, 25, 35, 50]
    skip_interaction_list = [True, False]

    configs = []
    param_combinations = []

    # 生成所有参数组合
    for n_subjects in n_subjects_list:
        for trials in trials_list:
            for skip_inter in skip_interaction_list:
                config = Step1Config(
                    design_csv_path=design_csv,
                    n_subjects=n_subjects,
                    trials_per_subject=trials,
                    skip_interaction=skip_inter,
                    output_dir=f"parameter_sweep/N{n_subjects}_T{trials}_inter{not skip_inter}",
                    merge=False,
                )
                configs.append(config)
                param_combinations.append(
                    {
                        "n_subjects": n_subjects,
                        "trials": trials,
                        "skip_interaction": skip_inter,
                        "total_budget": n_subjects * trials,
                    }
                )

    print(f"生成了 {len(configs)} 个配置组合")
    print(f"参数范围:")
    print(f"  被试数量: {n_subjects_list}")
    print(f"  每人试验: {trials_list}")
    print(f"  交互探索: {[True, False]}")
    print(
        f"  总预算范围: {min(c['total_budget'] for c in param_combinations)} - {max(c['total_budget'] for c in param_combinations)}"
    )

    # 批量执行
    print(f"\n开始批量执行...")
    batch_result = batch_step1(configs, "parameter_sweep_results")

    # 分析结果
    print(f"\n参数扫描结果:")
    print(f"  总配置数: {batch_result['total_configs']}")
    print(f"  成功: {batch_result['successful']}")
    print(f"  失败: {batch_result['failed']}")
    print(f"  成功率: {batch_result['summary']['success_rate']:.1%}")

    # 创建结果数据框
    results_data = []
    for i, result_info in enumerate(batch_result["results"]):
        config = result_info["config"]
        result = result_info["result"]

        adequacy_score = {
            "充分": 5,
            "刚好": 4,
            "基本满足": 3,
            "不足": 2,
            "严重不足": 1,
            "过度充足（可优化）": 4,
            "勉强": 2,
        }.get(result.get("adequacy", "N/A"), 0)

        results_data.append(
            {
                "配置ID": i + 1,
                "被试数量": config["n_subjects"],
                "每人试验": config["trials_per_subject"],
                "跳过交互": config["skip_interaction"],
                "总预算": config["n_subjects"] * config["trials_per_subject"],
                "预算评估": result.get("adequacy", "N/A"),
                "评估分数": adequacy_score,
                "成功": result["success"],
                "总样本数": result.get("budget", {}).get("total_samples", 0),
                "独特配置": result.get("budget", {}).get("unique_configs", 0),
                "覆盖率": (
                    result.get("budget", {}).get("unique_configs", 0) / 1200 * 100
                    if result["success"]
                    else 0
                ),
            }
        )

    df = pd.DataFrame(results_data)

    # 保存详细结果
    results_file = "parameter_sweep_detailed_results.csv"
    df.to_csv(results_file, index=False, encoding="utf-8")
    print(f"\n详细结果已保存到: {results_file}")

    # 显示最佳配置
    successful_df = df[df["成功"] == True]
    if not successful_df.empty:
        # 按评估分数和覆盖率排序
        best_configs = successful_df.nlargest(10, ["评估分数", "覆盖率"])

        print(f"\n最佳10个配置:")
        print(
            f"{'ID':<4} {'N':<4} {'Trials':<7} {'交互':<5} {'预算':<6} {'评估':<8} {'覆盖率':<8} {'分数'}"
        )
        print("-" * 60)
        for _, row in best_configs.iterrows():
            interaction = "否" if row["跳过交互"] else "是"
            print(
                f"{row['配置ID']:<4} {row['被试数量']:<4} {row['每人试验']:<7} {interaction:<5} "
                f"{row['总预算']:<6} {row['预算评估']:<8} {row['覆盖率']:<8.1f}% {row['评估分数']}"
            )


def example_2_multiple_designs():
    """示例2：多个设计空间文件的批量处理"""
    print("\n" + "=" * 60)
    print("示例2：多个设计空间文件的批量处理")
    print("=" * 60)

    # 假设有多个设计空间文件（这里用同一个文件演示，实际使用时可以是不同的文件）
    design_files = [
        {
            "name": "6vars_full",
            "path": "D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv",
        }
        # 可以添加更多设计文件
        # {
        #     "name": "reduced_design",
        #     "path": "path/to/reduced_design.csv"
        # },
        # {
        #     "name": "expanded_design",
        #     "path": "path/to/expanded_design.csv"
        # }
    ]

    # 为每个设计文件创建多个配置
    configs = []

    for design_info in design_files:
        # 为每个设计创建不同的参数配置
        for n_subjects in [5, 10, 15]:
            for trials in [25, 40]:
                for skip_inter in [True, False]:
                    config = Step1Config(
                        design_csv_path=design_info["path"],
                        n_subjects=n_subjects,
                        trials_per_subject=trials,
                        skip_interaction=skip_inter,
                        output_dir=f"multi_design/{design_info['name']}_N{n_subjects}_T{trials}_inter{not skip_inter}",
                        merge=False,
                    )
                    configs.append(config)

    print(f"为 {len(design_files)} 个设计文件生成了 {len(configs)} 个配置")

    # 批量执行
    print(f"\n开始批量执行...")
    batch_result = batch_step1(configs, "multi_design_results")

    # 分析结果
    print(f"\n多设计批量结果:")
    print(f"  总配置数: {batch_result['total_configs']}")
    print(f"  成功: {batch_result['successful']}")
    print(f"  失败: {batch_result['failed']}")
    print(f"  成功率: {batch_result['summary']['success_rate']:.1%}")

    # 按设计文件分组分析
    design_results = {}
    for i, result_info in enumerate(batch_result["results"]):
        config = result_info["config"]
        result = result_info["result"]
        file_path = config["design_csv_path"]

        # 简化文件名
        design_name = Path(file_path).stem

        if design_name not in design_results:
            design_results[design_name] = {
                "total": 0,
                "successful": 0,
                "adequacy_counts": {},
            }

        design_results[design_name]["total"] += 1
        if result["success"]:
            design_results[design_name]["successful"] += 1
            adequacy = result.get("adequacy", "N/A")
            design_results[design_name]["adequacy_counts"][adequacy] = (
                design_results[design_name]["adequacy_counts"].get(adequacy, 0) + 1
            )

    print(f"\n按设计文件分组的结果:")
    for design_name, stats in design_results.items():
        success_rate = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {design_name}:")
        print(f"    配置数: {stats['total']}")
        print(f"    成功率: {success_rate:.1%}")
        print(f"    预算评估分布:")
        for adequacy, count in stats["adequacy_counts"].items():
            percentage = (
                count / stats["successful"] * 100 if stats["successful"] > 0 else 0
            )
            print(f"      {adequacy}: {count} ({percentage:.1f}%)")


def example_3_optimization_search():
    """示例3：优化搜索 - 寻找最佳参数组合"""
    print("\n" + "=" * 60)
    print("示例3：优化搜索 - 寻找最佳参数组合")
    print("=" * 60)

    design_csv = "D:/WORKSPACE/python/aepsych-source/data/only_independences/data/only_independences/6vars_x1binary_x2x35level_x44level_x53level_x6binary_1200combinations.csv"

    # 定义搜索空间
    search_space = []

    # 被试数量：3-20
    for n_subjects in range(3, 21, 2):  # 3, 5, 7, ..., 19
        # 每人试验：15-60
        for trials in range(15, 61, 5):  # 15, 20, 25, ..., 60
            # 总预算约束：不超过1000次
            if n_subjects * trials <= 1000:
                for skip_inter in [True, False]:
                    search_space.append(
                        {
                            "n_subjects": n_subjects,
                            "trials": trials,
                            "skip_interaction": skip_inter,
                            "total_budget": n_subjects * trials,
                        }
                    )

    print(f"搜索空间大小: {len(search_space)} 个配置")

    # 创建配置
    configs = []
    for i, params in enumerate(search_space[:100]):  # 限制到前100个以避免过多计算
        config = Step1Config(
            design_csv_path=design_csv,
            n_subjects=params["n_subjects"],
            trials_per_subject=params["trials"],
            skip_interaction=params["skip_interaction"],
            output_dir=f"optimization_search/config_{i+1:03d}",
            merge=False,
        )
        configs.append(config)

    print(f"实际测试配置数: {len(configs)}")

    # 批量执行
    print(f"\n开始优化搜索...")
    batch_result = batch_step1(configs, "optimization_search_results")

    # 分析结果并寻找最优配置
    results_data = []
    for i, result_info in enumerate(batch_result["results"]):
        config = result_info["config"]
        result = result_info["result"]

        if result["success"]:
            # 计算综合评分
            adequacy_scores = {
                "充分": 100,
                "刚好": 90,
                "基本满足": 80,
                "勉强": 60,
                "不足": 40,
                "严重不足": 20,
                "过度充足（可优化）": 85,
            }

            adequacy_score = adequacy_scores.get(result.get("adequacy", "N/A"), 0)

            # 覆盖率评分（独特配置占总配置的比例）
            coverage_score = (
                result.get("budget", {}).get("unique_configs", 0) / 1200 * 100
            )

            # 预算效率评分（避免过度预算）
            budget_efficiency = 100 - min(config["total_budget"] / 1000 * 100, 100)

            # 综合评分
            composite_score = (
                adequacy_score * 0.5 + coverage_score * 0.3 + budget_efficiency * 0.2
            )

            results_data.append(
                {
                    "配置ID": i + 1,
                    "被试数量": config["n_subjects"],
                    "每人试验": config["trials_per_subject"],
                    "跳过交互": config["skip_interaction"],
                    "总预算": config["total_budget"],
                    "预算评估": result.get("adequacy", "N/A"),
                    "覆盖率": coverage_score,
                    "综合评分": composite_score,
                }
            )

    if results_data:
        # 转换为数据框并排序
        df = pd.DataFrame(results_data)
        df = df.sort_values("综合评分", ascending=False)

        # 保存结果
        optimization_file = "optimization_search_results.csv"
        df.to_csv(optimization_file, index=False, encoding="utf-8")
        print(f"\n优化搜索结果已保存到: {optimization_file}")

        # 显示最佳配置
        print(f"\n最佳5个配置（按综合评分）:")
        print(
            f"{'排名':<4} {'N':<4} {'Trials':<7} {'交互':<5} {'预算':<6} {'评估':<8} {'覆盖率':<8} {'综合分'}"
        )
        print("-" * 70)

        for rank, (_, row) in enumerate(df.head(5).iterrows(), 1):
            interaction = "否" if row["跳过交互"] else "是"
            print(
                f"{rank:<4} {row['被试数量']:<4} {row['每人试验']:<7} {interaction:<5} "
                f"{row['总预算']:<6} {row['预算评估']:<8} {row['覆盖率']:<8.1f}% {row['综合评分']:<.1f}"
            )

        # 显示预算-效果权衡分析
        print(f"\n预算-效果权衡分析:")
        print(
            f"  最低预算配置: {df.iloc[-1]['总预算']} 次 (N={df.iloc[-1]['被试数量']}, trials={df.iloc[-1]['每人试验']})"
        )
        print(
            f"  最高预算配置: {df.iloc[0]['总预算']} 次 (N={df.iloc[0]['被试数量']}, trials={df.iloc[0]['每人试验']})"
        )
        print(
            f"  最佳性价比: {df.iloc[0]['总预算']} 次预算获得 {df.iloc[0]['综合评分']:.1f} 分"
        )
    else:
        print("❌ 没有成功的结果")


def main():
    """主函数：运行所有批量处理示例"""
    print("Warmup Budget Check API 批量处理示例")
    print("=====================================")

    try:
        example_1_parameter_sweep()
        example_2_multiple_designs()
        example_3_optimization_search()

        print("\n" + "=" * 60)
        print("🎉 所有批量处理示例运行完成！")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
