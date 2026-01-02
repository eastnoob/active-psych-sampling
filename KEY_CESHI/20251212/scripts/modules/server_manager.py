#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AEPsych Server初始化和管理模块
负责Server创建、配置和验证
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Tuple
from aepsych.server import AEPsychServer


def initialize_server(
    config_path: Path,
    result_dir: Path,
    design_space,
    basegp_keypoints_path: Path,
    budget: int,
) -> Tuple[AEPsychServer, Dict[str, Any]]:
    """
    初始化AEPsych Server并加载配置

    Args:
        config_path: 配置文件路径
        result_dir: 结果目录
        design_space: 设计空间数组 (n_points, 6)
        basegp_keypoints_path: BaseGP黄金点JSON文件路径
        budget: 总采样预算

    Returns:
        (server, config_info): Server实例和配置信息字典
    """
    print("\n" + "=" * 80)
    print("步骤3: 初始化 AEPsych Server")
    print("=" * 80)

    db_path = result_dir / "experiment.db"

    try:
        # 读取配置文件
        with open(config_path, "r", encoding="utf-8") as f:
            config_str = f.read()

        # 动态替换 pool_points
        pool_points_str = str(design_space.tolist())
        pool_points_line = f"pool_points = {pool_points_str}"

        if "[PoolBasedGenerator]" in config_str:
            if "pool_points =" in config_str:
                config_str = re.sub(
                    r"pool_points\s*=\s*\[\[.*?\]\]", pool_points_line, config_str
                )
            else:
                config_str = config_str.replace(
                    "[PoolBasedGenerator]", f"[PoolBasedGenerator]\n{pool_points_line}"
                )
        else:
            pool_gen_section = f"\n[PoolBasedGenerator]\n{pool_points_line}\n\n"
            if "[EURAnovaMultiAcqf]" in config_str:
                config_str = config_str.replace(
                    "[EURAnovaMultiAcqf]", pool_gen_section + "[EURAnovaMultiAcqf]"
                )
            else:
                config_str += pool_gen_section

        print(f"✓ 已设置 pool_points: {design_space.shape[0]} 个候选点")

        # 加载并转换3个BaseGP黄金点 (BaseGP → INI空间)
        with open(basegp_keypoints_path, "r") as f:
            keypoints_data = json.load(f)

        def convert_basegp_to_ini(point_dict):
            """Convert BaseGP parameter space to INI space (auto-detects format)."""
            # Detect format based on keys
            if 'x1_CeilingHeight' in point_dict:
                # i9csy format
                ceiling_map = {2.8: 0.0, 4.0: 1.0, 8.5: 2.0}
                grid_map = {6.5: 0.0, 8.0: 1.0}
                return [
                    ceiling_map[point_dict["x1_CeilingHeight"]],  # x0
                    grid_map[point_dict["x2_GridModule"]],  # x1
                    float(point_dict["x3_OuterFurniture"]),  # x2 (already 0,1,2)
                    float(point_dict["x4_VisualBoundary"]),  # x3 (already 0,1,2)
                    float(point_dict["x5_PhysicalBoundary"]),  # x4 (already 0,1)
                    float(point_dict["x6_InnerFurniture"]),  # x5 (already 0,1,2)
                ]
            else:
                # 6vars format
                return [
                    float(point_dict["x1_binary"]),  # x0
                    float(point_dict["x2_5level_discrete"]) - 1.0,  # x1: 1-5 → 0-4
                    round(float(point_dict["x3_5level_decimal"]) * 4),  # x2: 0-1 → 0-4
                    float(point_dict["x4_4level_categorical"]),  # x3
                    float(point_dict["x5_3level_categorical"]),  # x4
                    float(point_dict["x6_binary"]),  # x5
                ]

        golden_best = convert_basegp_to_ini(keypoints_data["x_best_prior"])
        golden_worst = convert_basegp_to_ini(keypoints_data["x_worst_prior"])
        golden_maxstd = convert_basegp_to_ini(keypoints_data["x_max_std"])

        golden_points = [golden_best, golden_worst, golden_maxstd]
        golden_points_str = str(golden_points).replace("'", "")

        # 替换INI中的golden points
        old_points_line = "points = [[0.0, 4.0, 2.0, 1.0, 0.0, 0.0]]"
        new_points_line = f"points = {golden_points_str}"
        config_str = config_str.replace(old_points_line, new_points_line)

        if "[ManualGenerator]" not in config_str:
            manual_gen_section = f"\n[ManualGenerator]\n{new_points_line}\n\n"
            config_str = config_str.replace(
                "[init_strat]", manual_gen_section + "[init_strat]"
            )

        print(f"\n{'='*60}")
        print(f"✓ 加载3个BaseGP黄金点:")
        print(f"  1. Best  (mean={keypoints_data['best_mean']:.3f}): {golden_best}")
        print(f"  2. Worst (mean={keypoints_data['worst_mean']:.3f}): {golden_worst}")
        print(f"  3. MaxStd (std={keypoints_data['max_std']:.3f}): {golden_maxstd}")
        print(f"{'='*60}\n")

        # 动态替换 total_budget, tau_n_max, tau_n_min
        warmup_budget = 3  # 3个BaseGP黄金点
        eur_budget = budget - warmup_budget
        tau_n_max_value = max(int(eur_budget * 0.7), 5)  # 至少5
        tau_n_min_value = max(3, tau_n_max_value - 5)  # 确保 tau_n_min < tau_n_max
        if tau_n_min_value >= tau_n_max_value:
            tau_n_min_value = max(1, tau_n_max_value - 2)

        config_str = config_str.replace(
            "total_budget = 50", f"total_budget = {eur_budget}"
        )
        config_str = config_str.replace(
            "tau_n_max = 35", f"tau_n_max = {tau_n_max_value}"
        )
        config_str = config_str.replace(
            "tau_n_min = 10", f"tau_n_min = {tau_n_min_value}"
        )
        config_str = config_str.replace("max_asks = 50", f"max_asks = {eur_budget}")
        print(
            f"✓ 预算分配: warmup={warmup_budget} (golden), EUR={eur_budget}, 总计={budget}"
        )
        print(
            f"✓ 已替换 total_budget={eur_budget}, tau_n_max={tau_n_max_value}, tau_n_min={tau_n_min_value}"
        )

        # 创建 Server
        server = AEPsychServer(database_path=str(db_path))
        print(f"✓ Server 已创建")

        # 配置 Server
        setup_message = {
            "type": "setup",
            "message": {"config_str": config_str},
        }

        response = server.handle_request(setup_message)
        print(f"✓ 配置已加载: {config_path.name}")
        print(f"✓ Strategy ID: {response}")

        # 验证组件类型
        if hasattr(server, "strats") and len(server.strats) > 0:
            strat = server.strats[0]
            print(f"✓ Generator: {type(strat.generator).__name__}")
            if hasattr(strat.generator, "acqf"):
                print(f"✓ Acquisition: {type(strat.generator.acqf).__name__}")
            if hasattr(strat, "model"):
                print(f"✓ Model: {type(strat.model).__name__}")

        config_info = {
            "warmup_budget": warmup_budget,
            "eur_budget": eur_budget,
            "total_budget": budget,
            "tau_n_max": tau_n_max_value,
            "golden_points": golden_points,
            "config_path": str(config_path),
            "db_path": str(db_path),
        }

        return server, config_info

    except Exception as e:
        print(f"✗ Server 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        raise


def verify_server_components(server) -> Dict[str, str]:
    """
    验证Server的组件配置

    Args:
        server: AEPsych Server实例

    Returns:
        components: 组件名称字典
    """
    components = {
        "server": "✓",
        "generator": "✗",
        "acquisition": "✗",
        "model": "✗",
    }

    try:
        if hasattr(server, "_strats") and len(server._strats) > 0:
            strat = server._strats[0]

            if hasattr(strat, "generator"):
                components["generator"] = type(strat.generator).__name__

            if hasattr(strat.generator, "acqf") or hasattr(
                strat.generator, "_acqf_instance"
            ):
                acqf = (
                    strat.generator._acqf_instance
                    if hasattr(strat.generator, "_acqf_instance")
                    else strat.generator.acqf
                )
                if acqf is not None:
                    components["acquisition"] = type(acqf).__name__

            if hasattr(strat, "model") and strat.model is not None:
                components["model"] = type(strat.model).__name__

    except Exception as e:
        print(f"组件验证错误: {e}")

    return components
