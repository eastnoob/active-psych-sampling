#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
三模式核心-2b系统测试脚本
"""

import sys
import os
from pathlib import Path

# 设置编码 - Windows 兼容性修复
os.environ["PYTHONIOENCODING"] = "utf-8"

# 处理 Windows PowerShell 的编码问题
if sys.platform == "win32":
    # 强制使用 UTF-8 编码，避免 GBK 乱码
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
else:
    # Linux/Mac 直接重新配置
    sys.stdout.reconfigure(encoding="utf-8")

# 添加core目录到路径
sys.path.insert(0, str(Path(__file__).parent / "core"))

from warmup_sampler import WarmupSampler
from warmup_budget_estimator import WarmupBudgetEstimator

# 设计空间路径
DESIGN_CSV = (
    Path(__file__).parent.parent.parent
    / "data"
    / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv"
)


def test_three_modes():
    """测试三种模式"""

    print("=" * 80)
    print("三模式Core-2b系统测试")
    print("=" * 80)
    print()

    # 初始化采样器
    sampler = WarmupSampler(str(DESIGN_CSV))

    # 评估预算
    n_subjects = 5
    trials_per_subject = 20

    print(
        f"配置: {n_subjects}人 × {trials_per_subject}次 = {n_subjects * trials_per_subject}总trials"
    )
    print()

    adequacy, budget = sampler.evaluate_budget(
        n_subjects=n_subjects,
        trials_per_subject=trials_per_subject,
        skip_interaction=False,
    )

    print(f"预算充足性: {adequacy}")
    print(f"Core-2b分配: {budget.get('core2b_configs', 0)}个配置")
    print()

    # 指定的交互对（索引从0开始）
    interaction_pairs = [(3, 4), (0, 1)]

    # ========== 测试模式1: Free (当前默认) ==========
    print("=" * 80)
    print("模式1: FREE (自由探索所有交互对)")
    print("=" * 80)
    print()

    try:
        output_dir_1 = Path(__file__).parent / "test_output_mode_free"
        output_dir_1.mkdir(exist_ok=True)

        sampler.generate_samples(
            budget=budget,
            output_dir=str(output_dir_1),
            merge=False,
            interaction_mode="free",
            interaction_pairs_to_explore=interaction_pairs,
            min_config_per_pair=2,
        )

        print("[OK] 模式1生成成功")
        print()
    except Exception as e:
        print(f"[ERROR] 模式1失败: {e}")
        import traceback

        traceback.print_exc()
        print()

    # ========== 测试模式2: Specified Only ==========
    print("=" * 80)
    print("模式2: SPECIFIED_ONLY (仅探索指定的交互对)")
    print("=" * 80)
    print()

    try:
        output_dir_2 = Path(__file__).parent / "test_output_mode_specified"
        output_dir_2.mkdir(exist_ok=True)

        sampler.generate_samples(
            budget=budget,
            output_dir=str(output_dir_2),
            merge=False,
            interaction_mode="specified_only",
            interaction_pairs_to_explore=interaction_pairs,
            min_config_per_pair=2,
        )

        print("[OK] 模式2生成成功")
        print()
    except Exception as e:
        print(f"[ERROR] 模式2失败: {e}")
        import traceback

        traceback.print_exc()
        print()

    # ========== 测试模式3: Hybrid (推荐) ==========
    print("=" * 80)
    print("模式3: HYBRID (指定对优先 + 自由探索) - 推荐")
    print("=" * 80)
    print()

    try:
        output_dir_3 = Path(__file__).parent / "test_output_mode_hybrid"
        output_dir_3.mkdir(exist_ok=True)

        sampler.generate_samples(
            budget=budget,
            output_dir=str(output_dir_3),
            merge=False,
            interaction_mode="hybrid",
            interaction_pairs_to_explore=interaction_pairs,
            min_config_per_pair=2,
        )

        print("[OK] 模式3生成成功")
        print()
    except Exception as e:
        print(f"[ERROR] 模式3失败: {e}")
        import traceback

        traceback.print_exc()
        print()

    print("=" * 80)
    print("测试完成！")
    print("=" * 80)
    print()
    print("对比分析:")
    print(f"  模式1 (Free)         -> {output_dir_1}")
    print(f"  模式2 (Specified)    -> {output_dir_2}")
    print(f"  模式3 (Hybrid)       -> {output_dir_3}")
    print()
    print("建议: 对大多数场景使用模式3 (Hybrid)")


if __name__ == "__main__":
    test_three_modes()
