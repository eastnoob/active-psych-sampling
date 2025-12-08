#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EUR验证 - 使用复制的被试群体

本脚本基于 test/is_EUR_work/run_eur_verification_sps.py
但使用从 reproduce_subject_cluster.py 创建的被试群体

使用方法：
    python run_eur_with_reproduced_subjects.py \
        --subject_spec phase1_analysis_output/202512011547/step1_5/result/reproduced_subjects/subject_cluster_specs.json \
        --subject_id 1 \
        --budget 50
"""

import sys
import io
import argparse
import json
from pathlib import Path
import numpy as np

# 修复编码
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# 导入被试模拟器
from subject_simulator_v2.linear import LinearSubject

# ============================================================================
# 命令行参数
# ============================================================================
parser = argparse.ArgumentParser(description="EUR验证 - 使用复制的被试")
parser.add_argument(
    "--subject_spec",
    type=str,
    required=True,
    help="被试规格JSON文件路径（来自 reproduce_subject_cluster.py）",
)
parser.add_argument(
    "--subject_id",
    type=int,
    default=1,
    help="要使用的被试ID（1-N）",
)
parser.add_argument(
    "--budget",
    type=int,
    default=50,
    help="采样次数（默认50）",
)
parser.add_argument(
    "--config",
    type=str,
    default="eur_config_sps.ini",
    help="EUR配置文件（默认: eur_config_sps.ini）",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="输出目录（默认: eur_results/被试ID_时间戳）",
)
args = parser.parse_args()

print("\n" + "=" * 80)
print("EUR验证 - 使用复制的被试".center(80))
print("=" * 80)

# ============================================================================
# 步骤1: 加载被试规格
# ============================================================================
print("\n步骤1: 加载被试规格")
print("-" * 80)

spec_path = Path(args.subject_spec)
if not spec_path.exists():
    print(f"[错误] 被试规格文件不存在: {spec_path}")
    sys.exit(1)

with open(spec_path, 'r', encoding='utf-8') as f:
    cluster_data = json.load(f)

# 查找指定的被试
subject_spec = None
for spec in cluster_data['subjects']:
    if spec['subject_id'] == args.subject_id:
        subject_spec = spec
        break

if subject_spec is None:
    print(f"[错误] 找不到被试ID={args.subject_id}")
    print(f"可用被试ID: {[s['subject_id'] for s in cluster_data['subjects']]}")
    sys.exit(1)

print(f"[OK] 已加载被试 {args.subject_id}")
print(f"  种子: {subject_spec['seed']}")
print(f"  主效应权重: {subject_spec['subject_weights'][:3]}...")  # 只显示前3个
print(f"  Bias: {subject_spec['bias']:.4f}")

# ============================================================================
# 步骤2: 创建被试对象
# ============================================================================
print("\n步骤2: 创建被试对象")
print("-" * 80)

# 重建交互权重字典（从字符串键恢复）
interaction_weights = {}
for key_str, weight in subject_spec['interaction_weights'].items():
    idx1, idx2 = map(int, key_str.split(','))
    interaction_weights[(idx1, idx2)] = weight

# 创建 LinearSubject 对象
oracle = LinearSubject(
    weights=np.array(subject_spec['subject_weights']),
    interaction_weights=interaction_weights,
    bias=subject_spec['bias'],
    noise_std=0.0,  # 确定性输出（可选：添加噪声模拟测量误差）
    likert_levels=subject_spec['likert_levels'],
    likert_sensitivity=subject_spec['likert_sensitivity'],
    seed=subject_spec['seed']
)

print(f"[OK] 被试对象创建成功")
print(f"  类型: {type(oracle).__name__}")
print(f"  特征数: {len(subject_spec['subject_weights'])}")
print(f"  交互项数: {len(interaction_weights)}")

# 打印被试模型规格
print(f"\n{'=' * 60}")
print(f"【被试模型规格 - Subject {args.subject_id}】")
print(f"{'=' * 60}")
print(f"  特征数量: {len(subject_spec['subject_weights'])}")
print(f"  Likert级别: {subject_spec['likert_levels']}")
print(f"  噪声标准差: 0.0 (确定性)")
print(f"\n  主效应权重:")
for i, w in enumerate(subject_spec['subject_weights']):
    print(f"    x{i}: {w:+.6f}")
print(f"\n  交互项权重:")
for (i, j), w in interaction_weights.items():
    print(f"    x{i}×x{j}: {w:+.6f}")
print(f"\n  Bias: {subject_spec['bias']:.6f}")
print(f"{'=' * 60}\n")

# ============================================================================
# 步骤3: 调用完整的 EUR 验证流程
# ============================================================================
print("\n步骤3: 运行 EUR 验证")
print("-" * 80)
print(f"  Budget: {args.budget}")
print(f"  Config: {args.config}")
print()
print("[提示] 现在可以继续集成完整的 EUR 验证代码...")
print("       你可以复制 test/is_EUR_work/run_eur_verification_sps.py 的")
print("       步骤1-6（设计空间加载、Server初始化、采样循环等）")
print()

# ============================================================================
# 临时输出：保存被试信息到结果目录
# ============================================================================
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.output_dir:
    result_dir = Path(args.output_dir)
else:
    result_dir = Path(__file__).parent / "eur_results" / f"subject_{args.subject_id}_{timestamp}"
result_dir.mkdir(parents=True, exist_ok=True)

# 保存使用的被试规格
subject_info_file = result_dir / "subject_info.json"
with open(subject_info_file, 'w', encoding='utf-8') as f:
    json.dump(subject_spec, f, indent=2, ensure_ascii=False)

print(f"[OK] 被试信息已保存至: {subject_info_file}")
print(f"[OK] 结果目录: {result_dir}")
print()

print("=" * 80)
print("[提示] 如需完整的 EUR 验证，请添加以下内容：")
print("  1. 加载设计空间（从 only_independences CSV）")
print("  2. 初始化 AEPsych Server")
print("  3. 运行采样循环（使用上面创建的 oracle 对象）")
print("  4. 效应识别验证")
print("  5. 预测质量评估")
print("=" * 80)
