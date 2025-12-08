#!/usr/bin/env python3
"""
【最小化模拟脚本】CSV → 被试模拟 → CSV

【核心流程】
  1. 输入：subject_*.csv 文件（行 = 试次，列 = x1~x6 自变量）
  2. 处理：为每个被试创建一个 SingleOutputLatentSubject 模型实例
  3. 操作：遍历每一行，调用被试模型生成连续响应值 y
  4. 输出：新增 y 列后保存为 CSV（subject_*_result.csv 和 combined_results.csv）

【可配置参数】
  • interaction_pairs: 显式指定的交互项对，例如 [(0,1), (1,2)]
    - (0,1) 表示 x1 与 x2 的交互项
    - (1,3) 表示 x2 与 x4 的交互项
  • num_interactions: 额外随机生成的交互项数量（0 = 仅用配置的交互对）
  • interaction_scale: 交互项权重的标准差（控制交互项强度）

【为什么是最小化？】
  原始 run_simulation.py 有 1121 行，其中 75% 是无关代码：
  • SIMULATION_REPORT.md 生成（300+ 行）
  • 模型描述 markdown 写入（100+ 行）
  • 分布统计与百分位映射（100+ 行）
  • STEP1 采样（WarmupSampler）集成（100+ 行）
  • DEBUG 打印（100+ 行）
  • 其他报告与参数比较（200+ 行）

  本脚本只保留核心 CSV→模型→CSV 管道，280 行代码。
"""

import argparse
from pathlib import Path
import sys
import numpy as np

# ============================================================================
# 配置参数（在命令行中可被覆盖）
# ============================================================================

# 【交互项配置】显式指定哪些因子对有交互效应
#   例如：[(3,4), (0,1), (1,3)] 表示
#   - x4 与 x5 的交互（索引 3,4）
#   - x1 与 x2 的交互（索引 0,1）
#   - x2 与 x4 的交互（索引 1,3）
INTERACTION_PAIRS = [(3, 4), (0, 1), (1, 3)]

# 【交互项权重尺度】交互项系数从 N(0, interaction_scale²) 采样
#   0.4 = 中等强度的交互效应
INTERACTION_SCALE = 0.4

# 【群体权重分布】所有被试围绕这个分布采样个体权重
#   population_std = 0.4: 权重波动范围较大，被试间差异明显
POPULATION_STD = 0.4

# 【群体权重均值】权重分布的中心
POPULATION_MEAN = 0.0

# 【个体偏差程度】个体权重 = 群体权重 + N(0, individual_std) 的随机偏差
#   1.0 表示个体偏差 = 100% * population_std = 0.4
INDIVIDUAL_STD_PERCENT = 1.0

# 【是否使用潜变量】True = 潜变量模型，False = 直接固定权重线性模型
USE_LATENT = False

# 【随机种子】控制被试参数的可重复性
SEED = 42


# ============================================================================
# 导入与初始化
# ============================================================================


def import_subject_module():
    """
    【导入被试模型类】

    尝试多种方式导入 MixedEffectsLatentSubject：
    1. 直接 import（依赖已安装）
    2. 向上搜索 subject 目录并加入 sys.path
    3. 最后尝试向上6层找项目根目录

    【返回】
    MixedEffectsLatentSubject 类
    """
    try:
        from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject

        return MixedEffectsLatentSubject
    except Exception:
        pass

    # 向上搜索 subject 目录
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "subject").is_dir():
            sys.path.insert(0, str(p))
            try:
                from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject

                return MixedEffectsLatentSubject
            except Exception:
                continue

    fallback = Path(__file__).resolve().parents[6]
    sys.path.insert(0, str(fallback))
    from subject.MixedEffectsLatentSubject import MixedEffectsLatentSubject

    return MixedEffectsLatentSubject


# 【导入主类】MixedEffectsLatentSubject 是被试模型的基类
MixedEffectsLatentSubject = import_subject_module()

# 【导入包装类】SingleOutputLatentSubject 将输出强制为单个标量值
try:
    from single_output_subject import SingleOutputLatentSubject
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from single_output_subject import SingleOutputLatentSubject


# 【因子映射表】将分类变量转换为数值编码
# x4: "low"=0, "mid"=1, "high"=2, "max"=3
# x5: "A"=0, "B"=1, "C"=2
X4_MAP = {"low": 0, "mid": 1, "high": 2, "max": 3}
X5_MAP = {"A": 0, "B": 1, "C": 2}


# ============================================================================
# 工具函数
# ============================================================================


def convert_row_to_features(row):
    """
    【将 CSV 行转换为特征向量】

    从 dict（pandas 行）或列表/元组（原始 CSV）中提取特征值，
    并将分类变量（x4, x5）转换为数值。

    【输入】
    row: dict 或 list/tuple
         dict 形式：{"x1_binary": 0, "x2_5level_discrete": 1, ...}
         list 形式：[0, 1, 2.5, "low", "A", 1]

    【输出】
    numpy 数组，形状 (6,)，值为 [x1, x2, x3, x4, x5, x6]

    【例子】
    row = {"x1_binary": 0, "x4_4level_categorical": "high", ...}
    → [0.0, ..., 2.0, ..., ...]  # x4_map["high"] = 2
    """
    if isinstance(row, dict):
        x1 = float(row.get("x1_binary", 0))
        x2 = float(row.get("x2_5level_discrete", 0))
        x3 = float(row.get("x3_5level_decimal", 0))
        x4_val = row.get("x4_4level_categorical", "low")
        x4 = float(X4_MAP.get(x4_val, 0))
        x5_val = row.get("x5_3level_categorical", "A")
        x5 = float(X5_MAP.get(x5_val, 0))
        x6 = float(row.get("x6_binary", 0))
    else:
        # CSV 行（列表或元组）
        x1, x2, x3, x4_str, x5_str, x6 = row[:6]
        x4 = float(X4_MAP.get(x4_str, 0))
        x5 = float(X5_MAP.get(x5_str, 0))
    return np.array([x1, x2, x3, x4, x5, x6], dtype=float)


def read_csv_rows(csv_path: Path):
    """
    【读取 CSV 文件】

    优先使用 pandas（如果可用），否则用原生 csv 模块。

    【输入】
    csv_path: Path 对象，指向要读取的 CSV 文件

    【输出】
    rows: list of dict，每个 dict 是一行数据
         [{"x1_binary": 0, "x2_5level_discrete": 1, ...}, ...]

    【优点】
    返回 dict 列表便于后续用列名访问，避免列索引混乱
    """
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
        return [dict(row) for _, row in df.iterrows()]
    except Exception:
        # 降级：原生 csv 模块
        rows = []
        with open(csv_path) as f:
            header = next(f).strip().split(",")
            for line in f:
                values = line.strip().split(",")
                rows.append(dict(zip(header, values)))
        return rows


def write_csv_results(rows, csv_path: Path):
    """
    【写入 CSV 文件】

    将结果（包含 y 列）写入 CSV 文件。

    【输入】
    rows: list of dict，每个 dict 包含所有列（包括原始列和新增 y 列）
    csv_path: Path 对象，输出文件路径

    【例子】
    rows = [
        {"x1_binary": 0, ..., "y": 2.5},
        {"x1_binary": 1, ..., "y": 3.1},
        ...
    ]
    write_csv_results(rows, Path("output.csv"))
    → 写入 CSV，列顺序可能会变（pandas 默认按字典 key 排序）

    【注意】
    如果行数为 0，函数直接返回（不创建空文件）
    """
    if not rows:
        return

    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)


# ============================================================================
# 主函数
# ============================================================================


def run(
    input_dir: Path,
    output_dir: Path = None,
    seed: int = SEED,
    interaction_pairs=None,
    interaction_scale=None,
):
    """
    【主模拟函数】

    核心流程：
    1. 在 input_dir 中查找所有 subject_*.csv 文件
    2. 对每个文件：
       a. 读取数据（包含 x1~x6 自变量）
       b. 创建 SingleOutputLatentSubject 被试模型（配置交互项）
       c. 遍历每一行，用被试模型生成 y 值
       d. 保存结果到 subject_*_result.csv
    3. 合并所有被试数据到 combined_results.csv
    4. 保存最后一个被试的模型规格到 model_spec.txt（用于查阅）

    【输入参数】
    input_dir: Path
        输入目录，必须包含 subject_1.csv, subject_2.csv, ... 等文件

    output_dir: Path or None
        输出目录。如果为 None，默认为 input_dir/result

    seed: int
        随机种子。被试种子为 seed + 被试号（确保被试间差异但可重复）

    interaction_pairs: list of tuple or None
        交互项对，例如 [(3, 4), (0, 1), (1, 3)]
        如果为 None，使用全局变量 INTERACTION_PAIRS

    interaction_scale: float or None
        交互项权重尺度，例如 0.4
        如果为 None，使用全局变量 INTERACTION_SCALE

    【被试号与种子的关系】
    subject_1.csv → subject_seed = seed + 1 = 43
    subject_2.csv → subject_seed = seed + 2 = 44
    ...
    这样保证了每个被试有不同的随机参数，但整体可重复。

    【配置交互项的方式】
    通过函数参数 interaction_pairs, interaction_scale 传入，或使用全局变量。
    例如：
        interaction_pairs = [(3, 4), (0, 1), (1, 3)]
        interaction_scale = 0.4
    表示有 3 个交互项，权重从 N(0, 0.4²) 采样。

    【输出文件说明】
    • subject_*_result.csv: 每个被试的原始数据 + y 列
    • combined_results.csv: 所有被试的合并数据（新增 subject 列）
    • model_spec.txt: 最后一个被试的模型参数（dict 格式）
    """
    # 【使用参数或全局变量】
    if interaction_pairs is None:
        interaction_pairs = INTERACTION_PAIRS
    if interaction_scale is None:
        interaction_scale = INTERACTION_SCALE

    if output_dir is None:
        output_dir = input_dir / "result"

    output_dir.mkdir(exist_ok=True)

    # 【步骤1】查找所有 subject_*.csv 文件
    csv_files = sorted(list(input_dir.glob("subject_*.csv")))
    if not csv_files:
        print(f"No subject_*.csv files found in {input_dir}")
        return

    print(f"Found {len(csv_files)} subject files")

    # 【步骤2】创建基础被试实例，用于获取默认参数
    # （这一步可选，只是为了参考；实际每个被试都会创建独立实例）
    base = MixedEffectsLatentSubject(
        num_features=6,
        num_observed_vars=1,
        num_latent_vars=2,
        seed=seed,
    )

    combined_results = []
    subjects_info = []  # 【收集每个被试的模型参数和误差】

    # 【步骤3】处理每个被试
    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"\nProcessing {csv_path.name}...")

        # 【读取被试数据】
        rows = read_csv_rows(csv_path)
        subject_seed = seed + idx

        # 【创建被试对象】
        # 关键参数：
        # - use_latent=USE_LATENT: 是否使用潜变量（True=复杂，False=简单）
        # - interaction_pairs: 交互项对（例如 [(3,4), (0,1), (1,3)]）
        # - interaction_scale: 交互项权重尺度（控制交互强度）
        subject = SingleOutputLatentSubject(
            num_features=6,
            seed=subject_seed,
            use_latent=USE_LATENT,
            population_mean=POPULATION_MEAN,
            population_std=POPULATION_STD,
            individual_std_percent=INDIVIDUAL_STD_PERCENT,
            interaction_pairs=interaction_pairs,
            interaction_scale=interaction_scale,
        )

        # 【为每一行生成 y 值】
        # 遍历所有试次，调用被试模型生成响应值
        output_rows = []
        predictions = []  # 用于计算误差
        for row in rows:
            X = convert_row_to_features(row)
            # subject(X) 返回标量响应值 y
            y = subject(X)
            predictions.append(y)
            # 保留原始列，新增 y 列
            output_rows.append({**row, "y": y})

        # 【计算误差统计】
        y_mean = np.mean(predictions)
        y_std = np.std(predictions)
        y_min = np.min(predictions)
        y_max = np.max(predictions)

        # 【获取模型参数】
        model_spec = subject.get_model_spec()

        # 【收集被试信息】
        subject_name = csv_path.stem
        subjects_info.append(
            {
                "subject": subject_name,
                "n_trials": len(rows),
                "y_mean": y_mean,
                "y_std": y_std,
                "y_min": y_min,
                "y_max": y_max,
                "model_spec": model_spec,
            }
        )

        # 【保存被试结果】
        output_path = output_dir / f"{csv_path.stem}_result.csv"
        write_csv_results(output_rows, output_path)
        print(f"  → Saved {output_path.name}")

        # 【加入合并结果】
        # 新增 subject 列用于后续区分不同被试
        for r in output_rows:
            combined_results.append({**r, "subject": csv_path.stem})

    # 【步骤4】保存合并结果
    combined_path = output_dir / "combined_results.csv"
    write_csv_results(combined_results, combined_path)
    print(f"\nSaved combined results: {combined_path}")

    # 【步骤5】保存模型规格（参考用）
    # 这是最后一个被试的模型参数字典，包含：
    # - 群体权重 (population_weights)
    # - 个体偏差 (individual_deviation)
    # - 交互项 (interaction_terms)
    # - 其他参数（num_features, noise_std 等）
    spec_path = output_dir / "model_spec.txt"
    with open(spec_path, "w") as f:
        f.write(str(subject.get_model_spec()))
    print(f"Saved model spec: {spec_path}")

    # 【步骤6】返回被试信息供外部调用】
    return subjects_info


# ============================================================================
# 命令行入口
# ============================================================================

if __name__ == "__main__":
    """
    【快速开始】
    
    前置条件：input_dir 中必须有 subject_*.csv 文件
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【用法 1】最简单 - 自动输出到 input_dir/result
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_simulation_minimal.py d:\WORKSPACE\python\aepsych-source\data\linear
    
    输出：
      • d:\WORKSPACE\python\aepsych-source\data\linear\result\subject_1_result.csv
      • d:\WORKSPACE\python\aepsych-source\data\linear\result\subject_2_result.csv
      • d:\WORKSPACE\python\aepsych-source\data\linear\result\combined_results.csv
      • d:\WORKSPACE\python\aepsych-source\data\linear\result\model_spec.txt
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【用法 2】指定输出目录
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_simulation_minimal.py d:\WORKSPACE\python\aepsych-source\data\linear \
        --output_dir d:\output\my_results
    
    输出全部写入 d:\output\my_results 中
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【用法 3】自定义交互项
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_simulation_minimal.py d:\data\linear \
        --interaction_pairs "[(0,1), (2,3)]" \
        --interaction_scale 0.5
    
    说明：
      • interaction_pairs: 交互项对。(0,1) 表示 x1*x2, (2,3) 表示 x3*x4
      • interaction_scale: 交互项权重从 N(0, 0.5²) 采样（默认 0.4）
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【用法 4】自定义种子（确保可重复性）
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    python run_simulation_minimal.py d:\data\linear --seed 999
    
    说明：
      • 默认 seed=42
      • 改为 999 将生成完全不同的被试参数
      • 同一个 seed 始终生成相同的参数（可重复）
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【输出文件说明】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    1. subject_1_result.csv（被试 1 的结果）
       列：[x1_binary, x2_5level_discrete, x3_5level_decimal, x4_4level_categorical, 
            x5_3level_categorical, x6_binary, y]
       行数 = subject_1.csv 的行数
       y 列 = 被试模型的输出（连续值）
    
    2. combined_results.csv（所有被试合并）
       列：[..., y, subject]
       行数 = 所有 subject_*.csv 行数之和
       subject 列 = 标记该行属于 subject_1, subject_2, ... 等
    
    3. model_spec.txt（最后一个被试的模型规格）
       内容 = Python dict 格式的模型参数
       例如：{
         'population_weights': [0.12, -0.08, 0.25, ...],
         'interaction_terms': {'x1*x2': 0.19, 'x2*x4': -0.05, ...},
         'noise_std': 0.1,
         ...
       }
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    【故障排除】
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Q: "No subject_*.csv files found"
    A: 检查 input_dir 路径是否正确，确保包含 subject_1.csv, subject_2.csv, ... 等文件
    
    Q: 生成的 y 列都是 NaN 或者错误值
    A: 检查 CSV 列名是否正确（脚本期望列名为 x1_binary, x2_5level_discrete, ...）
       如果列名不同，修改 convert_row_to_features() 函数中的列名映射
    
    Q: 想禁用所有交互项
    A: 使用 --interaction_pairs "[]"
       python run_simulation_minimal.py d:\data\linear --interaction_pairs "[]"
    """
    parser = argparse.ArgumentParser(description="Run subject simulation on CSV files")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=".",
        help="Input directory containing subject_*.csv files",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: input_dir/result)",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help=f"Random seed (default: {SEED})"
    )
    parser.add_argument(
        "--interaction_pairs",
        type=str,
        default=None,
        help="Interaction pairs, e.g., '[(0,1), (1,2)]' or '(0,1) (1,2)'",
    )
    parser.add_argument(
        "--interaction_scale",
        type=float,
        default=INTERACTION_SCALE,
        help=f"Interaction weight scale (default: {INTERACTION_SCALE})",
    )

    args = parser.parse_args()

    # 【解析命令行参数】
    if args.interaction_pairs:
        try:
            # 尝试 eval（支持 "[(0,1), (1,2)]" 格式）
            INTERACTION_PAIRS = eval(args.interaction_pairs)
        except Exception:
            # 降级：尝试解析 "(0,1) (1,2)" 格式
            pairs_strs = args.interaction_pairs.strip().split()
            INTERACTION_PAIRS = [
                tuple(map(int, p.strip("()").split(","))) for p in pairs_strs
            ]

    if args.interaction_scale:
        INTERACTION_SCALE = args.interaction_scale

    # 【执行主函数】
    input_path = Path(args.input_dir).resolve()
    output_path = Path(args.output_dir).resolve() if args.output_dir else None
    run(input_path, output_path, args.seed)
