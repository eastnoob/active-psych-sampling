"""
批量创建同一群体的模拟被试

使用已有的群体参数（population weights）和固定的交互效应，
批量生成多个具有个体差异的被试。

使用方法：
    python reproduce_subject_cluster.py --base_dir phase1_analysis_output/202512011547/step1_5/result --n_subjects 20
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# 添加tools路径
tools_path = Path(__file__).parent.parent.parent / "tools"
sys.path.insert(0, str(tools_path))

from subject_simulator_v2.linear import LinearSubject


def load_population_params(result_dir: Path) -> dict:
    """
    从result目录加载群体参数

    Args:
        result_dir: Step1.5的result目录路径

    Returns:
        包含群体参数的字典
    """
    config_file = result_dir / "fixed_weights_auto.json"

    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    with open(config_file) as f:
        config = json.load(f)

    return {
        "population_weights": np.array(config["global"][0]),
        "interaction_weights": {
            tuple(map(int, k.split(','))): v
            for k, v in config.get("interactions", {}).items()
        },
        "bias": config.get("bias", 0.0),
        "method": config.get("method", "unknown")
    }


def create_subject_cluster(
    population_params: dict,
    n_subjects: int,
    individual_std: float = 0.125,  # 个体差异标准差
    likert_levels: int = 5,
    likert_sensitivity: float = 2.0,
    base_seed: int = 100,  # 使用不同的种子避免与原始数据重复
    output_dir: Path = None
) -> list:
    """
    批量创建同一群体的被试

    Args:
        population_params: 群体参数（从load_population_params获取）
        n_subjects: 要创建的被试数量
        individual_std: 个体偏差标准差（推荐0.1-0.15）
        likert_levels: Likert量表等级数
        likert_sensitivity: Likert转换灵敏度
        base_seed: 基础随机种子
        output_dir: 输出目录（如果提供，会保存被试参数）

    Returns:
        被试对象列表
    """
    print("=" * 80)
    print("批量创建同一群体的模拟被试")
    print("=" * 80)
    print()
    print("群体参数:")
    print(f"  Population weights: {population_params['population_weights']}")
    print(f"  Bias: {population_params['bias']:.4f}")
    print(f"  Interaction pairs: {list(population_params['interaction_weights'].keys())}")
    print(f"  Method: {population_params['method']}")
    print()
    print("个体差异参数:")
    print(f"  Individual std: {individual_std}")
    print(f"  Number of subjects: {n_subjects}")
    print(f"  Base seed: {base_seed}")
    print()

    subjects = []
    subject_specs = []

    n_features = len(population_params['population_weights'])

    for i in range(n_subjects):
        subject_id = i + 1
        subject_seed = base_seed + i

        # 设置被试专用随机数生成器
        rng = np.random.RandomState(subject_seed)

        # 生成个体偏差
        individual_deviation = rng.normal(0, individual_std, size=n_features)

        # 被试权重 = 群体权重 + 个体偏差
        subject_weights = population_params['population_weights'] + individual_deviation

        # 创建被试对象
        subject = LinearSubject(
            weights=subject_weights,
            interaction_weights=population_params['interaction_weights'],
            bias=population_params['bias'],
            noise_std=0.0,  # 确定性输出
            likert_levels=likert_levels,
            likert_sensitivity=likert_sensitivity,
            seed=subject_seed
        )

        subjects.append(subject)

        # 记录规格
        spec = {
            "subject_id": subject_id,
            "seed": subject_seed,
            "population_weights": population_params['population_weights'].tolist(),
            "individual_deviation": individual_deviation.tolist(),
            "subject_weights": subject_weights.tolist(),
            "bias": population_params['bias'],
            "interaction_weights": {
                f"{k[0]},{k[1]}": v
                for k, v in population_params['interaction_weights'].items()
            },
            "likert_levels": likert_levels,
            "likert_sensitivity": likert_sensitivity
        }
        subject_specs.append(spec)

        print(f"  [OK] Subject {subject_id} created (seed={subject_seed})")

    print()
    print(f"[OK] 成功创建 {n_subjects} 个被试")
    print()

    # 保存被试规格
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON格式
        specs_file = output_dir / "subject_cluster_specs.json"
        with open(specs_file, 'w', encoding='utf-8') as f:
            json.dump({
                "population_params": {
                    "weights": population_params['population_weights'].tolist(),
                    "bias": population_params['bias'],
                    "interactions": {
                        f"{k[0]},{k[1]}": v
                        for k, v in population_params['interaction_weights'].items()
                    }
                },
                "individual_std": individual_std,
                "base_seed": base_seed,
                "n_subjects": n_subjects,
                "subjects": subject_specs
            }, f, indent=2, ensure_ascii=False)

        print(f"[保存] 被试规格已保存至: {specs_file}")

        # 保存简化摘要
        summary_file = output_dir / "subject_cluster_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("被试群体摘要\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"群体参数:\n")
            f.write(f"  Population weights: {population_params['population_weights']}\n")
            f.write(f"  Bias: {population_params['bias']:.4f}\n")
            f.write(f"  Individual std: {individual_std}\n\n")
            f.write(f"被试数量: {n_subjects}\n")
            f.write(f"Base seed: {base_seed}\n\n")
            f.write("各被试权重偏差:\n")
            for spec in subject_specs:
                f.write(f"  Subject {spec['subject_id']}: {spec['individual_deviation']}\n")

        print(f"[保存] 摘要已保存至: {summary_file}")
        print()

    return subjects


def test_subjects_on_design_space(
    subjects: list,
    design_space_csv: Path,
    output_dir: Path = None
):
    """
    在设计空间上测试被试响应

    Args:
        subjects: 被试对象列表
        design_space_csv: 设计空间CSV文件
        output_dir: 输出目录
    """
    print("=" * 80)
    print("在设计空间上测试被试响应")
    print("=" * 80)
    print()

    # 读取设计空间
    design_df = pd.read_csv(design_space_csv)
    print(f"设计空间: {len(design_df)} 个点")
    print(f"特征列: {list(design_df.columns)}")
    print()

    # 为每个被试生成响应
    for i, subject in enumerate(subjects, 1):
        print(f"  测试 Subject {i}...", end='')

        # 计算响应
        responses = []
        for _, row in design_df.iterrows():
            x = row.values
            y = subject(x)
            responses.append(y)

        design_df[f'y_subject_{i}'] = responses

        # 统计
        mean_y = np.mean(responses)
        std_y = np.std(responses)
        print(f" mean={mean_y:.2f}, std={std_y:.2f}")

    print()

    # 统计被试间差异
    response_cols = [f'y_subject_{i}' for i in range(1, len(subjects)+1)]
    subject_means = [design_df[col].mean() for col in response_cols]

    print("被试间差异:")
    print(f"  Mean of means: {np.mean(subject_means):.3f}")
    print(f"  SD of means: {np.std(subject_means, ddof=1):.3f}")
    print(f"  Range: {min(subject_means):.2f} - {max(subject_means):.2f}")
    print()

    # 保存结果
    if output_dir:
        output_file = output_dir / "design_space_responses.csv"
        design_df.to_csv(output_file, index=False)
        print(f"[保存] 响应数据已保存至: {output_file}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="批量创建同一群体的模拟被试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Step1.5的result目录路径（包含fixed_weights_auto.json）"
    )

    parser.add_argument(
        "--n_subjects",
        type=int,
        default=20,
        help="要创建的被试数量（默认: 20）"
    )

    parser.add_argument(
        "--individual_std",
        type=float,
        default=0.125,
        help="个体差异标准差（默认: 0.125）"
    )

    parser.add_argument(
        "--base_seed",
        type=int,
        default=100,
        help="基础随机种子（默认: 100，避免与原始数据重复）"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认: base_dir/reproduced_subjects）"
    )

    parser.add_argument(
        "--test_design_space",
        type=str,
        default=None,
        help="（可选）设计空间CSV路径，用于测试被试响应"
    )

    parser.add_argument(
        "--likert_levels",
        type=int,
        default=5,
        help="Likert量表等级数（默认: 5）"
    )

    parser.add_argument(
        "--likert_sensitivity",
        type=float,
        default=2.0,
        help="Likert转换灵敏度（默认: 2.0）"
    )

    args = parser.parse_args()

    # 路径处理
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"[错误] 目录不存在: {base_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "reproduced_subjects"

    # 1. 加载群体参数
    print()
    print("步骤1: 加载群体参数")
    print("-" * 80)
    try:
        population_params = load_population_params(base_dir)
        print(f"[OK] 成功加载群体参数从: {base_dir}")
        print()
    except Exception as e:
        print(f"[错误] 加载失败: {e}")
        sys.exit(1)

    # 2. 批量创建被试
    print("步骤2: 批量创建被试")
    print("-" * 80)
    try:
        subjects = create_subject_cluster(
            population_params=population_params,
            n_subjects=args.n_subjects,
            individual_std=args.individual_std,
            likert_levels=args.likert_levels,
            likert_sensitivity=args.likert_sensitivity,
            base_seed=args.base_seed,
            output_dir=output_dir
        )
    except Exception as e:
        print(f"[错误] 创建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 3. （可选）在设计空间上测试
    if args.test_design_space:
        print("步骤3: 在设计空间上测试被试")
        print("-" * 80)
        design_space_path = Path(args.test_design_space)
        if not design_space_path.exists():
            print(f"[警告] 设计空间文件不存在: {design_space_path}")
        else:
            try:
                test_subjects_on_design_space(
                    subjects=subjects,
                    design_space_csv=design_space_path,
                    output_dir=output_dir
                )
            except Exception as e:
                print(f"[错误] 测试失败: {e}")
                import traceback
                traceback.print_exc()

    print("=" * 80)
    print("[OK] All Done!")
    print("=" * 80)
    print()
    print(f"输出目录: {output_dir}")
    print(f"创建了 {args.n_subjects} 个被试")
    print()


if __name__ == "__main__":
    main()
