#!/usr/bin/env python3
"""
Warmup Adapter for Subject Simulator V2

完全兼容 extensions/warmup_budget_check/core/simulation_runner.py 的接口，
使用Subject Simulator V2替换旧的MixedEffectsLatentSubject实现。

优势：
- 修复了3个严重bug（交互项缺失、Likert公式错误、fixed_weights被忽略）
- 完整参数保存（所有参数都保存到JSON）
- 可选正态性检查（避免单调/偏斜数据）
- 简洁可靠的线性模型

使用方法（在quick_start.py中）：
    # 替换原来的导入
    # from core.simulation_runner import run as simulate_responses

    # 改为
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from subject_simulator_v2.adapters.warmup_adapter import run as simulate_responses
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict, Any

# 添加subject_simulator_v2到路径
_adapter_dir = Path(__file__).resolve().parent
_v2_root = _adapter_dir.parent
sys.path.insert(0, str(_v2_root.parent))

from subject_simulator_v2 import ClusterGenerator, LinearSubject


def run(
    input_dir: Path,
    seed: int = 42,
    output_mode: str = "individual",
    clean: bool = False,
    interaction_pairs: Optional[List[Tuple[int, int]]] = None,
    num_interactions: int = 0,
    interaction_scale: float = 1.0,
    # 旧参数（部分不支持）
    use_latent: bool = False,  # ⚠ V2不支持潜变量，此参数被忽略
    output_type: str = "continuous",  # continuous/likert
    likert_levels: int = 5,
    likert_mode: str = "tanh",  # tanh/percentile
    likert_sensitivity: float = 1.0,
    population_mean: float = 0.0,
    population_std: float = 0.05,
    individual_std_percent: float = 0.8,
    individual_corr: float = 0.0,  # ⚠ V2不支持特征相关，此参数被忽略
    fixed_weights_file: Optional[str] = None,
    # 新参数（模型显示）
    print_model: bool = False,
    save_model_summary: bool = False,
    model_summary_format: str = "txt",
    # V2新增参数
    ensure_normality: bool = True,  # 是否检查正态性
    bias: float = 0.0,  # 截距（可调整响应中心）
    noise_std: float = 0.0,  # 试次内噪声
    design_space_csv: Optional[str] = None,  # 完整设计空间CSV路径（用于bias计算）
    # V3新增参数（交互作为特征方法）
    interaction_as_features: bool = True,  # ⭐ 默认启用V3方法（更优的分布）
    interaction_x3x4_weight: float = 0.12,  # x3*x4交互权重（强，可探测）
    interaction_x0x1_weight: float = -0.02,  # x0*x1交互权重（弱，平衡分布）
):
    """
    使用Subject Simulator V2模拟被试作答

    完全兼容原simulation_runner.py的接口，输出格式一致。

    Args:
        input_dir: Step 1生成的采样方案目录（包含subject_*.csv）
        seed: 随机种子
        output_mode: 输出模式 - "individual"/"combined"/"both"
        clean: 是否清理result目录
        interaction_pairs: 交互效应对，如 [(3,4), (0,1)]
        num_interactions: ⚠ 不支持，请使用interaction_pairs明确指定
        interaction_scale: 交互权重标准差

        # 旧参数映射
        use_latent: ⚠ 不支持潜变量，此参数被忽略
        output_type: "continuous" 或 "likert"
        likert_levels: Likert等级数（仅当output_type="likert"时有效）
        likert_mode: "tanh"（标准）或 "percentile"（强制均匀）
        likert_sensitivity: Likert灵敏度（越大越集中于中心）
        population_mean: 群体权重均值
        population_std: 群体权重标准差
        individual_std_percent: 个体偏差比例（实际std = population_std * 此值）
        individual_corr: ⚠ 不支持特征相关，此参数被忽略
        fixed_weights_file: 固定权重文件路径（JSON格式）

        # 新参数
        print_model: 是否打印模型规格到控制台
        save_model_summary: 是否保存模型摘要文件
        model_summary_format: 摘要格式 - "txt"/"md"/"both"
        ensure_normality: 是否检查正态性（V2新增）
        bias: 截距（可调整响应中心，V2新增）
              ⭐ 重要：如果 output_type="likert" 且 bias=0.0（默认），
              将自动计算合适的bias值，使Likert响应分布正态。
              计算公式：bias = -mean(design_space @ population_weights + interactions)
              如果需要手动控制，请明确指定非0值。
        noise_std: 试次内噪声标准差（V2新增）
        design_space_csv: 完整设计空间CSV路径（V2新增）
              ⭐ 用途：
              1. bias自动计算（基于完整设计空间，而非25个采样点）
              2. 正态性检查（ensure_normality=True时使用）
              默认：None（使用subject_1.csv作为设计空间）
              推荐：指定完整设计空间（如324点），确保bias计算准确

    Returns:
        None（输出保存到 input_dir/result/ 目录）

    输出文件：
        - result/subject_1.csv, subject_2.csv, ... (output_mode="individual"或"both")
        - result/combined_results.csv (output_mode="combined"或"both")
        - result/subject_1_spec.json, subject_2_spec.json, ... (完整参数)
        - result/cluster_summary.json (集群摘要)
        - result/subject_1_model.md, ... (兼容旧格式的参数文件)
        - result/MODEL_SUMMARY.txt/md (如果save_model_summary=True)
    """

    # V3方法分支：交互作为显式特征（默认启用）
    if interaction_as_features:
        # 导入v3实现
        from . import warmup_adapter_v3
        return warmup_adapter_v3.run(
            input_dir=input_dir,
            seed=seed,
            output_mode=output_mode,
            clean=clean,
            interaction_x3x4_weight=interaction_x3x4_weight,
            interaction_x0x1_weight=interaction_x0x1_weight,
            output_type=output_type,
            likert_levels=likert_levels,
            likert_sensitivity=likert_sensitivity,
            population_mean=population_mean,
            population_std=population_std,
            individual_std_percent=individual_std_percent,
            noise_std=noise_std,
            design_space_csv=design_space_csv,
            print_model=print_model,
            save_model_summary=save_model_summary,
        )

    # 以下是V2旧实现（当interaction_as_features=False时使用）
    # 参数警告
    if use_latent:
        print("[警告] Subject Simulator V2不支持潜变量模型，use_latent参数被忽略")
    if individual_corr != 0.0:
        print("[警告] Subject Simulator V2不支持特征相关，individual_corr参数被忽略")
    if num_interactions > 0:
        print("[警告] num_interactions参数不支持，请使用interaction_pairs明确指定交互对")

    # 验证输入目录
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # 查找subject CSV文件
    subject_csvs = sorted(input_dir.glob("subject_*.csv"))
    if not subject_csvs:
        raise FileNotFoundError(f"No subject_*.csv files found in {input_dir}")

    n_subjects = len(subject_csvs)
    print(f"Found {n_subjects} subject CSV files")

    # 读取采样设计空间（从第一个CSV，用于响应生成）
    df_first = pd.read_csv(subject_csvs[0])

    # 转换分类变量为数值
    categorical_mappings = {}
    for col in df_first.columns:
        if df_first[col].dtype == 'object':
            # 分类变量
            unique_vals = sorted(df_first[col].unique())
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
            categorical_mappings[col] = mapping
            df_first[col] = df_first[col].map(mapping)
            print(f"  Categorical mapping for {col}: {mapping}")

    sampling_design_space = df_first.values.astype(float)
    n_features = sampling_design_space.shape[1]

    print(f"Sampling design space: {sampling_design_space.shape[0]} points, {n_features} features")

    # 读取完整设计空间（用于bias计算和正态性检查）
    if design_space_csv and Path(design_space_csv).exists():
        print(f"Loading full design space from: {design_space_csv}")
        df_full = pd.read_csv(design_space_csv)

        # 应用相同的分类变量映射
        for col in df_full.columns:
            if col in categorical_mappings:
                df_full[col] = df_full[col].map(categorical_mappings[col])

        full_design_space = df_full.values.astype(float)
        print(f"Full design space: {full_design_space.shape[0]} points, {full_design_space.shape[1]} features")
    else:
        # 默认使用采样设计空间
        full_design_space = sampling_design_space
        if design_space_csv:
            print(f"[Warning] design_space_csv not found: {design_space_csv}")
            print(f"[Warning] Using sampling design space ({sampling_design_space.shape[0]} points) instead")

    print()

    # 准备输出目录
    result_dir = input_dir / "result"
    if clean and result_dir.exists():
        import shutil
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 处理fixed_weights（扩展格式：包含interactions和bias）
    population_weights_override = None
    fixed_interaction_weights = None
    fixed_bias = None

    if fixed_weights_file and Path(fixed_weights_file).exists():
        print(f"Loading fixed weights from: {fixed_weights_file}")
        with open(fixed_weights_file, 'r') as f:
            fixed_data = json.load(f)

            # 加载主效应权重
            if "global" in fixed_data:
                # 格式: {"global": [[w1, w2, ...]]}
                population_weights_override = np.array(fixed_data["global"][0])
                print(f"  Loaded population weights: {population_weights_override}")

            # 加载交互项权重（V2扩展格式）
            if "interactions" in fixed_data:
                fixed_interaction_weights = {}
                for key, value in fixed_data["interactions"].items():
                    # key格式: "3,4" -> (3, 4)
                    i, j = map(int, key.split(','))
                    fixed_interaction_weights[(i, j)] = value
                print(f"  Loaded interaction weights: {fixed_interaction_weights}")

            # 加载bias（V2扩展格式）
            if "bias" in fixed_data:
                fixed_bias = fixed_data["bias"]
                print(f"  Loaded bias: {fixed_bias}")

    # 参数映射
    individual_std = population_std * individual_std_percent

    # 自动bias计算（针对Likert输出）
    actual_bias = bias

    # 如果fixed_weights中已有bias，优先使用
    if fixed_bias is not None and bias == 0.0:
        actual_bias = fixed_bias
        print(f"\n[Using Fixed Bias]")
        print(f"  Bias from fixed_weights file: {fixed_bias:.2f}")
        print()
    elif output_type == "likert" and bias == 0.0:
        # 自动计算bias（基于完整设计空间）
        # 如果有fixed_weights，使用它；否则生成临时population_weights
        if population_weights_override is not None:
            temp_weights = population_weights_override
        else:
            # 生成临时的population_weights用于计算
            np.random.seed(seed)
            temp_weights = np.random.normal(
                population_mean, population_std, size=n_features
            )

        # 计算设计空间的连续输出（包含主效应和交互效应）
        continuous_outputs = full_design_space @ temp_weights

        # 添加交互效应的贡献
        if fixed_interaction_weights:
            # 使用fixed的交互权重
            interaction_contrib = np.zeros(len(full_design_space))
            for (i, j), weight in fixed_interaction_weights.items():
                interaction_contrib += weight * full_design_space[:, i] * full_design_space[:, j]
            continuous_outputs += interaction_contrib
        elif interaction_pairs:
            # 生成临时的interaction_weights（用于bias计算）
            interaction_contrib = np.zeros(len(full_design_space))
            np.random.seed(seed + 1)  # 使用不同的seed
            for (i, j) in interaction_pairs:
                # 交互权重从正态分布采样
                interaction_weight = np.random.normal(0, interaction_scale)
                interaction_contrib += interaction_weight * full_design_space[:, i] * full_design_space[:, j]
            continuous_outputs += interaction_contrib

        # 自动bias = 负的均值（将输出居中于0，使tanh有效范围）
        auto_bias = -continuous_outputs.mean()
        actual_bias = auto_bias

        print(f"\n[Auto Bias Calculation]")
        print(f"  Design space: {full_design_space.shape[0]} points")
        print(f"  Continuous output range: [{continuous_outputs.min():.2f}, {continuous_outputs.max():.2f}]")
        print(f"  Continuous output mean: {continuous_outputs.mean():.2f}")
        print(f"  Auto-calculated bias: {auto_bias:.2f}")
        print(f"  (This centers the output at 0 for proper Likert conversion)")
        print()

    # 处理likert_mode
    actual_sensitivity = likert_sensitivity

    if output_type == "likert" and likert_mode == "percentile":
        # percentile模式：强制均匀分布
        # 策略：减小sensitivity，调整bias使分布更平坦
        actual_sensitivity = likert_sensitivity * 0.5  # 降低灵敏度
        print(f"[Likert Mode] percentile: adjusted sensitivity {likert_sensitivity} -> {actual_sensitivity}")

    # 确定Likert参数
    actual_likert_levels = likert_levels if output_type == "likert" else None

    # 创建ClusterGenerator
    print(f"\nGenerating subject cluster...")
    print(f"  Population mean: {population_mean}")
    print(f"  Population std: {population_std}")
    print(f"  Individual std: {individual_std} ({individual_std_percent} * {population_std})")
    print(f"  Interaction pairs: {interaction_pairs}")
    print(f"  Interaction scale: {interaction_scale}")
    print(f"  Bias: {actual_bias}")
    print(f"  Noise std: {noise_std}")
    if actual_likert_levels:
        print(f"  Likert levels: {actual_likert_levels}")
        print(f"  Likert sensitivity: {actual_sensitivity}")
        print(f"  Likert mode: {likert_mode}")
    print(f"  Ensure normality: {ensure_normality}")
    print()

    generator = ClusterGenerator(
        design_space=full_design_space,  # 使用完整设计空间进行正态性检查
        n_subjects=n_subjects,
        population_mean=population_mean,
        population_std=population_std,
        individual_std=individual_std,
        interaction_pairs=interaction_pairs or [],
        interaction_scale=interaction_scale,
        bias=actual_bias,
        noise_std=noise_std,
        likert_levels=actual_likert_levels,
        likert_sensitivity=actual_sensitivity,
        ensure_normality=ensure_normality,
        max_retries=20,
        seed=seed
    )

    # 如果有固定权重，手动覆盖
    if population_weights_override is not None:
        print("[Override] Using fixed population weights from file")
        # 直接设置（需要在generate_cluster前）
        # 这里我们生成后再手动覆盖被试权重

    # 生成集群
    cluster = generator.generate_cluster(str(result_dir))

    subjects = cluster['subjects']

    # 如果有fixed_weights，覆盖所有被试参数
    if population_weights_override is not None:
        print("[Override] Applying fixed parameters to all subjects")
        for subject in subjects:
            subject.weights = population_weights_override.copy()
            # 如果有fixed_interaction_weights，也覆盖
            if fixed_interaction_weights is not None:
                subject.interaction_weights = fixed_interaction_weights.copy()
                print(f"  Subject {subjects.index(subject)+1}: interaction_weights updated")
            # bias已在ClusterGenerator中设置，无需覆盖

    # 读取每个subject CSV并生成响应
    print(f"\nGenerating responses...")
    all_results = []

    for i, (csv_path, subject) in enumerate(zip(subject_csvs, subjects), start=1):
        # 读取采样点（保留原始格式）
        df_original = pd.read_csv(csv_path)

        # 创建数值化副本用于模型输入
        df_numeric = df_original.copy()
        for col, mapping in categorical_mappings.items():
            if col in df_numeric.columns:
                df_numeric[col] = df_numeric[col].map(mapping)

        X = df_numeric.values.astype(float)

        # 生成响应
        y = np.array([subject(x) for x in X])

        # 添加响应列到原始格式DataFrame
        df_result = df_original.copy()
        df_result['y'] = y

        # 保存individual CSV（如果需要）
        if output_mode in ["individual", "both"]:
            output_csv = result_dir / f"subject_{i}.csv"
            df_result.to_csv(output_csv, index=False)

        # 添加subject列用于combined
        df_result.insert(0, 'subject', f'subject_{i}')
        all_results.append(df_result)

        print(f"  Subject {i}: {len(y)} responses generated")

    # 保存combined CSV（如果需要）
    if output_mode in ["combined", "both"]:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_csv = result_dir / "combined_results.csv"
        combined_df.to_csv(combined_csv, index=False)
        print(f"\n  Combined results: {len(combined_df)} total responses")

    # 保存兼容旧格式的subject_X_model.md
    print(f"\nSaving model specifications...")
    _save_legacy_model_md(subjects, result_dir, seed, n_features)

    # 保存fixed_weights_auto.json（V2扩展格式：包含interactions和bias）
    avg_weights = np.mean([s.weights for s in subjects], axis=0)

    # 收集交互项权重（平均值）
    avg_interactions = {}
    if subjects and hasattr(subjects[0], 'interaction_weights') and subjects[0].interaction_weights:
        # 收集所有被试的交互项权重
        all_interaction_keys = set()
        for subject in subjects:
            if subject.interaction_weights:
                all_interaction_keys.update(subject.interaction_weights.keys())

        # 计算平均值
        for key in all_interaction_keys:
            weights_for_key = [
                s.interaction_weights.get(key, 0.0)
                for s in subjects
                if s.interaction_weights
            ]
            avg_interactions[f"{key[0]},{key[1]}"] = np.mean(weights_for_key)

    # 获取bias（所有被试相同）
    avg_bias = subjects[0].bias if subjects else 0.0

    # 保存扩展格式
    fixed_weights_data = {
        "global": [avg_weights.tolist()],
        "interactions": avg_interactions,
        "bias": avg_bias
    }

    fixed_weights_json = result_dir / "fixed_weights_auto.json"
    with open(fixed_weights_json, 'w', encoding='utf-8') as f:
        json.dump(fixed_weights_data, f, indent=2)

    # 打印和保存模型摘要
    if print_model or save_model_summary:
        _print_and_save_model_summary(
            subjects=subjects,
            result_dir=result_dir,
            config={
                'seed': seed,
                'output_type': output_type,
                'likert_levels': likert_levels,
                'likert_mode': likert_mode,
                'likert_sensitivity': likert_sensitivity,
                'population_mean': population_mean,
                'population_std': population_std,
                'individual_std_percent': individual_std_percent,
                'interaction_pairs': interaction_pairs,
                'interaction_scale': interaction_scale,
                'bias': actual_bias,
                'noise_std': noise_std,
            },
            print_to_console=print_model,
            save_to_file=save_model_summary,
            format=model_summary_format
        )

    print(f"\n[OK] Simulation completed!")
    print(f"  Output directory: {result_dir}")
    print(f"  Files:")
    if output_mode in ["individual", "both"]:
        print(f"    - subject_1.csv ... subject_{n_subjects}.csv")
    if output_mode in ["combined", "both"]:
        print(f"    - combined_results.csv")
    print(f"    - subject_1_spec.json ... subject_{n_subjects}_spec.json (V2 format)")
    print(f"    - subject_1_model.md ... subject_{n_subjects}_model.md (legacy format)")
    print(f"    - cluster_summary.json")
    print(f"    - fixed_weights_auto.json")


def _save_legacy_model_md(subjects, result_dir, seed, n_features):
    """保存兼容旧格式的subject_X_model.md"""
    for i, subject in enumerate(subjects, start=1):
        md_path = result_dir / f"subject_{i}_model.md"

        lines = [
            f"# Model spec for subject_{i}",
            "",
            f"- seed: {seed + i - 1}",
            f"- num_features: {n_features}",
            f"- use_latent: False",
            f"- individual_std: 0.0",  # V2不分别存储，统一在cluster_summary中
            f"- weight_range: 0.5",
            "",
            "## Fixed Weights (deterministic outputs)",
            "",
            "### Observed var 1",
            ""
        ]

        for j, w in enumerate(subject.weights):
            lines.append(f"- x{j}: {w:.5f}")

        lines.append("")

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def _print_and_save_model_summary(
    subjects,
    result_dir,
    config,
    print_to_console=True,
    save_to_file=True,
    format="txt"
):
    """打印和保存模型摘要"""

    # 计算平均权重
    avg_weights = np.mean([s.weights for s in subjects], axis=0)

    lines = [
        "=" * 80,
        "Subject Simulator V2 - Model Summary",
        "=" * 80,
        "",
        "## Configuration",
        f"- Random seed: {config['seed']}",
        f"- Output type: {config['output_type']}",
    ]

    if config['output_type'] == 'likert':
        lines.extend([
            f"- Likert levels: {config['likert_levels']}",
            f"- Likert mode: {config['likert_mode']}",
            f"- Likert sensitivity: {config['likert_sensitivity']}",
        ])

    lines.extend([
        "",
        "## Population Parameters",
        f"- Population mean: {config['population_mean']}",
        f"- Population std: {config['population_std']}",
        f"- Individual std: {config['population_std'] * config['individual_std_percent']:.4f} ({config['individual_std_percent']} * {config['population_std']})",
        f"- Bias: {config['bias']}",
        f"- Noise std: {config['noise_std']}",
        "",
        "## Interaction Effects",
    ])

    if config['interaction_pairs']:
        lines.append(f"- Specified pairs: {len(config['interaction_pairs'])}")
        for idx, (i, j) in enumerate(config['interaction_pairs'], 1):
            lines.append(f"  {idx}. x{i} × x{j}")
    else:
        lines.append("- No interaction pairs")

    lines.append(f"- Interaction scale: {config['interaction_scale']}")
    lines.extend([
        "",
        "## Average Population Weights",
        ""
    ])

    for i, w in enumerate(avg_weights):
        lines.append(f"  x{i}: {w:+.5f}")

    lines.extend([
        "",
        "## Individual Subjects",
        f"See detailed specs: {result_dir}/subject_*_spec.json",
        "See legacy format: {result_dir}/subject_*_model.md",
        "",
        "=" * 80
    ])

    # 打印到控制台
    if print_to_console:
        print()
        for line in lines:
            print(line)
        print()

    # 保存到文件
    if save_to_file:
        if format in ["txt", "both"]:
            txt_path = result_dir / "MODEL_SUMMARY.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"[Saved] Model summary: {txt_path}")

        if format in ["md", "both"]:
            md_path = result_dir / "MODEL_SUMMARY.md"
            # Markdown格式调整
            md_lines = [line.replace("## ", "### ").replace("# ", "## ") for line in lines]
            md_lines[0] = "# " + md_lines[0].strip("=").strip()
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_lines))
            print(f"[Saved] Model summary: {md_path}")


# 兼容性别名
simulate_responses = run


if __name__ == "__main__":
    # 测试代码
    print("Warmup Adapter for Subject Simulator V2")
    print("This module provides drop-in replacement for simulation_runner.py")
    print()
    print("Usage in quick_start.py:")
    print("  from subject_simulator_v2.adapters.warmup_adapter import run as simulate_responses")
