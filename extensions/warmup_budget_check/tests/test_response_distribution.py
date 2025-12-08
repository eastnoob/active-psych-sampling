#!/usr/bin/env python3
"""
测试响应分布 - 使用实际设计空间验证模拟被试的响应分布

目的:
1. 使用实际设计空间 (i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv)
2. 生成多个模拟被试的响应
3. 分析响应分布是否符合预期 (正态性、偏移等)
4. 可视化分布并生成统计报告
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from collections import Counter

# 添加项目路径
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "tools"))

# 导入新的v2模拟被试模块
try:
    from subject_simulator_v2 import LinearSubject, ClusterGenerator
except ImportError as e:
    print(f"[错误] 无法导入 subject_simulator_v2: {e}")
    sys.exit(1)


def load_design_space(csv_path: str) -> tuple[pd.DataFrame, list[str]]:
    """加载设计空间并提取因子列"""
    df = pd.read_csv(csv_path)
    print(f"\n� 设计空间: {Path(csv_path).name}")
    print(f"   总行数: {len(df)}")
    print(f"   列: {list(df.columns)}")

    # 智能识别 x1, x2, ... xN 格式的因子列
    import re
    factor_cols = [c for c in df.columns if re.match(r'^x\d+', c)]

    if not factor_cols:
        print("[警告] 未找到 x1, x2... 格式的列，使用所有列")
        factor_cols = list(df.columns)
    else:
        excluded = set(df.columns) - set(factor_cols)
        if excluded:
            print(f"   ✅ 智能过滤: 使用 {len(factor_cols)} 个因子列")
            print(f"   ❌ 排除的列: {', '.join(sorted(excluded))}")

    return df, factor_cols


def encode_design_space(df: pd.DataFrame, factor_cols: list[str]) -> tuple[np.ndarray, dict]:
    """编码设计空间 (分类变量 -> 数值)"""
    df_factors = df[factor_cols].copy()
    encodings = {}

    for col in df_factors.columns:
        if df_factors[col].dtype == 'object':
            unique_vals = sorted(df_factors[col].dropna().unique())
            mapping = {v: i for i, v in enumerate(unique_vals)}
            encodings[col] = mapping
            df_factors[col] = df_factors[col].map(mapping)
        elif df_factors[col].dtype == 'bool':
            df_factors[col] = df_factors[col].astype(int)

    X = df_factors.values.astype(float)
    return X, encodings


def simulate_subjects(
    X: np.ndarray,
    n_subjects: int = 10,
    seed: int = 42,
    population_mean: float = 0.0,
    population_std: float = 0.25,
    individual_std_percent: float = 0.5,
    likert_levels: int = 5,
    likert_mode: str = "tanh",
    likert_sensitivity: float = 2.0,
    interaction_pairs: list = None,
    interaction_scale: float = 0.25,
) -> tuple[np.ndarray, list]:
    """模拟多个被试的响应 (使用v2 LinearSubject)

    返回:
        responses: (n_subjects, n_trials) 响应矩阵
        subjects: 被试对象列表
    """
    n_trials = X.shape[0]
    n_features = X.shape[1]

    print(f"\n�� 模拟 {n_subjects} 个被试...")
    print(f"   试次数: {n_trials}")
    print(f"   特征数: {n_features}")
    print(f"   群体分布: N({population_mean}, {population_std}²)")
    print(f"   个体偏差: {individual_std_percent} × {population_std} = {individual_std_percent * population_std}")
    print(f"   Likert: {likert_levels}级, mode={likert_mode}, sensitivity={likert_sensitivity}")

    # 生成群体权重 (共享)
    np.random.seed(seed)
    population_weights = np.random.normal(
        population_mean, population_std, size=(n_features,)
    )

    # 构建交互权重字典
    interaction_weights_dict = None
    if interaction_pairs:
        n_interactions = len(interaction_pairs)
        interaction_weights_array = np.random.normal(
            0.0, interaction_scale, size=(n_interactions,)
        )
        # 转换为 {(i,j): weight} 字典格式
        interaction_weights_dict = {
            tuple(pair): float(weight)
            for pair, weight in zip(interaction_pairs, interaction_weights_array)
        }
        print(f"   交互项: {len(interaction_pairs)} 对, scale={interaction_scale}")

    # 创建被试集群生成器
    # 注意: ClusterGenerator需要design_space参数,我们这里简化处理
    # 直接生成个体权重
    subjects = []
    responses = np.zeros((n_subjects, n_trials))

    for subj_idx in range(n_subjects):
        # 为每个被试生成个体权重
        np.random.seed(seed + subj_idx)
        individual_weights = np.random.normal(
            population_mean,
            np.sqrt(population_std**2 + (population_std * individual_std_percent)**2),
            size=(n_features,)
        )

        # 创建LinearSubject
        subject = LinearSubject(
            weights=individual_weights,
            interaction_weights=interaction_weights_dict,
            bias=0.0,
            noise_std=0.0,  # 试次内无额外噪声
            likert_levels=likert_levels,
            likert_sensitivity=likert_sensitivity,
            seed=seed + subj_idx,
        )
        subjects.append(subject)

        # 对每个试次生成响应
        for trial_idx in range(n_trials):
            x = X[trial_idx, :]
            y = subject(x)
            responses[subj_idx, trial_idx] = y

    print(f"   ✅ 模拟完成")
    return responses, subjects


def analyze_distribution(responses: np.ndarray, likert_levels: int = 5) -> dict:
    """分析响应分布统计特性"""
    all_responses = responses.flatten()

    # 基础统计
    stats = {
        "mean": float(np.mean(all_responses)),
        "std": float(np.std(all_responses)),
        "median": float(np.median(all_responses)),
        "min": float(np.min(all_responses)),
        "max": float(np.max(all_responses)),
        "skewness": float(_skewness(all_responses)),
        "kurtosis": float(_kurtosis(all_responses)),
    }

    # Likert分布
    if likert_levels > 0:
        counter = Counter(all_responses)
        likert_dist = {int(k): int(v) for k, v in sorted(counter.items())}
        stats["likert_distribution"] = likert_dist

        # 计算百分比
        total = len(all_responses)
        likert_percent = {k: v/total*100 for k, v in likert_dist.items()}
        stats["likert_percent"] = {k: round(v, 2) for k, v in likert_percent.items()}

    # 被试间差异
    subject_means = np.mean(responses, axis=1)
    stats["between_subject_std"] = float(np.std(subject_means))
    stats["between_subject_range"] = float(np.max(subject_means) - np.min(subject_means))

    return stats


def _skewness(x: np.ndarray) -> float:
    """计算偏度"""
    x = x - np.mean(x)
    m3 = np.mean(x**3)
    m2 = np.mean(x**2)
    return m3 / (m2**1.5 + 1e-10)


def _kurtosis(x: np.ndarray) -> float:
    """计算峰度 (excess kurtosis)"""
    x = x - np.mean(x)
    m4 = np.mean(x**4)
    m2 = np.mean(x**2)
    return m4 / (m2**2 + 1e-10) - 3


def print_report(stats: dict, output_path: Path = None):
    """打印分析报告"""
    print("\n" + "="*80)
    print("� 响应分布分析报告")
    print("="*80)

    print("\n� 基础统计:")
    print(f"   均值:   {stats['mean']:.3f}")
    print(f"   标准差: {stats['std']:.3f}")
    print(f"   中位数: {stats['median']:.3f}")
    print(f"   范围:   [{stats['min']:.1f}, {stats['max']:.1f}]")

    print("\n� 分布形状:")
    print(f"   偏度 (Skewness): {stats['skewness']:.3f}")
    skew_interp = "左偏" if stats['skewness'] < -0.5 else ("右偏" if stats['skewness'] > 0.5 else "对称")
    print(f"      → {skew_interp}分布")

    print(f"   峰度 (Kurtosis):  {stats['kurtosis']:.3f}")
    kurt_interp = "厚尾" if stats['kurtosis'] > 0 else "薄尾"
    print(f"      → {kurt_interp}分布")

    if "likert_distribution" in stats:
        print("\n� Likert分布:")
        for level in sorted(stats["likert_distribution"].keys()):
            count = stats["likert_distribution"][level]
            percent = stats["likert_percent"][level]
            bar_length = int(percent / 2)  # 每个#代表2%
            bar = "█" * bar_length
            print(f"   {level}: {bar} {percent:.1f}% (n={count})")

    print("\n� 被试间差异:")
    print(f"   被试均值的标准差: {stats['between_subject_std']:.3f}")
    print(f"   被试均值的范围:   {stats['between_subject_range']:.3f}")

    print("\n✅ 分布评估:")
    # 评估正态性
    is_symmetric = abs(stats['skewness']) < 0.5
    is_normal_kurt = abs(stats['kurtosis']) < 1.0
    print(f"   对称性: {'✅ 通过' if is_symmetric else '❌ 偏斜'}")
    print(f"   峰度正常: {'✅ 通过' if is_normal_kurt else '❌ 异常'}")

    # 评估Likert分布合理性 (中间值应该多，极端值少)
    if "likert_percent" in stats:
        middle = stats["likert_percent"].get(3, 0)  # 假设5级量表
        extremes = stats["likert_percent"].get(1, 0) + stats["likert_percent"].get(5, 0)
        is_reasonable = middle > extremes
        print(f"   Likert合理性: {'✅ 中间值多于极端值' if is_reasonable else '❌ 极端值过多'}")

    print("\n" + "="*80)

    # 保存JSON
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "distribution_stats.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n� 统计结果已保存: {json_path}")


def main():
    """主函数"""
    # 配置
    design_csv = str(Path(__file__).resolve().parents[3] / "data" / "i9csy65bljq14ovww2v91-6532622b_JBmIu2QSKA.csv")
    output_dir = Path(__file__).parent / "output" / "distribution_test"

    config = {
        "n_subjects": 20,  # 测试20个被试
        "seed": 42,
        "population_mean": 0.0,
        "population_std": 0.25,
        "individual_std_percent": 0.5,
        "likert_levels": 5,
        "likert_mode": "tanh",
        "likert_sensitivity": 2.0,
        "interaction_pairs": [(3, 4), (0, 1)],
        "interaction_scale": 0.25,
    }

    print("\n" + "="*80)
    print("[TEST] 响应分布测试")
    print("="*80)

    # 1. 加载设计空间
    df, factor_cols = load_design_space(design_csv)

    # 2. 编码
    X, encodings = encode_design_space(df, factor_cols)
    if encodings:
        print(f"\n[ENCODE] 编码的列: {list(encodings.keys())}")

    # 3. 模拟被试
    responses, subjects = simulate_subjects(X, **config)

    # 4. 分析分布
    stats = analyze_distribution(responses, config["likert_levels"])

    # 5. 打印报告
    print_report(stats, output_dir)

    # 6. 保存响应数据样本
    sample_df = df[factor_cols].head(10).copy()
    for i in range(min(5, config["n_subjects"])):
        sample_df[f"subject_{i+1}"] = responses[i, :10]

    sample_csv = output_dir / "sample_responses.csv"
    sample_df.to_csv(sample_csv, index=False)
    print(f"[SAVE] 响应样本已保存: {sample_csv}")

    print("\n[DONE] 测试完成!\n")


if __name__ == "__main__":
    main()
