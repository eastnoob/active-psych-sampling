"""
Cluster generator for creating groups of subjects
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import json

from .linear import LinearSubject
from .validators import check_normality, get_distribution_stats


class ClusterGenerator:
    """
    被试集群生成器

    确保：
    1. 群体共性：共享population_weights
    2. 个体差异：individual_deviation ~ N(0, σ_ind)
    3. 正态性保障：响应分布不单调

    示例：
        >>> gen = ClusterGenerator(
        ...     design_space=design_space_np,
        ...     n_subjects=5,
        ...     population_std=0.3,
        ...     individual_std=0.1,
        ...     interaction_pairs=[(3,4), (0,1)],
        ...     ensure_normality=True
        ... )
        >>> cluster = gen.generate_cluster(output_dir="output/cluster_001")
    """

    def __init__(
        self,
        design_space: np.ndarray,
        n_subjects: int = 5,
        population_mean: float = 0.0,
        population_std: float = 0.3,
        individual_std: float = 0.1,
        interaction_pairs: Optional[List[Tuple[int, int]]] = None,
        interaction_scale: float = 0.25,
        bias: float = 0.0,
        noise_std: float = 0.0,
        likert_levels: int = 5,
        likert_sensitivity: float = 2.0,
        ensure_normality: bool = True,
        max_retries: int = 20,
        seed: int = 42
    ):
        """
        初始化集群生成器

        Args:
            design_space: 设计空间 (N, n_features)
            n_subjects: 被试数量
            population_mean: 群体权重均值
            population_std: 群体权重标准差（窄分布）
            individual_std: 个体偏差标准差（更窄）
            interaction_pairs: 交互效应对 [(i,j), ...]
            interaction_scale: 交互效应权重标准差
            bias: 截距
            noise_std: 试次内噪声标准差
            likert_levels: Likert等级数
            likert_sensitivity: Likert转换灵敏度
            ensure_normality: 是否检查正态性
            max_retries: 正态性检查失败时的最大重试次数
            seed: 随机种子
        """
        self.design_space = np.array(design_space)
        self.n_subjects = n_subjects
        self.population_mean = population_mean
        self.population_std = population_std
        self.individual_std = individual_std
        self.interaction_pairs = interaction_pairs or []
        self.interaction_scale = interaction_scale
        self.bias = bias
        self.noise_std = noise_std
        self.likert_levels = likert_levels
        self.likert_sensitivity = likert_sensitivity
        self.ensure_normality = ensure_normality
        self.max_retries = max_retries
        self.seed = seed

        # 初始化随机数生成器
        np.random.seed(self.seed)

        self.n_features = self.design_space.shape[1]

    def generate_cluster(self, output_dir: str) -> Dict[str, Any]:
        """
        生成被试集群并保存

        Args:
            output_dir: 输出目录

        Returns:
            {
                "population_weights": np.ndarray,
                "interaction_weights": dict,
                "subjects": List[LinearSubject],
                "output_dir": Path
            }
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Generating cluster with {self.n_subjects} subjects...")
        print(f"Output directory: {output_path}")

        # 1. 生成群体权重（正态分布）
        population_weights = np.random.normal(
            self.population_mean,
            self.population_std,
            size=self.n_features
        )

        # 2. 生成交互效应权重
        interaction_weights = self._generate_interactions()

        # 3. 为每个被试生成individual weights
        subjects = []
        subject_specs = []

        for i in range(self.n_subjects):
            subject_id = i + 1
            subject_seed = self.seed + i

            subject, spec = self._generate_subject(
                population_weights=population_weights,
                interaction_weights=interaction_weights,
                subject_id=subject_id,
                subject_seed=subject_seed
            )

            subjects.append(subject)
            subject_specs.append(spec)

            # 保存被试参数
            subject.save(output_path / f"subject_{subject_id}_spec.json")
            print(f"  [OK] Subject {subject_id} generated")

        # 4. 生成响应数据并保存CSV
        self._generate_responses(subjects, output_path)

        # 5. 保存集群摘要
        self._save_cluster_summary(
            population_weights=population_weights,
            interaction_weights=interaction_weights,
            subject_specs=subject_specs,
            output_path=output_path
        )

        print(f"\n[OK] Cluster generation completed!")
        print(f"  Files saved in: {output_path}")

        return {
            "population_weights": population_weights,
            "interaction_weights": interaction_weights,
            "subjects": subjects,
            "output_dir": output_path
        }

    def _generate_interactions(self) -> Dict[Tuple[int, int], float]:
        """生成交互效应权重"""
        interaction_weights = {}

        for (i, j) in self.interaction_pairs:
            # 正态分布采样
            weight = np.random.normal(0, self.interaction_scale)
            interaction_weights[(i, j)] = weight

        return interaction_weights

    def _generate_subject(
        self,
        population_weights: np.ndarray,
        interaction_weights: Dict,
        subject_id: int,
        subject_seed: int
    ) -> Tuple[LinearSubject, Dict]:
        """
        生成单个被试，确保正态性

        Returns:
            (subject, spec_dict)
        """
        # 设置被试专用随机种子
        np.random.seed(subject_seed)

        for attempt in range(self.max_retries):
            # 个体权重 = 群体权重 + 个体偏差
            individual_deviation = np.random.normal(
                0, self.individual_std, size=self.n_features
            )
            weights = population_weights + individual_deviation

            # 创建被试
            subject = LinearSubject(
                weights=weights,
                interaction_weights=interaction_weights,
                bias=self.bias,
                noise_std=self.noise_std,
                likert_levels=self.likert_levels,
                likert_sensitivity=self.likert_sensitivity,
                seed=subject_seed
            )

            # 在设计空间上采样评估分布（用于统计或正态性检查）
            responses = [subject(x) for x in self.design_space]

            # 正态性检查
            if not self.ensure_normality:
                break

            validation = check_normality(responses)

            if validation["passed"]:
                break

            if attempt == self.max_retries - 1:
                print(f"  Warning: Subject {subject_id} failed normality check after {self.max_retries} retries")
                print(f"           Reason: {validation['reason']}")
                print(f"           Using population weights (no deviation)")

                # 使用群体权重（无偏差）作为保底
                subject = LinearSubject(
                    weights=population_weights,
                    interaction_weights=interaction_weights,
                    bias=self.bias,
                    noise_std=self.noise_std,
                    likert_levels=self.likert_levels,
                    likert_sensitivity=self.likert_sensitivity,
                    seed=subject_seed
                )
                responses = [subject(x) for x in self.design_space]

        # 生成spec
        stats = get_distribution_stats(responses)
        spec = subject.to_dict()
        spec["subject_id"] = f"subject_{subject_id}"
        spec["response_statistics"] = stats

        return subject, spec

    def _generate_responses(self, subjects: List[LinearSubject], output_path: Path):
        """生成所有被试的响应数据并保存CSV"""
        # 合并数据
        all_data = []

        for subject_id, subject in enumerate(subjects, start=1):
            # 为每个被试生成响应
            responses = [subject(x) for x in self.design_space]

            # 构造DataFrame
            df = pd.DataFrame(self.design_space)
            df['y'] = responses
            df['subject'] = f"subject_{subject_id}"

            # 保存单独文件
            df.to_csv(output_path / f"subject_{subject_id}.csv", index=False)

            all_data.append(df)

        # 合并所有被试数据
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_path / "combined_results.csv", index=False)

    def _save_cluster_summary(
        self,
        population_weights: np.ndarray,
        interaction_weights: Dict,
        subject_specs: List[Dict],
        output_path: Path
    ):
        """保存集群摘要"""
        # 收集所有响应统计
        overall_distribution = {}
        for spec in subject_specs:
            dist = spec["response_statistics"]["distribution"]
            for level, count in dist.items():
                level_str = str(level)
                overall_distribution[level_str] = overall_distribution.get(level_str, 0) + count

        summary = {
            "n_subjects": self.n_subjects,
            "design_space_size": len(self.design_space),
            "n_features": self.n_features,
            "population_weights": population_weights.tolist(),
            "interaction_weights": {
                f"{i},{j}": float(w) for (i, j), w in interaction_weights.items()
            },
            "population_mean": self.population_mean,
            "population_std": self.population_std,
            "individual_std": self.individual_std,
            "interaction_scale": self.interaction_scale,
            "bias": self.bias,
            "noise_std": self.noise_std,
            "likert_levels": self.likert_levels,
            "likert_sensitivity": self.likert_sensitivity,
            "overall_response_distribution": overall_distribution,
            "seed": self.seed,
        }

        with open(output_path / "cluster_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
