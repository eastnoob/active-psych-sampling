"""
单元测试：Core-2a D-optimal采样功能

测试目标：
1. D-optimal方法基本功能
2. 选择的配置数量正确
3. 无重复选择
4. D-efficiency值合理
5. Fallback机制正常工作
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# 添加模块路径
warmup_path = Path(__file__).parent.parent / "extensions" / "warmup_budget_check"
sys.path.insert(0, str(warmup_path))
sys.path.insert(0, str(warmup_path / "core"))

from warmup_sampler import WarmupSampler


class TestCore2aDOptimal:
    """Core-2a D-optimal采样测试套件"""

    @pytest.fixture
    def simple_design_space(self):
        """创建简单的设计空间"""
        # 6因子，每个因子3个水平
        n_factors = 6
        levels = 3

        # 全因子设计
        factor_values = []
        for i in range(n_factors):
            factor_values.append(np.repeat(np.arange(levels), levels**(n_factors-i-1)))

        factor_values = [fv[:levels**n_factors] for fv in factor_values]

        df = pd.DataFrame({
            f"x{i+1}": np.tile(factor_values[i], levels**n_factors // len(factor_values[i]))[:levels**n_factors]
            for i in range(n_factors)
        })

        df = df.head(100)  # 限制到100个配置

        # 保存为临时CSV
        tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(tmp_file.name, index=False)
        tmp_file.close()

        yield tmp_file.name

        # 清理
        os.unlink(tmp_file.name)

    @pytest.fixture
    def mixed_design_space(self):
        """创建混合类型的设计空间"""
        np.random.seed(42)
        n_configs = 50

        df = pd.DataFrame({
            "numeric1": np.random.uniform(0, 10, n_configs),
            "numeric2": np.random.uniform(-5, 5, n_configs),
            "categorical": np.random.choice(["A", "B", "C", "D"], n_configs),
            "boolean": np.random.choice([True, False], n_configs),
            "ordinal": np.random.choice([1, 2, 3, 4, 5], n_configs),
        })

        # 保存为临时CSV
        tmp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv')
        df.to_csv(tmp_file.name, index=False)
        tmp_file.close()

        yield tmp_file.name

        # 清理
        os.unlink(tmp_file.name)

    def test_doptimal_basic_functionality(self, simple_design_space):
        """测试D-optimal基本功能"""
        sampler = WarmupSampler(simple_design_space)

        n_configs = 20
        used_indices = set()

        selected = sampler._select_doptimal_configs(n_configs, used_indices)

        # 验证返回的配置数量
        assert len(selected) == n_configs, f"Expected {n_configs} configs, got {len(selected)}"

        # 验证无重复
        assert len(set(selected)) == len(selected), "Selected configs contain duplicates"

        # 验证索引有效
        for idx in selected:
            assert idx in sampler.design_df.index, f"Invalid index {idx}"

    def test_doptimal_no_overlap_with_used(self, simple_design_space):
        """测试D-optimal不选择已使用的配置"""
        sampler = WarmupSampler(simple_design_space)

        # 标记一些配置为已使用
        used_indices = set(range(10))
        n_configs = 15

        selected = sampler._select_doptimal_configs(n_configs, used_indices)

        # 验证无重叠
        overlap = set(selected) & used_indices
        assert len(overlap) == 0, f"Selected configs overlap with used: {overlap}"

    def test_doptimal_with_limited_candidates(self, simple_design_space):
        """测试候选集不足时的行为"""
        sampler = WarmupSampler(simple_design_space)

        # 标记大部分配置为已使用
        used_indices = set(range(90))  # 只剩10个可用
        n_configs = 20  # 需求20个

        selected = sampler._select_doptimal_configs(n_configs, used_indices)

        # 应该返回所有可用的
        available = len(simple_design_space) - len(used_indices)
        assert len(selected) == available, f"Expected {available} configs, got {len(selected)}"

    def test_normalize_design_to_candidates(self, mixed_design_space):
        """测试设计矩阵标准化功能"""
        sampler = WarmupSampler(mixed_design_space)

        available_indices = list(range(len(mixed_design_space)))
        candidates = sampler._normalize_design_to_candidates(available_indices)

        # 验证形状
        assert candidates.shape[0] == len(mixed_design_space), "Row count mismatch"
        assert candidates.shape[1] > 0, "No features generated"

        # 验证数值范围（应该在[0,1]）
        assert np.all(candidates >= 0), "Candidates contain negative values"
        assert np.all(candidates <= 1), "Candidates contain values > 1"

    def test_stratified_fallback(self, simple_design_space):
        """测试分层采样fallback"""
        sampler = WarmupSampler(simple_design_space)

        n_configs = 20
        used_indices = set()

        selected = sampler._select_stratified_configs(n_configs, used_indices)

        # 验证基本属性
        assert len(selected) == n_configs
        assert len(set(selected)) == len(selected)

        # 验证覆盖度
        coverage = sampler._compute_factor_coverage(selected)
        assert 0 <= coverage <= 1, f"Coverage {coverage} out of range"
        assert coverage > 0.5, f"Coverage {coverage} too low"

    def test_factor_coverage_computation(self, simple_design_space):
        """测试因子覆盖度计算"""
        sampler = WarmupSampler(simple_design_space)

        # 测试空集
        coverage_empty = sampler._compute_factor_coverage([])
        assert coverage_empty == 0.0

        # 测试单个配置
        coverage_single = sampler._compute_factor_coverage([0])
        assert 0 < coverage_single <= 1

        # 测试多个配置（应该增加覆盖度）
        coverage_multi = sampler._compute_factor_coverage(list(range(20)))
        assert coverage_multi > coverage_single

    def test_doptimal_with_categorical_explosion(self):
        """测试大量分类变量的情况"""
        # 创建有大量类别的设计空间
        np.random.seed(42)
        n_configs = 50
        n_categories = 20  # 超过阈值10

        df = pd.DataFrame({
            "cat_many": np.random.choice([f"cat_{i}" for i in range(n_categories)], n_configs),
            "numeric": np.random.uniform(0, 1, n_configs),
        })

        sampler = WarmupSampler(df)

        # 应该能正常处理（使用序数编码）
        available_indices = list(range(len(df)))
        candidates = sampler._normalize_design_to_candidates(available_indices)

        assert candidates.shape[0] == n_configs
        assert candidates.shape[1] == 2  # cat_many(1列) + numeric(1列)

    def test_integration_with_five_step_sampling(self, simple_design_space):
        """集成测试：验证D-optimal在五步采样中的工作"""
        sampler = WarmupSampler(simple_design_space)

        # 模拟预算
        budget = {
            "core1_configs": 8,
            "core2a_configs": 15,
            "core2b_configs": 10,
            "boundary_configs": 12,
            "lhs_configs": 10,
            "skip_interaction": False,
        }

        n_subjects = 3

        # 执行采样（这会调用_select_doptimal_configs）
        samples = sampler.generate_samples(
            n_subjects=n_subjects,
            trials_per_subject=20,
            budget=budget,
            interaction_mode="free",
        )

        # 验证输出
        assert len(samples) == n_subjects
        for subject_id, df in samples.items():
            assert "subject_id" in df.columns
            assert len(df) > 0


class TestCore2aCore2bSeparation:
    """测试Core-2a和Core-2b的正确分离"""

    @pytest.fixture
    def design_space(self):
        """创建测试设计空间"""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            "x1": np.random.uniform(0, 1, n),
            "x2": np.random.uniform(0, 1, n),
            "x3": np.random.choice([0, 1, 2], n),
            "x4": np.random.choice([0, 1, 2], n),
            "x5": np.random.uniform(-1, 1, n),
            "x6": np.random.uniform(-1, 1, n),
        })

    def test_separation_in_five_step(self, design_space, capsys):
        """测试Core-2a和Core-2b在五步采样中的分离"""
        sampler = WarmupSampler(design_space)

        budget = {
            "core1_configs": 8,
            "core2a_configs": 16,
            "core2b_configs": 21,
            "boundary_configs": 10,
            "lhs_configs": 10,
            "skip_interaction": False,
        }

        samples = sampler.generate_samples(
            n_subjects=5,
            trials_per_subject=25,
            budget=budget,
            interaction_mode="hybrid",
            interaction_pairs_to_explore=[(0, 1), (3, 4)],
        )

        # 捕获输出检查日志
        captured = capsys.readouterr()

        # 验证日志中有Core-2a和Core-2b的分离信息
        assert "[Core-2a]" in captured.out, "Missing Core-2a log"
        assert "[Core-2b]" in captured.out, "Missing Core-2b log"
        assert "D-optimal" in captured.out, "Missing D-optimal mention"
        assert "D-efficiency" in captured.out, "Missing D-efficiency output"

    def test_core2a_count_correct(self, design_space):
        """验证Core-2a选择的配置数量正确"""
        sampler = WarmupSampler(design_space)

        n_core2a = 20
        used_indices = set(range(10))

        selected = sampler._select_doptimal_configs(n_core2a, used_indices)

        assert len(selected) == n_core2a


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
