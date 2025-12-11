#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ordinal parameter support in CustomPoolBasedGenerator

测试覆盖：
1. 直接指定 values 的 ordinal 参数
2. min_value + max_value + step 配置
3. min_value + max_value + num_levels 配置
4. 混合 categorical 和 ordinal 参数
5. Pool 生成的笛卡尔积正确性
"""

import pytest
import torch
import tempfile
from pathlib import Path
import sys

# Add extensions to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestOrdinalPoolGeneration:
    """测试 ordinal 参数的 pool 生成"""

    def test_ordinal_direct_values(self, tmp_path):
        """测试直接指定 values 的 ordinal 参数"""
        from aepsych.config import Config
        from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator

        # 创建测试配置文件
        config_content = """
[common]
parnames = [ceiling_height]
stimuli_per_trial = 1
outcome_types = binary

[ceiling_height]
par_type = custom_ordinal_mono
values = [2.0, 2.5, 3.5]

[CustomPoolBasedGenerator]
acqf = qUpperConfidenceBound
acqf_kwargs = {}
"""
        config_file = tmp_path / "test_ordinal_values.ini"
        config_file.write_text(config_content)

        config = Config(config_fnames=[str(config_file)])

        # 生成 pool
        pool_points, categorical_mappings = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 验证 pool 形状和内容
        assert pool_points.shape == (3, 1), f"Expected shape (3, 1), got {pool_points.shape}"
        assert torch.allclose(pool_points, torch.tensor([[2.0], [2.5], [3.5]]), atol=1e-5)

        # Ordinal 参数不应该有 categorical mappings
        assert len(categorical_mappings) == 0

    def test_ordinal_min_max_step(self, tmp_path):
        """测试 min_value + max_value + step 配置"""
        from aepsych.config import Config
        from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator

        config_content = """
[common]
parnames = [temperature]
stimuli_per_trial = 1
outcome_types = binary

[temperature]
par_type = custom_ordinal
min_value = 16
max_value = 24
step = 2

[CustomPoolBasedGenerator]
acqf = qUpperConfidenceBound
acqf_kwargs = {}
"""
        config_file = tmp_path / "test_ordinal_step.ini"
        config_file.write_text(config_content)

        config = Config(config_fnames=[str(config_file)])
        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 应生成 [16, 18, 20, 22, 24]
        expected = torch.tensor([[16.], [18.], [20.], [22.], [24.]])
        assert pool_points.shape == (5, 1)
        assert torch.allclose(pool_points, expected, atol=1e-5)

    def test_ordinal_min_max_num_levels(self, tmp_path):
        """测试 min_value + max_value + num_levels 配置"""
        from aepsych.config import Config
        from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator

        config_content = """
[common]
parnames = [brightness]
stimuli_per_trial = 1
outcome_types = binary

[brightness]
par_type = ordinal
min_value = 0.0
max_value = 1.0
num_levels = 5

[CustomPoolBasedGenerator]
acqf = qUpperConfidenceBound
acqf_kwargs = {}
"""
        config_file = tmp_path / "test_ordinal_levels.ini"
        config_file.write_text(config_content)

        config = Config(config_fnames=[str(config_file)])
        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 应生成 [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
        assert pool_points.shape == (5, 1)
        assert torch.allclose(pool_points, expected, atol=1e-5)

    def test_mixed_categorical_ordinal(self, tmp_path):
        """测试混合 categorical 和 ordinal 参数"""
        from aepsych.config import Config
        from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator

        config_content = """
[common]
parnames = [height, color]
stimuli_per_trial = 1
outcome_types = binary

[height]
par_type = custom_ordinal
min_value = 2.0
max_value = 3.0
num_levels = 3

[color]
par_type = categorical
choices = ['red', 'blue']

[CustomPoolBasedGenerator]
acqf = qUpperConfidenceBound
acqf_kwargs = {}
"""
        config_file = tmp_path / "test_mixed.ini"
        config_file.write_text(config_content)

        config = Config(config_fnames=[str(config_file)])
        pool_points, categorical_mappings = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 笛卡尔积: 3 heights × 2 colors = 6 combinations
        # Height: [2.0, 2.5, 3.0]
        # Color: [0, 1] (indices for categorical)
        assert pool_points.shape == (6, 2), f"Expected (6, 2), got {pool_points.shape}"

        # 验证第一维是 ordinal 值 (2.0, 2.5, 3.0)
        unique_heights = torch.unique(pool_points[:, 0])
        assert torch.allclose(unique_heights, torch.tensor([2.0, 2.5, 3.0]), atol=1e-5)

        # 验证第二维是 categorical indices (0, 1)
        unique_colors = torch.unique(pool_points[:, 1])
        assert torch.allclose(unique_colors, torch.tensor([0.0, 1.0]), atol=1e-5)

        # Categorical 应该有 mapping，ordinal 不应该有
        assert 1 in categorical_mappings  # color (param_idx=1)
        assert 0 not in categorical_mappings  # height (ordinal, no mapping)

    def test_ordinal_minimum_values_error(self, tmp_path):
        """测试 ordinal 至少需要2个值的验证"""
        from aepsych.config import Config
        from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator

        config_content = """
[common]
parnames = [single_value]
stimuli_per_trial = 1
outcome_types = binary

[single_value]
par_type = ordinal
values = [5.0]

[CustomPoolBasedGenerator]
acqf = qUpperConfidenceBound
acqf_kwargs = {}
"""
        config_file = tmp_path / "test_single_value.ini"
        config_file.write_text(config_content)

        config = Config(config_fnames=[str(config_file)])

        # 应抛出错误
        with pytest.raises(ValueError, match="至少需要2个值"):
            CustomPoolBasedGenerator._generate_pool_from_config(config)

    def test_ordinal_missing_config_error(self, tmp_path):
        """测试缺少必需配置的错误"""
        from aepsych.config import Config
        from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator

        config_content = """
[common]
parnames = [incomplete]
stimuli_per_trial = 1
outcome_types = binary

[incomplete]
par_type = ordinal
# 没有提供 values 或 min/max 配置

[CustomPoolBasedGenerator]
acqf = qUpperConfidenceBound
acqf_kwargs = {}
"""
        config_file = tmp_path / "test_incomplete.ini"
        config_file.write_text(config_content)

        config = Config(config_fnames=[str(config_file)])

        # 应抛出错误
        with pytest.raises(ValueError, match="必须指定以下之一"):
            CustomPoolBasedGenerator._generate_pool_from_config(config)

    def test_three_ordinal_parameters_cartesian_product(self, tmp_path):
        """测试3个 ordinal 参数的笛卡尔积"""
        from aepsych.config import Config
        from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator

        config_content = """
[common]
parnames = [height, width, depth]
stimuli_per_trial = 1
outcome_types = binary

[height]
par_type = ordinal
values = [1.0, 2.0]

[width]
par_type = ordinal
values = [3.0, 4.0, 5.0]

[depth]
par_type = ordinal
values = [6.0, 7.0]

[CustomPoolBasedGenerator]
acqf = qUpperConfidenceBound
acqf_kwargs = {}
"""
        config_file = tmp_path / "test_three_params.ini"
        config_file.write_text(config_content)

        config = Config(config_fnames=[str(config_file)])
        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 笛卡尔积: 2 × 3 × 2 = 12 combinations
        assert pool_points.shape == (12, 3)

        # 验证每个维度的 unique 值
        assert torch.allclose(torch.unique(pool_points[:, 0]), torch.tensor([1.0, 2.0]), atol=1e-5)
        assert torch.allclose(torch.unique(pool_points[:, 1]), torch.tensor([3.0, 4.0, 5.0]), atol=1e-5)
        assert torch.allclose(torch.unique(pool_points[:, 2]), torch.tensor([6.0, 7.0]), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
