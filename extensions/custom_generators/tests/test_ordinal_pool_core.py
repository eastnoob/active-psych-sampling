#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 ordinal 参数池生成测试

直接测试 _generate_pool_from_config 的核心逻辑，
不依赖完整的 Config 校验
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add extensions to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from extensions.custom_generators.custom_pool_based_generator import CustomPoolBasedGenerator


class MockConfig:
    """模拟 Config 对象用于测试"""

    def __init__(self, sections):
        self.sections = sections

    def get(self, section, option, fallback=None):
        if section in self.sections and option in self.sections[section]:
            return self.sections[section][option]
        return fallback

    def has_option(self, section, option):
        return section in self.sections and option in self.sections[section]


class TestOrdinalPoolGenerationCore:
    """核心 ordinal pool 生成测试（不依赖完整 Config）"""

    def test_ordinal_direct_values(self):
        """测试直接指定 values 的 ordinal 参数"""
        config = MockConfig({
            "common": {
                "parnames": '["ceiling_height"]'
            },
            "ceiling_height": {
                "par_type": "custom_ordinal_mono",
                "values": "[2.0, 2.5, 3.5]"
            }
        })

        pool_points, categorical_mappings = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 验证 pool 形状和内容
        assert pool_points.shape == (3, 1), f"Expected shape (3, 1), got {pool_points.shape}"
        assert torch.allclose(pool_points, torch.tensor([[2.0], [2.5], [3.5]], dtype=torch.float32), atol=1e-5)

        # Ordinal 参数不应该有 categorical mappings
        assert len(categorical_mappings) == 0

    def test_ordinal_min_max_step(self):
        """测试 min_value + max_value + step 配置"""
        config = MockConfig({
            "common": {
                "parnames": '["temperature"]'
            },
            "temperature": {
                "par_type": "custom_ordinal",
                "min_value": "16",
                "max_value": "24",
                "step": "2"
            }
        })

        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 应生成 [16, 18, 20, 22, 24]
        expected = torch.tensor([[16.], [18.], [20.], [22.], [24.]], dtype=torch.float32)
        assert pool_points.shape == (5, 1)
        assert torch.allclose(pool_points, expected, atol=1e-5)

    def test_ordinal_min_max_num_levels(self):
        """测试 min_value + max_value + num_levels 配置"""
        config = MockConfig({
            "common": {
                "parnames": '["brightness"]'
            },
            "brightness": {
                "par_type": "ordinal",
                "min_value": "0.0",
                "max_value": "1.0",
                "num_levels": "5"
            }
        })

        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 应生成 [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float32)
        assert pool_points.shape == (5, 1)
        assert torch.allclose(pool_points, expected, atol=1e-5)

    def test_mixed_categorical_ordinal(self):
        """测试混合 categorical 和 ordinal 参数"""
        config = MockConfig({
            "common": {
                "parnames": '["height", "color"]'
            },
            "height": {
                "par_type": "custom_ordinal",
                "min_value": "2.0",
                "max_value": "3.0",
                "num_levels": "3"
            },
            "color": {
                "par_type": "categorical",
                "choices": "['red', 'blue']"
            }
        })

        pool_points, categorical_mappings = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 笛卡尔积: 3 heights × 2 colors = 6 combinations
        # Height: [2.0, 2.5, 3.0]
        # Color: [0, 1] (indices for categorical)
        assert pool_points.shape == (6, 2), f"Expected (6, 2), got {pool_points.shape}"

        # 验证第一维是 ordinal 值 (2.0, 2.5, 3.0)
        unique_heights = torch.unique(pool_points[:, 0])
        assert len(unique_heights) == 3
        assert torch.allclose(unique_heights, torch.tensor([2.0, 2.5, 3.0], dtype=torch.float32), atol=1e-5)

        # 验证第二维是 categorical indices (0, 1)
        unique_colors = torch.unique(pool_points[:, 1])
        assert torch.allclose(unique_colors, torch.tensor([0.0, 1.0], dtype=torch.float32), atol=1e-5)

        # Categorical 应该有 mapping（但是 string categorical 不应该）
        # 在这个测试中，color 是 string 类型，所以不会有 mapping
        assert 0 not in categorical_mappings  # height (ordinal, no mapping)

    def test_ordinal_minimum_values_error(self):
        """测试 ordinal 至少需要2个值的验证"""
        config = MockConfig({
            "common": {
                "parnames": '["single_value"]'
            },
            "single_value": {
                "par_type": "ordinal",
                "values": "[5.0]"
            }
        })

        # 应抛出错误
        with pytest.raises(ValueError, match="至少需要2个值"):
            CustomPoolBasedGenerator._generate_pool_from_config(config)

    def test_ordinal_missing_config_error(self):
        """测试缺少必需配置的错误"""
        config = MockConfig({
            "common": {
                "parnames": '["incomplete"]'
            },
            "incomplete": {
                "par_type": "ordinal"
                # 没有提供 values 或 min/max 配置
            }
        })

        # 应抛出错误
        with pytest.raises(ValueError, match="必须指定以下之一"):
            CustomPoolBasedGenerator._generate_pool_from_config(config)

    def test_three_ordinal_parameters(self):
        """测试3个 ordinal 参数的笛卡尔积"""
        config = MockConfig({
            "common": {
                "parnames": '["height", "width", "depth"]'
            },
            "height": {
                "par_type": "ordinal",
                "values": "[1.0, 2.0]"
            },
            "width": {
                "par_type": "ordinal",
                "values": "[3.0, 4.0, 5.0]"
            },
            "depth": {
                "par_type": "ordinal",
                "values": "[6.0, 7.0]"
            }
        })

        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 笛卡尔积: 2 × 3 × 2 = 12 combinations
        assert pool_points.shape == (12, 3)

        # 验证每个维度的 unique 值
        assert torch.allclose(torch.unique(pool_points[:, 0]), torch.tensor([1.0, 2.0], dtype=torch.float32), atol=1e-5)
        assert torch.allclose(torch.unique(pool_points[:, 1]), torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32), atol=1e-5)
        assert torch.allclose(torch.unique(pool_points[:, 2]), torch.tensor([6.0, 7.0], dtype=torch.float32), atol=1e-5)

    def test_ordinal_levels_string_labels(self):
        """测试 levels（字符串标签）配置"""
        config = MockConfig({
            "common": {
                "parnames": '["agreement"]'
            },
            "agreement": {
                "par_type": "ordinal",
                "levels": "strongly_disagree, disagree, neutral, agree, strongly_agree"
            }
        })

        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 应生成整数序列 [0, 1, 2, 3, 4]
        expected = torch.tensor([[0.], [1.], [2.], [3.], [4.]], dtype=torch.float32)
        assert pool_points.shape == (5, 1)
        assert torch.allclose(pool_points, expected, atol=1e-5)

    def test_ordinal_auto_sorting(self):
        """测试未排序的 values 会被自动排序"""
        config = MockConfig({
            "common": {
                "parnames": '["unsorted"]'
            },
            "unsorted": {
                "par_type": "ordinal",
                "values": "[3.5, 1.0, 2.5]"  # 未排序
            }
        })

        pool_points, _ = CustomPoolBasedGenerator._generate_pool_from_config(config)

        # 应被自动排序为 [1.0, 2.5, 3.5]
        expected = torch.tensor([[1.0], [2.5], [3.5]], dtype=torch.float32)
        assert pool_points.shape == (3, 1)
        assert torch.allclose(pool_points, expected, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
