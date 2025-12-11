#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for CustomOrdinal Transform

测试覆盖：
1. 规范化映射正确性
2. Transform/untransform 往返一致性
3. 边界变换
4. 配置解析
5. 边界情况处理
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# 添加 extensions 到 path
project_root = Path(__file__).parent.parent
ext_path = project_root / "extensions" / "dynamic_eur_acquisition"
if str(ext_path) not in sys.path:
    sys.path.insert(0, str(ext_path))

from transforms.ops.custom_ordinal import CustomOrdinal


class TestCustomOrdinalBasic:
    """基础功能测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [2.0, 2.5, 3.5]}
        )

        assert ordinal.indices == [0]
        assert 0 in ordinal.values
        assert len(ordinal.values[0]) == 3

    def test_initialization_with_level_names(self):
        """测试带标签初始化"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [1, 2, 3]},
            level_names={0: ["low", "medium", "high"]}
        )

        assert ordinal.level_names[0] == ["low", "medium", "high"]

    def test_values_sorting(self):
        """测试值自动排序"""
        with pytest.warns(UserWarning, match="not sorted"):
            ordinal = CustomOrdinal(
                indices=[0],
                values={0: [3.5, 2.0, 2.5]}  # 未排序
            )

        # 应该被自动排序
        assert ordinal.values[0] == [2.0, 2.5, 3.5]

    def test_minimum_values(self):
        """测试最少值要求（至少2个）"""
        with pytest.raises(ValueError, match="at least 2 values"):
            CustomOrdinal(
                indices=[0],
                values={0: [1.0]}  # 只有1个值
            )


class TestNormalizedMappings:
    """规范化映射测试"""

    def test_normalized_values_range(self):
        """测试规范化值在 [0, 1] 范围"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [2.0, 2.5, 3.5]}
        )

        norm_vals = ordinal.normalized_values[0]
        assert np.min(norm_vals) == pytest.approx(0.0)
        assert np.max(norm_vals) == pytest.approx(1.0)

    def test_normalized_spacing_preservation(self):
        """测试间距比例保留"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [2.0, 2.5, 3.5]}  # 间距：0.5, 1.0
        )

        norm_vals = ordinal.normalized_values[0]

        # 计算规范化后的间距
        spacing_norm = np.diff(norm_vals)

        # 物理间距比例：0.5:1.0 = 1:2
        # 规范化后应保持此比例
        ratio = spacing_norm[1] / spacing_norm[0]
        assert ratio == pytest.approx(2.0, rel=1e-6)

    def test_equal_spacing(self):
        """测试等间距序列"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [10, 20, 30, 40, 50]}
        )

        norm_vals = ordinal.normalized_values[0]
        spacing = np.diff(norm_vals)

        # 等间距序列，规范化后间距应相等
        assert np.allclose(spacing, spacing[0])

    def test_same_values(self):
        """测试所有值相同的情况"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [5.0, 5.0, 5.0]}
        )

        norm_vals = ordinal.normalized_values[0]

        # 所有值相同，应归一化为 0
        assert np.allclose(norm_vals, 0.0)


class TestTransformMethods:
    """Transform/untransform 方法测试"""

    def test_transform_basic(self):
        """测试基础 transform"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [2.0, 2.5, 3.5]}
        )

        # 物理值
        X_phys = torch.tensor([[2.0], [2.5], [3.5]], dtype=torch.float32)

        # Transform
        X_norm = ordinal.transform(X_phys)

        # 预期规范化值
        expected = torch.tensor([[0.0], [0.333333], [1.0]], dtype=torch.float32)

        assert torch.allclose(X_norm, expected, atol=1e-5)

    def test_untransform_basic(self):
        """测试基础 untransform"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [2.0, 2.5, 3.5]}
        )

        # 规范化值
        X_norm = torch.tensor([[0.0], [0.333333], [1.0]], dtype=torch.float32)

        # Untransform
        X_phys = ordinal.untransform(X_norm)

        # 预期物理值
        expected = torch.tensor([[2.0], [2.5], [3.5]], dtype=torch.float32)

        assert torch.allclose(X_phys, expected, atol=1e-5)

    def test_roundtrip_consistency(self):
        """测试往返一致性：transform → untransform → 原值"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [10.0, 15.0, 25.0, 40.0]}
        )

        # 原始物理值
        X_original = torch.tensor(
            [[10.0], [15.0], [25.0], [40.0]],
            dtype=torch.float32
        )

        # Transform → Untransform
        X_norm = ordinal.transform(X_original)
        X_reconstructed = ordinal.untransform(X_norm)

        assert torch.allclose(X_original, X_reconstructed, atol=1e-5)

    def test_nearest_neighbor_matching(self):
        """测试最近邻匹配（非精确值）"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [1.0, 2.0, 3.0]}
        )

        # 非精确物理值（应匹配到最近值）
        X_approx = torch.tensor([[1.1], [1.9], [2.6]], dtype=torch.float32)

        with pytest.warns(UserWarning, match="not in ordinal values"):
            X_norm = ordinal.transform(X_approx)

        # 验证匹配到最近值
        expected_phys = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
        expected_norm = ordinal.transform(expected_phys)

        assert torch.allclose(X_norm, expected_norm, atol=1e-5)

    def test_batch_transform(self):
        """测试批量 transform"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [0.0, 0.5, 1.0]}
        )

        # 批量数据
        X_batch = torch.tensor(
            [[0.0], [0.5], [1.0], [0.0], [1.0]],
            dtype=torch.float32
        )

        X_norm = ordinal.transform(X_batch)

        assert X_norm.shape == X_batch.shape
        assert torch.min(X_norm) >= 0.0
        assert torch.max(X_norm) <= 1.0


class TestBoundsTransform:
    """边界变换测试"""

    def test_bounds_transform_basic(self):
        """测试基础边界变换"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [2.0, 3.0, 4.0]}
        )

        # 物理边界 [lb, ub]
        bounds = torch.tensor([[2.0], [4.0]], dtype=torch.float32)

        # Transform
        bounds_norm = ordinal.transform_bounds(bounds, epsilon=1e-6)

        # 规范化边界应接近 [-eps, 1.0+eps]
        assert bounds_norm[0, 0] == pytest.approx(-1e-6, abs=1e-7)
        assert bounds_norm[1, 0] == pytest.approx(1.0 + 1e-6, abs=1e-7)

    def test_bounds_transform_custom_epsilon(self):
        """测试自定义 epsilon"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [0.0, 1.0]}
        )

        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
        bounds_norm = ordinal.transform_bounds(bounds, epsilon=0.1)

        assert bounds_norm[0, 0] == pytest.approx(-0.1, abs=1e-7)
        assert bounds_norm[1, 0] == pytest.approx(1.1, abs=1e-7)


class TestConfigParsing:
    """配置解析测试"""

    def test_direct_values(self):
        """测试直接指定 values"""
        config = None  # Mock
        options = {"values": [2.0, 2.5, 3.5]}

        config_dict = CustomOrdinal.get_config_options(config, "height", options)

        assert config_dict["indices"] == [0]
        assert 0 in config_dict["values"]
        assert config_dict["values"][0] == [2.0, 2.5, 3.5]

    def test_min_max_step(self):
        """测试 min_value + max_value + step"""
        options = {
            "min_value": 10.0,
            "max_value": 20.0,
            "step": 2.5
        }

        config_dict = CustomOrdinal.get_config_options(None, "param", options)

        values = config_dict["values"][0]

        # 应生成 [10.0, 12.5, 15.0, 17.5, 20.0]
        expected = [10.0, 12.5, 15.0, 17.5, 20.0]
        assert np.allclose(values, expected, atol=1e-6)

    def test_min_max_num_levels(self):
        """测试 min_value + max_value + num_levels"""
        options = {
            "min_value": 0.0,
            "max_value": 1.0,
            "num_levels": 5
        }

        config_dict = CustomOrdinal.get_config_options(None, "param", options)

        values = config_dict["values"][0]

        # 应生成 [0.0, 0.25, 0.5, 0.75, 1.0]
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]
        assert np.allclose(values, expected, atol=1e-6)

    def test_levels_string(self):
        """测试 levels（字符串标签）"""
        options = {
            "levels": "low, medium, high"
        }

        config_dict = CustomOrdinal.get_config_options(None, "param", options)

        values = config_dict["values"][0]
        level_names = config_dict["level_names"][0]

        # 应生成整数序列
        assert values == [0, 1, 2]
        assert level_names == ["low", "medium", "high"]

    def test_levels_list(self):
        """测试 levels（列表）"""
        options = {
            "levels": ["strongly_disagree", "disagree", "neutral", "agree", "strongly_agree"]
        }

        config_dict = CustomOrdinal.get_config_options(None, "param", options)

        values = config_dict["values"][0]
        level_names = config_dict["level_names"][0]

        assert values == [0, 1, 2, 3, 4]
        assert len(level_names) == 5

    def test_missing_required_options(self):
        """测试缺少必需选项"""
        with pytest.raises(ValueError, match="Must specify one of"):
            CustomOrdinal.get_config_options(None, "param", {})


class TestMultipleDimensions:
    """多维度测试"""

    def test_two_dimensions(self):
        """测试两个 ordinal 维度"""
        ordinal = CustomOrdinal(
            indices=[0, 1],
            values={
                0: [2.0, 3.0, 4.0],
                1: [10.0, 15.0, 20.0]
            }
        )

        X_phys = torch.tensor(
            [[2.0, 10.0], [3.0, 15.0], [4.0, 20.0]],
            dtype=torch.float32
        )

        X_norm = ordinal.transform(X_phys)

        # 第一维：[0.0, 0.5, 1.0]
        # 第二维：[0.0, 0.5, 1.0]
        expected = torch.tensor(
            [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
            dtype=torch.float32
        )

        assert torch.allclose(X_norm, expected, atol=1e-5)

    def test_mixed_ordinal_continuous(self):
        """测试 ordinal + 连续混合"""
        ordinal = CustomOrdinal(
            indices=[0],  # 只有第一维是 ordinal
            values={0: [1.0, 2.0, 3.0]}
        )

        # [ordinal, continuous]
        X_mixed = torch.tensor(
            [[1.0, 0.5], [2.0, 0.7], [3.0, 0.9]],
            dtype=torch.float32
        )

        X_transformed = ordinal.transform(X_mixed)

        # 第一维应被规范化，第二维保持不变
        assert X_transformed[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert X_transformed[1, 0] == pytest.approx(0.5, abs=1e-5)
        assert X_transformed[2, 0] == pytest.approx(1.0, abs=1e-5)

        # 第二维不变
        assert X_transformed[0, 1] == pytest.approx(0.5, abs=1e-5)
        assert X_transformed[1, 1] == pytest.approx(0.7, abs=1e-5)
        assert X_transformed[2, 1] == pytest.approx(0.9, abs=1e-5)


class TestEdgeCases:
    """边界情况测试"""

    def test_very_small_spacing(self):
        """测试极小间距"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [1.0, 1.0001, 1.0002]}
        )

        X = torch.tensor([[1.0], [1.0001], [1.0002]], dtype=torch.float32)
        X_norm = ordinal.transform(X)

        # 应正常归一化
        assert torch.min(X_norm) == pytest.approx(0.0, abs=1e-5)
        assert torch.max(X_norm) == pytest.approx(1.0, abs=1e-5)

    def test_negative_values(self):
        """测试负值"""
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: [-10.0, -5.0, 0.0, 5.0]}
        )

        X = torch.tensor([[-10.0], [-5.0], [0.0], [5.0]], dtype=torch.float32)
        X_norm = ordinal.transform(X)

        expected = torch.tensor([[0.0], [0.333333], [0.666667], [1.0]], dtype=torch.float32)
        assert torch.allclose(X_norm, expected, atol=1e-5)

    def test_large_number_of_levels(self):
        """测试大量水平"""
        values = list(range(0, 100, 2))  # 50 个水平
        ordinal = CustomOrdinal(
            indices=[0],
            values={0: values}
        )

        X = torch.tensor([[v] for v in values], dtype=torch.float32)
        X_norm = ordinal.transform(X)

        # 检查归一化范围
        assert torch.min(X_norm) == pytest.approx(0.0, abs=1e-5)
        assert torch.max(X_norm) == pytest.approx(1.0, abs=1e-5)

        # 检查往返一致性
        X_reconstructed = ordinal.untransform(X_norm)
        assert torch.allclose(X, X_reconstructed, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
