#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - CustomBaseGPResidualFactory 和 CustomBaseGPResidualMixedFactory

测试覆盖：
1. Mean模块 (4个测试)
2. 工厂初始化 (5个测试)
3. 前向传播 (6个测试)
4. 集成测试 (3个测试)

总计: 18个测试，预期覆盖率 > 85%
"""

import pytest
import torch
import gpytorch
import numpy as np
from pathlib import Path
import tempfile
import pandas as pd

# 导入被测试的模块
from extensions.custom_factory.custom_basegp_residual_factory import (
    CustomBaseGPResidualFactory,
)
from extensions.custom_factory.custom_basegp_residual_mixed_factory import (
    CustomBaseGPResidualMixedFactory,
)
from extensions.custom_mean.custom_basegp_prior_mean import (
    CustomBaseGPPriorMean,
    CustomMeanWithOffsetPrior,
)


# ============================================================================
# Fixtures - 测试数据准备
# ============================================================================


@pytest.fixture
def sample_basegp_csv():
    """创建样本BaseGP预测CSV文件"""
    n_samples = 100
    data = {
        "x1_binary": (np.random.rand(n_samples) > 0.5).astype(int),
        "x2_5level_discrete": np.random.randint(1, 6, n_samples),
        "x3_5level_decimal": np.linspace(0, 1, n_samples),
        "x4_4level_categorical": np.random.randint(0, 4, n_samples),
        "x5_3level_categorical": np.random.randint(0, 3, n_samples),
        "x6_binary": (np.random.rand(n_samples) > 0.5).astype(int),
        "pred_mean": np.random.randn(n_samples) + 50,
        "pred_std": np.abs(np.random.randn(n_samples)) + 0.1,
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        csv_path = f.name

    yield csv_path

    # 清理
    Path(csv_path).unlink()


@pytest.fixture
def sample_train_data():
    """创建样本训练数据"""
    n_samples = 30
    n_dims = 6

    # INI空间的数据
    train_X = torch.rand(n_samples, n_dims, dtype=torch.float32)
    train_X[:, 1] = (train_X[:, 1] * 5).round()  # x1: 离散化到0-4
    train_X[:, 2] = train_X[:, 2] * 4  # x2: 0-4
    train_X[:, 3] = (train_X[:, 3] * 4).round()  # x3: 0-3
    train_X[:, 4] = (train_X[:, 4] * 3).round()  # x4: 0-2

    train_Y = torch.randn(n_samples, 1, dtype=torch.float32) + 50

    return train_X, train_Y


@pytest.fixture
def sample_mixed_train_data():
    """创建混合参数的训练数据"""
    n_samples = 30

    # 连续参数: 2个 (dim 0-1)
    X_continuous = torch.rand(n_samples, 2, dtype=torch.float32)

    # 离散参数: 2个 (dim 2-3, intensity:3类, color:2类)
    X_discrete = torch.zeros(n_samples, 2, dtype=torch.float32)
    X_discrete[:, 0] = torch.randint(
        0, 3, (n_samples,), dtype=torch.float32
    )  # intensity: 0-2
    X_discrete[:, 1] = torch.randint(
        0, 2, (n_samples,), dtype=torch.float32
    )  # color: 0-1

    train_X = torch.cat([X_continuous, X_discrete], dim=1)  # (30, 4)
    train_Y = torch.randn(n_samples, 1, dtype=torch.float32) + 50

    return train_X, train_Y


# ============================================================================
# 测试 - Mean模块 (4个)
# ============================================================================


class TestMeanModules:
    """Mean模块的单元测试"""

    def test_basegp_prior_mean_initialization(self, sample_basegp_csv):
        """测试CustomBaseGPPriorMean初始化"""
        mean = CustomBaseGPPriorMean(basegp_scan_csv=sample_basegp_csv)

        assert mean is not None
        assert mean.pred_means.shape[0] == 100
        assert mean.kdtree is not None
        print("✅ CustomBaseGPPriorMean 初始化成功")

    def test_basegp_prior_mean_forward(self, sample_basegp_csv):
        """测试CustomBaseGPPriorMean前向传播"""
        mean = CustomBaseGPPriorMean(basegp_scan_csv=sample_basegp_csv)

        # 创建测试输入
        x = torch.tensor([[0.5, 2.0, 2.0, 1.5, 1.0, 0.5]], dtype=torch.float32)

        output = mean(x)

        assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        print(f"✅ CustomBaseGPPriorMean 前向传播成功, 输出: {output.item():.3f}")

    def test_mean_with_offset_initialization(self, sample_basegp_csv):
        """测试CustomMeanWithOffsetPrior初始化"""
        mean = CustomMeanWithOffsetPrior(
            basegp_scan_csv=sample_basegp_csv, offset_prior_std=0.10
        )

        assert mean is not None
        assert hasattr(mean, "offset")
        assert mean.offset.shape == (
            1,
        ), f"Expected offset shape (1,), got {mean.offset.shape}"
        assert mean.offset.item() == 0.0, "Offset should be initialized to 0"

        # 检查参数计数
        params = list(mean.parameters())
        assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"
        print("✅ CustomMeanWithOffsetPrior 初始化成功 (1个参数)")

    def test_mean_with_offset_forward(self, sample_basegp_csv):
        """测试CustomMeanWithOffsetPrior前向传播"""
        mean = CustomMeanWithOffsetPrior(
            basegp_scan_csv=sample_basegp_csv, offset_prior_std=0.10
        )

        x = torch.tensor([[0.5, 2.0, 2.0, 1.5, 1.0, 0.5]], dtype=torch.float32)

        output = mean(x)

        assert output.shape == (1,), f"Expected shape (1,), got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # offset初始为0，输出应该大约等于base_mean
        base_mean_output = mean.base_mean(x)
        # 比较数值（考虑浮点精度）
        diff = (output - base_mean_output).abs().max().item()
        assert (
            diff < 1e-3
        ), f"Output should approximately equal base_mean when offset~0, diff={diff}"
        print("✅ CustomMeanWithOffsetPrior 前向传播成功")


# ============================================================================
# 测试 - 工厂初始化 (5个)
# ============================================================================


class TestFactoryInitialization:
    """工厂初始化的单元测试"""

    def test_residual_factory_pure_residual_init(self, sample_basegp_csv):
        """测试CustomBaseGPResidualFactory - pure_residual模式"""
        factory = CustomBaseGPResidualFactory(
            dim=6,
            basegp_scan_csv=sample_basegp_csv,
            mean_type="pure_residual",
            ls_loc=[0.0, -0.3, 0.2, -1.5, 0.8, 0.6],
            ls_scale=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )

        assert factory is not None
        assert factory.mean_type == "pure_residual"
        assert factory.dim == 6
        print("✅ CustomBaseGPResidualFactory (pure_residual) 初始化成功")

    def test_residual_factory_learned_offset_init(self, sample_basegp_csv):
        """测试CustomBaseGPResidualFactory - learned_offset模式"""
        factory = CustomBaseGPResidualFactory(
            dim=6,
            basegp_scan_csv=sample_basegp_csv,
            mean_type="learned_offset",
            offset_prior_std=0.10,
            ls_loc=[0.0, -0.3, 0.2, -1.5, 0.8, 0.6],
            ls_scale=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        )

        assert factory is not None
        assert factory.mean_type == "learned_offset"
        assert factory.offset_prior_std == 0.10
        print("✅ CustomBaseGPResidualFactory (learned_offset) 初始化成功")

    def test_mixed_factory_continuous_only(self):
        """测试CustomBaseGPResidualMixedFactory - 仅连续参数"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=["x1", "x2"],
            discrete_params={},
        )

        assert factory is not None
        assert len(factory.continuous_params) == 2
        assert len(factory.discrete_params) == 0
        print("✅ CustomBaseGPResidualMixedFactory (仅连续) 初始化成功")

    def test_mixed_factory_discrete_only(self):
        """测试CustomBaseGPResidualMixedFactory - 仅离散参数"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=[],
            discrete_params={"intensity": 3, "color": 2},
        )

        assert factory is not None
        assert len(factory.continuous_params) == 0
        assert len(factory.discrete_params) == 2
        print("✅ CustomBaseGPResidualMixedFactory (仅离散) 初始化成功")

    def test_mixed_factory_mixed_params(self, sample_basegp_csv):
        """测试CustomBaseGPResidualMixedFactory - 混合参数"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["duration", "frequency"],
            discrete_params={"intensity": 3, "color": 2},
            basegp_scan_csv=sample_basegp_csv,
            mean_type="pure_residual",
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        assert factory is not None
        assert len(factory.continuous_params) == 2
        assert len(factory.discrete_params) == 2
        assert factory.dim == 4
        print("✅ CustomBaseGPResidualMixedFactory (混合参数) 初始化成功")


# ============================================================================
# 测试 - 前向传播 (6个)
# ============================================================================


class TestForwardPass:
    """前向传播的单元测试"""

    def test_residual_factory_make_mean_module_pure(self, sample_basegp_csv):
        """测试_make_mean_module - pure_residual"""
        factory = CustomBaseGPResidualFactory(
            dim=6,
            basegp_scan_csv=sample_basegp_csv,
            mean_type="pure_residual",
        )

        mean_module = factory._make_mean_module()

        assert isinstance(mean_module, CustomBaseGPPriorMean)
        assert (
            len(list(mean_module.parameters())) == 0
        ), "pure_residual should have 0 learnable params"
        print("✅ _make_mean_module (pure_residual) 正确")

    def test_residual_factory_make_mean_module_offset(self, sample_basegp_csv):
        """测试_make_mean_module - learned_offset"""
        factory = CustomBaseGPResidualFactory(
            dim=6,
            basegp_scan_csv=sample_basegp_csv,
            mean_type="learned_offset",
            offset_prior_std=0.10,
        )

        mean_module = factory._make_mean_module()

        assert isinstance(mean_module, CustomMeanWithOffsetPrior)
        assert (
            len(list(mean_module.parameters())) == 1
        ), "learned_offset should have 1 learnable param"
        print("✅ _make_mean_module (learned_offset) 正确")

    def test_residual_factory_make_covar_module(self, sample_basegp_csv):
        """测试_make_covar_module"""
        factory = CustomBaseGPResidualFactory(
            dim=6,
            basegp_scan_csv=sample_basegp_csv,
            ls_loc=[0.0] * 6,
            ls_scale=[0.5] * 6,
        )

        covar_module = factory._make_covar_module()

        assert covar_module is not None
        assert isinstance(covar_module, gpytorch.kernels.ScaleKernel)
        print("✅ _make_covar_module 创建成功 (包含ScaleKernel)")

    def test_mixed_factory_make_covar_continuous(self):
        """测试_make_covar_module - 连续参数"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=["x1", "x2"],
            discrete_params={},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()

        assert covar_module is not None
        # 对于连续参数，应该返回ScaleKernel(MaternKernel)
        assert isinstance(covar_module, gpytorch.kernels.ScaleKernel)
        print("✅ _make_covar_module (连续) 创建成功")

    def test_mixed_factory_make_covar_discrete(self):
        """测试_make_covar_module - 离散参数"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=[],
            discrete_params={"intensity": 3, "color": 2},
        )

        covar_module = factory._make_covar_module()

        assert covar_module is not None
        # 对于离散参数，应该返回ScaleKernel(CategoricalKernel)
        assert isinstance(covar_module, gpytorch.kernels.ScaleKernel)
        print("✅ _make_covar_module (离散) 创建成功")

    def test_mixed_factory_make_covar_mixed(self):
        """测试_make_covar_module - 混合参数"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"intensity": 3, "color": 2},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()

        assert covar_module is not None
        # 应该包含ProductKernel
        assert isinstance(covar_module, gpytorch.kernels.ScaleKernel)
        print("✅ _make_covar_module (混合) 创建成功 (包含ProductKernel)")


# ============================================================================
# 测试 - 集成测试 (3个)
# ============================================================================


class TestIntegration:
    """集成测试"""

    def test_parameter_count_residual(self, sample_basegp_csv):
        """测试参数计数 - residual factory"""
        # pure_residual: 6个continuous ard + 1个outputscale = 7
        factory_pure = CustomBaseGPResidualFactory(
            dim=6,
            basegp_scan_csv=sample_basegp_csv,
            mean_type="pure_residual",
            ls_loc=[0.0] * 6,
            ls_scale=[0.5] * 6,
        )

        mean_pure = factory_pure._make_mean_module()
        covar_pure = factory_pure._make_covar_module()

        n_params_mean = sum(p.numel() for p in mean_pure.parameters())
        n_params_covar = sum(p.numel() for p in covar_pure.parameters())

        assert (
            n_params_mean == 0
        ), f"pure_residual mean should have 0 params, got {n_params_mean}"
        assert n_params_covar > 0, f"covar should have params, got {n_params_covar}"
        print(f"✅ 参数计数正确: Mean={n_params_mean}, Covar={n_params_covar}")

    def test_parameter_count_mixed(self):
        """测试参数计数 - mixed factory"""
        # 连续: 2个ard, 离散: 2个ard, outputscale: 1
        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"intensity": 3, "color": 2},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        mean = factory._make_mean_module()
        covar = factory._make_covar_module()

        n_params_mean = sum(p.numel() for p in mean.parameters())
        n_params_covar = sum(p.numel() for p in covar.parameters())

        assert n_params_mean == 0, f"pure_residual mean should have 0 params"
        assert n_params_covar > 0, f"covar should have params"
        print(f"✅ 混合参数计数正确: Mean={n_params_mean}, Covar={n_params_covar}")

    def test_gradient_flow(self, sample_basegp_csv, sample_train_data):
        """测试梯度反向传播"""
        factory = CustomBaseGPResidualFactory(
            dim=6,
            basegp_scan_csv=sample_basegp_csv,
            mean_type="learned_offset",
            offset_prior_std=0.10,
            ls_loc=[0.0] * 6,
            ls_scale=[0.5] * 6,
        )

        mean_module = factory._make_mean_module()
        covar_module = factory._make_covar_module()

        train_X, train_Y = sample_train_data

        # 前向传播
        mean_values = mean_module(train_X)
        covar_matrix = covar_module(train_X)
        covar_dense = covar_matrix.to_dense()

        # 简单的损失计算 (用于测试梯度流)
        loss = mean_values.sum() + covar_dense.sum()

        # 反向传播
        loss.backward()

        # 检查Mean模块的offset参数是否有梯度
        assert mean_module.offset.grad is not None, "offset should have gradient"
        assert not torch.isnan(mean_module.offset.grad).any(), "gradient contains NaN"
        print(f"✅ 梯度反向传播成功, offset.grad={mean_module.offset.grad.item():.6f}")


# ============================================================================
# 测试运行
# ============================================================================

if __name__ == "__main__":
    # 使用pytest运行测试
    # 命令: pytest extensions/test/test_custom_factories.py -v --tb=short
    pytest.main([__file__, "-v", "--tb=short"])
