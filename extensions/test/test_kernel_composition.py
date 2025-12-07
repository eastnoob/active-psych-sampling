#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - ProductKernel组合和核心逻辑

测试覆盖:
1. ProductKernel验证 (4个测试)
2. 核心逻辑验证 (3个测试)
3. 兼容性测试 (2个测试)

总计: 9个测试
"""

import pytest
import torch
import gpytorch
from botorch.models.kernels import CategoricalKernel

from extensions.custom_factory.custom_basegp_residual_mixed_factory import (
    CustomBaseGPResidualMixedFactory,
)


# ============================================================================
# 测试 - ProductKernel验证 (4个)
# ============================================================================


class TestProductKernelComposition:
    """ProductKernel组合的单元测试"""

    def test_product_kernel_mixed_params(self):
        """测试混合参数时ProductKernel的创建"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"intensity": 3, "color": 2},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()

        # 检查外层是ScaleKernel
        assert isinstance(
            covar_module, gpytorch.kernels.ScaleKernel
        ), f"Expected ScaleKernel, got {type(covar_module)}"

        # 检查内层是ProductKernel
        base_kernel = covar_module.base_kernel
        assert isinstance(
            base_kernel, gpytorch.kernels.ProductKernel
        ), f"Expected ProductKernel, got {type(base_kernel)}"

        print("✅ ProductKernel结构正确 (ScaleKernel > ProductKernel)")

    def test_kernel_composition_kernels(self):
        """测试ProductKernel内部的子核心"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"intensity": 3, "color": 2},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()
        base_kernel = covar_module.base_kernel

        # ProductKernel应该有2个子核
        assert (
            len(base_kernel.kernels) == 2
        ), f"Expected 2 kernels in ProductKernel, got {len(base_kernel.kernels)}"

        kernel1_type = type(base_kernel.kernels[0])
        kernel2_type = type(base_kernel.kernels[1])

        # 一个应该是MaternKernel，一个应该是CategoricalKernel
        has_matern = issubclass(
            kernel1_type, gpytorch.kernels.MaternKernel
        ) or issubclass(kernel2_type, gpytorch.kernels.MaternKernel)
        has_categorical = issubclass(kernel1_type, CategoricalKernel) or issubclass(
            kernel2_type, CategoricalKernel
        )

        assert has_matern, "ProductKernel should contain MaternKernel"
        assert has_categorical, "ProductKernel should contain CategoricalKernel"

        print(
            f"✅ ProductKernel子核心正确: {kernel1_type.__name__} × {kernel2_type.__name__}"
        )

    def test_continuous_kernel_ard(self):
        """测试连续参数核心的ARD"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=["x1", "x2"],
            discrete_params={},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()
        base_kernel = covar_module.base_kernel

        # 对于仅连续参数，base_kernel应该是MaternKernel (被ScaleKernel包装了)
        if isinstance(base_kernel, gpytorch.kernels.MaternKernel):
            matern = base_kernel
        else:
            # 可能被ProductKernel包装了
            matern = base_kernel.kernels[0]

        # 检查ARD
        assert hasattr(matern, "ard_num_dims"), "MaternKernel should have ard_num_dims"
        assert (
            matern.ard_num_dims == 2
        ), f"Expected ard_num_dims=2, got {matern.ard_num_dims}"

        print(f"✅ 连续参数核心ARD正确: ard_num_dims={matern.ard_num_dims}")

    def test_discrete_kernel_ard(self):
        """测试离散参数核心的ARD"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=[],
            discrete_params={"intensity": 3, "color": 2},
        )

        covar_module = factory._make_covar_module()
        base_kernel = covar_module.base_kernel

        # 对于仅离散参数，base_kernel应该是CategoricalKernel
        if isinstance(base_kernel, CategoricalKernel):
            categorical = base_kernel
        else:
            # 可能被ProductKernel包装了
            categorical = base_kernel.kernels[0]

        # 检查ARD
        assert hasattr(
            categorical, "ard_num_dims"
        ), "CategoricalKernel should have ard_num_dims"
        assert (
            categorical.ard_num_dims == 2
        ), f"Expected ard_num_dims=2, got {categorical.ard_num_dims}"

        print(f"✅ 离散参数核心ARD正确: ard_num_dims={categorical.ard_num_dims}")


# ============================================================================
# 测试 - 核心逻辑验证 (3个)
# ============================================================================


class TestCoreLogic:
    """核心逻辑验证的单元测试"""

    def test_kernel_forward_continuous(self):
        """测试连续参数核心的前向传播"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=["x1", "x2"],
            discrete_params={},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()

        # 创建测试数据
        x = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.2, 0.2]], dtype=torch.float32)

        # 前向传播
        covar_matrix = covar_module(x)

        # 转换为dense矩阵便于操作
        covar_dense = covar_matrix.to_dense()

        # 检查输出形状
        assert covar_dense.shape == torch.Size(
            [3, 3]
        ), f"Expected shape (3, 3), got {covar_dense.shape}"

        # 检查对称性
        assert torch.allclose(
            covar_dense, covar_dense.transpose(0, 1), atol=1e-5
        ), "Covariance matrix should be symmetric"

        # 检查正定性 (对角线元素应该为正)
        assert (
            torch.diag(covar_dense) > 0
        ).all(), "Diagonal elements should be positive"

        # 检查没有NaN/Inf
        assert not torch.isnan(covar_dense).any(), "Covariance matrix contains NaN"
        assert not torch.isinf(covar_dense).any(), "Covariance matrix contains Inf"

        print(f"✅ 连续参数核心前向传播正确: shape={covar_dense.shape}")

    def test_kernel_forward_discrete(self):
        """测试离散参数核心的前向传播"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=[],
            discrete_params={"intensity": 3, "color": 2},
        )

        covar_module = factory._make_covar_module()

        # 创建离散测试数据 (values: 0-2 and 0-1)
        x = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        # 前向传播
        covar_matrix = covar_module(x)

        # 转换为dense矩阵便于操作
        covar_dense = covar_matrix.to_dense()

        # 检查输出形状
        assert covar_dense.shape == torch.Size(
            [4, 4]
        ), f"Expected shape (4, 4), got {covar_dense.shape}"

        # 检查对称性
        assert torch.allclose(
            covar_dense, covar_dense.transpose(0, 1), atol=1e-5
        ), "Covariance matrix should be symmetric"

        # 检查没有NaN/Inf
        assert not torch.isnan(covar_dense).any(), "Covariance matrix contains NaN"
        assert not torch.isinf(covar_dense).any(), "Covariance matrix contains Inf"

        print(f"✅ 离散参数核心前向传播正确: shape={covar_dense.shape}")

    def test_kernel_forward_mixed(self):
        """测试混合参数的前向传播"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"intensity": 3, "color": 2},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()

        # 混合参数: [连续, 连续, 离散, 离散]
        x = torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.3, 0.7, 1.0, 1.0],
                [0.2, 0.2, 2.0, 0.0],
            ],
            dtype=torch.float32,
        )

        # 前向传播
        covar_matrix = covar_module(x)

        # 转换为dense矩阵便于操作
        covar_dense = covar_matrix.to_dense()

        # 检查输出形状
        assert covar_dense.shape == torch.Size(
            [3, 3]
        ), f"Expected shape (3, 3), got {covar_dense.shape}"

        # 检查对称性
        assert torch.allclose(
            covar_dense, covar_dense.transpose(0, 1), atol=1e-5
        ), "Covariance matrix should be symmetric"

        # 检查没有NaN/Inf
        assert not torch.isnan(covar_dense).any(), "Covariance matrix contains NaN"
        assert not torch.isinf(covar_dense).any(), "Covariance matrix contains Inf"

        print(f"✅ 混合参数前向传播正确: shape={covar_dense.shape}")


# ============================================================================
# 测试 - 兼容性测试 (2个)
# ============================================================================


class TestCompatibility:
    """兼容性测试"""

    def test_gradients_enabled(self):
        """测试梯度是否可以启用"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=2,
            continuous_params=["x1", "x2"],
            discrete_params={},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()

        # 创建需要梯度的输入
        x = torch.tensor(
            [[0.5, 0.5], [0.3, 0.7]], dtype=torch.float32, requires_grad=True
        )

        # 前向传播
        covar_matrix = covar_module(x)
        covar_dense = covar_matrix.to_dense()

        # 计算简单的标量损失
        loss = covar_dense.mean()

        # 反向传播
        loss.backward()

        # 检查梯度
        assert x.grad is not None, "Input should have gradient"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"

        print("✅ 梯度反向传播正确")

    def test_eval_mode(self):
        """测试评估模式"""
        factory = CustomBaseGPResidualMixedFactory(
            dim=4,
            continuous_params=["x1", "x2"],
            discrete_params={"intensity": 3, "color": 2},
            ls_loc=[0.0, -0.3],
            ls_scale=[0.5, 0.5],
        )

        covar_module = factory._make_covar_module()

        # 切换到eval模式
        covar_module.eval()

        # 创建测试数据
        x = torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.3, 0.7, 1.0, 1.0],
            ],
            dtype=torch.float32,
        )

        # 前向传播
        with torch.no_grad():
            covar_matrix = covar_module(x)
            covar_dense = covar_matrix.to_dense()

        # 检查输出
        assert covar_dense.shape == torch.Size([2, 2])
        assert not torch.isnan(covar_dense).any()

        print("✅ 评估模式正确")


# ============================================================================
# 测试运行
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
