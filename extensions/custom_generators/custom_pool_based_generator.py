#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Pool-Based Generator for AEPsych with Acquisition Function Support

This generator selects points from a predefined pool of candidate points using
acquisition functions to intelligently choose the most informative points.
It's useful for pool-based active learning scenarios where you have a fixed set
of candidate points and want to select the best ones based on model uncertainty
or other criteria.
"""

from typing import Any, Optional
import sys
import os

# Add temp_aepsych to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "temp_aepsych"))

import torch
from aepsych.config import Config
from aepsych.models.base import AEPsychModelMixin
from aepsych.utils import _process_bounds
from aepsych.generators.base import AcqfGenerator
from botorch.acquisition import AcquisitionFunction
from loguru import logger


class CustomPoolBasedGenerator(AcqfGenerator):
    """
    Generator that samples from a predefined pool of points using acquisition functions.

    This generator is designed for pool-based active learning where you have a fixed
    set of candidate points. It uses acquisition functions to intelligently select
    the most informative points from the pool based on model uncertainty or other
    criteria defined by the acquisition function.

    When a model is available, the generator evaluates the acquisition function on all
    available points in the pool and selects the ones with the highest acquisition
    values. When no model is available, it falls back to sequential or random selection.

    Attributes:
        pool_points (torch.Tensor): The pool of candidate points to sample from.
        lb (torch.Tensor): Lower bounds of each parameter.
        ub (torch.Tensor): Upper bounds of each parameter.
        dim (int): Dimensionality of the parameter space.
        acqf (AcquisitionFunction): Acquisition function to use for point selection.
        acqf_kwargs (dict): Additional arguments for the acquisition function.
        allow_resampling (bool): Whether to allow resampling of already used points.
        shuffle (bool): Whether to shuffle the pool initially (used when no model).
        seed (int, optional): Random seed for reproducibility.
    """

    _requires_model = True  # Changed to True since we use acquisition functions

    def __init__(
        self,
        lb: torch.Tensor,
        ub: torch.Tensor,
        pool_points: torch.Tensor,
        acqf: AcquisitionFunction,
        acqf_kwargs: Optional[dict[str, Any]] = None,
        dim: Optional[int] = None,
        allow_resampling: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize PoolBasedGenerator.

        Args:
            lb (torch.Tensor): Lower bounds of each parameter.
            ub (torch.Tensor): Upper bounds of each parameter.
            pool_points (torch.Tensor): The pool of candidate points [n_points x dim].
            acqf (AcquisitionFunction): Acquisition function to use for point selection.
            acqf_kwargs (dict, optional): Extra arguments for the acquisition function.
            dim (int, optional): Dimensionality of the parameter space.
                If None, it is inferred from lb and ub.
            allow_resampling (bool): Whether to allow resampling of already used points.
                Default is False.
            shuffle (bool): Whether to shuffle the pool initially (used when no model
                is available). Default is True.
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        super().__init__(acqf=acqf, acqf_kwargs=acqf_kwargs)
        self.seed = seed
        self.lb, self.ub, self.dim = _process_bounds(lb, ub, dim)
        self.allow_resampling = allow_resampling

        # 【新增】缓存采集函数实例，确保诊断信息的一致性
        self._acqf_instance = None
        self._acqf_instance_model = None
        self._last_model_train_size = None  # 追踪模型训练样本数,用于检测refit
        self._last_train_tensor_id = (
            None  # 追踪train_inputs[0]的tensor ID,更可靠的refit检测
        )

        # Validate pool_points
        if pool_points is None or len(pool_points) == 0:
            raise ValueError("pool_points must be a non-empty tensor")

        # Convert to tensor if needed
        if not isinstance(pool_points, torch.Tensor):
            pool_points = torch.tensor(pool_points, dtype=torch.float32)

        # Ensure pool_points is 2D
        if len(pool_points.shape) == 1:
            pool_points = pool_points.unsqueeze(0)

        # Check dimensionality
        if pool_points.shape[1] != self.dim:
            raise ValueError(
                f"pool_points dimensionality ({pool_points.shape[1]}) "
                f"does not match specified dim ({self.dim})"
            )

        # Optionally shuffle the pool
        if shuffle:
            if seed is not None:
                torch.manual_seed(seed)
            perm = torch.randperm(len(pool_points))
            pool_points = pool_points[perm]

        self.pool_points = pool_points
        self._used_indices = set()
        self._current_idx = 0
        self.max_asks = len(self.pool_points) if not allow_resampling else None

    def gen(
        self,
        num_points: int = 1,
        model: Optional[AEPsychModelMixin] = None,
        fixed_features: Optional[dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Query next point(s) from the pool using acquisition function.

        Uses the acquisition function to evaluate all available points in the pool
        and selects the ones with the highest acquisition values. This enables
        intelligent point selection based on model uncertainty or other criteria.

        Args:
            num_points (int): Number of points to query. Defaults to 1.
            model (AEPsychModelMixin): Fitted model to use for acquisition function evaluation.
            fixed_features (dict[int, float], optional): Fixed features for generation.
                Currently not supported for PoolBasedGenerator.
            **kwargs: Additional arguments (ignored, for API compatibility).

        Returns:
            torch.Tensor: Next set of point(s) to evaluate [num_points x dim].

        Raises:
            RuntimeError: If pool is exhausted and resampling is not allowed.
        """
        if fixed_features is not None and len(fixed_features) != 0:
            logger.warning(
                f"Cannot fix features when generating from {self.__class__.__name__}"
            )

        # Exclude historical points from model training data
        excluded_count = 0
        if model is not None:
            excluded_count = self._exclude_historical_points(model)
            if excluded_count > 0:
                logger.info(f"[PoolGen] 已排除 {excluded_count} 个新的历史采样点")

        # Get available points
        available_indices = self._get_available_indices()

        if len(available_indices) == 0:
            if self.allow_resampling:
                # Reset and allow resampling
                logger.info("Pool exhausted, resampling from beginning")
                self._used_indices.clear()
                self._current_idx = 0
                available_indices = self._get_available_indices()
            else:
                raise RuntimeError(
                    f"Pool exhausted! Requested {num_points} points but pool is empty. "
                    "Set allow_resampling=True to enable resampling."
                )

        # Limit num_points to available points
        actual_num_points = min(num_points, len(available_indices))
        if actual_num_points < num_points:
            logger.warning(
                f"Requested {num_points} points but only {actual_num_points} available in pool. "
                f"Returning {actual_num_points} points."
            )

        available_points = self.pool_points[available_indices]

        # If model is provided, use acquisition function to select points  
        if model is not None:
            model.eval()

            # 【调试】检查收到的 model
            import sys

            if hasattr(model, "train_inputs") and model.train_inputs:
                print(
                    f"[PoolGen.gen] Received model_id={id(model)}, train_inputs[0].shape={model.train_inputs[0].shape}",
                    file=sys.stderr,
                )
                if hasattr(model, "_base_obj"):
                    print(
                        f"[PoolGen.gen] model._base_obj id={id(model._base_obj)}, _base_obj._train_inputs[0].shape={model._base_obj._train_inputs[0].shape}",
                        file=sys.stderr,
                    )

            # Move points to model device if needed
            if hasattr(model, "device"):
                available_points = available_points.to(model.device)

            # 【修复】智能acqf缓存：只在model refit时重新创建
            # 检测model是否refit：比较训练样本数
            current_train_size = (
                model.train_inputs[0].shape[0]
                if hasattr(model, "train_inputs") and model.train_inputs
                else 0
            )
            logger.debug(
                f"[PoolGen] model_id={id(model)}, train_inputs[0]_id={id(model.train_inputs[0]) if hasattr(model, 'train_inputs') and model.train_inputs else 'None'}, current_train_size={current_train_size}, _last_model_train_size={self._last_model_train_size}"
            )
            need_recreate = (
                self._acqf_instance is None
                or self._last_model_train_size is None
                or current_train_size != self._last_model_train_size
            )
            logger.debug(
                f"[PoolGen] need_recreate={need_recreate}, _acqf_instance={'None' if self._acqf_instance is None else type(self._acqf_instance).__name__}"
            )

            if need_recreate:
                # 【关键修复】在重新创建acqf之前，保存旧weight_engine的r_t状态
                # 这样新acqf可以继承历史参数变化信息，实现r_t的正确追踪
                old_weight_engine_state = None
                old_sps_tracker_state = None
                if self._acqf_instance is not None and hasattr(
                    self._acqf_instance, "weight_engine"
                ):
                    old_we = self._acqf_instance.weight_engine
                    old_weight_engine_state = {
                        "_prev_core_params": getattr(old_we, "_prev_core_params", None),
                        "_initial_param_norm": getattr(
                            old_we, "_initial_param_norm", None
                        ),
                        "_r_t_smoothed": getattr(old_we, "_r_t_smoothed", None),
                        "_cached_r_t": getattr(old_we, "_cached_r_t", None),
                        "_cached_r_t_n_train": getattr(
                            old_we, "_cached_r_t_n_train", -1
                        ),
                    }
                    # 保存SPS tracker状态（如果存在）
                    if (
                        hasattr(old_we, "sps_tracker")
                        and old_we.sps_tracker is not None
                    ):
                        old_sps_tracker_state = {
                            "prev_predictions": getattr(
                                old_we.sps_tracker, "prev_predictions", None
                            ),
                            "r_t_smoothed": getattr(
                                old_we.sps_tracker, "r_t_smoothed", None
                            ),
                        }
                    logger.debug(
                        f"[PoolGen] Saved old weight_engine state: _r_t_smoothed={old_weight_engine_state['_r_t_smoothed']}, sps_state={'exists' if old_sps_tracker_state else 'None'}"
                    )

                acqf = self._instantiate_acquisition_fn(model)
                self._acqf_instance = acqf
                self._last_model_train_size = current_train_size

                # 【关键修复】恢复weight_engine状态到新acqf
                if old_weight_engine_state is not None and hasattr(
                    acqf, "weight_engine"
                ):
                    new_we = acqf.weight_engine
                    # 恢复_prev_core_params以计算参数变化率
                    # 这是r_t追踪的基础，没有这个状态r_t永远从1.0开始
                    if old_weight_engine_state["_prev_core_params"] is not None:
                        new_we._prev_core_params = old_weight_engine_state[
                            "_prev_core_params"
                        ]
                    if old_weight_engine_state["_initial_param_norm"] is not None:
                        new_we._initial_param_norm = old_weight_engine_state[
                            "_initial_param_norm"
                        ]
                    # 恢复_r_t_smoothed以维持EMA平滑的连续性
                    if old_weight_engine_state["_r_t_smoothed"] is not None:
                        new_we._r_t_smoothed = old_weight_engine_state["_r_t_smoothed"]

                    # 恢复SPS tracker状态（如果存在）
                    if (
                        old_sps_tracker_state is not None
                        and hasattr(new_we, "sps_tracker")
                        and new_we.sps_tracker is not None
                    ):
                        if old_sps_tracker_state["prev_predictions"] is not None:
                            new_we.sps_tracker.prev_predictions = old_sps_tracker_state[
                                "prev_predictions"
                            ]
                        if old_sps_tracker_state["r_t_smoothed"] is not None:
                            new_we.sps_tracker.r_t_smoothed = old_sps_tracker_state[
                                "r_t_smoothed"
                            ]

                    logger.debug(
                        f"[PoolGen] Restored weight_engine state: _prev_core_params={'exists' if new_we._prev_core_params is not None else 'None'}, _r_t_smoothed={new_we._r_t_smoothed}"
                    )

                # 同步训练状态
                if hasattr(acqf, "weight_engine"):
                    acqf.weight_engine.update_training_status(
                        current_train_size, fitted=True
                    )
                logger.debug(
                    f"[PoolGen] Created new acqf, train_size={current_train_size}, synced weight_engine"
                )
            else:
                acqf = self._acqf_instance
                # Update weight_engine with current training status
                if hasattr(acqf, "weight_engine"):
                    acqf.weight_engine.update_training_status(
                        current_train_size, fitted=True
                    )
                logger.debug(
                    f"[PoolGen] Reusing cached acqf, train_size={current_train_size}"
                )

            # Evaluate acquisition function on all available points
            with torch.no_grad():
                # Acquisition functions expect (batch_size, q, dim) for batch evaluation
                # We evaluate each point individually
                acqf_values = []
                for point in available_points:
                    # Reshape to (1, 1, dim) for single point evaluation
                    point_reshaped = point.unsqueeze(0).unsqueeze(0)
                    try:
                        acqf_value = acqf(point_reshaped)
                        acqf_values.append(acqf_value.item())
                    except Exception as e:
                        # If acquisition function fails, use 0 as fallback
                        logger.warning(
                            f"Acquisition function evaluation failed for point {point}: {e}"
                        )
                        acqf_values.append(0.0)

                acqf_values = torch.tensor(acqf_values)

            # Select points with highest acquisition values
            _, top_indices = torch.topk(acqf_values, k=actual_num_points, largest=True)
            selected_pool_indices = available_indices[top_indices]
            selected_points = self.pool_points[selected_pool_indices]

            logger.debug(
                "PoolBasedGenerator used=%d available=%d picked=%s",
                len(self._used_indices),
                len(available_indices),
                selected_pool_indices.tolist(),
            )

            # 计算总的已使用点数（包括刚选中的点）
            total_used = len(self._used_indices)
            logger.info(
                f"[PoolGen] 选中 {actual_num_points} 个点: 索引={selected_pool_indices.tolist()}, "
                f"总已用点={total_used}个, 剩余={len(self.pool_points)-total_used}个, "
                f"采集函数={self.acqf.__name__}"
            )
        else:
            # Fallback: sequential selection if no model (shouldn't happen with _requires_model=True)
            logger.warning("No model provided, using sequential selection")
            selected_pool_indices = available_indices[:actual_num_points]
            selected_points = available_points[:actual_num_points]

        # Mark as used
        self._used_indices.update(selected_pool_indices.tolist())

        # Store the last selected indices for external access
        self.last_selected_indices = selected_pool_indices.tolist()

        logger.debug(
            "PoolBasedGenerator used_indices now=%s",
            sorted(self._used_indices),
        )

        return selected_points

    def _get_available_indices(self) -> torch.Tensor:
        """
        Get indices of points that haven't been used yet.

        Returns:
            torch.Tensor: Tensor of available indices.
        """
        all_indices = torch.arange(len(self.pool_points))
        if len(self._used_indices) == 0:
            return all_indices

        used_tensor = torch.tensor(list(self._used_indices))
        mask = torch.ones(len(self.pool_points), dtype=torch.bool)
        mask[used_tensor] = False
        return all_indices[mask]

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: Optional[str] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Extract configuration options for the generator from a Config object.

        Args:
            config (Config): Config object to extract options from.
            name (str, optional): Name of the generator section in config.
            options (dict[str, Any], optional): Existing options to update.

        Returns:
            dict[str, Any]: Dictionary of options to initialize the generator.
        """
        # Fix: Pre-populate acqf if it's not in options yet
        # This prevents KeyError when parent class tries to access options["acqf"]
        if options is None:
            options = {}

        name = name or cls.__name__

        # Read acqf from config if not already in options
        if "acqf" not in options:
            try:
                acqf_class = config.getobj(name, "acqf")
                options["acqf"] = acqf_class
            except Exception:
                pass  # Let parent class handle the error

        options = super().get_config_options(config, name, options)

        # Handle potential shape issues with pool_points from config
        if "pool_points" in options:
            pool_points = options["pool_points"]
            if len(pool_points.shape) == 3:
                # Configs have a reasonable natural input method that produces incorrect tensors
                options["pool_points"] = pool_points.swapaxes(-1, -2).squeeze(0)

        return options

    @property
    def finished(self) -> bool:
        """
        Check if the generator has exhausted its pool.

        Returns:
            bool: True if pool is exhausted and resampling is not allowed, False otherwise.
        """
        if self.allow_resampling:
            return False
        return len(self._used_indices) >= len(self.pool_points)

    def reset(self) -> None:
        """Reset the generator to its initial state, clearing used indices."""
        self._used_indices.clear()
        self._current_idx = 0
        logger.info("PoolBasedGenerator reset - all points available again")

    def get_acqf_instance(self):
        """【新增】获取缓存的采集函数实例（用于诊断）"""
        return self._acqf_instance

    def _exclude_historical_points(self, model: AEPsychModelMixin) -> int:
        """
        排除模型训练数据中已使用的历史点
        
        Args:
            model: 训练好的AEPsych模型
            
        Returns:
            int: 排除的历史点数量
        """
        if not hasattr(model, 'train_inputs') or not model.train_inputs:
            return 0
            
        # 获取历史训练点
        train_inputs = model.train_inputs[0]  # [n_points, n_dim]
        if train_inputs is None or len(train_inputs) == 0:
            return 0
            
        logger.debug(f"[PoolGen] 检查 {len(train_inputs)} 个历史训练点")
        
        # 将训练点匹配到设计空间索引
        excluded_indices = self._match_points_to_pool_indices(train_inputs)
        
        # 更新已使用索引集合
        original_used_count = len(self._used_indices)
        self._used_indices.update(excluded_indices)
        newly_excluded = len(self._used_indices) - original_used_count
        
        if newly_excluded > 0:
            logger.debug(f"[PoolGen] 新排除历史点索引: {sorted(excluded_indices - set(range(original_used_count)))}")
        
        return newly_excluded
    
    def _match_points_to_pool_indices(self, train_points: torch.Tensor) -> set[int]:
        """
        将训练点匹配到池中对应的索引
        
        Args:
            train_points: 训练点张量 [n_points, n_dim]
            
        Returns:
            set[int]: 匹配到的池索引集合
        """
        matched_indices = set()
        tolerance = 1e-6  # 浮点数比较容差
        
        for train_point in train_points:
            # 计算与池中所有点的距离
            distances = torch.norm(self.pool_points - train_point.unsqueeze(0), dim=1)
            min_distance, closest_idx = torch.min(distances, dim=0)
            
            # 如果距离足够小，认为是匹配的点
            if min_distance.item() < tolerance:
                matched_indices.add(closest_idx.item())
                logger.debug(f"[PoolGen] 匹配: 训练点 {train_point.tolist()[:3]}... -> 池索引 {closest_idx.item()} (距离: {min_distance.item():.2e})")
            else:
                logger.warning(f"[PoolGen] 未匹配: 训练点 {train_point.tolist()[:3]}... (最小距离: {min_distance.item():.2e})")
        
        return matched_indices


# Register the CustomPoolBasedGenerator with the Config system
Config.register_object(CustomPoolBasedGenerator)
