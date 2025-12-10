"""
Sampling History Manager

Tracks and manages sampling history to exclude previously sampled points.
"""

import torch
from typing import Set, Optional
from loguru import logger


class HistoryManager:
    """Manages sampling history tracking and exclusion."""

    def __init__(
        self,
        pool_points: torch.Tensor,
        dim: int,
    ):
        """
        Initialize HistoryManager.

        Args:
            pool_points: All pool points
            dim: Dimensionality
        """
        self.pool_points = pool_points
        self.dim = dim
        self._used_indices: Set[int] = set()
        self._aepsych_server: Optional = None

    def exclude_historical_points_from_history(
        self, train_inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Exclude historical points from the training history.

        Removes points from train_inputs that have already been sampled
        from the pool, preventing duplication in the training dataset.

        Args:
            train_inputs: Full training history [n_samples, n_dim]

        Returns:
            torch.Tensor: Filtered training inputs without duplicates
        """
        try:
            if train_inputs is None or train_inputs.shape[0] == 0:
                logger.debug("[HistoryMgr] Empty training inputs")
                return train_inputs

            # Get indices of already used points from pool
            used_pool_indices = self.get_used_pool_indices(train_inputs)

            if not used_pool_indices:
                logger.debug("[HistoryMgr] No duplicates found in training history")
                return train_inputs

            # Create mask for points NOT in pool
            mask = torch.ones(train_inputs.shape[0], dtype=torch.bool)
            for idx in used_pool_indices:
                # Find position of this point in train_inputs
                for i, train_point in enumerate(train_inputs):
                    distances = torch.norm(self.pool_points[idx] - train_point)
                    if distances.item() < 1e-6:  # Match found
                        mask[i] = False
                        logger.debug(
                            f"[HistoryMgr] Excluded duplicate: index {idx} position {i}"
                        )

            filtered_inputs = train_inputs[mask]
            logger.info(
                f"[HistoryMgr] Excluded {(~mask).sum().item()} duplicates, "
                f"keeping {filtered_inputs.shape[0]} points"
            )
            return filtered_inputs

        except Exception as e:
            logger.error(f"[HistoryMgr] Failed to exclude historical points: {e}")
            return train_inputs

    def get_used_pool_indices(self, train_inputs: torch.Tensor) -> Set[int]:
        """
        Find which pool indices are present in training inputs.

        Args:
            train_inputs: Training inputs [n_samples, n_dim]

        Returns:
            set[int]: Set of pool indices found in training inputs
        """
        used_pool_indices = set()
        tolerance = 1e-6

        for pool_idx, pool_point in enumerate(self.pool_points):
            for train_point in train_inputs:
                distance = torch.norm(pool_point - train_point)
                if distance.item() < tolerance:
                    used_pool_indices.add(pool_idx)
                    break

        return used_pool_indices

    def get_sampling_history_from_server(self) -> Optional[torch.Tensor]:
        """
        Extract sampling history from AEPsych server.

        Returns:
            torch.Tensor: Historical sampling points or None
        """
        try:
            if self._aepsych_server is None:
                logger.debug("[HistoryMgr] No server configured")
                return None

            if not hasattr(self._aepsych_server, "db"):
                logger.debug("[HistoryMgr] Server has no database")
                return None

            query = """
            SELECT param_name, param_value, iteration_id 
            FROM param_data 
            ORDER BY iteration_id, param_name
            """

            result = self._aepsych_server.db.execute_sql_query(query, {})
            if not result:
                logger.debug("[HistoryMgr] No sampling history in server")
                return None

            param_dict = {}
            for param_name, param_value, iteration_id in result:
                clean_param_name = param_name.strip("'\"")
                if iteration_id not in param_dict:
                    param_dict[iteration_id] = {}
                param_dict[iteration_id][clean_param_name] = float(param_value)

            iteration_ids = sorted(param_dict.keys())
            if not iteration_ids:
                return None

            param_names = sorted(param_dict[iteration_ids[0]].keys())
            if not param_names:
                return None

            sampling_data = []
            for iteration_id in iteration_ids:
                point = [
                    param_dict[iteration_id].get(pname, 0.0) for pname in param_names
                ]
                sampling_data.append(point)

            if sampling_data:
                sampling_history = torch.tensor(sampling_data, dtype=torch.float32)
                logger.debug(
                    f"[HistoryMgr] Extracted {len(sampling_history)} points from server"
                )
                return sampling_history

        except Exception as e:
            logger.debug(f"[HistoryMgr] Failed to get server history: {e}")

        return None

    def set_aepsych_server(self, server) -> None:
        """Set the AEPsych server instance for database access."""
        self._aepsych_server = server
        logger.debug(f"[HistoryMgr] Server configured: {type(server)}")

    def add_used_index(self, idx: int) -> None:
        """Add used pool index."""
        self._used_indices.add(idx)

    def get_used_indices(self) -> Set[int]:
        """Get set of used indices."""
        return self._used_indices.copy()

    def clear_used_indices(self) -> None:
        """Clear used indices (reset)."""
        self._used_indices.clear()
