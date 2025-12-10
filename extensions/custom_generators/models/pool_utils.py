"""
Pool-based Generator Utility Functions

Handles pool index management, server discovery, and point matching.
"""

import torch
from typing import Optional, Set
import inspect
from loguru import logger


def get_available_indices(
    used_indices: Set[int],
    pool_size: int,
    pool_points: torch.Tensor,
    historical_points: Set,
) -> torch.Tensor:
    """
    Get indices of points available for selection.

    Excludes both:
    - Points already used in current run (_used_indices)
    - Points in historical database (_historical_points)

    Args:
        used_indices: Set of indices used in current run
        pool_size: Total number of points in pool
        pool_points: All pool points tensor
        historical_points: Set of historical points from database

    Returns:
        torch.Tensor: Tensor of available indices
    """
    all_indices = torch.arange(pool_size)

    # Exclude used points
    excluded_mask = torch.ones(pool_size, dtype=torch.bool)
    if len(used_indices) > 0:
        used_tensor = torch.tensor(list(used_indices))
        excluded_mask[used_tensor] = False

    available_after_used = all_indices[excluded_mask]

    # Exclude historical points
    if len(historical_points) > 0:
        available_points = pool_points[available_after_used]
        points_np = (
            available_points.cpu().numpy()
            if available_points.is_cuda
            else available_points.numpy()
        )

        historical_mask = torch.ones(len(available_points), dtype=torch.bool)
        for i, point in enumerate(points_np):
            # 【修复】不应该sorted,直接使用原始tuple以匹配_historical_points的存储格式
            point_tuple = tuple(point)
            if point_tuple in historical_points:
                historical_mask[i] = False
                logger.debug(f"[PoolUtils] Excluding historical point: {point_tuple}")

        available_indices = available_after_used[historical_mask]
    else:
        available_indices = available_after_used

    return available_indices


def find_aepsych_server() -> Optional:
    """
    Intelligently locate AEPsych server instance through multiple methods.

    Tries to find server by:
    1. Global module search
    2. Call stack inspection

    Returns:
        AEPsychServer instance or None
    """
    try:
        # Method 1: Search through modules
        import sys

        for name, obj in sys.modules.items():
            if hasattr(obj, "__dict__"):
                for attr_name, attr_val in obj.__dict__.items():
                    if (
                        hasattr(attr_val, "db")
                        and hasattr(attr_val, "handle_request")
                        and "AEPsychServer" in str(type(attr_val))
                    ):
                        logger.debug(
                            f"[PoolUtils] Found server via module {name}.{attr_name}"
                        )
                        return attr_val

        # Method 2: Search call stack
        for frame_info in inspect.stack():
            frame_locals = frame_info.frame.f_locals
            frame_globals = frame_info.frame.f_globals

            # Check local variables
            for var_name, var_val in frame_locals.items():
                if (
                    hasattr(var_val, "db")
                    and hasattr(var_val, "handle_request")
                    and "AEPsychServer" in str(type(var_val))
                ):
                    logger.debug(f"[PoolUtils] Found server via stack local {var_name}")
                    return var_val

            # Check global variables
            for var_name, var_val in frame_globals.items():
                if (
                    hasattr(var_val, "db")
                    and hasattr(var_val, "handle_request")
                    and "AEPsychServer" in str(type(var_val))
                ):
                    logger.debug(
                        f"[PoolUtils] Found server via stack global {var_name}"
                    )
                    return var_val

    except Exception as e:
        logger.debug(f"[PoolUtils] Failed to find server: {e}")

    return None


def get_sampling_history_from_server(server) -> Optional[torch.Tensor]:
    """
    Extract sampling history from AEPsych server database.

    IMPORTANT: Only returns points that exist in the database. These may include
    raw/untransformed values from EUR or other sources that are NOT in the pool.
    The calling code (CustomPoolBasedGenerator) is responsible for:
    1. Matching these points to pool indices
    2. Handling non-matching points via nearest-neighbor correction
    3. Ensuring used indices are tracked for deduplication

    Args:
        server: AEPsychServer instance

    Returns:
        torch.Tensor: Historical sampling points [n_points, n_dim] or None
    """
    try:
        if not hasattr(server, "db") or server.db is None:
            logger.debug("[PoolUtils] Server has no database connection")
            return None

        # Query raw sampling data
        query = """
        SELECT param_name, param_value, iteration_id 
        FROM param_data 
        ORDER BY iteration_id, param_name
        """

        result = server.db.execute_sql_query(query, {})
        rows = result

        if not rows:
            logger.debug("[PoolUtils] No sampling history in database")
            return None

        # Parse into coordinate format
        param_dict = {}
        for param_name, param_value, iteration_id in rows:
            clean_param_name = param_name.strip("'\"")

            if iteration_id not in param_dict:
                param_dict[iteration_id] = {}
            param_dict[iteration_id][clean_param_name] = float(param_value)

        if not param_dict:
            return None

        iteration_ids = sorted(param_dict.keys())
        param_names = (
            sorted(param_dict[iteration_ids[0]].keys()) if iteration_ids else []
        )

        if not param_names:
            return None

        sampling_data = []
        for iteration_id in iteration_ids:
            point = [param_dict[iteration_id].get(pname, 0.0) for pname in param_names]
            sampling_data.append(point)

        if sampling_data:
            sampling_history = torch.tensor(sampling_data, dtype=torch.float32)
            logger.debug(
                f"[PoolUtils] Extracted {len(sampling_history)} historical points from server "
                f"(NOTE: These may include non-pool values requiring constraint/matching)"
            )
            return sampling_history

    except Exception as e:
        logger.debug(f"[PoolUtils] Failed to extract sampling history: {e}")

    return None


def match_points_to_pool_indices(
    train_points: torch.Tensor, pool_points: torch.Tensor
) -> Set[int]:
    """
    Match training points to corresponding pool indices.

    Args:
        train_points: Training points [n_points, n_dim]
        pool_points: Pool points [pool_size, n_dim]

    Returns:
        set[int]: Set of matched pool indices
    """
    matched_indices = set()
    tolerance = 1e-6  # Floating point comparison tolerance

    for train_point in train_points:
        # Compute distance to all pool points
        distances = torch.norm(pool_points - train_point.unsqueeze(0), dim=1)
        min_distance, closest_idx = torch.min(distances, dim=0)

        # Match if distance is small enough
        if min_distance.item() < tolerance:
            matched_indices.add(closest_idx.item())
            logger.debug(
                f"[PoolUtils] Matched: {train_point.tolist()[:3]}... -> "
                f"pool index {closest_idx.item()} (dist: {min_distance.item():.2e})"
            )
        else:
            logger.warning(
                f"[PoolUtils] No match: {train_point.tolist()[:3]}... "
                f"(min dist: {min_distance.item():.2e})"
            )

    return matched_indices
