"""
Deduplication Database Manager

Handles all deduplication database lifecycle operations:
- Mode 1: Manual path (string) - Persistent SQLite
- Mode 2: None - Temporary in-memory
- Mode 3: Tuple auto-naming - Auto-generated path
"""

import sqlite3
from pathlib import Path
from typing import Optional, Set, Tuple
import torch
from loguru import logger


class DedupDatabaseManager:
    """Manages deduplication database for tracking historical sampling points."""

    def __init__(
        self,
        dedup_database_path: Optional[str | Tuple | list] = None,
        dim: Optional[int] = None,
    ):
        """
        Initialize DedupDatabaseManager.

        Args:
            dedup_database_path: Database configuration (str, tuple, or None)
            dim: Dimensionality of parameter space
        """
        self.dedup_database_path = dedup_database_path
        self.dim = dim
        self._dedup_conn: Optional[sqlite3.Connection] = None
        self._is_temp_db: bool = False
        self._historical_points: Set[Tuple[float, ...]] = set()

    def initialize(self) -> None:
        """Initialize deduplication database with three-mode support."""
        try:
            if self.dedup_database_path is None:
                # Mode 2: Temporary in-memory database
                self._dedup_conn = sqlite3.connect(":memory:")
                self._is_temp_db = True
                logger.info(
                    "[DedupDB] Mode 2: Temporary in-memory database (current run only)"
                )

            elif isinstance(self.dedup_database_path, str):
                # Mode 1: Manual path (string)
                db_path = Path(self.dedup_database_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)

                # Delete old file and create new one
                if db_path.exists():
                    db_path.unlink()
                    logger.info(f"[DedupDB] Deleted old database: {db_path}")

                self._dedup_conn = sqlite3.connect(str(db_path))
                self._is_temp_db = False
                logger.info(f"[DedupDB] Mode 1: Using persistent database: {db_path}")

            elif isinstance(self.dedup_database_path, (list, tuple)):
                # Mode 3: Tuple auto-naming
                db_path_str = self._generate_db_path(self.dedup_database_path)
                db_path = Path(db_path_str)
                db_path.parent.mkdir(parents=True, exist_ok=True)

                # Delete old file and create new one
                if db_path.exists():
                    db_path.unlink()
                    logger.info(f"[DedupDB] Deleted old database: {db_path}")

                self._dedup_conn = sqlite3.connect(str(db_path))
                self._is_temp_db = False
                logger.info(f"[DedupDB] Mode 3: Using auto-named database: {db_path}")
            else:
                raise ValueError(
                    f"dedup_database_path must be str, tuple, or None, got: "
                    f"{type(self.dedup_database_path)}"
                )

            # Create standard param_data table (AEPsych architecture)
            self._dedup_conn.execute(
                """
                CREATE TABLE IF NOT EXISTS param_data (
                    param_name TEXT NOT NULL,
                    param_value REAL NOT NULL,
                    iteration_id INTEGER
                )
                """
            )
            self._dedup_conn.commit()
            logger.debug("[DedupDB] param_data table initialized")

            # Load historical points
            self._load_historical_points()

        except Exception as e:
            logger.error(f"[DedupDB] Initialization failed: {e}")
            # Fallback to in-memory database
            self._dedup_conn = sqlite3.connect(":memory:")
            self._is_temp_db = True
            logger.warning("[DedupDB] Fallback to temporary in-memory database")

    def _generate_db_path(self, config_value) -> str:
        """
        Generate database path from tuple configuration (Mode 3).

        Args:
            config_value: Tuple format (subject_id, run_id) or
                         (subject_id, run_id, save_dir)

        Returns:
            Generated database path
        """
        if isinstance(config_value, (list, tuple)):
            if len(config_value) == 2:
                subject_id, run_id = config_value
                save_dir = "./data"
            elif len(config_value) == 3:
                subject_id, run_id, save_dir = config_value
            else:
                raise ValueError(
                    f"Tuple must have 2 or 3 elements, got: {len(config_value)}"
                )

            # Generate path: {save_dir}/{subject_id}_{run_id}_dedup.db
            db_path = Path(save_dir) / f"{subject_id}_{run_id}_dedup.db"
            return str(db_path)
        else:
            raise ValueError(
                f"Cannot parse config value type {type(config_value)}: {config_value}"
            )

    def _load_historical_points(self) -> None:
        """Load historical sampling points from deduplication database."""
        try:
            if not self._dedup_conn:
                return

            cursor = self._dedup_conn.execute(
                "SELECT param_name, param_value, iteration_id FROM param_data "
                "ORDER BY iteration_id, param_name"
            )

            current_iteration = None
            current_point_tuple = []
            historical_points_set = set()

            for param_name, param_value, iteration_id in cursor.fetchall():
                if iteration_id != current_iteration:
                    if current_point_tuple:
                        historical_points_set.add(tuple(sorted(current_point_tuple)))
                    current_iteration = iteration_id
                    current_point_tuple = []

                current_point_tuple.append(param_value)

            # Add last point
            if current_point_tuple:
                historical_points_set.add(tuple(sorted(current_point_tuple)))

            self._historical_points = historical_points_set
            logger.info(
                f"[DedupDB] Loaded {len(historical_points_set)} historical sampling points"
            )

        except Exception as e:
            logger.error(f"[DedupDB] Failed to load historical points: {e}")
            self._historical_points = set()

    def record_points(self, points: torch.Tensor) -> None:
        """
        Record selected sampling points to deduplication database.

        Args:
            points (torch.Tensor): Selected sampling points [num_points x dim]
        """
        try:
            if not self._dedup_conn:
                return

            if points.shape[0] == 0:
                return

            # Get current max iteration_id
            cursor = self._dedup_conn.execute(
                "SELECT MAX(iteration_id) FROM param_data"
            )
            max_iter = cursor.fetchone()[0]
            next_iter = (max_iter or 0) + 1

            # Insert each point's parameters
            points_np = points.cpu().numpy() if points.is_cuda else points.numpy()

            for point_idx, point in enumerate(points_np):
                current_iter = next_iter + point_idx
                point_tuple = tuple(point)

                # Create parameter entry for each dimension
                for dim_idx, value in enumerate(point):
                    param_name = f"param_{dim_idx}"

                    self._dedup_conn.execute(
                        "INSERT INTO param_data (param_name, param_value, iteration_id) "
                        "VALUES (?, ?, ?)",
                        (param_name, float(value), current_iter),
                    )

                # Update in-memory historical points set
                self._historical_points.add(point_tuple)

            self._dedup_conn.commit()
            logger.debug(f"[DedupDB] Recorded {points.shape[0]} points to database")

        except Exception as e:
            logger.error(f"[DedupDB] Failed to record points: {e}")

    def close(self) -> None:
        """Close database connection."""
        try:
            if self._dedup_conn:
                self._dedup_conn.close()
                self._dedup_conn = None
        except Exception as e:
            logger.debug(f"[DedupDB] Error closing database: {e}")

    def get_historical_points(self) -> Set[Tuple[float, ...]]:
        """Get set of historical sampling points."""
        return self._historical_points.copy()

    @property
    def is_temp_db(self) -> bool:
        """Check if using temporary database."""
        return self._is_temp_db

    def __del__(self):
        """Destructor: close database connection."""
        self.close()
