"""Ball localisation: each robot's own perceived ball position.

A robot's ball belief comes from its own detections, tracked independently
per robot and keyed by player_id (see ``Context.ball``'s docstring in
``framework/types.py``).
"""

from __future__ import annotations

import threading
import time

from ..framework.types import BallState
from ..vision.types import FieldDetection
from .config import BALL_DETECTION_MAX_AGE_SEC


__all__ = ["PerRobotBallTracker"]


class PerRobotBallTracker:
    """Track each robot's own latest ball detection independently."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[int, FieldDetection] = {}

    def update(self, player_id: int, detection: FieldDetection) -> None:
        with self._lock:
            self._latest[player_id] = detection

    def get_all(self) -> dict[int, BallState]:
        """Return every robot's own fresh ball estimate, keyed by player_id."""
        now = time.monotonic()
        with self._lock:
            latest = dict(self._latest)
        return {
            pid: BallState(x=det.x, y=det.y, last_seen_at=now, confidence=det.confidence)
            for pid, det in latest.items()
            if now - det.timestamp <= BALL_DETECTION_MAX_AGE_SEC
        }
