"""Ball localisation: each robot's own perceived ball position.

Deliberately NOT fused or averaged across robots -- a robot's ball belief
comes only from its own detections, exactly like each robot's own self-pose
comes only from its own odometry (see ``odometry.dead_reckoning``). Pooling
robots' independent (and independently drifting) beliefs into one shared
value hides disagreement instead of surfacing it, and lets one robot's bad
reading silently corrupt another's. ``Context.ball`` is keyed by player_id
for exactly this reason -- see its docstring in ``framework/types.py``.
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
