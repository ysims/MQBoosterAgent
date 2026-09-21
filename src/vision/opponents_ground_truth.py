"""Opponent position tracking.

The sim's own vision detector (``detection_extension``) reports the ball,
goalposts, and field markers, but never other robots -- so there is
currently no vision-based way to perceive opponents. This tracks each
opponent's field-frame pose from the simulator's own pose topic for that
robot instead, so ``Context.opponents`` isn't left permanently empty.

This is a lightweight helper subscribing on ``VisionContextSource``'s own
node (passed in), not a standalone ``ContextSource`` -- it owns no node,
executor, or spin thread of its own.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from geometry_msgs.msg import Pose2D as RosPose2D
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

from ..framework.config import SoccerConfig
from ..framework.types import Pose2D, RobotState


__all__ = ["OpponentGroundTruthTracker"]


class OpponentGroundTruthTracker:
    """Track each opponent's pose via ``/team{id}/{robot_name}/...``."""

    def __init__(self, node: Any, config: SoccerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._opponents: dict[int, RobotState] = {
            pid: RobotState(player_id=pid)
            for pid in range(1, len(config.opponent_robot_names) + 1)
        }

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._subscriptions = [
            node.create_subscription(
                RosPose2D,
                self._robot_topic(name, "soccer/sim/ground_truth/robot_pose"),
                self._make_pose_cb(pid),
                qos,
            )
            for pid, name in enumerate(config.opponent_robot_names, start=1)
        ]

    def get_snapshot(self) -> dict[int, RobotState]:
        with self._lock:
            return dict(self._opponents)

    def _make_pose_cb(self, player_id: int):
        def callback(msg: Any) -> None:
            pose = Pose2D(x=float(msg.x), y=float(msg.y), theta=float(msg.theta))
            with self._lock:
                self._opponents[player_id] = RobotState(
                    player_id=player_id, pose=pose, last_seen_at=time.monotonic(),
                )
        return callback

    def _robot_topic(self, robot_name: str, suffix: str) -> str:
        return self._join(f"team{self._config.team_id}", robot_name, suffix)

    @staticmethod
    def _join(*parts: str) -> str:
        clean = [p.strip("/") for p in parts if p.strip("/")]
        return "/" + "/".join(clean)
