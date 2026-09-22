"""Landmark-based localisation: correcting dead-reckoned pose with goalposts.

Wraps ``OdomAnchoredLocaliser`` for the baseline pose and separately
subscribes to the same robot's detections topic to also receive goalpost
sightings. Each visible goalpost gives a bearing and range (via the same
apparent-size technique used for the ball, see
``vision.projection.estimate_position_from_apparent_size``), which
resolves to a field-frame position under the current pose; matching that
against the nearest of the four known goalpost positions and blending the
resulting offset into a running correction keeps the reported pose closer
to the true one than dead reckoning alone.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

from ..framework.types import ADULT_FIELD_DIMENSIONS, Pose2D
from ..odometry.dead_reckoning import OdomAnchoredLocaliser
from ..vision.config import GOALPOST_DIAMETER_M
from ..vision.projection import estimate_position_from_apparent_size
from .config import LANDMARK_CORRECTION_GAIN


__all__ = ["GoalpostCorrectedLocaliser"]


_log = logging.getLogger(__name__)

_GOALPOST_CLASS_ID = "Goalpost"

# The four goalposts sit at the corners of both goal mouths.
_KNOWN_GOALPOSTS: list[tuple[float, float]] = [
    (sign_x * ADULT_FIELD_DIMENSIONS.length / 2.0,
     sign_y * ADULT_FIELD_DIMENSIONS.goal_width / 2.0)
    for sign_x in (-1.0, 1.0)
    for sign_y in (-1.0, 1.0)
]


class GoalpostCorrectedLocaliser:
    """Self-localiser combining odometry with goalpost-based correction."""

    def __init__(
        self, node: Any, topic: str, anchor_x: float, anchor_y: float,
        mirrored: bool = False,
    ) -> None:
        self._odom = OdomAnchoredLocaliser(node, topic, anchor_x, anchor_y, mirrored)
        self._lock = threading.Lock()
        self._correction_x = 0.0
        self._correction_y = 0.0

        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )
        from vision_msgs.msg import Detection2DArray

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        detections_topic = _detections_topic_from_odom_topic(topic)
        self._sub = node.create_subscription(
            Detection2DArray, detections_topic, self._on_detections, qos,
        )

    def get_pose(self) -> Pose2D | None:
        pose = self._odom.get_pose()
        if pose is None:
            return None
        with self._lock:
            dx, dy = self._correction_x, self._correction_y
        return Pose2D(x=pose.x + dx, y=pose.y + dy, theta=pose.theta)

    def _on_detections(self, msg: Any) -> None:
        pose = self._odom.get_pose()
        if pose is None:
            return

        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        offsets: list[tuple[float, float]] = []
        for det in msg.detections:
            if not det.results:
                continue
            if det.results[0].hypothesis.class_id != _GOALPOST_CLASS_ID:
                continue

            bbox = det.bbox
            # A goalpost is a thin, tall cylinder: its width in the image
            # constrains distance via its known diameter, but its height
            # reflects the post's own tall extent, not distance -- so only
            # the width is used here, unlike the ball's averaged width/height.
            robot_frame = estimate_position_from_apparent_size(
                bbox.center.position.x, bbox.size_x, GOALPOST_DIAMETER_M,
            )
            if robot_frame is None:
                continue
            forward, left = robot_frame

            observed_x = pose.x + forward * cos_t - left * sin_t
            observed_y = pose.y + forward * sin_t + left * cos_t

            known_x, known_y = min(
                _KNOWN_GOALPOSTS,
                key=lambda p: math.hypot(p[0] - observed_x, p[1] - observed_y),
            )
            offsets.append((known_x - observed_x, known_y - observed_y))

        if not offsets:
            return

        avg_dx = sum(o[0] for o in offsets) / len(offsets)
        avg_dy = sum(o[1] for o in offsets) / len(offsets)
        with self._lock:
            self._correction_x += LANDMARK_CORRECTION_GAIN * (avg_dx - self._correction_x)
            self._correction_y += LANDMARK_CORRECTION_GAIN * (avg_dy - self._correction_y)


def _detections_topic_from_odom_topic(odom_topic: str) -> str:
    """Derive the sibling detections topic from an odom topic string.

    Both are built from the same robot name (see
    ``VisionContextSource._flat_robot_topic``): ``{robot_name}/odom`` and
    ``{robot_name}/soccer/sim/vision/detections``.
    """
    prefix = odom_topic.rsplit("/", 1)[0]
    return f"{prefix}/soccer/sim/vision/detections"
