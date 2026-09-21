"""Dead-reckoning self-localiser, designed for direct user editing.

This is the file to edit when improving self-localisation: swap
``OdomAnchoredLocaliser`` for a real state estimator (EKF/particle filter
fusing IMU/odometry/detected landmarks) that corrects drift instead of
dead-reckoning forever. It satisfies the ``Localiser`` protocol in
``localisation/protocols.py`` and is wired in as the default by
``strategy/main.py``'s ``localiser_class`` hook -- swap the class there and
nothing else in the framework needs to change.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from ..framework.types import Pose2D
from ..localisation.config import LOCALISER_STALE_SEC
from ..utils.geom import normalize_angle


__all__ = ["OdomAnchoredLocaliser"]


_log = logging.getLogger(__name__)


class OdomAnchoredLocaliser:
    """Dead-reckon field-frame self-pose from raw wheel/gait odometry.

    ``/robot{N}/odom`` (``nav_msgs/Odometry``) is a relative signal: it boots
    at an arbitrary origin, not the field frame, and drifts. This class
    calibrates that origin against a fixed, pre-measured field-frame anchor
    (see ``ODOM_FIELD_ANCHOR`` in ``localisation/config.py``, captured once
    via a live measurement taken at INITIAL-state spawn -- never at runtime)
    and reports ``odom + anchor`` thereafter.

    For team 1, this calibration reduces to a pure translation: live
    calibration showed ``/robot{N}/odom`` boots at position (0, 0) with its
    yaw already equal to team1's field-frame theta (no rotation offset), so
    ``field_theta = odom_yaw`` and ``field_x/y = odom_x/y + anchor``.

    ``/robot{N}/odom`` is a raw per-robot signal, not team-relative -- it
    uses one fixed world convention for every robot regardless of team. But
    each team's own field frame is team-relative (``+x`` toward *that*
    team's opponent goal; see ``utils/geom.py``), and since both teams start
    in an equivalent-looking formation from their own side (a
    competition-fairness requirement), team1's own frame and team2's own
    frame are related by a 180 degree rotation about the field center, not a
    mirror/reflection -- a true axis flip would invert handedness (clockwise
    vs. counterclockwise), which isn't physically consistent for two views
    of the same field. Passing ``mirrored=True`` (for any team other than
    team1) applies that inverse rotation before adding the anchor:
    ``field_x/y = anchor - odom_x/y``, ``field_theta = odom_yaw + pi``. The
    anchor constants themselves need no change between teams, since they are
    already team-relative.

    Odom drifts significantly even while standing still (bipedal balance
    sway), so this is a crude dead-reckoning estimate that degrades over a
    match -- correcting that drift with a proper filter (EKF/particle filter)
    fusing ``/imu/data`` and other available signals is the natural next step.
    """

    def __init__(
        self, node: Any, topic: str, anchor_x: float, anchor_y: float,
        mirrored: bool = False,
    ) -> None:
        self._anchor_x = anchor_x
        self._anchor_y = anchor_y
        self._mirrored = mirrored
        self._lock = threading.Lock()
        self._pose: Pose2D | None = None
        self._last_msg_at: float | None = None

        from nav_msgs.msg import Odometry
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._sub = node.create_subscription(Odometry, topic, self._on_odom, qos)

    def _on_odom(self, msg: Any) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = 2.0 * math.atan2(orientation.z, orientation.w)
        if self._mirrored:
            pose = Pose2D(
                x=self._anchor_x - float(position.x),
                y=self._anchor_y - float(position.y),
                theta=normalize_angle(yaw + math.pi),
            )
        else:
            pose = Pose2D(
                x=float(position.x) + self._anchor_x,
                y=float(position.y) + self._anchor_y,
                theta=yaw,
            )
        with self._lock:
            self._pose = pose
            self._last_msg_at = time.monotonic()

    def get_pose(self) -> Pose2D | None:
        with self._lock:
            if self._pose is None or self._last_msg_at is None:
                return None
            if time.monotonic() - self._last_msg_at > LOCALISER_STALE_SEC:
                return None
            return self._pose
