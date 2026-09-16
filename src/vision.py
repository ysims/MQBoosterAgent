"""Perception implementations, designed for direct user editing.

This is the file to edit when improving perception: swap
``OdomAnchoredLocaliser`` for a real state estimator (EKF/particle filter
fusing IMU/odometry/detected landmarks) that corrects drift instead of
dead-reckoning forever. It satisfies the ``Localiser`` protocol in
``framework/vision_types.py`` and is wired in as the default by ``main.py``'s
``localiser_class`` hook -- swap the class there and nothing else in the
framework needs to change.

The sim's own ``detection_extension`` supplies the raw perception signal --
a pixel bounding box around the ball, per camera, respecting real
field-of-view and occlusion (see ``vision_source.py``'s module docstring)
-- but turning that bounding box into a 3D position is on us, same as it
would be with a real camera: no depth sensor, so ``estimate_ball_position``
below estimates distance from how large the ball *appears* (a smaller
bounding box means farther away), then projects that into the robot's own
body frame. ``vision_source.py`` calls this for every ball detection and
rotates the result into field-frame coordinates using the robot's own pose.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from .framework.types import Pose2D
from .framework.vision_types import Detection2D
from .utils.geom import normalize_angle
from .param import (
    BALL_DIAMETER_M,
    CAMERA_CX,
    CAMERA_FX,
    LOCALISER_STALE_SEC,
    MIN_RELIABLE_APPARENT_PX,
)


__all__ = ["OdomAnchoredLocaliser", "estimate_ball_position"]


_log = logging.getLogger(__name__)


def estimate_ball_position(detection: Detection2D) -> tuple[float, float] | None:
    """Estimate the ball's robot-frame (forward, left) position from its bbox.

    No depth sensor -- distance comes from the ball's *apparent* size via the
    standard similar-triangles relationship: a real object of known size
    ``BALL_DIAMETER_M`` projects to a smaller bounding box the farther away
    it is, in direct proportion to the camera's focal length:

        distance = (true_size * focal_length) / apparent_size_px

    That distance, plus the bounding box's horizontal pixel offset from the
    image center, gives the lateral (camera-frame x, standard pinhole
    projection): ``x_cam = (x_px - cx) * distance / fx``. The vertical pixel
    offset isn't needed: we only want the ball's ground-plane position, not
    its height. Converting to the robot's own body frame (+x forward, +y
    left) assumes the camera is mounted at the robot's own origin, facing
    straight ahead -- a real robot would also need a fixed mount offset
    here, but the K1's isn't currently calibrated, so this keeps that
    assumption explicit rather than guessing.

    Returns ``None`` for a degenerate (zero-size) bounding box.
    """
    apparent_px = (detection.w_px + detection.h_px) / 2.0
    if apparent_px <= MIN_RELIABLE_APPARENT_PX:
        return None

    distance_m = (BALL_DIAMETER_M * CAMERA_FX) / apparent_px
    x_cam = (detection.x_px - CAMERA_CX) * distance_m / CAMERA_FX

    # Camera optical frame (x right, z forward) -> robot body frame
    # (x forward, y left).
    forward = distance_m
    left = -x_cam
    return forward, left


class OdomAnchoredLocaliser:
    """Dead-reckon field-frame self-pose from raw wheel/gait odometry.

    Working basic default: there is no free "sim gives you a good pose"
    topic -- ``/robot{N}/odom`` (``nav_msgs/Odometry``) is the real signal a
    robot has, and it is relative: it boots at an arbitrary origin, not the
    field frame, and drifts. This class calibrates that origin against a
    fixed, pre-measured field-frame anchor (see ``ODOM_FIELD_ANCHOR`` in
    ``param.py``, captured once by comparing ``/robot{N}/odom`` against the
    sim's ground-truth topic at INITIAL-state spawn -- never at runtime) and
    reports ``odom + anchor`` thereafter.

    For team 1, this calibration reduces to a pure translation: live
    calibration showed ``/robot{N}/odom`` boots at position (0, 0) with its
    yaw already equal to team1's field-frame theta (no rotation offset), so
    ``field_theta = odom_yaw`` and ``field_x/y = odom_x/y + anchor``.

    ``/robot{N}/odom`` is a raw per-robot signal, not team-relative -- unlike
    the sim's own ground-truth topics, it uses one fixed world convention for
    every robot regardless of team. But each team's own field frame is
    team-relative (``+x`` toward *that* team's opponent goal; see
    ``utils/geom.py``), and since both teams start in an equivalent-looking
    formation from their own side (a competition-fairness requirement),
    team1's own frame and team2's own frame are related by a 180 degree
    rotation about the field center, not a mirror/reflection -- a true axis
    flip would invert handedness (clockwise vs. counterclockwise), which
    isn't physically consistent for two views of the same field. Passing
    ``mirrored=True`` (for any team other than team1) applies that inverse
    rotation before adding the anchor: ``field_x/y = anchor - odom_x/y``,
    ``field_theta = odom_yaw + pi``. The anchor constants themselves need no
    change between teams, since they are already team-relative.

    Odom drifts significantly even while standing still (bipedal balance
    sway), so this is a crude dead-reckoning estimate that degrades over a
    match -- the real upgrade path is fusing ``/imu/data`` and
    vision-derived field-line detections into an actual filter (EKF/particle
    filter) that periodically corrects the drift instead of trusting odom
    forever.
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
