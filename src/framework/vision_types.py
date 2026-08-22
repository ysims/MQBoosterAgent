"""Vision + localisation data contract: pure dataclasses, no ROS dependency.

This module has no ROS or boosteros dependencies, mirroring the rule in
``types.py``, so it can be imported, tested, and reloaded alone (e.g. from
``src/vision.py`` on a development machine without ROS installed).

Coordinate conventions:
- Pixel frame: ``x_px``/``y_px`` are image-space pixel coordinates (origin
  top-left, +x right, +y down), matching OpenCV / sensor_msgs/Image.
- Camera optical frame: +x right, +y down, +z forward (REP 103 optical
  convention), matching CameraInfo's ``K`` intrinsics matrix.
- Robot body frame: +x forward, +y left, +z up, matching ``Pose2D``.
- Field frame: team-relative, +x toward the opponent's goal, matching
  ``framework.types.Context``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .types import Pose2D


__all__ = [
    "CameraExtrinsics",
    "CameraIntrinsics",
    "Detection2D",
    "Detector",
    "FieldDetection",
    "Localiser",
    "project_to_field",
]


# ----------------------------------------------------------------------
# Detection and camera geometry types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Detection2D:
    """One pixel-space detection produced by a :class:`Detector`."""

    x_px: float
    y_px: float
    w_px: float
    h_px: float
    label: str  # "ball" | "robot"
    confidence: float
    camera_id: str
    timestamp: float


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics, taken directly from a CameraInfo ``K`` matrix."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class CameraExtrinsics:
    """Fixed robot-frame offset of the camera's optical center.

    ``yaw`` is the camera's forward-axis heading relative to the robot's own
    forward axis (0 means the camera looks straight ahead).
    """

    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class FieldDetection:
    """A detection projected into field-frame coordinates.

    ``label`` is carried over from the source :class:`Detection2D` so that
    downstream fusion (ball) and tracking (opponents) can filter by type.
    ``source_robot_id`` records which of our own robots produced this
    detection, for debugging and per-robot fusion weighting.
    """

    x: float
    y: float
    label: str
    confidence: float
    source_robot_id: int
    timestamp: float


def project_to_field(
    detection: Detection2D,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    robot_pose: Pose2D,
    source_robot_id: int,
) -> FieldDetection:
    """Project one pixel detection to a field-frame point.

    Pipeline: pixel -> camera ray (pinhole back-projection at ``depth_m``) ->
    robot body frame (via ``extrinsics``) -> field frame (via ``robot_pose``).
    Only the ground-plane (x, y) component is kept; camera/robot height is not
    modeled since ``Pose2D`` and the field frame are both 2D.
    """

    # Pixel -> camera optical frame (x right, y down, z forward).
    x_cam = (detection.x_px - intrinsics.cx) * depth_m / intrinsics.fx
    z_cam = depth_m

    # Camera optical frame -> robot body frame (x forward, y left), rotated by
    # the camera's fixed yaw offset and translated by its mount position.
    forward_local = z_cam
    left_local = -x_cam
    cos_c, sin_c = math.cos(extrinsics.yaw), math.sin(extrinsics.yaw)
    rx = extrinsics.x + forward_local * cos_c - left_local * sin_c
    ry = extrinsics.y + forward_local * sin_c + left_local * cos_c

    # Robot body frame -> field frame, using the robot's own field pose.
    cos_t, sin_t = math.cos(robot_pose.theta), math.sin(robot_pose.theta)
    field_x = robot_pose.x + rx * cos_t - ry * sin_t
    field_y = robot_pose.y + rx * sin_t + ry * cos_t

    return FieldDetection(
        x=field_x,
        y=field_y,
        label=detection.label,
        confidence=detection.confidence,
        source_robot_id=source_robot_id,
        timestamp=detection.timestamp,
    )


# ----------------------------------------------------------------------
# Pluggable perception protocols
# ----------------------------------------------------------------------


class Detector(Protocol):
    """Per-robot object detector, constructed with no arguments.

    ``VisionContextSource`` builds one instance per robot as
    ``detector_class()`` and calls ``detect`` on every synchronized RGB+depth
    pair; the detector itself should be stateless or self-contained (no ROS
    access) so it stays independently testable.
    """

    def detect(
        self, rgb: object, depth: object, intrinsics: CameraIntrinsics,
    ) -> list[Detection2D]: ...


class Localiser(Protocol):
    """Per-robot self-pose estimator.

    ``VisionContextSource`` constructs one per robot as
    ``localiser_class(node, topic, anchor_x, anchor_y)`` -- ``topic`` is the
    robot's raw odometry topic and ``anchor_x``/``anchor_y`` calibrate its
    arbitrary boot-time origin against the robot's known field-frame
    starting position (see ``ODOM_FIELD_ANCHOR`` in ``param.py``).
    Implementations own whatever ROS subscription(s) they need (given
    ``node``) and answer ``get_pose()`` from cached state. Returning ``None``
    signals "no current pose estimate" (e.g. never received, or stale) and
    propagates to ``Context.teammates[id].pose``.
    """

    def get_pose(self) -> Pose2D | None: ...
