"""Vision + localisation data contract: pure dataclasses, no ROS dependency.

This module has no ROS or boosteros dependencies, mirroring the rule in
``types.py``, so it can be imported, tested, and reloaded alone (e.g. from
``src/vision.py`` on a development machine without ROS installed).

Coordinate conventions:
- Pixel frame: ``x_px``/``y_px`` are image-space pixel coordinates (origin
  top-left, +x right, +y down), matching the bounding boxes reported by
  ``/soccer/sim/vision/detections``.
- Robot body frame: +x forward, +y left, +z up, matching ``Pose2D``.
- Field frame: team-relative, +x toward the opponent's goal, matching
  ``framework.types.Context``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .types import Pose2D


__all__ = [
    "Detection2D",
    "FieldDetection",
    "Localiser",
]


@dataclass(frozen=True)
class Detection2D:
    """A pixel-space bounding-box detection.
    
    ``w_px``/``h_px`` are the bounding box's pixel width/height -- how large
    the object *appears*, which is what lets a monocular camera estimate
    distance at all (see ``src/vision.py``'s ``estimate_ball_position``).
    """

    x_px: float
    y_px: float
    w_px: float
    h_px: float
    label: str
    confidence: float


@dataclass(frozen=True)
class FieldDetection:
    """One object detection, already resolved to field-frame coordinates.

    ``label`` lets downstream fusion (ball) and tracking (opponents) filter
    by type. ``source_robot_id`` records which of our own robots produced
    this detection, for debugging and per-robot fusion weighting.
    """

    x: float
    y: float
    label: str
    confidence: float
    source_robot_id: int
    timestamp: float


class Localiser(Protocol):
    """Per-robot self-pose estimator.

    ``VisionContextSource`` constructs one per robot as
    ``localiser_class(node, topic, anchor_x, anchor_y, mirrored)`` --
    ``topic`` is the robot's raw odometry topic, ``anchor_x``/``anchor_y``
    calibrate its arbitrary boot-time origin against the robot's known
    field-frame starting position (see ``ODOM_FIELD_ANCHOR`` in
    ``param.py``), and ``mirrored`` selects whether that team's field frame
    needs the 180 degree rotation relative to team1's (see
    ``VisionContextSource._create_robots``). Implementations own whatever ROS
    subscription(s) they need (given ``node``) and answer ``get_pose()`` from
    cached state. Returning ``None`` signals "no current pose estimate" (e.g.
    never received, or stale) and propagates to ``Context.teammates[id].pose``.
    """

    def __init__(
        self, node: object, topic: str,
        anchor_x: float, anchor_y: float, mirrored: bool,
    ) -> None: ...

    def get_pose(self) -> Pose2D | None: ...
