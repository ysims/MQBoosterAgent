"""Vision data contract: pure dataclasses, no ROS dependency.

This module has no ROS or boosteros dependencies, mirroring the rule in
``framework/types.py``, so it can be imported, tested, and reloaded alone.

Coordinate conventions:
- Pixel frame: ``x_px``/``y_px`` are image-space pixel coordinates (origin
  top-left, +x right, +y down), matching the bounding boxes reported by
  ``/soccer/sim/vision/detections``.
- Field frame: team-relative, +x toward the opponent's goal, matching
  ``framework.types.Context``.
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["Detection2D", "FieldDetection"]


@dataclass(frozen=True)
class Detection2D:
    """A pixel-space bounding-box detection.

    ``w_px``/``h_px`` are the bounding box's pixel width/height -- how large
    the object *appears*, which is what lets a monocular camera estimate
    distance at all (see ``vision/ball_detection.py``'s
    ``estimate_ball_position``).
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

    ``label`` lets downstream code filter by type. ``source_robot_id``
    records which of our own robots produced this detection, for debugging
    and per-robot tracking.
    """

    x: float
    y: float
    label: str
    confidence: float
    source_robot_id: int
    timestamp: float
