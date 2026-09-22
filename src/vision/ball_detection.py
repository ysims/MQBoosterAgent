"""Ball perception: turning a bounding box into a robot-frame position.

The sim's own ``detection_extension`` supplies the raw signal -- a pixel
bounding box around the ball, per camera, respecting real field-of-view and
occlusion (see ``framework/vision_source.py``'s module docstring) -- but
turning that bounding box into a 3D position is on us, same as it would be
with a real camera: no depth sensor, so ``estimate_ball_position`` below
estimates distance from how large the ball *appears* (a smaller bounding box
means farther away), then projects that into the robot's own body frame.
``vision_source.py`` calls this for every ball detection and rotates the
result into field-frame coordinates using the robot's own pose.
"""

from __future__ import annotations

from .config import BALL_DIAMETER_M
from .projection import estimate_position_from_apparent_size
from .types import Detection2D


__all__ = ["estimate_ball_position"]


def estimate_ball_position(detection: Detection2D) -> tuple[float, float] | None:
    """Estimate the ball's robot-frame (forward, left) position from its bbox.

    The ball projects to a roughly circular bounding box regardless of
    viewing angle, so its width and height are averaged into one apparent
    size. The vertical pixel offset isn't used: we only want the ball's
    ground-plane position, not its height.
    """
    apparent_px = (detection.w_px + detection.h_px) / 2.0
    return estimate_position_from_apparent_size(detection.x_px, apparent_px, BALL_DIAMETER_M)
