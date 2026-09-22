"""Ball perception: turning a bounding box into a robot-frame position.

This is the file to edit when improving ball perception. The sim's own
``detection_extension`` supplies the raw signal -- a pixel bounding box
around the ball, per camera, respecting real field-of-view and occlusion
(see ``framework/vision_source.py``'s module docstring) -- but turning that
bounding box into a 3D position is on us, same as it would be with a real
camera: no depth sensor, so ``estimate_ball_position`` below estimates
distance from how large the ball *appears* (a smaller bounding box means
farther away), then projects that into the robot's own body frame.
``vision_source.py`` calls this for every ball detection and rotates the
result into field-frame coordinates using the robot's own pose.
"""

from __future__ import annotations

from .config import BALL_DIAMETER_M, CAMERA_CX, CAMERA_FX, MIN_RELIABLE_APPARENT_PX
from .types import Detection2D


__all__ = ["estimate_ball_position"]


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
    here, but the K1's isn't currently calibrated.

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
