"""Shared camera projection math: turning an apparent size into a position.

Used by anything that estimates a robot-frame position from how large a
known-size object appears in a pixel bounding box -- ``ball_detection.py``
for the ball, ``localisation.landmark_localisation`` for goalposts.
"""

from __future__ import annotations

from .config import CAMERA_CX, CAMERA_FX, MIN_RELIABLE_APPARENT_PX


__all__ = ["estimate_position_from_apparent_size"]


def estimate_position_from_apparent_size(
    x_px: float, apparent_px: float, true_size_m: float,
) -> tuple[float, float] | None:
    """Estimate a robot-frame (forward, left) position from an apparent size.

    No depth sensor -- distance comes from an object's *apparent* size via
    the standard similar-triangles relationship: an object of known size
    ``true_size_m`` projects to a smaller bounding box dimension the farther
    away it is, in direct proportion to the camera's focal length:

        distance = (true_size * focal_length) / apparent_size_px

    That distance, plus the pixel offset of ``x_px`` from the image center,
    gives the lateral (camera-frame x, standard pinhole projection):
    ``x_cam = (x_px - cx) * distance / fx``. Converting to the robot's own
    body frame (+x forward, +y left) assumes the camera is mounted at the
    robot's own origin, facing straight ahead -- a real robot would also
    need a fixed mount offset here, but the K1's isn't currently calibrated.

    ``apparent_px`` is left for the caller to choose (e.g. averaging a
    bounding box's width and height for a roughly spherical object, or just
    the width for a thin vertical one), since the right choice depends on
    the object's shape, not this projection math. Returns ``None`` when
    ``apparent_px`` is too small to trust.
    """
    if apparent_px <= MIN_RELIABLE_APPARENT_PX:
        return None

    distance_m = (true_size_m * CAMERA_FX) / apparent_px
    x_cam = (x_px - CAMERA_CX) * distance_m / CAMERA_FX

    # Camera optical frame (x right, z forward) -> robot body frame
    # (x forward, y left).
    forward = distance_m
    left = -x_cam
    return forward, left
