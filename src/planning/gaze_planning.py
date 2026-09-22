"""Gaze planning: deciding where to point the head to keep the ball in view.

A pure decision function, like ``kick_planning`` -- it chooses a head
pitch/yaw, but never issues commands. ``strategy.player.Player.look_at``
calls into this and passes the result to ``motion``.
"""

from __future__ import annotations

import math

from ..framework.types import Pose2D
from ..utils.geom import clamp, normalize_angle
from .config import HEAD_LOOK_DOWN_RANGE_M, HEAD_PITCH_FAR, HEAD_PITCH_NEAR, HEAD_YAW_MAX


__all__ = ["plan_head_angle"]


def plan_head_angle(
    pose: Pose2D | None, target: tuple[float, float],
) -> tuple[float, float] | None:
    """Return ``(pitch, yaw)`` to point the head at ``target``, or None.

    The camera is fixed to the head; a level, forward-looking head loses
    the ball once it's close enough to sit below the camera's field of
    view -- the closer the target, the steeper the pitch needed to keep it
    in frame. ``target`` is usually a walk/chase target rather than a
    confirmed ball sighting, since by the time the ball is about to drop
    out of view is exactly when this needs to already be looking down.
    """
    if pose is None:
        return None

    tx, ty = target
    dx, dy = tx - pose.x, ty - pose.y
    distance = math.hypot(dx, dy)

    ramp = clamp(1.0 - distance / HEAD_LOOK_DOWN_RANGE_M, 0.0, 1.0)
    pitch = HEAD_PITCH_FAR + (HEAD_PITCH_NEAR - HEAD_PITCH_FAR) * ramp

    target_heading = math.atan2(dy, dx)
    yaw = clamp(
        normalize_angle(target_heading - pose.theta),
        -HEAD_YAW_MAX, HEAD_YAW_MAX,
    )
    return pitch, yaw
