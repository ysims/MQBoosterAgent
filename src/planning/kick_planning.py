"""Kick planning: aim/power decisions and defensive block-path projection.

These are pure decision functions -- they choose *where* and *how hard* to
kick, or *where* to stand to block, but never issue commands.
``strategy.player.Player`` calls into these and passes the result to
``motion``. ``ball`` is always taken as an explicit parameter rather than
read off ``context``, since the caller already has its own resolved reading
to pass in (see ``Context.ball``'s docstring in ``framework/types.py``).
"""

from __future__ import annotations

import math

from ..framework.types import BallState, Context
from ..utils.geom import angle_to, clamp, dist, opponent_goal
from .config import KICK_POWER_BACKFIELD, KICK_POWER_DEFAULT


__all__ = [
    "plan_kick",
    "in_backfield",
    "kick_can_score",
    "block_path_projection",
]


def plan_kick(
    context: Context | None, ball: BallState | None,
) -> tuple[float, float] | None:
    """Calculate kick direction and power.

    Aim from the current ball position toward the opponent's goal center.
    Return ``(kick_direction, kick_power)``, or None when the ball or context
    is unavailable.
    """
    if context is None or ball is None:
        return None

    kick_target = opponent_goal(context)
    kick_direction = angle_to(ball.x, ball.y, *kick_target)
    kick_power = (
        KICK_POWER_BACKFIELD if in_backfield(ball)
        else KICK_POWER_DEFAULT
    )
    return kick_direction, kick_power


def in_backfield(ball: BallState | None) -> bool:
    """Return whether the ball is in our half, where kicks use more power."""
    if ball is None:
        return False
    return ball.x < 0


def kick_can_score(
    context: Context | None, ball: BallState | None, kick_direction: float,
) -> bool:
    """Return whether a straight kick in ``kick_direction`` can score."""
    if context is None or ball is None:
        return False

    goal_x = context.field.length / 2.0
    dx = math.cos(kick_direction)
    dy = math.sin(kick_direction)
    if dx <= 1e-6:
        return False

    half_goal = context.field.goal_width / 2.0
    if half_goal <= 0.0:
        return False
    if ball.x >= goal_x:
        y_at_goal = ball.y
    else:
        t = (goal_x - ball.x) / dx
        if t < 0.0:
            return False
        y_at_goal = ball.y + dy * t
    return -half_goal <= y_at_goal <= half_goal


def block_path_projection(
    context: Context | None,
    ball: BallState | None,
    pose_xy: tuple[float, float] | None,
    opponent_id: int,
) -> tuple[float, float, float, float] | None:
    """Project a player position onto an opponent-to-ball segment.

    Return ``(x, y, perpendicular_distance, segment_parameter_t)``.
    """
    opponent = context.opponents.get(opponent_id) if context is not None else None
    if context is None or pose_xy is None or ball is None or opponent is None:
        return None
    if opponent.pose is None:
        return None

    px, py = pose_xy
    ax, ay = opponent.pose.x, opponent.pose.y
    bx, by = ball.x, ball.y
    vx, vy = bx - ax, by - ay
    length2 = vx * vx + vy * vy
    if length2 < 1e-6:
        return None

    raw_t = ((px - ax) * vx + (py - ay) * vy) / length2
    t = clamp(raw_t, 0.0, 1.0)
    tx = ax + vx * t
    ty = ay + vy * t
    return tx, ty, dist(px, py, tx, ty), raw_t
