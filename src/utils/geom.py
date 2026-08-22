"""Stateless geometry helpers operating on Context or coordinates.

These framework-provided examples can be read, changed, or forked. Users can
also add utilities such as out-of-bounds checks. These calculations remain
valid across strategies and are shared by navigation and strategy code.

Coordinate system: team-relative field view, +x toward the opponent's goal,
-x toward our goal, with the field center at (0, 0).
"""

from __future__ import annotations

import math

from ..framework.types import Context
from ..param import GOAL_TARGET_DEPTH_M


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def normalize_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def deg2rad(deg: float) -> float:
    """Convert degrees to radians."""
    return math.radians(deg)


def rad2deg(rad: float) -> float:
    """Convert radians to degrees."""
    return math.degrees(rad)


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def angle_to(fx: float, fy: float, tx: float, ty: float) -> float:
    """Return the field-relative angle from (fx, fy) to (tx, ty)."""
    return math.atan2(ty - fy, tx - fx)


def opponent_goal(ctx: Context) -> tuple[float, float]:
    """Return a point behind the opponent's goal line for stable aiming."""
    return (ctx.field.length / 2.0 + GOAL_TARGET_DEPTH_M, 0.0)


def own_goal(ctx: Context) -> tuple[float, float]:
    """Return the center of our goal, the defensive reference point."""
    return (-ctx.field.length / 2.0, 0.0)


def own_goal_area_center(ctx: Context) -> tuple[float, float]:
    """Return the default goalkeeper position at our goal-area center.

    The goal area extends ``goal_area_length`` along +x from our goal line, so
    its center lies half that distance inside the field.
    """
    return (-ctx.field.length / 2.0 + ctx.field.goal_area_length / 2.0, 0.0)


def clamp_inside_field(
    ctx: Context, x: float, y: float, margin: float = 2.0,
) -> tuple[float, float]:
    """Clamp (x, y) inside the rectangular field with the given margin."""
    half_l = ctx.field.length / 2.0 - margin
    half_w = ctx.field.width / 2.0 - margin
    return (clamp(x, -half_l, half_l), clamp(y, -half_w, half_w))
