"""Pure, testable obstacle collection and single-obstacle detouring.

Ported from the legacy MotionController path-detour layer. It forms a corridor
from the start to the target, finds the first blocking circular obstacle, and
generates a via point beside it. The caller remembers the chosen side across
frames; ``walk_to`` stores it in ``self._avoid_side``.

Obstacles are optional; for example, the ball is an obstacle only during an
opponent restart. See the ``ball`` and ``robots`` flags on collect_obstacles.
Radii and related constants are configurable here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..framework.types import Context
from ..param import (
    BALL_OBSTACLE_RADIUS,
    GOAL_DEPTH,
    NET_RADIUS,
    NET_STEP,
    OPPONENT_RADIUS,
    POST_RADIUS,
    SAFETY_MARGIN,
    START_IGNORE,
    TARGET_IGNORE,
    TEAMMATE_RADIUS,
)


__all__ = ["Obstacle", "collect_obstacles", "goal_obstacles", "detour"]


@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float


def collect_obstacles(
    context: Context,
    exclude_id: int,
    *,
    ball: bool,
    robots: bool,
    goals: bool = False,
) -> list[Obstacle]:
    """Collect circular ball, robot, and goal obstacles as requested."""
    obstacles: list[Obstacle] = []
    if ball and context.ball is not None:
        obstacles.append(
            Obstacle(context.ball.x, context.ball.y, BALL_OBSTACLE_RADIUS)
        )
    if robots:
        for r in context.opponents.values():
            if r.pose is not None:
                obstacles.append(Obstacle(r.pose.x, r.pose.y, OPPONENT_RADIUS))
        for tid, r in context.teammates.items():
            if tid != exclude_id and r.pose is not None:
                obstacles.append(Obstacle(r.pose.x, r.pose.y, TEAMMATE_RADIUS))
    if goals:
        obstacles.extend(goal_obstacles(context))
    return obstacles


def goal_obstacles(context: Context) -> list[Obstacle]:
    """Model both goals as impassable U shapes using posts and net samples."""
    f = context.field
    half_l = f.length / 2.0
    half_gw = f.goal_width / 2.0
    obstacles: list[Obstacle] = []
    for sign_x in (-1.0, 1.0):
        front_x = sign_x * half_l
        back_x = sign_x * (half_l + GOAL_DEPTH)
        for sign_y in (-1.0, 1.0):               # Four posts: two front and two rear.
            obstacles.append(Obstacle(front_x, sign_y * half_gw, POST_RADIUS))
            obstacles.append(Obstacle(back_x, sign_y * half_gw, POST_RADIUS))
        # Back net.
        obstacles += _sample_segment(
            back_x, -half_gw, back_x, half_gw, NET_STEP, NET_RADIUS,
        )
        # Side nets.
        for sign_y in (-1.0, 1.0):
            obstacles += _sample_segment(
                front_x, sign_y * half_gw, back_x, sign_y * half_gw,
                NET_STEP, NET_RADIUS,
            )
    return obstacles


def _sample_segment(
    x0: float, y0: float, x1: float, y1: float, step: float, radius: float,
) -> list[Obstacle]:
    """Sample circular obstacles along a segment, excluding post endpoints."""
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= step:
        return []
    n = max(1, int(length / step) - 1)
    return [
        Obstacle(
            x0 + (x1 - x0) * (i + 1) / (n + 1),
            y0 + (y1 - y0) * (i + 1) / (n + 1),
            radius,
        )
        for i in range(n)
    ]


def detour(
    sx: float, sy: float, tx: float, ty: float,
    obstacles: list[Obstacle],
    side_hint: float | None,
) -> tuple[tuple[float, float], float | None]:
    """Detour around the first obstacle on the path from start to target.

    Return the target, possibly replaced with a via point, and the chosen side.
    With no obstacle, return the original target and None so the caller can
    clear its side memory. ``side_hint`` preserves the previous frame's side
    and prevents oscillation.
    """
    blocker = _first_blocking_obstacle(sx, sy, tx, ty, obstacles)
    if blocker is None:
        return (tx, ty), None
    side = side_hint if side_hint is not None else _choose_side(sx, sy, tx, ty, blocker)
    via = _via_point(sx, sy, tx, ty, blocker, side)
    return via, side


def _first_blocking_obstacle(
    sx: float, sy: float, tx: float, ty: float, obstacles: list[Obstacle],
) -> Obstacle | None:
    """Find the nearest obstacle that actually blocks the path corridor."""
    seg_dx, seg_dy = tx - sx, ty - sy
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len < 1e-6:
        return None
    dir_x, dir_y = seg_dx / seg_len, seg_dy / seg_len
    left_x, left_y = -dir_y, dir_x
    best: Obstacle | None = None
    best_along = 0.0
    for obs in obstacles:
        rel_x, rel_y = obs.x - sx, obs.y - sy
        along = rel_x * dir_x + rel_y * dir_y
        if along <= START_IGNORE or along >= seg_len - TARGET_IGNORE:
            continue
        lateral = abs(rel_x * left_x + rel_y * left_y)
        if lateral >= obs.radius + SAFETY_MARGIN:
            continue
        if best is None or along < best_along:
            best, best_along = obs, along
    return best


def _choose_side(
    sx: float, sy: float, tx: float, ty: float, obstacle: Obstacle,
) -> float:
    """Choose the shorter detour: right (-1) or left (+1) of the path."""
    seg_dx, seg_dy = tx - sx, ty - sy
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len < 1e-6:
        return 1.0
    left_x, left_y = -seg_dy / seg_len, seg_dx / seg_len
    lateral = (obstacle.x - sx) * left_x + (obstacle.y - sy) * left_y
    return -1.0 if lateral > 0.0 else 1.0


def _via_point(
    sx: float, sy: float, tx: float, ty: float,
    obstacle: Obstacle, side_sign: float,
) -> tuple[float, float]:
    """Create a via point offset from the obstacle's projection on the path."""
    seg_dx, seg_dy = tx - sx, ty - sy
    seg_len = math.hypot(seg_dx, seg_dy)
    if seg_len < 1e-6:
        return (tx, ty)
    dir_x, dir_y = seg_dx / seg_len, seg_dy / seg_len
    left_x, left_y = -dir_y, dir_x
    along = (obstacle.x - sx) * dir_x + (obstacle.y - sy) * dir_y
    closest_x, closest_y = sx + dir_x * along, sy + dir_y * along
    offset = obstacle.radius + SAFETY_MARGIN
    return (
        closest_x + left_x * side_sign * offset,
        closest_y + left_y * side_sign * offset,
    )
