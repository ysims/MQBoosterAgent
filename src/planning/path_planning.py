"""Path planning: obstacle modeling, a global A* planner, and local VFH.

``strategy.player.Player.walk_to`` prefers the global A* planner
(:func:`plan_global_path`) and falls back to the local VFH direction scan
(:func:`plan_local_heading`) when no path is found or the planner is
disabled. Both operate on the same :class:`Obstacle` circles collected by
:func:`collect_obstacles`.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from ..framework.types import BallState, Context, Pose2D
from ..utils.geom import dist
from .config import (
    BALL_OBSTACLE_RADIUS,
    GLOBAL_FIELD_MARGIN_M,
    GLOBAL_GRID_RESOLUTION_M,
    GLOBAL_OBSTACLE_MARGIN_M,
    GLOBAL_PATH_LOOKAHEAD_M,
    GOAL_DEPTH,
    NET_RADIUS,
    NET_STEP,
    OPPONENT_RADIUS,
    PLAN_CLEARANCE,
    PLAN_LOOKAHEAD,
    PLAN_MAX_OFFSET,
    PLAN_STEP,
    POST_RADIUS,
    SAFETY_MARGIN,
    START_IGNORE,
    TARGET_IGNORE,
    TEAMMATE_RADIUS,
)


__all__ = [
    "Obstacle",
    "collect_obstacles",
    "goal_obstacles",
    "detour",
    "plan_global_path",
    "plan_local_heading",
    "path_waypoint",
]


# ======================================================================
# Obstacle modeling
# ======================================================================


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
    ball_state: BallState | None = None,
) -> list[Obstacle]:
    """Collect circular ball, robot, and goal obstacles as requested.

    ``ball_state`` is the calling player's own ball belief; pass it
    explicitly when ``ball=True``.
    """
    obstacles: list[Obstacle] = []
    if ball and ball_state is not None:
        obstacles.append(
            Obstacle(ball_state.x, ball_state.y, BALL_OBSTACLE_RADIUS)
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


# ======================================================================
# Global path planner (A* on an 8-neighbor grid)
# ======================================================================


def plan_global_path(
    context: Context,
    start: tuple[float, float],
    target: tuple[float, float],
    obstacles: list[Obstacle],
) -> list[tuple[float, float]] | None:
    """Return an A* path from start to target, or None if no path is found."""
    min_x, max_x, min_y, max_y = _bounds(context)
    sx, sy = _clamp_point(start[0], start[1], min_x, max_x, min_y, max_y)
    tx, ty = _clamp_point(target[0], target[1], min_x, max_x, min_y, max_y)

    start_idx = _to_idx(sx, sy, min_x, min_y)
    goal_idx = _to_idx(tx, ty, min_x, min_y)
    if start_idx == goal_idx:
        return [(tx, ty)]

    open_heap: list[tuple[float, int, tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (0.0, counter, start_idx))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start_idx: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal_idx:
            return _reconstruct(came_from, current, min_x, min_y, (tx, ty))
        closed.add(current)

        for neighbor, step_cost in _neighbors(current):
            x, y = _to_xy(neighbor, min_x, min_y)
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue
            if (
                neighbor != start_idx
                and neighbor != goal_idx
                and _blocked(x, y, obstacles)
            ):
                continue
            tentative = g_score[current] + step_cost
            if tentative >= g_score.get(neighbor, math.inf):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            heapq.heappush(
                open_heap,
                (tentative + _heuristic(neighbor, goal_idx), counter, neighbor),
            )

    return None


def _bounds(context: Context) -> tuple[float, float, float, float]:
    half_l = context.field.length / 2.0 - GLOBAL_FIELD_MARGIN_M
    half_w = context.field.width / 2.0 - GLOBAL_FIELD_MARGIN_M
    return -half_l, half_l, -half_w, half_w


def _clamp_point(
    x: float,
    y: float,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> tuple[float, float]:
    return (max(min_x, min(max_x, x)), max(min_y, min(max_y, y)))


def _to_idx(x: float, y: float, min_x: float, min_y: float) -> tuple[int, int]:
    return (
        int(round((x - min_x) / GLOBAL_GRID_RESOLUTION_M)),
        int(round((y - min_y) / GLOBAL_GRID_RESOLUTION_M)),
    )


def _to_xy(idx: tuple[int, int], min_x: float, min_y: float) -> tuple[float, float]:
    return (
        min_x + idx[0] * GLOBAL_GRID_RESOLUTION_M,
        min_y + idx[1] * GLOBAL_GRID_RESOLUTION_M,
    )


def _neighbors(idx: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    x, y = idx
    out: list[tuple[tuple[int, int], float]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            cost = math.sqrt(2.0) if dx != 0 and dy != 0 else 1.0
            out.append(((x + dx, y + dy), cost))
    return out


def _blocked(x: float, y: float, obstacles: list[Obstacle]) -> bool:
    for obs in obstacles:
        if math.hypot(x - obs.x, y - obs.y) <= obs.radius + GLOBAL_OBSTACLE_MARGIN_M:
            return True
    return False


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _reconstruct(
    came_from: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
    min_x: float,
    min_y: float,
    target: tuple[float, float],
) -> list[tuple[float, float]]:
    indices = [current]
    while current in came_from:
        current = came_from[current]
        indices.append(current)
    indices.reverse()
    path = [_to_xy(idx, min_x, min_y) for idx in indices]
    path[-1] = target
    return _smooth_path(path)


def _smooth_path(path: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    prev_dx = 0
    prev_dy = 0
    for i in range(1, len(path)):
        dx = _sign(path[i][0] - path[i - 1][0])
        dy = _sign(path[i][1] - path[i - 1][1])
        if i > 1 and (dx, dy) != (prev_dx, prev_dy):
            smoothed.append(path[i - 1])
        prev_dx, prev_dy = dx, dy
    smoothed.append(path[-1])
    return smoothed


def _sign(v: float) -> int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def path_waypoint(
    pose: Pose2D, path: list[tuple[float, float]],
) -> tuple[float, float]:
    """Pick a short lookahead waypoint from a planned global path."""
    if not path:
        return (pose.x, pose.y)
    prev = (pose.x, pose.y)
    points = path[1:] if len(path) > 1 else path
    for point in points:
        seg_len = dist(prev[0], prev[1], point[0], point[1])
        if seg_len >= GLOBAL_PATH_LOOKAHEAD_M:
            ratio = GLOBAL_PATH_LOOKAHEAD_M / max(seg_len, 1e-6)
            return (
                prev[0] + (point[0] - prev[0]) * ratio,
                prev[1] + (point[1] - prev[1]) * ratio,
            )
        prev = point
    return path[-1]


# ======================================================================
# Local path planner (VFH direction scan), used when the global planner
# is disabled or finds no path
# ======================================================================


def _heading_clearance(
    px: float, py: float, heading: float, obstacles: list[Obstacle],
) -> float:
    """Return clearance along a lookahead ray for local obstacle avoidance.

    Only obstacles ahead of (px, py), with projection t > 0, block this heading.
    Ignoring obstacles behind or to the rear prevents a nearby rear obstacle
    from reducing clearance in every direction. Return infinity when clear;
    larger values mean more space and negative values indicate a collision.
    """
    ux, uy = math.cos(heading), math.sin(heading)
    min_clear = math.inf
    for obs in obstacles:
        t = (obs.x - px) * ux + (obs.y - py) * uy
        if t <= 0.0:
            continue                      # Rear obstacles do not block this heading.
        if t > PLAN_LOOKAHEAD:
            t = PLAN_LOOKAHEAD
        nx, ny = px + ux * t, py + uy * t
        clear = math.hypot(obs.x - nx, obs.y - ny) - obs.radius
        if clear < min_clear:
            min_clear = clear
    return min_clear


def plan_local_heading(
    player_id: int, pose: Pose2D, goal_dir: float, obstacles: list[Obstacle],
) -> float:
    """Choose the clearest candidate heading nearest the target direction.

    Candidates are tested by increasing absolute offset from the target.
    Player ID parity chooses which side is tried first, breaking symmetry
    when two players avoid each other.
    """
    sign_first = 1.0 if player_id % 2 == 0 else -1.0
    best_h = goal_dir
    best_clear = -math.inf

    offsets = [0.0]
    k = 1
    while k * PLAN_STEP <= PLAN_MAX_OFFSET + 1e-9:
        offsets.append(sign_first * k * PLAN_STEP)
        offsets.append(-sign_first * k * PLAN_STEP)
        k += 1

    for off in offsets:
        h = goal_dir + off
        clear = _heading_clearance(pose.x, pose.y, h, obstacles)
        if clear >= PLAN_CLEARANCE:
            return h
        if clear > best_clear:
            best_clear, best_h = clear, h
    return best_h
