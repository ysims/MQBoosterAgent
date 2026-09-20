"""Utility layer with framework examples and user-defined helpers.

These pure, stateless functions depend only on the framework.types data
contract and can be reused independently. Users may edit this package or add
modules for tasks such as out-of-bounds detection or pass scoring.

- geom: geometry helpers (opponent_goal, dist, angle_to, clamp, and
  clamp_inside_field)

Obstacle avoidance and path planning (Obstacle, collect_obstacles, and the
global/local planners) live in the planning package, since they are
decisions rather than pure math. Movement and readiness operations such as
``walk_to``, ``face_to``, and ``ensure_ready`` issue commands to a player and
require cross-frame state, so they are Player methods in
strategy/player.py rather than utilities.
"""

from .geom import (
    angle_to,
    clamp,
    clamp_inside_field,
    deg2rad,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
    own_goal_area_center,
    rad2deg,
)

__all__ = [
    "angle_to",
    "clamp",
    "clamp_inside_field",
    "deg2rad",
    "dist",
    "normalize_angle",
    "opponent_goal",
    "own_goal",
    "own_goal_area_center",
    "rad2deg",
]
