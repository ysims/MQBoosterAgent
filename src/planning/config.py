"""Planning tuning constants: path planners, obstacle geometry, kick targets."""

from __future__ import annotations

import math


# ======================================================================
# Kick power
# ======================================================================

# Kick power during normal play.
KICK_POWER_DEFAULT = 5.0
KICK_POWER_BACKFIELD = 5.0

# ======================================================================
# Kick-target geometry
# ======================================================================

GOAL_TARGET_DEPTH_M = 0.25              # Kick target depth inside the goal (m)

# ======================================================================
# Obstacle geometry
# ======================================================================

BALL_OBSTACLE_RADIUS = 0.5              # Ball obstacle radius (m)
OPPONENT_RADIUS = 0.55                  # Opponent robot radius (m)
TEAMMATE_RADIUS = 0.48                  # Teammate robot radius (m)
SAFETY_MARGIN = 0.22                    # General safety margin (m)

GOAL_DEPTH = 0.6                        # Goal depth used for obstacle modeling (m)
POST_RADIUS = 0.18                      # Goal-post radius (m)
NET_RADIUS = 0.20                       # Goal-net sample radius (m)
NET_STEP = 0.35                         # Goal-net sampling interval (m)

START_IGNORE = 0.0                      # Ignore obstacles near the start (m)
TARGET_IGNORE = 0.0                     # Ignore obstacles near the target (m)

# ======================================================================
# Global path planner (A* grid planner)
# ======================================================================

USE_GLOBAL_PATH_PLANNER = True          # Fall back to local VFH when disabled
GLOBAL_GRID_RESOLUTION_M = 0.35         # Grid cell size; smaller is finer (m)
GLOBAL_FIELD_MARGIN_M = 0.25            # Extra field margin for boundary paths (m)
GLOBAL_OBSTACLE_MARGIN_M = 0.10         # Additional obstacle inflation radius (m)
GLOBAL_PATH_LOOKAHEAD_M = 0.9           # Lookahead distance along the path (m)

# ======================================================================
# Local path planner (VFH direction scan)
# ======================================================================

PLAN_LOOKAHEAD = 1.2                    # Forward probe length (m)
PLAN_CLEARANCE = 0.35                   # Minimum clearance for a candidate (m)
PLAN_STEP = math.radians(15)            # Candidate direction scan step (rad)
PLAN_MAX_OFFSET = math.radians(100)     # Maximum offset from target heading (rad)

# ======================================================================
# Gaze planning (head tracking)
# ======================================================================

# set_head_angle(pitch, yaw): positive pitch is down, positive yaw is left,
# both in radians. The camera is fixed to the head, not the torso, and a
# forward-level head loses sight of the ball once it's close enough to sit
# below the camera's forward-facing field of view -- observed live as
# "ball unknown" warnings specifically at close range, right when aiming a
# kick matters most. Pitch ramps from HEAD_PITCH_FAR to HEAD_PITCH_NEAR as
# the walk/chase target gets closer than HEAD_LOOK_DOWN_RANGE_M, on the
# assumption that such a target is a reasonable proxy for "where the ball
# probably still is" even after it drops out of view. These are
# conservative starting values, not measured against the K1's actual joint
# limits -- set_head_angle() doesn't error on an out-of-limit value, it
# just silently stops responding, so retune down rather than up if the
# head appears to stop moving.
HEAD_PITCH_FAR = 0.25           # Head pitch (rad) at/beyond HEAD_LOOK_DOWN_RANGE_M
HEAD_PITCH_NEAR = 0.55          # Head pitch (rad) at distance 0
HEAD_LOOK_DOWN_RANGE_M = 2.5    # Distance over which pitch ramps far->near (m)
HEAD_YAW_MAX = 0.6              # Clamp on head yaw (rad) either direction

# ======================================================================
# Search planning (reacquiring a lost ball)
# ======================================================================

# How long to trust a remembered ball position after this player's own
# detection lapses. Longer than BALL_DETECTION_MAX_AGE_SEC (the per-robot
# detection freshness window in localisation/config.py) on purpose --
# search only starts once a reading has ALREADY gone stale by that window,
# so this needs enough margin to cover a real, if brief, tracking gap
# rather than immediately giving up.
SEARCH_MEMORY_SEC = 5.0

SEARCH_TURN_RATE = 1.0  # In-place turn rate (rad/s) once memory goes stale

# ======================================================================
# Visualization
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18
