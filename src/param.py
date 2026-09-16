"""Central configuration point for strategy tuning.

This module contains tunable strategy, movement, kicking, and obstacle
avoidance parameters. Existing values were moved here from their original
modules; make future parameter adjustments here first.
"""

from __future__ import annotations

import math


# ======================================================================
# Kick power
# ======================================================================

# Player.kick() clamps power to this range. Unsupported values may cause a fall.
KICK_POWER_MIN = 1.0
KICK_POWER_MAX = 10.0

# Kick power during normal play.
KICK_POWER_DEFAULT = 5.0
KICK_POWER_BACKFIELD = 5.0
KICK_POWER_OUR_KICKOFF = 5.0


# ======================================================================
# Player movement control
# ======================================================================

ARRIVE_DIST = 0.15             # Target arrival threshold (m)
OMNI_DIST = 1.0                # Use omnidirectional control below this distance
TURN_THRESHOLD = 0.5           # Turn in place above this heading error (rad)
MAX_LINEAR = 2.0               # Maximum commanded forward speed (m/s)
MAX_ANGULAR = 2.0              # Maximum commanded angular speed (rad/s)
LINEAR_GAIN = 1.5              # Translation proportional gain
ANGULAR_GAIN = 2.0             # Rotation proportional gain


# ======================================================================
# Player kicking and shot planning
# ======================================================================

KICK_ENTER_M = 2.0             # Enter kicking state below this ball distance
KICK_EXIT_M = 2.5              # Leave kicking state above this ball distance
CHASE_BEHIND_M = 0.35          # Distance behind the ball while chasing


# ======================================================================
# Player skill parameters: guarding and support
# ======================================================================

GUARD_FACE_BALL = True
GUARD_THREAT_ENTER_X = -1.0
GUARD_THREAT_EXIT_X = -0.7

SUPPORT_DIST_M = 3.0

# ======================================================================
# Normal-phase strategy
# ======================================================================

ATTACKER_KEEP_DIST_MARGIN_M = 0.3  # Prevent attacker-selection oscillation

FALLEN_COST = 10.0  # Distance penalty assigned to a fallen player (m)


# ======================================================================
# Kickoff and set-play strategy
# ======================================================================

# Kickoff
KICKOFF_STAGE_M = 2.0
KICKOFF_FRONT_MARGIN = 0.1
KICKOFF_LATERAL_TOL = 0.35

CENTER_LEAVE_DIST_M = 0.15 # Distance from center at which the ball counts as moved

OPP_SET_WALL_DIST_M = 2.0 # Blocking distance during an opponent kickoff


# ======================================================================
# Positioning and avoidance
# ======================================================================

OPPONENT_RESTART_AVOID_M = 1.65
CIRCLE_MARGIN_M = 0.3


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
# Vision / localization tuning
# ======================================================================

# Fixed pinhole camera intrinsics (pixels), calibrated from the sim's own
# camera at 320x240 resolution. estimate_ball_position() only needs the
# horizontal axis (ground-plane position, not height); fy/cy would matter
# for anything using vertical pixel position, e.g. a ground-plane-
# intersection method.
CAMERA_FX = 216.5
CAMERA_CX = 168.0

# Ball's diameter (m), used with CAMERA_FX to turn "how large the ball's
# bounding box appears" into a distance estimate (similar triangles):
# distance = (BALL_DIAMETER_M * CAMERA_FX) / apparent_size_px.
BALL_DIAMETER_M = 0.19

# Minimum apparent bounding-box size (px) to trust as a real detection
# rather than sensor noise.
MIN_RELIABLE_APPARENT_PX = 6.0

# OdomAnchoredLocaliser: treat the relayed pose as gone after this long silent.
LOCALISER_STALE_SEC = 1.0

# Odom -> field-frame calibration anchor, captured live at INITIAL-state spawn
# for this exact deployment (team1, robot1-3; see /robot{N}/odom vs
# /team1/robot{N}/soccer/sim/ground_truth/robot_pose, both read at match reset
# before any motion). /robot{N}/odom reliably boots at (x=0, y=0, yaw) with
# yaw already equal to field-frame theta (no rotation offset observed to
# ~0.001 rad), so only a fixed (x, y) translation per robot is needed:
#   field_x = odom_x + anchor_x, field_y = odom_y + anchor_y, field_theta = odom_yaw
# Re-derive these if deployed as team2 or with different robot_names.
ODOM_FIELD_ANCHOR: dict[int, tuple[float, float]] = {
    1: (-3.99997, 4.97330),
    2: (-4.99997, 4.97330),
    3: (-5.99998, 4.97335),
}

# Ball fusion across robots' detections.
BALL_DETECTION_MAX_AGE_SEC = 0.5    # Ignore ball detections older than this

# Opponent nearest-neighbor tracker.
OPPONENT_TRACK_MATCH_DIST_M = 0.75      # Max distance to match an existing track
OPPONENT_TRACK_MAX_MISS_FRAMES = 15     # Drop a track after this many missed ticks

# ======================================================================
# Visualization
# ======================================================================

KICK_TARGET_MARK_SIZE_M = 0.18
