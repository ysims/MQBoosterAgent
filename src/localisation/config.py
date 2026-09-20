"""Localisation tuning constants: odom calibration anchor and freshness."""

from __future__ import annotations


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
