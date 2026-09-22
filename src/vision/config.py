"""Vision tuning constants: camera intrinsics and object geometry."""

from __future__ import annotations


# Fixed pinhole camera intrinsics (pixels), calibrated from the sim's own
# camera at 320x240 resolution. estimate_ball_position() only needs the
# horizontal axis (ground-plane position, not height); fy/cy would matter
# for anything using vertical pixel position, e.g. a ground-plane-
# intersection method.
CAMERA_FX = 216.5
CAMERA_CX = 163.0

# Ball's diameter (m), used with CAMERA_FX to turn "how large the ball's
# bounding box appears" into a distance estimate (similar triangles):
# distance = (BALL_DIAMETER_M * CAMERA_FX) / apparent_size_px.
BALL_DIAMETER_M = 0.205

# Minimum apparent bounding-box size (px) to trust as a real detection
# rather than sensor noise.
MIN_RELIABLE_APPARENT_PX = 3.0

# Goalpost diameter (m): a thin, tall cylinder, so only the bounding box's
# width constrains distance via its known diameter -- the height instead
# reflects the post's own tall extent, not distance.
GOALPOST_DIAMETER_M = 0.10
