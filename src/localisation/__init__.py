"""Localisation layer: turning raw sensing into field-frame position estimates.

- protocols: the ``Localiser`` contract that ``odometry``'s dead-reckoning
  class (and any future self-pose estimator) satisfies
- config: odom calibration anchor and freshness constants
- ball_localisation: each robot's own ball position estimate
- landmark_localisation: self-pose combining odometry with goalpost-based
  correction
"""
