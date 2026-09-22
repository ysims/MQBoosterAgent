"""Vision layer: turning the sim's detections into usable positions.

- types: pixel-space and field-frame detection dataclasses (data contract,
  no ROS dependency)
- config: camera intrinsics and object geometry constants
- projection: shared apparent-size -> robot-frame position math, used for
  any object of known physical size
- ball_detection: bbox -> robot-frame ball position, via ``projection``
- opponents_ground_truth: opponent position tracking, since the sim's
  detector never reports other robots (see that module's docstring)
"""
