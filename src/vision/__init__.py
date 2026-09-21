"""Vision layer: turning the sim's detections into usable positions.

- types: pixel-space and field-frame detection dataclasses (data contract,
  no ROS dependency)
- config: camera intrinsics and ball geometry constants
- ball_detection: bbox -> robot-frame ball position (the file to edit when
  improving ball perception)
- opponents_ground_truth: opponent position tracking, since the sim's
  detector never reports other robots (see that module's docstring)
"""
