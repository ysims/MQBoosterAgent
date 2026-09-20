"""Planning layer: deciding a path, a kick, or a gaze, between motion and
strategy.

- config: path-planner, obstacle-geometry, kick-target, and gaze tuning
  constants
- path_planning: obstacle collection, a global A* grid planner, and the
  local VFH heading fallback used by ``strategy.player.Player.walk_to``
- kick_planning: aim/power decisions for a kick, and defensive block-path
  projection
- gaze_planning: head pitch/yaw decisions to keep the ball in the camera's
  field of view, used by ``strategy.player.Player.look_at``
"""
