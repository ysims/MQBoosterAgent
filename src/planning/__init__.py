"""Planning layer: deciding a path or a kick, between motion and strategy.

- config: path-planner, obstacle-geometry, and kick-target tuning constants
- path_planning: obstacle collection, a global A* grid planner, and the
  local VFH heading fallback used by ``strategy.player.Player.walk_to``
- kick_planning: aim/power decisions for a kick, and defensive block-path
  projection
"""
