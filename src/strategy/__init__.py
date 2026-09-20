"""Strategy layer: role behavior and match-level decision making.

- config: guard/support/kickoff and positioning tuning constants
- player: the Player control handle -- attack/guard/support/take_kickoff and
  movement orchestration (walk_to/face_to/ensure_ready), calling into
  ``planning`` for decisions and ``motion`` for execution
- main: SoccerSimAgent entry point, the Phase state machine, and ``_act_*``
  team-wide role assignment
"""
