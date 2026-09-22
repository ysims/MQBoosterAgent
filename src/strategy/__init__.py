"""Strategy layer: role behavior and match-level decision making.

- config: guard/support/kickoff and positioning tuning constants
- player: the Player control handle -- attack/guard/support/take_kickoff and
  movement orchestration (walk_to/face_to/ensure_ready), calling into
  ``planning`` for decisions and ``motion`` for execution
- phase: match phase classification from the GameController state
- selection: choosing which player attacks, guards, or supports
- actions: per-phase team behavior, dispatched by phase
- debug: the live JSON snapshot stream and state-change log
- main: SoccerSimAgent entry point, dispatching ``play()`` through the
  above
"""
