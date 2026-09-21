"""Search planning: deciding where to look/walk when the ball isn't visible.

A pure decision function, like ``kick_planning`` and ``gaze_planning`` -- it
chooses a target to search around, but never issues commands.
``strategy.player.Player._search_for_ball`` calls into this and passes the
result to ``motion`` (via ``walk_to``/``look_at``), or falls back to an
in-place turn when there's nothing usable to go on.
"""

from __future__ import annotations

from .config import SEARCH_MEMORY_SEC


__all__ = ["plan_search_target"]


def plan_search_target(
    last_seen: tuple[float, float] | None,
    last_seen_at: float | None,
    now: float | None,
) -> tuple[float, float] | None:
    """Return where to walk/look to try to reacquire the ball, or None.

    None means the remembered position (if any) is stale enough that it's
    no longer a useful place to look -- the caller should fall back to
    something else, such as turning in place to sweep the camera.
    """
    if last_seen is None or last_seen_at is None or now is None:
        return None
    if now - last_seen_at > SEARCH_MEMORY_SEC:
        return None
    return last_seen
