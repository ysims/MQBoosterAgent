"""Live debugging: a JSON snapshot stream and a state-change log.

See ``framework/debug_stream.py`` (the sender, a fire-and-forget UDP
socket) and ``scripts/debug_viz.py`` (the standalone matplotlib receiver
that runs on the host, outside any container) for the live-view side of
this, and ``scripts/debug_monitor.py`` for a plain-text log viewer over
the log stream ``log_state_changes`` writes to.
"""

from __future__ import annotations

import logging

from ..framework.types import Context
from .player import Player


__all__ = ["stream_debug_state", "log_state_changes"]


_log = logging.getLogger(__name__)


def stream_debug_state(context: Context, players: list[Player]) -> None:
    """Send a compact snapshot for the external visual debugger."""
    from ..framework import debug_stream

    payload = {
        "now": context.now,
        "team_id": context.team_id,
        "field": {"length": context.field.length, "width": context.field.width},
        "ball": {
            str(pid): {"x": b.x, "y": b.y} for pid, b in context.ball.items()
        },
        "teammates": {
            str(p.id): {
                "x": p.pose.x, "y": p.pose.y, "theta": p.pose.theta,
                "action": p.action, "mode": p.mode, "fall": p.fall_down_state,
                "kicking": p.is_kicking,
            }
            for p in players if p.pose is not None
        },
        "opponents": {
            str(pid): {"x": r.pose.x, "y": r.pose.y}
            for pid, r in context.opponents.items() if r.pose is not None
        },
    }
    debug_stream.send(payload)


def log_state_changes(players: list[Player], store) -> None:
    """Log a line whenever a player's action/mode/fall state changes.

    A live viewer can easily miss a one-frame freeze or a rapid role flip;
    this gives a readable, retroactive timeline in the logs for exactly
    that, without needing to grep raw per-frame noise.
    """
    prev: dict[int, tuple] = getattr(store, "debug_state", None) or {}
    cur: dict[int, tuple] = {}
    for p in players:
        state = (p.action, p.mode, p.fall_down_state, p.is_kicking)
        cur[p.id] = state
        old = prev.get(p.id)
        if old is not None and old != state:
            _log.info(
                "player %d state change: action %s->%s mode %s->%s "
                "fall %s->%s kicking %s->%s",
                p.id, old[0], state[0], old[1], state[1],
                old[2], state[2], old[3], state[3],
            )
    store.debug_state = cur
