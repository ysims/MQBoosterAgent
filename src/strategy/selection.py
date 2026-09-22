"""Role selection: choosing which player attacks, guards, or supports."""

from __future__ import annotations

import math

from ..framework.types import Context
from ..utils.geom import dist, own_goal
from .config import (
    ATTACKER_KEEP_DIST_MARGIN_M,
    ATTACKER_SWITCH_COOLDOWN_SEC,
    FALLEN_COST,
    GUARD_KEEP_DIST_MARGIN_M,
)
from .player import Player


__all__ = [
    "clear_normal_sticky",
    "select_closest_attacker",
    "select_attacker",
    "select_closest_guard",
]


def clear_normal_sticky(store) -> None:
    store.normal_attacker = None
    store.attacker_locked_until = 0.0
    store.normal_guard = None


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Return a player's distance to its own perceived ball position."""
    ball = context.ball.get(p.id)
    return (
        dist(p.pose.x, p.pose.y, ball.x, ball.y) + _fallen_time_cost(p)
        if ball is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest(
    players: list[Player],
    dist_fn,
    preferred_id: int | None,
    keep_margin: float,
) -> Player:
    """Select the player with the lowest distance under ``dist_fn``, sticky.

    Without stickiness, two candidates at a near-equal distance can flip
    the winner every frame from tiny position noise -- both players then
    keep swapping which role/target they're walking toward, never settling,
    and can end up walking straight into each other. The previously-selected
    player (``preferred_id``) keeps the role unless another player is more
    than ``keep_margin`` closer.
    """
    ranked = [(p, dist_fn(p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if preferred is not None and preferred[1] <= best_dist + keep_margin:
        return preferred[0]
    return best


def select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """Select the player with the lowest effective distance to the ball.

    ``players`` must be nonempty, ready, and have known poses. Normal play and
    kickoffs share this selection logic.
    """
    return _select_closest(
        players,
        lambda p: _player_dist_to_ball(context, p),
        preferred_id,
        ATTACKER_KEEP_DIST_MARGIN_M,
    )


def select_attacker(context: Context, players: list[Player], store) -> Player | None:
    """Select the attacker: distance-margin sticky, plus a switch cooldown.

    See ATTACKER_SWITCH_COOLDOWN_SEC's docstring for why the cooldown is
    needed on top of the distance margin alone -- without it, the current
    attacker's own ball loss (an instant jump to infinite distance) would
    trigger a switch the very same frame, every time.

    While the cooldown is active, a locked attacker that's momentarily
    absent from ``players`` (e.g. one bad frame of ensure_ready() right
    after a kick's recoil, not an actual long-term problem) returns None
    rather than picking a substitute -- reassigning here, even briefly,
    would both hand the role away AND start a fresh cooldown for whoever
    got it, defeating the entire point of locking it in the first place.
    No attacker acting for a frame is a much smaller cost than losing the
    role outright over a single-frame blip.
    """
    preferred_id = getattr(store, "normal_attacker", None)
    locked_until = getattr(store, "attacker_locked_until", 0.0)

    if context.now < locked_until:
        return next((p for p in players if p.id == preferred_id), None)

    attacker = select_closest_attacker(context, players, preferred_id)
    if attacker.id != preferred_id:
        store.attacker_locked_until = context.now + ATTACKER_SWITCH_COOLDOWN_SEC
    return attacker


def select_closest_guard(
    context: Context,
    players: list[Player],
    preferred_id: int | None,
) -> Player:
    """Select the player closest to our own goal, sticky like the attacker."""
    gx, gy = own_goal(context)
    return _select_closest(
        players,
        lambda p: dist(p.pose.x, p.pose.y, gx, gy),
        preferred_id,
        GUARD_KEEP_DIST_MARGIN_M,
    )
