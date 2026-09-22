"""Per-phase team behavior: assigning and executing each player's role."""

from __future__ import annotations

from ..framework.types import Context, SetPlay
from .config import KICK_POWER_OUR_KICKOFF
from .phase import Phase, get_set_play_type
from .player import Player
from .selection import select_attacker, select_closest_attacker, select_closest_guard


__all__ = [
    "act_normal",
    "act_our_kickoff",
    "act_opp_kickoff",
    "act_our_set_play",
    "act_opp_set_play",
    "act_ready",
]


def act_normal(context: Context, players: list[Player], store) -> None:
    """Assign the nearest player to attack, the next to guard, and others to support.

    ``players`` contains ready players with known poses. Assign and execute
    their roles directly here.
    """
    if not players:
        return

    attacker = select_attacker(context, players, store)
    rest = players
    if attacker is not None:
        # Only update the stored id when we actually have a live attacker --
        # a None result means the locked attacker is just momentarily
        # unready, and store.normal_attacker must keep pointing at it so
        # the lock still means something once it returns (see
        # select_attacker's docstring).
        store.normal_attacker = attacker.id
        attacker.action = "attack"
        attacker.attack()
        rest = [p for p in players if p is not attacker]

    # The remaining player nearest our goal becomes the guard.
    if rest:
        guard = select_closest_guard(
            context, rest, getattr(store, "normal_guard", None),
        )
        store.normal_guard = guard.id
        guard.guard()
        rest = [p for p in rest if p is not guard]

    # All other players support.
    for p in rest:
        p.action = "support"
        p.support()


def act_our_kickoff(context: Context, players: list[Player], store) -> None:
    """Lock the nearest kickoff taker, assign a guard, and leave others in support."""
    if not players:
        return

    active_ids = {p.id for p in players}
    if store.prev_phase != Phase.OUR_KICKOFF or store.kickoff_taker not in active_ids:
        # Select a new taker when entering the kickoff phase.
        store.kickoff_taker = select_closest_attacker(context, players).id

    attacker_id = store.kickoff_taker
    attacker = next((p for p in players if p.id == attacker_id), None)
    if attacker is None:
        return

    attacker.action = "kickoff"
    attacker.kick(0.1, KICK_POWER_OUR_KICKOFF)

    rest = [p for p in players if p is not attacker]
    if rest:
        guard = select_closest_guard(
            context, rest, getattr(store, "kickoff_guard", None),
        )
        store.kickoff_guard = guard.id
        guard.guard()
        rest = [p for p in rest if p is not guard]

    for p in rest:
        p.action = "stay"
        p.stop()


def act_opp_kickoff(context: Context, players: list[Player], store) -> None:
    """Guard with one player while the others wait outside the center circle."""
    if not players:
        return
    guard = select_closest_guard(
        context, players, getattr(store, "opp_kickoff_guard", None),
    )
    store.opp_kickoff_guard = guard.id
    guard.guard()

    rest = [p for p in players if p is not guard]
    r = context.field.circle_radius
    slots = [(-r - 0.5, 0.0), (-r - 2.0, 0.5)]
    for p, target in zip(rest, slots):
        p.action = "opp_kickoff:ready"
        p.walk_to(target, avoid_ball=True, avoid_robots=True)


def act_our_set_play(context: Context, players: list[Player], store) -> None:
    """Dispatch our set play by type. TODO: add strategy; defaults to normal."""
    set_play = get_set_play_type(context)
    if set_play == SetPlay.THROW_IN:
        act_normal(context, players, store)
        return
    if set_play == SetPlay.CORNER_KICK:
        act_normal(context, players, store)
        return
    if set_play == SetPlay.GOAL_KICK:
        act_normal(context, players, store)
        return
    act_normal(context, players, store)


def act_opp_set_play(context: Context, players: list[Player], store) -> None:
    """Handle an opponent set play. TODO: add strategy; defaults to normal."""
    act_normal(context, players, store)


def act_ready(context: Context, players: list[Player]) -> None:
    """Move each player to a ready position.

    Positions are assigned by stable player ID, not by position in
    ``players`` -- that list only contains whichever players are currently
    ready/active this frame (see ``main.SoccerSimAgent.play``), which can
    vary tick to tick (e.g. a player still transitioning through "prepare"
    mode is excluded). Indexing into it directly would send whichever
    player happens to be first into the "player 1" ready slot, regardless
    of who that actually is -- looking like two players swapped roles.
    """
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    field = context.field
    by_id = {p.id: p for p in players}

    if our_kickoff:
        # Our kickoff priority: center-circle edge, goal-area center, then left side.
        positions = {
            1: (-field.circle_radius, 0.0),
            2: (-field.length / 2.0 + field.goal_area_length, 0.0),
            3: (-0.5, field.circle_radius + 2),
        }
    else:
        # Opponent kickoff priority: outside center circle, goal area, then penalty area.
        positions = {
            1: (-field.circle_radius - 0.5, 0.0),
            2: (-field.length / 2.0 + field.goal_area_length, 0.0),
            3: (-field.length / 2.0 + field.penalty_area_length, 0.0),
        }

    for pid, target in positions.items():
        p = by_id.get(pid)
        if p is None:
            continue
        p.action = "ready"
        p.walk_to(target, face=0.0, avoid_ball=True, avoid_robots=True)
