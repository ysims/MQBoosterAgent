"""SoccerSim strategy entry point and primary match logic.

Structure, from high level to low level:
- main.py (this file): match strategy. ``play()`` dispatches through the Phase
  state machine to ``_act_*`` functions. Each function selects an attacker,
  usually the player nearest the ball, and invokes Player actions directly.
- player.py: Player control handle and high-level actions such as ``attack``,
  ``take_kickoff``, ``move_to_position``, and ``walk_to``. Add new skills there.
- planning/: path and kick decisions (where to walk, where and how hard to kick).
- motion/: action-level chassis and kick commands.
- odometry/, vision/, localisation/: perception and self-pose estimation.
- framework/: platform plumbing that users normally do not modify.

To change the playing style, primarily edit the Phase state machine, ``_act_*``
behaviors, and positioning formulas in this file.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from ..framework.agent import SoccerAgentMixin
from ..framework.types import KICKING_TEAM_NONE, Context, GameState, SetPlay
from ..odometry.dead_reckoning import OdomAnchoredLocaliser
from ..utils.geom import dist, opponent_goal, own_goal
from ..vision.ball_detection import estimate_ball_position
from .config import (
    ATTACKER_KEEP_DIST_MARGIN_M,
    ATTACKER_SWITCH_COOLDOWN_SEC,
    FALLEN_COST,
    GUARD_KEEP_DIST_MARGIN_M,
    KICK_POWER_OUR_KICKOFF,
)
from .player import Player


_log = logging.getLogger(__name__)


# ======================================================================
# Phase state machine: match phase classification
# ======================================================================


class Phase(Enum):
    """Top-level match phase controlling normal play, restarts, and stops."""
    NORMAL = "normal"              # Normal contest while PLAYING
    OUR_KICKOFF = "our_kickoff"    # Our kickoff during early SET/PLAYING
    OPP_KICKOFF = "opp_kickoff"    # Opponent kickoff; keep clear
    OUR_SET_PLAY = "our_set_play"  # Our free kick, corner, or goal kick
    OPP_SET_PLAY = "opp_set_play"  # Opponent set play; keep clear
    READY = "ready"                # Move into READY positions
    STOPPED = "stopped"            # Non-kickoff SET, INITIAL, FINISHED, or stopped


def get_phase(context: Context) -> Phase:
    """Determine the current phase from the GameController state."""
    g = context.game
    if g is None:
        return Phase.STOPPED

    state = g.state

    # READY: move to ready positions.
    if state == GameState.READY:
        return Phase.READY

    # PLAYING: normal play or an active kickoff/set play.
    if state == GameState.PLAYING and not g.stopped:
        # For a set play, kicking_team identifies the team taking it.
        if g.set_play != SetPlay.NONE and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_SET_PLAY
            else:
                return Phase.OPP_SET_PLAY

        # During the kickoff countdown, kicking_team identifies the taker.
        if g.secondary_time > 0 and g.kicking_team != KICKING_TEAM_NONE:
            our_team = context.team_id
            if g.kicking_team == our_team:
                return Phase.OUR_KICKOFF
            else:
                return Phase.OPP_KICKOFF

        # Normal play.
        return Phase.NORMAL

    # Remain stationary during SET, INITIAL, FINISHED, or a stoppage.
    return Phase.STOPPED

def get_set_play_type(context: Context) -> SetPlay:
    """Return the active set-play type, or ``SetPlay.NONE`` when unavailable.

    This reads the GameController's ``set_play`` field without identifying the
    taking team. :func:`get_phase` distinguishes OUR_SET_PLAY from OPP_SET_PLAY;
    this function only identifies the type.

    The seven possible values from framework.types.SetPlay are:
    - ``NONE``: no set play, such as normal play or a kickoff
    - ``DIRECT_FREE_KICK``: a direct free kick that may score immediately
    - ``INDIRECT_FREE_KICK``: another player must touch the ball before a goal
    - ``PENALTY_KICK``: a penalty kick
    - ``THROW_IN``: a kick-in from touch
    - ``GOAL_KICK``: a goal kick
    - ``CORNER_KICK``: a corner kick
    """
    g = context.game
    if g is None:
        return SetPlay.NONE
    return g.set_play


# ======================================================================
# Agent entry point
# ======================================================================


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent."""

    player_class = Player
    localiser_class = OdomAnchoredLocaliser
    ball_position_estimator = staticmethod(estimate_ball_position)

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # Previous phase, used to detect transitions
        store.cur_phase = None
        store.kickoff_taker = None    # Locked taker ID, reselected for each kickoff
        store.normal_attacker = None
        store.attacker_locked_until = 0.0
        store.normal_guard = None
        store.kickoff_guard = None
        store.opp_kickoff_guard = None
        store.debug_state = None      # Last-seen per-player state, for change logging

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase

        # Draw per-frame visualizations.
        _analyze_and_draw(context, players, store)

        # Draw the current phase as a label just outside the field.
        from ..framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
        )
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.5,
            "player markers: green=walk  amber=switching mode  red=fallen/recovering",
            rgb=(0.8, 0.8, 0.8), ns="legend",
        )

        # Handle readiness and retain only players that can act this frame.
        # ensure_ready asynchronously gets up or switches to walk mode without
        # moving. Penalized players also prepare so they can rejoin immediately.
        # Penalized or unready players are excluded from role assignment to avoid
        # selecting an attacker who cannot move.
        active: list[Player] = []
        for p in players:
            ready = p.ensure_ready()
            if p.is_penalized:
                p.action = "penalized"     # May prepare, but cannot move
                p.stop()
            elif not ready:
                p.action = "fallen" if p.is_fallen else "switching_mode"
            elif p.pose is None:
                p.action = "no_pose"       # Exclude players with unknown positions
                p.stop()
            else:
                active.append(p)

        # Dispatch the team once per phase; _act_* owns team-wide role assignment.
        if phase == Phase.NORMAL:
            _act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            _clear_normal_sticky(store)
            _act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            _clear_normal_sticky(store)
            _act_opp_kickoff(context, active, store)
        elif phase == Phase.OUR_SET_PLAY:
            _clear_normal_sticky(store)
            _act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            _clear_normal_sticky(store)
            _act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            _clear_normal_sticky(store)
            _act_ready(context, active)
        elif phase == Phase.STOPPED:
            _clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        # Draw every teammate last, including penalized, unready, and stopped
        # players, so markers remain visible during states such as SET.
        for p in players:
            _draw_teammate_marker(p)

        _log_state_changes(players, store)
        _stream_debug_state(context, players)


def _stream_debug_state(context: Context, players: list[Player]) -> None:
    """Send a compact snapshot for the external visual debugger.

    See ``framework/debug_stream.py`` (the sender, a fire-and-forget UDP
    socket) and ``scripts/debug_viz.py`` (the standalone matplotlib
    receiver that runs on the host, outside any container).
    """
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


def _log_state_changes(players: list[Player], store) -> None:
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


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None
    store.attacker_locked_until = 0.0
    store.normal_guard = None


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Return a player's distance to its own perceived ball position.

    Each player uses its own belief, not a shared team-wide one -- see
    ``Context.ball``'s docstring.
    """
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


def _select_closest_attacker(
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


def _select_attacker(context: Context, players: list[Player], store) -> Player | None:
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

    attacker = _select_closest_attacker(context, players, preferred_id)
    if attacker.id != preferred_id:
        store.attacker_locked_until = context.now + ATTACKER_SWITCH_COOLDOWN_SEC
    return attacker


def _select_closest_guard(
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


def _act_normal(context: Context, players: list[Player], store) -> None:
    """Assign the nearest player to attack, the next to guard, and others to support.

    ``players`` contains ready players with known poses. Assign and execute
    their roles directly here.
    """
    if not players:
        return

    attacker = _select_attacker(context, players, store)
    rest = players
    if attacker is not None:
        # Only update the stored id when we actually have a live attacker --
        # a None result means the locked attacker is just momentarily
        # unready, and store.normal_attacker must keep pointing at it so
        # the lock still means something once it returns (see
        # _select_attacker's docstring).
        store.normal_attacker = attacker.id
        attacker.action = "attack"
        attacker.attack()
        rest = [p for p in players if p is not attacker]

    # The remaining player nearest our goal becomes the guard.
    if rest:
        guard = _select_closest_guard(
            context, rest, getattr(store, "normal_guard", None),
        )
        store.normal_guard = guard.id
        guard.guard()
        rest = [p for p in rest if p is not guard]

    # All other players support.
    for p in rest:
        p.action = "support"
        p.support()


def _act_our_kickoff(context: Context, players: list[Player], store) -> None:
    """Lock the nearest kickoff taker, assign a guard, and leave others in support."""
    if not players:
        return

    active_ids = {p.id for p in players}
    if store.prev_phase != Phase.OUR_KICKOFF or store.kickoff_taker not in active_ids:
        # Select a new taker when entering the kickoff phase.
        store.kickoff_taker = _select_closest_attacker(context, players).id

    attacker_id = store.kickoff_taker
    attacker = next((p for p in players if p.id == attacker_id), None)
    if attacker is None:
        return

    attacker.action = "kickoff"
    attacker.kick(0.1, KICK_POWER_OUR_KICKOFF)

    rest = [p for p in players if p is not attacker]
    if rest:
        guard = _select_closest_guard(
            context, rest, getattr(store, "kickoff_guard", None),
        )
        store.kickoff_guard = guard.id
        guard.guard()
        rest = [p for p in rest if p is not guard]

    for p in rest:
        p.action = "stay"
        p.stop()


def _act_opp_kickoff(context: Context, players: list[Player], store) -> None:
    """Guard with one player while the others wait outside the center circle."""
    if not players:
        return
    guard = _select_closest_guard(
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


def _act_our_set_play(context: Context, players: list[Player], store) -> None:
    """Dispatch our set play by type. TODO: add strategy; defaults to normal."""
    set_play = get_set_play_type(context)
    if set_play == SetPlay.THROW_IN:
        _act_normal(context, players, store)
        return
    if set_play == SetPlay.CORNER_KICK:
        _act_normal(context, players, store)
        return
    if set_play == SetPlay.GOAL_KICK:
        _act_normal(context, players, store)
        return
    _act_normal(context, players, store)


def _act_opp_set_play(context: Context, players: list[Player], store) -> None:
    """Handle an opponent set play. TODO: add strategy; defaults to normal."""
    _act_normal(context, players, store)


def _act_ready(context: Context, players: list[Player]) -> None:
    """Move each player to a ready position.

    Positions are assigned by stable player ID, not by position in
    ``players`` -- that list only contains whichever players are currently
    ready/active this frame (see ``play()``), which can vary tick to tick
    (e.g. a player still transitioning through "prepare" mode is excluded).
    Indexing into it directly would send whichever player happens to be
    first into the "player 1" ready slot, regardless of who that actually
    is -- looking like two players swapped roles.
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



# ======================================================================
# Match visualization: ball position and each player's distance to the ball
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """Draw teammates color-coded by readiness state, with a state label.

    This runs for every player each frame regardless of phase, penalty, or
    readiness -- so a robot stuck mid-recovery or mid-mode-switch stays
    visible instead of disappearing from the view. Marker color reflects
    ``mode``/``fall_down_state`` (see the legend drawn once per frame by
    ``play()``): red while fallen/recovering, amber while not yet in walk
    mode, green otherwise. The label adds the exact ``mode``/
    ``fall_down_state`` strings so a frozen or thrashing robot can be
    diagnosed directly from the viewer instead of grepping logs. Shape
    (cube vs sphere) additionally indicates the kicking state.
    """
    from ..framework import debugdraw

    if p.pose is None:
        return

    mode = p.mode or "none"
    fall = p.fall_down_state or "none"
    if p.is_fallen:
        rgb = (1.0, 0.15, 0.15)     # red: fallen or recovering
    elif mode != "walk":
        rgb = (1.0, 0.75, 0.0)      # amber: not yet in walk mode
    else:
        rgb = (0.2, 1.0, 0.3)       # green: normal and active

    if p.is_kicking:
        debugdraw.cube(p.pose.x, p.pose.y, rgb=rgb, scale=0.38, ns="teammate")
    else:
        debugdraw.point(p.pose.x, p.pose.y, rgb=rgb, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}\nmode={mode} fall={fall}"
    debugdraw.text(p.pose.x, p.pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """Draw each player's distance to its own perceived ball position.

    Each robot's ball belief is independent (see ``Context.ball``'s
    docstring), so this draws a distance label per teammate using that
    teammate's own reading rather than one shared value. The ball position
    itself is drawn per-robot by the runtime (see ``SoccerRuntime._draw_world``);
    there's no separate opponent distance here since we have no principled
    per-opponent ball reference to compare against.
    """
    from ..framework import debugdraw

    for p in players:
        ball = context.ball.get(p.id)
        if p.pose is None or ball is None:
            continue
        d = dist(p.pose.x, p.pose.y, ball.x, ball.y)
        debugdraw.text(
            p.pose.x + 0.3, p.pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
