"""SoccerSim strategy entry point and primary match logic.

Structure, from high level to low level:
- main.py (this file): match strategy. ``play()`` dispatches through the Phase
  state machine to ``_act_*`` functions. Each function selects an attacker,
  usually the player nearest the ball, and invokes Player actions directly.
- player.py: Player control handle and high-level actions such as ``attack``,
  ``take_kickoff``, ``move_to_position``, and ``walk_to``. Add new skills there.
- utils/: movement, geometry, and obstacle-avoidance helpers such as
  ``opponent_goal``, ``dist``, and ``angle_to``.
- framework/: platform plumbing that users normally do not modify.

To change the playing style, primarily edit the Phase state machine, ``_act_*``
behaviors, and positioning formulas in this file.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

from booster_agent_framework import AgentBase

from .framework.agent import SoccerAgentMixin
from .framework.types import KICKING_TEAM_NONE, Context, GameState, SetPlay
from .param import *
from .player import Player
from .utils import dist, opponent_goal, own_goal
from .vision import ColorLutDetector, OdomAnchoredLocaliser


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
    detector_class = ColorLutDetector
    localiser_class = OdomAnchoredLocaliser

    def init_store(self, store) -> None:
        _log.info("init_store called")
        store.prev_phase = None       # Previous phase, used to detect transitions
        store.cur_phase = None
        store.kickoff_taker = None    # Locked taker ID, reselected for each kickoff
        store.normal_attacker = None

    @staticmethod
    def play(context: Context, players: list[Player], store) -> None:
        phase = get_phase(context)
        store.prev_phase = store.cur_phase
        store.cur_phase = phase

        # Draw per-frame visualizations.
        _analyze_and_draw(context, players, store)

        # Draw the current phase as a label just outside the field.
        from .framework import debugdraw
        g = context.game
        game_state = g.state.value if g is not None else "none"
        set_play = g.set_play.value if g is not None else "none"
        secondary_time = g.secondary_time if g is not None else 0.0
        debugdraw.text(
            0.0, context.field.width / 2.0 + 0.2,
            f"phase={phase.value} state={game_state} set={set_play} secondary={secondary_time:.1f}",
            rgb=(1.0, 1.0, 0.0), ns="phase",
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
            _act_opp_kickoff(context, active)
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


def _clear_normal_sticky(store) -> None:
    store.normal_attacker = None


def _player_dist_to_ball(context: Context, p: Player) -> float:
    """Return a player's distance to the ball's current position."""
    ball = context.ball
    return (
        dist(p.pose.x, p.pose.y, ball.x, ball.y) + _fallen_time_cost(p)
        if ball is not None else math.inf
    )


def _fallen_time_cost(p: Player) -> float:
    return FALLEN_COST if p.is_fallen else 0.0


def _select_closest_attacker(
    context: Context,
    players: list[Player],
    preferred_id: int | None = None,
) -> Player:
    """Select the player with the lowest effective distance to the ball.

    ``players`` must be nonempty, ready, and have known poses. Normal play and
    kickoffs share this selection logic.
    """
    ranked = [(p, _player_dist_to_ball(context, p)) for p in players]
    best, best_dist = min(ranked, key=lambda item: item[1])
    preferred = next((item for item in ranked if item[0].id == preferred_id), None)
    if (
        preferred is not None
        and preferred[1] <= best_dist + ATTACKER_KEEP_DIST_MARGIN_M
    ):
        return preferred[0]
    return best


def _act_normal(context: Context, players: list[Player], store) -> None:
    """Assign the nearest player to attack, the next to guard, and others to support.

    ``players`` contains ready players with known poses. Assign and execute
    their roles directly here.
    """
    if not players:
        return

    attacker = _select_closest_attacker(
        context, players, getattr(store, "normal_attacker", None),
    )
    store.normal_attacker = attacker.id
    attacker.action = "attack"
    attacker.attack()

    # The remaining player nearest our goal becomes the guard.
    rest = [p for p in players if p is not attacker]
    if rest:
        gx, gy = own_goal(context)
        guard = min(rest, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
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
        gx, gy = own_goal(context)
        guard = min(rest, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
        guard.guard()
        rest = [p for p in rest if p is not guard]

    for p in rest:
        p.action = "stay"
        p.stop()


def _act_opp_kickoff(context: Context, players: list[Player]) -> None:
    """Guard with one player while the others wait outside the center circle."""
    if not players:
        return
    gx, gy = own_goal(context)
    guard = min(players, key=lambda p: dist(p.pose.x, p.pose.y, gx, gy))
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
    """Move each player to a ready position."""
    game = context.game
    our_kickoff = game is not None and game.kicking_team == context.team_id
    field = context.field
    # Our kickoff priority: center-circle edge, goal-area center, then left side.
    if our_kickoff:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-0.5, field.circle_radius + 2),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
    # Opponent kickoff priority: outside center circle, goal area, then penalty area.
    else:
        if len(players) >= 1:
            p1 = players[0]
            p1.action = "ready"
            p1.walk_to(
                (-field.circle_radius - 0.5, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 2:
            p2 = players[1]
            p2.action = "ready"
            p2.walk_to(
                (-field.length / 2.0 + field.goal_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )
        if len(players) >= 3:
            p3 = players[2]
            p3.action = "ready"
            p3.walk_to(
                (-field.length / 2.0 + field.penalty_area_length, 0.0),
                face=0.0,
                avoid_ball=True,
                avoid_robots=True,
            )



# ======================================================================
# Match visualization: ball position and each player's distance to the ball
# ======================================================================

def _draw_teammate_marker(p: Player) -> None:
    """Draw teammates in red, using cubes while kicking and spheres otherwise.

    This runs for every player each frame regardless of phase, penalty, or
    readiness. The label contains the player ID and current high-level action
    (``p.action``), with ``[KICK]`` appended while kicking. Shape also indicates
    whether the player is in the kicking state.
    """
    from .framework import debugdraw

    if p.pose is None:
        return
    red = (1.0, 0.2, 0.2)
    if p.is_kicking:
        debugdraw.cube(p.pose.x, p.pose.y, rgb=red, scale=0.38, ns="teammate")
    else:
        debugdraw.point(p.pose.x, p.pose.y, rgb=red, scale=0.3, ns="teammate")
    kick_tag = " [KICK]" if p.is_kicking else ""
    label = f"{p.id}:{p.action}{kick_tag}"
    debugdraw.text(p.pose.x, p.pose.y, label, rgb=(1.0, 0.9, 0.6), ns="teammate_id")


def _analyze_and_draw(context: Context, players: list[Player], store) -> None:
    """Calculate player-to-ball distances and draw them each frame.

    This no longer depends on the analysis module; distances use the ball's
    current position.
    """
    from .framework import debugdraw

    ball = context.ball

    # Nothing can be drawn when the ball is not visible.
    if ball is None:
        return

    # 1. Draw the ball's current position as a green point.
    debugdraw.point(ball.x, ball.y, rgb=(0.0, 1.0, 0.0), scale=0.2, ns="ball_current")

    # 2. Draw distance labels: red for teammates and blue for opponents.
    for p in players:
        if p.pose is None:
            continue
        d = dist(p.pose.x, p.pose.y, ball.x, ball.y)
        debugdraw.text(
            p.pose.x + 0.3, p.pose.y - 0.3, f"{d:.1f}m",
            rgb=(1.0, 0.6, 0.6), ns="dist_ours",
        )
    for r in context.opponents.values():
        if r.pose is None:
            continue
        d = dist(r.pose.x, r.pose.y, ball.x, ball.y)
        debugdraw.text(
            r.pose.x + 0.3, r.pose.y - 0.3, f"{d:.1f}m",
            rgb=(0.6, 0.6, 1.0), ns="dist_opp",
        )
