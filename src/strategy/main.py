"""SoccerSim strategy entry point.

Structure, from high level to low level:
- main.py (this file): the agent entry point. ``play()`` dispatches through
  the Phase state machine (phase.py) to per-phase behaviors (actions.py).
- actions.py: per-phase team behavior -- assigns roles (via selection.py)
  and invokes Player actions directly.
- selection.py: choosing which player attacks, guards, or supports.
- phase.py: match phase classification from the GameController state.
- debug.py: the live JSON snapshot stream and state-change log.
- player.py: Player control handle and high-level actions such as ``attack``,
  ``take_kickoff``, ``move_to_position``, and ``walk_to``. Add new skills there.
- planning/: path and kick decisions (where to walk, where and how hard to kick).
- motion/: action-level chassis and kick commands.
- odometry/, vision/, localisation/: perception and self-pose estimation.
- framework/: platform plumbing that users normally do not modify.

To change the playing style, primarily edit the Phase state machine and the
``act_*`` behaviors and positioning formulas in actions.py.
"""

from __future__ import annotations

import logging

from booster_agent_framework import AgentBase

from ..framework.agent import SoccerAgentMixin
from ..framework.types import Context
from ..localisation.landmark_localisation import GoalpostCorrectedLocaliser
from ..vision.ball_detection import estimate_ball_position
from .actions import (
    act_normal,
    act_opp_kickoff,
    act_opp_set_play,
    act_our_kickoff,
    act_our_set_play,
    act_ready,
)
from .debug import log_state_changes, stream_debug_state
from .phase import Phase, get_phase
from .player import Player
from .selection import clear_normal_sticky


_log = logging.getLogger(__name__)


class SoccerSimAgent(SoccerAgentMixin, AgentBase):
    """3v3 SoccerSim agent."""

    player_class = Player
    localiser_class = GoalpostCorrectedLocaliser
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

        # Dispatch the team once per phase; act_* owns team-wide role assignment.
        if phase == Phase.NORMAL:
            act_normal(context, active, store)
        elif phase == Phase.OUR_KICKOFF:
            clear_normal_sticky(store)
            act_our_kickoff(context, active, store)
        elif phase == Phase.OPP_KICKOFF:
            clear_normal_sticky(store)
            act_opp_kickoff(context, active, store)
        elif phase == Phase.OUR_SET_PLAY:
            clear_normal_sticky(store)
            act_our_set_play(context, active, store)
        elif phase == Phase.OPP_SET_PLAY:
            clear_normal_sticky(store)
            act_opp_set_play(context, active, store)
        elif phase == Phase.READY:
            clear_normal_sticky(store)
            act_ready(context, active)
        elif phase == Phase.STOPPED:
            clear_normal_sticky(store)
            for p in active:
                p.action = "stopped"
                p.stop()

        log_state_changes(players, store)
        stream_debug_state(context, players)
