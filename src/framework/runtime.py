"""Framework runtime for the 30 Hz loop, Context construction, and Players.

Context data comes from an injected ContextSource -- normally
``VisionContextSource``. Without one, development and test runs build an
empty Context each frame. This layer uniformly replaces stale data with None.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

from .config import SoccerConfig
from .types import (
    ADULT_FIELD_DIMENSIONS,
    BallState,
    Context,
    GameControlState,
    RobotState,
    WorldSnapshot,
)

if TYPE_CHECKING:
    from ..strategy.player import Player
    from .agent import SoccerAgentMixin


__all__ = ["ContextSource", "SoccerRuntime"]


_log = logging.getLogger(__name__)


class ContextSource(Protocol):
    """Source protocol supplying raw snapshots without coupling runtime to ROS."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_snapshot(self) -> WorldSnapshot: ...


class SoccerRuntime:
    """Manage the 30 Hz control loop and Player lifecycle.

    Users interact only with SoccerAgent, Player, Context, and ``play()`` and do
    not need to know this class exists.

    With ``context_source`` set to None, each frame receives an empty Context
    for development or testing. A ROS source produces real, freshness-filtered
    Context data.
    """

    def __init__(
        self,
        agent: "SoccerAgentMixin",
        context_source: "ContextSource | None" = None,
    ) -> None:
        self._agent = agent
        self._config: SoccerConfig = agent.config
        self._source = context_source
        self._store = SimpleNamespace()
        self._players: list[Player] = [
            agent.player_class(player_id=pid, config=self._config, _backend=None)
            for pid in self._config.player_ids
        ]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._init_store_called = False
        self._last_now: float | None = None
        self._tick_id = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            _log.info("runtime already running, ignore start")
            return
        if self._source is not None:
            self._source.start()
        if not self._init_store_called:
            self._agent.init_store(self._store)
            self._init_store_called = True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="soccer_runtime", daemon=True,
        )
        self._thread.start()
        _log.info(
            "SoccerRuntime started: team_id=%d control_hz=%.1f players=%d source=%s",
            self._config.team_id, self._config.control_hz, len(self._players),
            type(self._source).__name__ if self._source else "None",
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._source is not None:
            self._source.stop()
        self._close_backends()
        _log.info("SoccerRuntime stopped")

    def _close_backends(self) -> None:
        """Close every player's SDK backend."""
        for player in self._players:
            if player._backend is not None:
                try:
                    player._backend.close()
                except Exception as exc:
                    _log.warning(
                        "player %d backend close failed: %s", player.id, exc,
                    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        period = 1.0 / max(1.0, self._config.control_hz)
        while not self._stop.is_set():
            started_at = time.monotonic()
            try:
                self._tick(started_at)
            except Exception as exc:
                _log.exception("control loop tick failed: %s", exc)
                # Stop every player directly after an exception, bypassing play.
                for p in self._players:
                    try:
                        p.stop()
                    except Exception:
                        pass

            elapsed = time.monotonic() - started_at
            self._stop.wait(max(0.0, period - elapsed))

    def _tick(self, now: float) -> None:
        self._tick_id += 1
        dt = 0.0 if self._last_now is None else (now - self._last_now)
        self._last_now = now

        ctx = self._build_context(now, dt)
        for p in self._players:
            p.context = ctx

        # Invoke user play(); the framework does not prescribe its behavior.
        self._agent.play(ctx, self._players, self._store)

        # Log a heartbeat about every two seconds to confirm the loop and data path.
        if self._tick_id % 60 == 0:
            self._log_heartbeat(ctx, dt)

    def _log_heartbeat(self, ctx: Context, dt: float) -> None:
        ball_repr = (
            ",".join(
                f"{pid}:({b.x:.2f},{b.y:.2f})" for pid, b in ctx.ball.items()
            )
            or "none"
        )
        seen = sum(1 for r in ctx.teammates.values() if r.pose is not None)
        opp_seen = sum(1 for r in ctx.opponents.values() if r.pose is not None)
        _log.info(
            "tick #%d dt=%.3f game=%s ball=%s teammates_seen=%d/%d opponents_seen=%d/%d",
            self._tick_id, dt,
            "None" if ctx.game is None else ctx.game.state.value,
            ball_repr,
            seen, len(ctx.teammates),
            opp_seen, len(ctx.opponents),
        )

    # ------------------------------------------------------------------
    # Context construction and freshness filtering
    # ------------------------------------------------------------------

    def _build_context(self, now: float, dt: float) -> Context:
        snap = self._source.get_snapshot() if self._source is not None else WorldSnapshot()
        return Context(
            now=now,
            dt=dt,
            team_id=self._config.team_id,
            field=ADULT_FIELD_DIMENSIONS,
            game=self._fresh_game(snap.game, now),
            ball=self._fresh_ball_map(snap.ball, now),
            teammates={
                pid: self._fresh_robot(r, now) for pid, r in snap.teammates.items()
            },
            opponents={
                pid: self._fresh_robot(r, now) for pid, r in snap.opponents.items()
            },
        )

    def _fresh_game(
        self, game: GameControlState | None, now: float,
    ) -> GameControlState | None:
        if game is None:
            return None
        if now - game.last_seen_at > self._config.game_state_max_age_sec:
            return None
        return game

    def _fresh_ball_map(
        self, ball_map: dict[int, BallState], now: float,
    ) -> dict[int, BallState]:
        """Drop any per-robot ball reading older than ``ball_max_age_sec``."""
        return {
            pid: ball for pid, ball in ball_map.items()
            if now - ball.last_seen_at <= self._config.ball_max_age_sec
        }

    def _fresh_robot(self, robot: RobotState, now: float) -> RobotState:
        """Retain the robot but clear a stale pose; see documentation section 9.3."""
        if (
            robot.pose is not None
            and now - robot.last_seen_at > self._config.robot_pose_max_age_sec
        ):
            return dataclasses.replace(robot, pose=None)
        return robot
