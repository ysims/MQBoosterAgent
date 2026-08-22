"""BoosterRobot SDK wrapper providing one backend handle per player.

This Docker-only platform layer depends on boosteros.robots.booster and is
imported only where the SDK is installed. Player uses ``_backend`` for chassis,
kicking, and slow operations but does not import this module. The agent injects
it during runtime construction, keeping player.py platform-independent.

SDK method names match calls verified in the legacy implementation.

Slow operations such as ``request_mode`` and ``get_up`` are synchronous SDK
calls that can take seconds. Each backend runs them on a worker thread so the
main loop remains nonblocking. A single overwrite slot retains only the latest
request.

Users manage mode through request_mode; see section 5 of docs/new_design.md.
Velocity and kick commands are sent only when ``_mode == "walk"`` to avoid
repeated SDK 400 errors in other modes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, cast

from boosteros.robots.booster import BoosterRobot, SoccerKickManager


__all__ = ["RobotBackend"]


_log = logging.getLogger(__name__)

_GET_UP_THROTTLE_SEC = 1.0


class RobotBackend:
    """SDK wrapper and slow-operation worker for one player.

    The mixin creates and injects it into Player. Runtime closes it, including
    the worker, during shutdown.
    """

    def __init__(self, player_id: int, robot_name: str) -> None:
        self._player_id = player_id
        self._robot_name = robot_name
        self._robot = BoosterRobot(
            virtual_robot_name=robot_name,
            enable_tf_listener=False,
            timeout=10.0,
        )
        self._kick_manager = SoccerKickManager(self._robot)
        self._mode: str | None = None   # Confirmed SDK mode, updated by worker
        self._fall_down_state: str | None = None
        self._fall_down_recoverable: bool = False
        self._kicking = False

        # Slow-operation worker with one overwrite slot and a wake event.
        self._pending: tuple[str, object] | None = None
        self._slot_lock = threading.Lock()
        self._wake = threading.Event()
        self._worker_stop = threading.Event()
        self._last_get_up_at = 0.0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"backend_worker_{player_id}",
            daemon=True,
        )
        self._worker.start()

        _log.info(
            "RobotBackend created: player_id=%d robot_name=%s",
            player_id, robot_name,
        )

    def close(self) -> None:
        """Release SDK resources by stopping work, motion, and the connection."""
        self._worker_stop.set()
        self._wake.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        self.release_kick()
        try:
            self._robot.set_velocity(vx=0.0, vy=0.0, vyaw=0.0)
        except Exception as exc:
            _log.warning(
                "player %d set_velocity(0,0,0) on close failed: %s",
                self._player_id, exc,
            )
        try:
            close_fn = getattr(self._robot, "_close", None)
            if callable(close_fn):
                cast(Callable[[], None], close_fn)()
        except Exception as exc:
            _log.warning("player %d SDK close failed: %s", self._player_id, exc)

    @property
    def mode(self) -> str | None:
        """Return the confirmed SDK mode, updated after the worker switches it."""
        return self._mode

    @property
    def fall_down_state(self) -> str | None:
        """Return the SDK fall state, or None when unknown."""
        return self._fall_down_state

    @property
    def fall_down_recoverable(self) -> bool:
        return self._fall_down_recoverable

    # ------------------------------------------------------------------
    # Chassis control (step 1)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Set chassis velocity unless kicking or outside walk mode."""
        if self._kicking:
            return
        if self._mode != "walk":
            _log.debug(
                "player %d set_velocity skipped: mode=%s (call request_mode first)",
                self._player_id, self._mode,
            )
            return
        try:
            self._robot.set_velocity(vx=vx, vy=vy, vyaw=vyaw)
        except Exception as exc:
            _log.warning(
                "player %d set_velocity(%.3f,%.3f,%.3f) failed: %s",
                self._player_id, vx, vy, vyaw, exc,
            )

    # ------------------------------------------------------------------
    # Kicking (step 2); inputs use body coordinates
    # ------------------------------------------------------------------

    def kick(
        self, direction: float, power: float, ball_x: float, ball_y: float,
    ) -> None:
        """Start or update a body-frame kick while in walk mode."""
        if self._mode != "walk":
            _log.debug(
                "player %d kick skipped: mode=%s (call request_mode first)",
                self._player_id, self._mode,
            )
            return
        try:
            if not self._kicking:
                self._kick_manager.start()
                self._kicking = True
                _log.info("player %d kick started", self._player_id)
            self._kick_manager.update_command(direction=direction, power=power)
            self._kick_manager.update_ball(x=ball_x, y=ball_y)
        except Exception as exc:
            _log.warning("player %d kick failed: %s", self._player_id, exc)
            self._kicking = False

    def release_kick(self) -> None:
        """End the kick so the chassis accepts set_velocity again."""
        if not self._kicking:
            return
        try:
            self._kick_manager.stop()
            _log.info("player %d kick released", self._player_id)
        except Exception as exc:
            _log.warning("player %d kick stop failed: %s", self._player_id, exc)
        finally:
            self._kicking = False

    # ------------------------------------------------------------------
    # Slow operations (step 3), executed nonblockingly by the worker
    # ------------------------------------------------------------------

    def request_mode(self, mode: str) -> None:
        """Request an asynchronous SDK mode switch unless already in that mode."""
        if self._mode == mode:
            return
        self._enqueue(("mode", mode))

    def get_up(self) -> None:
        """Request get-up asynchronously, throttled to about once per second."""
        now = time.monotonic()
        if now - self._last_get_up_at < _GET_UP_THROTTLE_SEC:
            return
        self._last_get_up_at = now
        self._enqueue(("get_up", None))

    def _enqueue(self, intent: tuple[str, object]) -> None:
        with self._slot_lock:
            self._pending = intent   # Keep only the most recent intent.
        self._wake.set()

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            self._wake.wait(timeout=0.5)
            if self._worker_stop.is_set():
                break
            self._wake.clear()
            # Poll the actual SDK mode so _mode reflects reality. A simulation
            # restart can reset the robot out of walk mode; polling lets
            # ensure_ready detect that and request walk mode again.
            self._poll_mode()
            self._poll_fall_down_state()
            with self._slot_lock:
                intent = self._pending
                self._pending = None
            if intent is None:
                continue
            kind, arg = intent
            if kind == "mode":
                self._exec_set_mode(cast(str, arg))
            elif kind == "get_up":
                self._exec_get_up()

    def _poll_mode(self) -> None:
        try:
            mode = self._robot.get_mode()
        except Exception as exc:
            _log.debug("player %d get_mode failed: %s", self._player_id, exc)
            return
        if isinstance(mode, str):
            self._mode = mode
            if mode == "walk":
                self._fall_down_state = "normal"
                self._fall_down_recoverable = False

    def _poll_fall_down_state(self) -> None:
        if self._mode == "walk":
            return
        try:
            fall_down_state = self._robot.get_fall_down_state()
        except Exception as exc:
            _log.debug(
                "player %d get_fall_down_state failed: %s", self._player_id, exc,
            )
            return
        state_value = getattr(fall_down_state, "state", None)
        recoverable_value = getattr(fall_down_state, "recoverable", False)
        self._fall_down_state = state_value if isinstance(state_value, str) else None
        self._fall_down_recoverable = (
            recoverable_value if isinstance(recoverable_value, bool) else False
        )

    def _exec_set_mode(self, mode: str) -> None:
        try:
            self._robot.set_gait("soccer")
            self._robot.set_mode(mode)
            self._mode = mode   # Optimistic update; the next poll corrects it.
            _log.info("player %d entered %s mode", self._player_id, mode)
        except Exception as exc:
            _log.warning("player %d set_mode(%s) failed: %s", self._player_id, mode, exc)

    def _exec_get_up(self) -> None:
        try:
            self._robot.get_up()
            self._mode = None   # Mode is unknown after getting up; request it again.
            self._fall_down_state = None
            self._fall_down_recoverable = False
            _log.info("player %d get_up done", self._player_id)
        except Exception as exc:
            _log.warning("player %d get_up failed: %s", self._player_id, exc)
