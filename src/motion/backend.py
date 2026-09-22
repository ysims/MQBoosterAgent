"""Action-level robot backend: mode-gated velocity and kick commands.

Wraps a ``framework.robot_backend.SdkConnection`` (the raw SDK connection and
its slow-operation worker) with the action policy Player actually calls:
velocity and kick commands are only sent in walk mode, and kicking suppresses
velocity commands until released, mirroring a real kick's exclusive control
of the chassis.
"""

from __future__ import annotations

import logging

from ..framework.robot_backend import SdkConnection


__all__ = ["RobotBackend"]


_log = logging.getLogger(__name__)


class RobotBackend:
    """Action wrapper around one player's SDK connection."""

    def __init__(self, player_id: int, robot_name: str) -> None:
        self._player_id = player_id
        self._connection = SdkConnection(player_id, robot_name)
        self._kicking = False

    def close(self) -> None:
        self.release_kick()
        self._connection.close()

    @property
    def mode(self) -> str | None:
        return self._connection.mode

    @property
    def fall_down_state(self) -> str | None:
        return self._connection.fall_down_state

    @property
    def fall_down_recoverable(self) -> bool:
        return self._connection.fall_down_recoverable

    # ------------------------------------------------------------------
    # Chassis control
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Set chassis velocity unless kicking or outside walk mode."""
        if self._kicking:
            return
        if self._connection.mode != "walk":
            _log.debug(
                "player %d set_velocity skipped: mode=%s (call request_mode first)",
                self._player_id, self._connection.mode,
            )
            return
        self._connection.raw_set_velocity(vx, vy, vyaw)

    def set_head_angle(self, pitch: float, yaw: float) -> None:
        """Set head pitch/yaw unless outside walk mode (the SDK requires it)."""
        if self._connection.mode != "walk":
            _log.debug(
                "player %d set_head_angle skipped: mode=%s (call request_mode first)",
                self._player_id, self._connection.mode,
            )
            return
        self._connection.raw_set_head_angle(pitch, yaw)

    # ------------------------------------------------------------------
    # Kicking; inputs use body coordinates
    # ------------------------------------------------------------------

    def kick(
        self, direction: float, power: float, ball_x: float, ball_y: float,
    ) -> None:
        """Start or update a body-frame kick while in walk mode."""
        if self._connection.mode != "walk":
            _log.debug(
                "player %d kick skipped: mode=%s (call request_mode first)",
                self._player_id, self._connection.mode,
            )
            return
        try:
            if not self._kicking:
                self._connection.kick_manager.start()
                self._kicking = True
                _log.info("player %d kick started", self._player_id)
            self._connection.kick_manager.update_command(direction=direction, power=power)
            self._connection.kick_manager.update_ball(x=ball_x, y=ball_y)
        except Exception as exc:
            _log.warning("player %d kick failed: %s", self._player_id, exc)
            self._kicking = False

    def release_kick(self) -> None:
        """End the kick so the chassis accepts set_velocity again."""
        if not self._kicking:
            return
        try:
            self._connection.kick_manager.stop()
            _log.info("player %d kick released", self._player_id)
        except Exception as exc:
            _log.warning("player %d kick stop failed: %s", self._player_id, exc)
        finally:
            self._kicking = False

    # ------------------------------------------------------------------
    # Slow operations, delegated to the connection's worker
    # ------------------------------------------------------------------

    def request_mode(self, mode: str) -> None:
        self._connection.request_mode(mode)

    def get_up(self) -> None:
        self._connection.get_up()
