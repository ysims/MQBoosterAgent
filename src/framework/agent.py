"""SoccerAgent mixin providing framework behavior to the user's entry class.

Booster build validation requires ``booster_agent_framework.AgentBase`` to be
a direct base of the entry class and does not follow indirect inheritance.
Consequently, the framework cannot provide ``SoccerAgent(AgentBase)`` as the
only user base because SoccerAgent, rather than AgentBase, would be direct.

The solution is to place framework behavior in a mixin that does not inherit
AgentBase. User entry classes declare
``class MyAgent(SoccerAgentMixin, AgentBase)`` so AgentBase remains direct.

See section 8 of docs/new_design.md for the detailed API.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from booster_agent_framework import AgentFeatures

from .config import SoccerConfig
from .runtime import SoccerRuntime

if TYPE_CHECKING:
    from types import SimpleNamespace

    from ..player import Player
    from .types import Context
    from .vision_types import Detector, Localiser


__all__ = ["SoccerAgentMixin"]


_log = logging.getLogger(__name__)


class _PlatformLogHandler(logging.Handler):
    """Forward standard Python logging records to the Booster platform logger.

    The rclcpp-style platform logger only exposes ``.info``, ``.warn``, and
    ``.error`` methods. Standard logging has no handler by default and drops
    INFO records. This bridge routes framework and user logs to the console and
    log files correctly.

    Platform coupling is isolated here; runtime and player continue to use
    platform-independent standard logging.
    """

    def __init__(self, platform_logger: object) -> None:
        super().__init__()
        self._platform = platform_logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                self._platform.error(msg)  # type: ignore[attr-defined]
            elif record.levelno >= logging.WARNING:
                warn = getattr(self._platform, "warn", None) or getattr(
                    self._platform, "warning", None
                )
                (warn or self._platform.info)(msg)  # type: ignore[attr-defined]
            else:
                self._platform.info(msg)  # type: ignore[attr-defined]
        except Exception:
            self.handleError(record)


class SoccerAgentMixin:
    """Framework behavior mixin used alongside, but not derived from, AgentBase.

    Usage::

        from booster_agent_framework import AgentBase
        from .soccer.agent import SoccerAgentMixin

        class MyAgent(SoccerAgentMixin, AgentBase):
            player_class = MyPlayer

            @staticmethod
            def play(context, players, store): ...

            def init_store(self, store): ...

    The MRO is ``[MyAgent, SoccerAgentMixin, AgentBase, object]``, so
    ``super().__init__`` correctly reaches ``AgentBase.__init__``.
    """

    # ------------------------------------------------------------------
    # User-provided hooks
    # ------------------------------------------------------------------

    # Hook 1: Player class. main.py must set ``player_class = Player`` from
    # src.player. The framework injects it instead of importing user player.py.
    player_class: "type[Player]"

    # Hook 2: play, called every frame and a no-op by default.
    @staticmethod
    def play(
        context: "Context",
        players: "list[Player]",
        store: "SimpleNamespace",
    ) -> None:
        """Run at 30 Hz. Subclasses may override this default no-op."""

    # Hook 3: optional init_store, called once before the match.
    def init_store(self, store: "SimpleNamespace") -> None:
        """Subclasses may override this default no-op."""

    # Hook 4/5: perception defaults for perception_mode="vision" (the
    # default). main.py sets these explicitly from src.vision so the swap
    # point is obvious; ignored when perception_mode="ground_truth".
    detector_class: "type[Detector]"
    localiser_class: "type[Localiser]"

    # ------------------------------------------------------------------
    # Framework internals; users normally do not modify these
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Follow the MRO to AgentBase.__init__(AgentFeatures()).
        super().__init__(AgentFeatures())  # type: ignore[call-arg]
        self._setup_logging()
        self.config = SoccerConfig.from_env()
        source = self._create_context_source()
        self.runtime = SoccerRuntime(self, context_source=source)
        # Create and inject an SDK backend for each player.
        self._create_backends()
        _log.info(
            "SoccerAgent initialized: team_id=%d robots=%s",
            self.config.team_id, list(self.config.robot_names),
        )

    def _create_context_source(self):
        """Build the ContextSource per ``config.perception_mode``.

        Both imports are delayed Docker-only ROS imports, mirroring
        ``_create_backends``, so rclpy is not required for development.
        """
        if self.config.perception_mode == "ground_truth":
            from .ros_source import RosContextSource

            return RosContextSource(self.config)

        from .vision_source import VisionContextSource

        return VisionContextSource(
            self.config,
            detector_class=self.detector_class,
            localiser_class=self.localiser_class,
        )

    def _create_backends(self) -> None:
        """Create each player's SDK backend through a delayed Docker-only import."""
        from .robot_backend import RobotBackend

        for player in self.runtime._players:
            robot_name = self.config.robot_names[player.id - 1]
            player._backend = RobotBackend(player.id, robot_name)

    def _setup_logging(self) -> None:
        """Bridge standard logging to the platform logger.

        AgentBase provides ``self.logger`` after ``super().__init__``. Install
        the bridge once and remove an old bridge before repeated activation to
        prevent duplicate output.
        """

        platform_logger = getattr(self, "logger", None)
        if platform_logger is None:
            # Fall back to stderr if the theoretically required logger is absent.
            logging.basicConfig(level=logging.INFO)
            return
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        for handler in list(root.handlers):
            if isinstance(handler, _PlatformLogHandler):
                root.removeHandler(handler)
        bridge = _PlatformLogHandler(platform_logger)
        bridge.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root.addHandler(bridge)

    def on_agent_activated(self) -> None:
        _log.info("SoccerAgent activated")
        self.runtime.start()

    def on_agent_close(self) -> None:
        _log.info("SoccerAgent closing")
        self.runtime.stop()
