"""Self-localisation contract: the ``Localiser`` protocol.

No ROS or boosteros dependencies, mirroring the rule in ``framework/types.py``.
"""

from __future__ import annotations

from typing import Protocol

from ..framework.types import Pose2D


__all__ = ["Localiser"]


class Localiser(Protocol):
    """Per-robot self-pose estimator.

    ``VisionContextSource`` constructs one per robot as
    ``localiser_class(node, topic, anchor_x, anchor_y, mirrored)`` --
    ``topic`` is the robot's raw odometry topic, ``anchor_x``/``anchor_y``
    calibrate its arbitrary boot-time origin against the robot's known
    field-frame starting position (see ``ODOM_FIELD_ANCHOR`` in
    ``localisation/config.py``), and ``mirrored`` selects whether that
    team's field frame needs the 180 degree rotation relative to team1's
    (see ``VisionContextSource._create_robots``). Implementations own
    whatever ROS subscription(s) they need (given ``node``) and answer
    ``get_pose()`` from cached state. Returning ``None`` signals "no current
    pose estimate" (e.g. never received, or stale) and propagates to
    ``Context.teammates[id].pose``.
    """

    def __init__(
        self, node: object, topic: str,
        anchor_x: float, anchor_y: float, mirrored: bool,
    ) -> None: ...

    def get_pose(self) -> Pose2D | None: ...
