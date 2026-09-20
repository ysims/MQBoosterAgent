"""Player control handle for one robot, designed for direct user editing.

Platform primitives such as ``set_velocity``, ``kick``, ``release_kick``,
``request_mode``, and ``get_up`` delegate to the injected framework
``_backend`` (see ``motion.backend.RobotBackend``). State properties such as
``pose``, ``mode``, ``is_fallen``, and ``penalty`` read from ``self.context``
or the backend.

Movement behaviors such as ``walk_to``, ``face_to``, and ``ensure_ready`` are
also Player methods because they command this player and may need cross-frame
state for hysteresis or avoidance -- they call into ``planning`` for path and
kick decisions. Pure coordinate calculations such as ``dist``, ``angle_to``,
and goal coordinates belong in utils/geom.

Instances live for the entire match, while the framework replaces
``self.context`` every frame. Add custom skills directly to this class.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

from ..framework.types import BallState, Context, Penalty, Pose2D
from ..framework import debugdraw

from ..motion.config import (
    ANGULAR_GAIN,
    ARRIVE_DIST,
    PREPARE_SETTLE_SEC,
    CHASE_BEHIND_M,
    KICK_ENTER_M,
    KICK_EXIT_M,
    KICK_POWER_MAX,
    KICK_POWER_MIN,
    LINEAR_GAIN,
    MAX_ANGULAR,
    MAX_LINEAR,
    OMNI_DIST,
    TURN_THRESHOLD,
)
from ..planning.config import PLAN_LOOKAHEAD, USE_GLOBAL_PATH_PLANNER
from ..planning.kick_planning import plan_kick as _plan_kick
from ..planning.path_planning import (
    collect_obstacles,
    path_waypoint,
    plan_global_path,
    plan_local_heading,
)
from ..planning.config import KICK_POWER_DEFAULT
from ..utils.geom import (
    angle_to,
    clamp,
    dist,
    normalize_angle,
    opponent_goal,
    own_goal,
    own_goal_area_center,
)
from .config import (
    GUARD_FACE_BALL,
    KICKOFF_FRONT_MARGIN,
    KICKOFF_LATERAL_TOL,
    KICKOFF_STAGE_M,
    SUPPORT_DIST_M,
)

if TYPE_CHECKING:
    from ..framework.config import SoccerConfig


__all__ = ["Player"]


_log = logging.getLogger(__name__)


def _behind_ball(
    ball_x: float, ball_y: float, aim: tuple[float, float], offset: float,
) -> tuple[float, float]:
    """Return a position ``offset`` behind the ball relative to ``aim``.

    This is used as the chase target so the ball lies between the robot and
    ``aim``, naturally aligning the robot on arrival. If ``aim`` coincides with
    the ball and the direction is undefined, return the ball position.
    """
    dx, dy = aim[0] - ball_x, aim[1] - ball_y
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return (ball_x, ball_y)
    ux, uy = dx / d, dy / d              # Unit vector from the ball to aim.
    return (ball_x - ux * offset, ball_y - uy * offset)   # Move away from aim.


class Player:
    """Handle for one player. Add new skills here.
    """

    def __init__(
        self,
        player_id: int,
        config: "SoccerConfig",
        _backend: object | None,
    ) -> None:
        self.id: int = player_id
        self.config: "SoccerConfig" = config
        self._backend = _backend           # Robot control wrapper; rarely changed
        self.context: Context | None = None

        # Current high-level action for visualization and debugging. Strategy
        # dispatch updates it each frame; actions such as guard refine substates
        # internally. The visualization pass in main.py uses it as a label.
        self.action: str = "init"

        # SDK cache fields updated automatically by the framework backend.
        self._mode: str | None = None
        self._fall_down_state: str | None = None

        # When we first observed "prepare" mode, for PREPARE_SETTLE_SEC in
        # ensure_ready(); None means not currently tracking a prepare wait.
        self._prepare_entered_at: float | None = None

        # Cross-frame detour-side memory; None means no active detour.
        self._avoid_side: float | None = None

        # Cross-frame kick hysteresis state.
        self._kicking: bool = False

        # Cross-frame block/guard hysteresis state.
        self._block_pressing: bool = False
        self._guard_threatened: bool = False
        # Cross-frame state for driving the ball over the goal line.
        self._goal_line_push: bool = False
        # Cross-frame state for temporarily attacking when support gets stuck.
        self._support_last_pos: tuple[float, float] | None = None
        self._support_stationary_since: float | None = None
        self._support_last_update_at: float | None = None

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def is_kicking(self) -> bool:
        """Return whether the player is currently kicking."""
        return self._kicking

    @property
    def pose(self) -> Pose2D | None:
        ctx = self.context
        if ctx is None:
            return None
        robot = ctx.teammates.get(self.id)
        return None if robot is None else robot.pose

    @property
    def ball(self) -> BallState | None:
        """This player's own ball belief -- not shared with teammates.

        There is no team-wide "the" ball (see ``Context.ball``'s docstring);
        each robot's belief comes only from its own detections.
        """
        ctx = self.context
        if ctx is None:
            return None
        return ctx.ball.get(self.id)

    @property
    def mode(self) -> str | None:
        if self._backend is not None:
            return self._backend.mode
        return self._mode

    @property
    def is_fallen(self) -> bool:
        return self.fall_down_state not in (None, "normal")

    @property
    def fall_down_state(self) -> str | None:
        if self._backend is not None:
            return getattr(self._backend, "fall_down_state", None)
        return self._fall_down_state

    @property
    def penalty(self) -> Penalty:
        ctx = self.context
        if ctx is None or ctx.game is None:
            return Penalty.NONE
        state = ctx.game.get_player_state(self.config.team_id, self.id)
        return Penalty.NONE if state is None else state.penalty

    @property
    def is_penalized(self) -> bool:
        return self.penalty != Penalty.NONE

    # ------------------------------------------------------------------
    # Chassis control
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        if self._backend is None:
            _log.debug(
                "player %d set_velocity vx=%.3f vy=%.3f vyaw=%.3f (no backend)",
                self.id, vx, vy, vyaw,
            )
            return
        self._backend.set_velocity(vx, vy, vyaw)

    def stop(self) -> None:
        self.release_kick()
        self.set_velocity(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Kicking
    # ------------------------------------------------------------------

    def kick(
        self,
        kick_direction: float | None = None,
        power: float = KICK_POWER_DEFAULT,
    ) -> None:
        pose = self.pose
        if pose is None:
            _log.warning("player %d kick skipped: pose unknown", self.id)
            self.release_kick()
            return

        ball = self.ball
        if ball is None:
            # Release rather than just warn: kick() only sets self._kicking
            # True in the success path below, never clears it here, and
            # motion.backend.RobotBackend.set_velocity() unconditionally
            # drops commands while _kicking is True. A caller that keeps
            # calling kick() every frame regardless of ball visibility (e.g.
            # _act_our_kickoff, which has no ball check of its own) would
            # otherwise leave this player permanently frozen the moment its
            # own ball detection drops out mid-kick -- a real, observed
            # failure mode, not just a hypothetical one.
            _log.warning("player %d kick skipped: ball unknown", self.id)
            self.release_kick()
            return

        if kick_direction is None:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                _log.warning("player %d kick skipped: kick plan unavailable", self.id)
                self.release_kick()
                return
            kick_direction, power = kick_plan

        if self._backend is None:
            _log.debug(
                "player %d kick ball=(%.3f, %.3f) dir=%.3f (no backend)",
                self.id, ball.x, ball.y, kick_direction,
            )
            return
        # Transform field coordinates to body coordinates using the current pose.
        dx = ball.x - pose.x
        dy = ball.y - pose.y
        cos_t = math.cos(pose.theta)
        sin_t = math.sin(pose.theta)
        ball_x_body = dx * cos_t + dy * sin_t
        ball_y_body = -dx * sin_t + dy * cos_t
        kick_direction = normalize_angle(kick_direction)
        direction_body = normalize_angle(kick_direction - pose.theta)
        power_clamped = max(KICK_POWER_MIN, min(KICK_POWER_MAX, power))
        self._kicking = True
        self._backend.kick(direction_body, power_clamped, ball_x_body, ball_y_body)

    def plan_kick(self) -> tuple[float, float] | None:
        """Calculate kick direction and power; see planning.kick_planning."""
        return _plan_kick(self.context, self.ball)

    def release_kick(self) -> None:
        self._kicking = False  # Clear kick hysteresis and cube visualization.
        if self._backend is None:
            _log.debug("player %d release_kick (no backend)", self.id)
            return
        self._backend.release_kick()

    # ------------------------------------------------------------------
    # Slow operations, invoked asynchronously to avoid blocking
    # ------------------------------------------------------------------

    def request_mode(self, mode: str) -> None:
        if self._backend is None:
            _log.debug("player %d request_mode -> %s (no backend)", self.id, mode)
            return
        self._backend.request_mode(mode)

    def get_up(self) -> None:
        if self._backend is None:
            _log.debug("player %d get_up (no backend)", self.id)
            return
        self._backend.get_up()

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def ensure_ready(self) -> bool:
        """Recover from falls and switch to walk mode asynchronously.

        The SDK only allows ``damping -> prepare -> walk``, never
        ``damping -> walk`` directly (``set_mode`` raises if the transition
        isn't supported) -- a robot that spawns upright in ``"damping"``
        mode is never "fallen", so it needs the explicit "prepare" step
        here rather than reaching "prepare" as a side effect of ``get_up()``.

        Return whether the player is ready to act during this frame.
        """
        if self.is_fallen:
            self._prepare_entered_at = None
            self.get_up()
            return False
        if self.mode == "walk":
            self._prepare_entered_at = None
            return True
        if self.mode != "prepare":
            self._prepare_entered_at = None
            self.request_mode("prepare")
            return False
        # In "prepare": wait for it to physically settle before asking for
        # "walk" -- see PREPARE_SETTLE_SEC's docstring.
        now = time.monotonic()
        if self._prepare_entered_at is None:
            self._prepare_entered_at = now
            return False
        if now - self._prepare_entered_at < PREPARE_SETTLE_SEC:
            return False
        self.request_mode("walk")
        return False

    def face_to(self, target_theta: float) -> None:
        """Turn in place to the target heading."""
        if self.pose is None:
            self.stop()
            return
        err = normalize_angle(target_theta - self.pose.theta)
        self.set_velocity(0.0, 0.0, self._angular(err))

    def walk_to(
        self,
        target: tuple[float, float],
        *,
        face: float | None = None,
        avoid_ball: bool = False,
        avoid_robots: bool = False,
        arrive_dist: float = ARRIVE_DIST,
    ) -> bool:
        """Walk toward a target point and return whether it has been reached.

        With ``avoid_ball`` or ``avoid_robots`` enabled, ``planning`` collects
        obstacles and either finds a global A* path or falls back to a
        simplified VFH local planner that scans candidate headings and picks
        the one closest to the target that remains collision-free over
        ``PLAN_LOOKAHEAD``.

        Nearby targets use omnidirectional walking; distant targets use a
        turn-walk-turn sequence. ``face`` sets the heading near or at the target.
        """
        self.release_kick() # Walking is overridden while kicking, so release first.
        pose = self.pose
        if pose is None:
            self.stop()
            return False

        tx, ty = target
        dx = tx - pose.x
        dy = ty - pose.y
        distance = math.hypot(dx, dy)

        if distance < arrive_dist:
            # At the target, turn to the requested heading if needed.
            if face is not None:
                err = normalize_angle(face - pose.theta)
                if abs(err) > 0.1:
                    self.set_velocity(0.0, 0.0, self._angular(err))
                else:
                    self.stop()
            else:
                self.stop()
            return True

        # Prefer global A* planning and fall back to the legacy local planner.
        goal_dir = math.atan2(dy, dx)
        planned_path: list[tuple[float, float]] | None = None
        waypoint: tuple[float, float] | None = None
        if (avoid_ball or avoid_robots) and self.context is not None:
            obstacles = collect_obstacles(
                self.context, self.id,
                ball=avoid_ball, robots=avoid_robots,
                goals=(avoid_ball or avoid_robots),
                ball_state=self.ball,
            )
            if USE_GLOBAL_PATH_PLANNER:
                planned_path = plan_global_path(
                    self.context,
                    (pose.x, pose.y),
                    (tx, ty),
                    obstacles,
                )
                if planned_path is not None:
                    waypoint = path_waypoint(pose, planned_path)
                    heading = angle_to(pose.x, pose.y, waypoint[0], waypoint[1])
                else:
                    heading = plan_local_heading(self.id, pose, goal_dir, obstacles)
            else:
                heading = plan_local_heading(self.id, pose, goal_dir, obstacles)
        else:
            heading = goal_dir

        # Visualize the target (green), direct line (gray), planned heading
        # (yellow), and forward lookahead probe (cyan).
        debugdraw.point(tx, ty, rgb=(0.0, 1.0, 0.0), scale=0.15, ns="target")
        debugdraw.line([(pose.x, pose.y), (tx, ty)], rgb=(0.4, 0.4, 0.4), ns="to_target")
        if planned_path is not None and len(planned_path) >= 2:
            debugdraw.line(planned_path, rgb=(0.2, 0.8, 1.0), ns="global_path")
        if waypoint is not None:
            debugdraw.point(
                waypoint[0], waypoint[1],
                rgb=(0.2, 0.8, 1.0), scale=0.12, ns="global_waypoint",
            )
        debugdraw.arrow(
            pose.x, pose.y,
            pose.x + math.cos(heading) * 0.6, pose.y + math.sin(heading) * 0.6,
            rgb=(1.0, 1.0, 0.0), ns="heading",
        )
        debugdraw.line(
            [(pose.x, pose.y),
             (pose.x + math.cos(heading) * PLAN_LOOKAHEAD,
              pose.y + math.sin(heading) * PLAN_LOOKAHEAD)],
            rgb=(0.0, 0.8, 0.8), ns="lookahead",
        )

        if distance <= OMNI_DIST:
            # Nearby: translate along heading while turning toward face.
            wdx, wdy = math.cos(heading) * distance, math.sin(heading) * distance
            cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
            vx = LINEAR_GAIN * (wdx * cos_t + wdy * sin_t)
            vy = LINEAR_GAIN * (-wdx * sin_t + wdy * cos_t)
            speed = math.hypot(vx, vy)
            if speed > MAX_LINEAR:
                vx *= MAX_LINEAR / speed
                vy *= MAX_LINEAR / speed
            vyaw = (
                self._angular(normalize_angle(face - pose.theta))
                if face is not None else 0.0
            )
            self.set_velocity(vx, vy, vyaw)
        else:
            # Distant: turn, walk, and turn along heading.
            angle_err = normalize_angle(heading - pose.theta)
            if abs(angle_err) > TURN_THRESHOLD:
                self.set_velocity(0.0, 0.0, self._angular(angle_err))
            else:
                vx = clamp(
                    LINEAR_GAIN * distance * math.cos(angle_err), 0.0, MAX_LINEAR,
                )
                self.set_velocity(vx, 0.0, self._angular(angle_err))
        return False

    # ------------------------------------------------------------------
    # High-level actions called directly by the strategy in main.py
    # ------------------------------------------------------------------

    def attack(self, kick_target: tuple[float, float] | None = None) -> None:
        """Chase the ball and shoot toward ``kick_target`` or the opponent goal.

        Kick hysteresis enters below ENTER and exits above EXIT, where EXIT is
        greater than ENTER, to prevent threshold oscillation. Normal challenges
        approach the ball directly without avoidance. The chase target lies
        ``CHASE_BEHIND_M`` behind the ball on the ball-to-goal line, naturally
        aligning the player for a shot on arrival.
        """
        ball = self.ball
        if ball is None or self.pose is None or self.context is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(self.context)

        d = dist(self.pose.x, self.pose.y, ball.x, ball.y)
        self._kicking = d <= (KICK_EXIT_M if self._kicking else KICK_ENTER_M)
        if self._kicking:
            kick_plan = self.plan_kick()
            if kick_plan is None:
                self.stop()
                return
            kick_direction, kick_power = kick_plan
            self.kick(kick_direction, kick_power)
        else:
            self.release_kick()
            self.walk_to(
                _behind_ball(ball.x, ball.y, kick_target, CHASE_BEHIND_M)
            )

    def guard(self) -> None:
        """Guard from the center of the goal area.

        Fall back to that position when no active threat is available.
        """
        home = own_goal_area_center(self.context) if self.context is not None else None
        if home is None or self.pose is None:
            self.action = "guard:stop"
            self.stop()
            return

        ball = self.ball

        # Face the ball for faster reactions, or the opponent's goal if unseen.
        face = 0.0
        if GUARD_FACE_BALL and ball is not None:
            face = angle_to(
                self.pose.x, self.pose.y, ball.x, ball.y,
            )

        self.action = "guard:home"
        debugdraw.point(
            home[0], home[1], rgb=(0.0, 0.6, 1.0), scale=0.2, ns="guard_home",
        )
        self.walk_to(home, face=face, avoid_ball=True, avoid_robots=True)

    def support(self) -> None:
        """Support on the ball-to-own-goal line at ``SUPPORT_DIST_M``.

        This places the player on a blocking line between the ball and our goal.
        """
        ctx = self.context
        ball = self.ball
        if ctx is None or ball is None:
            self.stop()
            return

        gx, gy = own_goal(ctx)
        bx, by = (ball.x, ball.y)
        dx, dy = gx - bx, gy - by
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = -1.0, 0.0
        else:
            ux, uy = dx / d, dy / d
        along = min(SUPPORT_DIST_M, d)
        tx = bx + ux * along
        ty = by + uy * along

        # Stay 0.3 m inside our goal line and clamp laterally inside the field.
        half_l = ctx.field.length / 2.0
        half_w = ctx.field.width / 2.0 - 0.3
        tx = clamp(tx, -half_l + 0.3, half_l)
        ty = clamp(ty, -half_w, half_w)
        self.move_to_position((tx, ty))

    def take_kickoff(self, kick_target: tuple[float, float] | None = None) -> None:
        """Stage behind the ball for our restart, then approach and kick."""
        ball = self.ball
        if ball is None or self.pose is None or self.context is None:
            self.stop()
            return
        if kick_target is None:
            kick_target = opponent_goal(self.context)
        kick_dir = angle_to(ball.x, ball.y, *kick_target)
        cos_k, sin_k = math.cos(kick_dir), math.sin(kick_dir)
        rel_x, rel_y = self.pose.x - ball.x, self.pose.y - ball.y
        behind = rel_x * cos_k + rel_y * sin_k          # Negative is our side.
        lateral = abs(-rel_x * sin_k + rel_y * cos_k)
        from .config import KICKOFF_FRONT_MARGIN, KICKOFF_LATERAL_TOL, KICKOFF_STAGE_M
        if behind > KICKOFF_FRONT_MARGIN or lateral > KICKOFF_LATERAL_TOL:
            stage = (
                ball.x - cos_k * KICKOFF_STAGE_M,
                ball.y - sin_k * KICKOFF_STAGE_M,
            )
            self.release_kick()
            self.walk_to(stage, face=kick_dir, avoid_ball=True)
        else:
            self.attack(kick_target)

    def move_to_position(self, target: tuple[float, float] | None) -> None:
        """Move to a support/defense position while facing and avoiding the ball."""
        if target is None:
            self.stop()
            return
        face = None
        ball = self.ball
        if ball is not None and self.pose is not None:
            face = angle_to(self.pose.x, self.pose.y, ball.x, ball.y)
        self.release_kick()
        self.walk_to(target, face=face, avoid_ball=True, avoid_robots=True)

    @staticmethod
    def _angular(err: float) -> float:
        return clamp(ANGULAR_GAIN * err, -MAX_ANGULAR, MAX_ANGULAR)
