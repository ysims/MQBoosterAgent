"""Vision + localisation source: a ContextSource built on real sensor topics.

This Docker-only platform layer depends on rclpy, vision_msgs, nav_msgs, and
std_msgs and is imported only in a ROS environment, mirroring
``ros_source.py``. Unlike ``RosContextSource``, it never reads
``.../sim/ground_truth/...`` topics: teammate self-pose comes from a plugged
``Localiser`` per robot (default: :class:`src.vision.OdomAnchoredLocaliser`
dead-reckoning ``/robot{name}/odom`` against a pre-measured field anchor),
and ball state comes from the sim's own ``detection_extension``, which
publishes ``vision_msgs/Detection2DArray`` on
``/{robot_name}/soccer/sim/vision/detections`` -- a pixel bounding box per
object, computed by projecting true object positions through each camera's
real field-of-view and pose (respecting occlusion), same as a real camera
would see. Turning that bounding box into a 3D position is a plugged
``ball_position_estimator`` (default: :func:`src.vision.estimate_ball_position`),
matching the ``Localiser`` hook's pattern; this class only rotates/
translates the resulting robot-frame position into the field frame using
the robot's current pose. See ``scripts/patch_match_scene_detection.py``
for how this extension gets enabled on the 3v3 match scene, which omits it
by default.

Detection also reports ``Goalpost`` and field-marker positions at known,
fixed field locations, not currently consumed here -- a natural opening for
landmark-based localisation correction on top of ``OdomAnchoredLocaliser``'s
dead reckoning, as a follow-up.

Two semantic differences from ground-truth mode, both intentional:
1. Ball state is fused across whatever robots currently see it, rather than
   read from one authoritative topic.
2. ``detection_extension`` does not detect other robots at all (only Ball,
   Goalpost, and field markers), so ``Context.opponents`` is always empty
   under this source. The opponent-tracking machinery below is kept for a
   future detector that does report robots -- it is simply never fed.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String as RosString

from .config import SoccerConfig
from .game_codec import game_control_state_from_json
from .types import BallState, GameControlState, Pose2D, RobotState, WorldSnapshot
from .vision_types import Detection2D, FieldDetection, Localiser
from ..param import (
    BALL_DETECTION_MAX_AGE_SEC,
    ODOM_FIELD_ANCHOR,
    OPPONENT_TRACK_MATCH_DIST_M,
    OPPONENT_TRACK_MAX_MISS_FRAMES,
)


__all__ = ["VisionContextSource"]


_log = logging.getLogger(__name__)

# detection_extension's own class_id strings -> our internal FieldDetection
# labels. Only Ball is consumed today; Goalpost and field-marker names (see
# the module docstring) are left for a future landmark-localisation pass.
_LABEL_MAP = {"Ball": "ball"}


class _OpponentTrack:
    __slots__ = ("track_id", "x", "y", "last_seen_at", "missed", "dirty")

    def __init__(self, track_id: int, x: float, y: float, last_seen_at: float) -> None:
        self.track_id = track_id
        self.x = x
        self.y = y
        self.last_seen_at = last_seen_at
        self.missed = 0
        self.dirty = True


class _RobotVision:
    """Own one robot's detection subscription and localiser.

    Subscribes directly to the sim's own
    ``{robot_name}/soccer/sim/vision/detections`` topic via plain ``rclpy``
    -- a normal ROS topic, no SDK connection needed (see the module
    docstring for why this replaced a from-pixels RGB detector). Each
    incoming detection already carries the object's position in this
    robot's own body frame; converting that to field-frame coordinates only
    needs the robot's current pose from ``localiser``.
    """

    def __init__(
        self, player_id: int, localiser: Localiser,
        node: Any, source: "VisionContextSource", topic: str,
    ) -> None:
        self.player_id = player_id
        self.localiser = localiser
        self._source = source

        # Diagnostic counters, logged periodically so a live log dump shows
        # whether detections are arriving at all and how many are ball
        # detections, instead of only ever showing "ball=None" downstream.
        self._messages_seen = 0
        self._messages_no_pose = 0
        self._degenerate_bbox = 0
        self._ball_detections = 0
        self._logged_first_message = False

        from vision_msgs.msg import Detection2DArray

        self._sub = node.create_subscription(
            Detection2DArray, topic, self._on_detections, source._qos(depth=10),
        )

    def _on_detections(self, msg: Any) -> None:
        self._messages_seen += 1
        if not self._logged_first_message:
            self._logged_first_message = True
            _log.info(
                "robot %d vision: first detections message received (%d detections)",
                self.player_id, len(msg.detections),
            )

        pose = self.localiser.get_pose()
        if pose is None:
            self._messages_no_pose += 1
            self._maybe_log_stats()
            return

        now = time.monotonic()
        cos_t, sin_t = math.cos(pose.theta), math.sin(pose.theta)
        field_detections: list[FieldDetection] = []
        for det in msg.detections:
            if not det.results:
                continue
            hypothesis = det.results[0].hypothesis
            label = _LABEL_MAP.get(hypothesis.class_id)
            if label is None:
                continue

            bbox = det.bbox
            detection = Detection2D(
                x_px=bbox.center.position.x, y_px=bbox.center.position.y,
                w_px=bbox.size_x, h_px=bbox.size_y,
                label=label, confidence=float(hypothesis.score) if hypothesis.score else 1.0,
            )
            robot_frame = self._source._ball_position_estimator(detection)
            if robot_frame is None:
                self._degenerate_bbox += 1
                continue
            forward, left = robot_frame

            # Robot body frame (forward, left) -> field frame, using the
            # robot's own current pose.
            field_x = pose.x + forward * cos_t - left * sin_t
            field_y = pose.y + forward * sin_t + left * cos_t
            field_detections.append(FieldDetection(
                x=field_x, y=field_y, label=detection.label,
                confidence=detection.confidence,
                source_robot_id=self.player_id, timestamp=now,
            ))
            if label == "ball":
                self._ball_detections += 1

        if field_detections:
            self._source._ingest_detections(field_detections)
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        """Log a pipeline summary every ~90 messages (~3s at 30 Hz)."""
        if self._messages_seen % 90 != 0:
            return
        _log.info(
            "robot %d vision: messages=%d no_pose=%d degenerate_bbox=%d ball_detections=%d",
            self.player_id, self._messages_seen, self._messages_no_pose,
            self._degenerate_bbox, self._ball_detections,
        )


class VisionContextSource:
    """Build WorldSnapshots from real sensor topics rather than ground truth.

    Implements runtime's ContextSource protocol: ``start``, ``stop``, and
    ``get_snapshot``. Opponent dict keys are transient tracker-assigned IDs;
    see the module docstring.
    """

    def __init__(
        self, config: SoccerConfig, *,
        localiser_class: type[Localiser],
        ball_position_estimator: Callable[[Detection2D], tuple[float, float] | None],
    ) -> None:
        self._config = config
        self._localiser_class = localiser_class
        self._ball_position_estimator = ball_position_estimator
        self._lock = threading.RLock()

        self._ball_candidates: dict[int, FieldDetection] = {}
        self._opponent_tracks: dict[int, _OpponentTrack] = {}
        self._next_track_id = 1
        self._game: GameControlState | None = None

        self._robots: list[_RobotVision] = []
        self._teammate_ids: tuple[int, ...] = ()

        self._owns_context = False
        self._ros_context: Any = None
        self._node: Any = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._subscriptions: list[Any] = []
        self._started = False

    # ------------------------------------------------------------------
    # ContextSource protocol
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._node = self._create_node()
        self._create_robots()
        self._create_game_subscription()
        self._start_spin()
        from . import debugdraw
        from . import log_publisher
        debugdraw.install(self._node)
        log_publisher.install(self._node)
        _log.info(
            "VisionContextSource started: team_id=%d robots=%s",
            self._config.team_id, list(self._config.robot_names),
        )

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._stop_spin()
        self._destroy_node()

    def get_snapshot(self) -> WorldSnapshot:
        with self._lock:
            teammates = {
                pid: self._teammate_state(pid) for pid in self._teammate_ids
            }
            return WorldSnapshot(
                game=self._game,
                ball=self._fuse_ball(),
                teammates=teammates,
                opponents=self._prune_and_snapshot_tracks(),
            )

    # ------------------------------------------------------------------
    # Per-robot setup
    # ------------------------------------------------------------------

    def _create_robots(self) -> None:
        self._teammate_ids = self._config.player_ids
        # Team1's own frame needs no rotation relative to raw odom (confirmed
        # via live calibration); every other team's frame is team1's rotated
        # 180 degrees, per the fairness argument in OdomAnchoredLocaliser's
        # docstring. ODOM_FIELD_ANCHOR's constants are already team-relative,
        # so they're reused unchanged for every team.
        mirrored = self._config.team_id != 1
        for pid, name in enumerate(self._config.robot_names, start=1):
            anchor_x, anchor_y = ODOM_FIELD_ANCHOR.get(pid, (0.0, 0.0))
            localiser = self._localiser_class(
                self._node, self._flat_robot_topic(name, "odom"),
                anchor_x, anchor_y, mirrored,
            )
            topic = self._flat_robot_topic(name, "soccer/sim/vision/detections")
            self._robots.append(
                _RobotVision(pid, localiser, self._node, self, topic)
            )

    def _teammate_state(self, pid: int) -> RobotState:
        robot = next((r for r in self._robots if r.player_id == pid), None)
        if robot is None:
            return RobotState(player_id=pid)
        pose = robot.localiser.get_pose()
        if pose is None:
            return RobotState(player_id=pid)
        return RobotState(player_id=pid, pose=pose, last_seen_at=time.monotonic())

    def _create_game_subscription(self) -> None:
        self._subscriptions.append(
            self._node.create_subscription(
                RosString, self._config.game_controller_topic,
                self._game_cb, self._qos(depth=10),
            )
        )

    def _game_cb(self, msg: Any) -> None:
        try:
            game = game_control_state_from_json(str(msg.data))
        except ValueError as exc:
            _log.warning("ignore invalid GameController payload: %s", exc)
            return
        import dataclasses
        game = dataclasses.replace(game, last_seen_at=time.monotonic())
        with self._lock:
            self._game = game

    # ------------------------------------------------------------------
    # Detection ingestion: ball fusion and opponent tracking
    # ------------------------------------------------------------------

    def _ingest_detections(self, detections: list[FieldDetection]) -> None:
        with self._lock:
            for det in detections:
                if det.label == "ball":
                    self._ball_candidates[det.source_robot_id] = det
                elif det.label == "robot":
                    self._update_track(det)

    def _fuse_ball(self) -> BallState | None:
        """Pick the highest-confidence, then freshest, fresh ball candidate."""
        now = time.monotonic()
        fresh = [
            c for c in self._ball_candidates.values()
            if now - c.timestamp <= BALL_DETECTION_MAX_AGE_SEC
        ]
        if not fresh:
            return None
        best = max(fresh, key=lambda c: (c.confidence, c.timestamp))
        return BallState(x=best.x, y=best.y, last_seen_at=now, confidence=best.confidence)

    def _update_track(self, det: FieldDetection) -> None:
        """Greedy nearest-neighbor match against existing opponent tracks."""
        best_track: _OpponentTrack | None = None
        best_dist = OPPONENT_TRACK_MATCH_DIST_M
        for track in self._opponent_tracks.values():
            d = ((track.x - det.x) ** 2 + (track.y - det.y) ** 2) ** 0.5
            if d <= best_dist:
                best_dist = d
                best_track = track
        if best_track is None:
            track_id = self._next_track_id
            self._next_track_id += 1
            self._opponent_tracks[track_id] = _OpponentTrack(track_id, det.x, det.y, det.timestamp)
        else:
            best_track.x, best_track.y = det.x, det.y
            best_track.last_seen_at = det.timestamp
            best_track.dirty = True

    def _prune_and_snapshot_tracks(self) -> dict[int, RobotState]:
        """Age tracks by ~one runtime tick and drop long-missed ones.

        Called once per ``get_snapshot`` (one runtime tick), so ``missed``
        counts ticks without a matching detection, matching
        ``OPPONENT_TRACK_MAX_MISS_FRAMES``'s intent.
        """
        dead: list[int] = []
        result: dict[int, RobotState] = {}
        for track_id, track in self._opponent_tracks.items():
            if track.dirty:
                track.missed = 0
                track.dirty = False
            else:
                track.missed += 1
            if track.missed > OPPONENT_TRACK_MAX_MISS_FRAMES:
                dead.append(track_id)
                continue
            result[track_id] = RobotState(
                player_id=track_id,
                pose=Pose2D(x=track.x, y=track.y, theta=0.0),
                last_seen_at=track.last_seen_at,
            )
        for track_id in dead:
            del self._opponent_tracks[track_id]
        return result

    # ------------------------------------------------------------------
    # Topic names, shared with RosContextSource's convention
    # ------------------------------------------------------------------

    def _robot_topic(self, robot_name: str, suffix: str) -> str:
        if robot_name:
            return self._join(f"team{self._config.team_id}", robot_name, suffix)
        return self._join(f"team{self._config.team_id}", suffix)

    @staticmethod
    def _flat_robot_topic(robot_name: str, suffix: str) -> str:
        """Build a real-sensor topic name: flat, no team or soccer/sim prefix."""
        return VisionContextSource._join(robot_name, suffix)

    @staticmethod
    def _join(*parts: str) -> str:
        clean = [p.strip("/") for p in parts if p.strip("/")]
        return "/" + "/".join(clean)

    def _qos(self, depth: int) -> QoSProfile:
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=depth,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

    # ------------------------------------------------------------------
    # Node and executor lifecycle, mirrored from RosContextSource
    # ------------------------------------------------------------------

    def _create_node(self) -> Any:
        context = rclpy.get_default_context()
        self._ros_context = context
        if not rclpy.ok(context=context):
            context.init(args=None, initialize_logging=False)
            self._owns_context = True
        return rclpy.create_node("soccer_vision_bridge", context=context)

    def _start_spin(self) -> None:
        self._executor = SingleThreadedExecutor(context=self._ros_context)
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(
            target=self._spin, name="soccer_vision_source_spin", daemon=True,
        )
        self._spin_thread.start()

    def _spin(self) -> None:
        if self._executor is None:
            return
        try:
            self._executor.spin()
        except ExternalShutdownException:
            pass
        except Exception as exc:
            _log.warning("VisionContextSource spin failed: %s", exc)

    def _stop_spin(self) -> None:
        if self._executor is not None:
            try:
                self._executor.shutdown()
            except Exception as exc:
                _log.warning("executor shutdown failed: %s", exc)
        if self._spin_thread is not None and self._spin_thread.is_alive():
            self._spin_thread.join(timeout=2.0)
        self._spin_thread = None
        self._executor = None

    def _destroy_node(self) -> None:
        # Per-robot subscriptions (detections, and the localiser's own odom
        # subscription) are plain rclpy subscriptions on self._node; they are
        # cleaned up when the node itself is destroyed below, same as the
        # game-controller subscription in self._subscriptions.
        self._robots.clear()
        for sub in self._subscriptions:
            try:
                self._node.destroy_subscription(sub)
            except Exception:
                pass
        self._subscriptions.clear()
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception as exc:
                _log.warning("node destroy failed: %s", exc)
        if (
            self._owns_context
            and self._ros_context is not None
            and rclpy.ok(context=self._ros_context)
        ):
            self._ros_context.shutdown()
        self._node = None
