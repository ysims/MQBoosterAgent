"""Vision + localisation source: a ContextSource built on real sensor topics.

This Docker-only platform layer depends on rclpy, sensor_msgs, nav_msgs, and
std_msgs and is imported only in a ROS environment, mirroring
``ros_source.py``. Unlike ``RosContextSource``, it never reads
``.../sim/ground_truth/...`` topics: teammate self-pose comes from a plugged
``Localiser`` per robot (default: :class:`src.vision.OdomAnchoredLocaliser`
dead-reckoning ``/robot{name}/odom`` against a pre-measured field anchor),
and ball/opponent state is derived by running a plugged ``Detector``
(default: :class:`src.vision.ColorLutDetector`) on each robot's RGB-D camera
and projecting pixel detections into the field frame.

Per-robot topic naming for real sensors is flat -- just the robot's own
name, with no ``/team{id}/`` or ``/soccer/sim/`` prefix (confirmed live via
``ros2 topic list``; this differs from ``RosContextSource``'s ground-truth
topics, which *are* team-prefixed):
- RGB: ``/{robot_name}/rgbd_camera/rgb/image_compressed``
- depth: ``/{robot_name}/rgbd_camera/depth/image_raw``
- RGB camera info: ``/{robot_name}/rgbd_camera/rgb/camera_info``
- odometry: ``/{robot_name}/odom`` (``nav_msgs/Odometry``)

Two semantic differences from ground-truth mode, both intentional:
1. Ball state is fused across whatever robots currently see it, rather than
   read from one authoritative topic.
2. Opponent dict keys are transient tracker-assigned IDs, not stable
   opponent identities -- vision cannot tell opponents apart, only track
   "an opponent was here." Track IDs may change across a match; strategy
   code should not assume ``opponents[k]`` refers to the same physical robot
   over time.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import String as RosString

from .config import SoccerConfig
from .game_codec import game_control_state_from_json
from .types import BallState, GameControlState, Pose2D, RobotState, WorldSnapshot
from .vision_types import (
    CameraExtrinsics,
    CameraIntrinsics,
    Detector,
    FieldDetection,
    Localiser,
    project_to_field,
)
from ..param import (
    BALL_DETECTION_MAX_AGE_SEC,
    CAMERA_EXTRINSICS_X,
    CAMERA_EXTRINSICS_Y,
    CAMERA_EXTRINSICS_Z,
    CAMERA_EXTRINSICS_YAW,
    ODOM_FIELD_ANCHOR,
    OPPONENT_TRACK_MATCH_DIST_M,
    OPPONENT_TRACK_MAX_MISS_FRAMES,
)


__all__ = ["VisionContextSource"]


_log = logging.getLogger(__name__)

_DEFAULT_EXTRINSICS = CameraExtrinsics(
    x=CAMERA_EXTRINSICS_X, y=CAMERA_EXTRINSICS_Y,
    z=CAMERA_EXTRINSICS_Z, yaw=CAMERA_EXTRINSICS_YAW,
)


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
    """Own one robot's camera subscriptions, detector, and localiser."""

    def __init__(
        self, node: Any, source: "VisionContextSource",
        player_id: int, robot_name: str, detector: Detector, localiser: Localiser,
    ) -> None:
        self.player_id = player_id
        self.detector = detector
        self.localiser = localiser
        self._source = source
        self._lock = threading.Lock()
        self._depth: Any = None
        self._intrinsics: CameraIntrinsics | None = None

        img_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
        )
        topic = source._flat_robot_topic
        self.subscriptions = [
            node.create_subscription(
                CompressedImage, topic(robot_name, "rgbd_camera/rgb/image_compressed"),
                self._on_rgb, img_qos,
            ),
            node.create_subscription(
                Image, topic(robot_name, "rgbd_camera/depth/image_raw"),
                self._on_depth, img_qos,
            ),
            node.create_subscription(
                CameraInfo, topic(robot_name, "rgbd_camera/rgb/camera_info"),
                self._on_camera_info, source._qos(depth=1),
            ),
        ]

    def _on_camera_info(self, msg: Any) -> None:
        k = msg.k
        intrinsics = CameraIntrinsics(fx=float(k[0]), fy=float(k[4]), cx=float(k[2]), cy=float(k[5]))
        with self._lock:
            self._intrinsics = intrinsics

    def _on_depth(self, msg: Any) -> None:
        import numpy as np

        try:
            row_floats = msg.step // 4
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, row_floats))
            depth = arr[:, : msg.width]
        except Exception as exc:
            _log.warning("robot %d depth decode failed: %s", self.player_id, exc)
            return
        with self._lock:
            self._depth = depth

    def _on_rgb(self, msg: Any) -> None:
        import numpy as np
        import cv2

        try:
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                return
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            _log.warning("robot %d rgb decode failed: %s", self.player_id, exc)
            return

        with self._lock:
            depth = self._depth
            intrinsics = self._intrinsics
        if depth is None or intrinsics is None:
            return

        pose = self.localiser.get_pose()
        if pose is None:
            return

        try:
            detections = self.detector.detect(rgb, depth, intrinsics)
        except Exception as exc:
            _log.warning("robot %d detector failed: %s", self.player_id, exc)
            return

        field_detections = []
        for det in detections:
            depth_m = self._sample_depth(depth, det.x_px, det.y_px)
            if depth_m is None:
                continue
            field_detections.append(project_to_field(
                det, depth_m, intrinsics, _DEFAULT_EXTRINSICS, pose, self.player_id,
            ))
        if field_detections:
            self._source._ingest_detections(field_detections)

    @staticmethod
    def _sample_depth(depth: Any, x_px: float, y_px: float) -> float | None:
        import math

        h, w = depth.shape[:2]
        x, y = int(x_px), int(y_px)
        if x < 0 or y < 0 or x >= w or y >= h:
            return None
        value = float(depth[y, x])
        if not math.isfinite(value) or value <= 0.0:
            return None
        return value


class VisionContextSource:
    """Build WorldSnapshots from real sensor topics rather than ground truth.

    Implements runtime's ContextSource protocol: ``start``, ``stop``, and
    ``get_snapshot``. Opponent dict keys are transient tracker-assigned IDs;
    see the module docstring.
    """

    def __init__(
        self, config: SoccerConfig, *,
        detector_class: type[Detector], localiser_class: type[Localiser],
    ) -> None:
        self._config = config
        self._detector_class = detector_class
        self._localiser_class = localiser_class
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
        for pid, name in enumerate(self._config.robot_names, start=1):
            anchor_x, anchor_y = ODOM_FIELD_ANCHOR.get(pid, (0.0, 0.0))
            localiser = self._localiser_class(
                self._node, self._flat_robot_topic(name, "odom"), anchor_x, anchor_y,
            )
            detector = self._detector_class()
            self._robots.append(_RobotVision(self._node, self, pid, name, detector, localiser))

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
        for robot in self._robots:
            for sub in robot.subscriptions:
                try:
                    self._node.destroy_subscription(sub)
                except Exception:
                    pass
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
