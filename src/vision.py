"""Perception implementations, designed for direct user editing.

This is the file to edit when improving perception: swap ``ColorLutDetector``
for a learned detector (e.g. YOLO), or swap ``OdomAnchoredLocaliser`` for a
real state estimator (EKF/particle filter fusing IMU/odometry/vision
field-lines) that corrects drift instead of dead-reckoning forever. Both classes
satisfy the ``Detector``/``Localiser`` protocols in
``framework/vision_types.py`` and are wired in as the defaults by
``main.py``'s ``detector_class``/``localiser_class`` hooks -- swap the class
there and nothing else in the framework needs to change.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from .framework.types import Pose2D
from .framework.vision_types import CameraIntrinsics, Detection2D
from .param import (
    BALL_HSV_LOWER,
    BALL_HSV_UPPER,
    DETECTION_MIN_CONFIDENCE,
    LOCALISER_STALE_SEC,
    MIN_BALL_BLOB_AREA_PX,
    MIN_ROBOT_BLOB_AREA_PX,
    ROBOT_HSV_LOWER,
    ROBOT_HSV_UPPER,
)


__all__ = ["ColorLutDetector", "OdomAnchoredLocaliser"]


_log = logging.getLogger(__name__)


class ColorLutDetector:
    """HSV color-threshold ball + opponent-jersey detector.

    Working basic default: threshold the ball's orange and the opponents'
    jersey color, then treat each large-enough connected blob as one
    detection. Good enough to bootstrap the framework end to end; a real
    project should upgrade this to a learned detector (e.g. YOLO) without
    changing anything outside this class, since ``VisionContextSource`` only
    depends on the ``Detector`` protocol (``detect(rgb, depth, intrinsics)``).

    ``cv2`` is imported lazily inside ``detect`` so importing this module (and
    the rest of the framework) does not require OpenCV to be installed. If
    ``cv2`` is unavailable, ``_detect_numpy`` provides a much cruder
    numpy-only ball-only fallback so the framework still produces *some*
    signal rather than silently detecting nothing.
    """

    def detect(
        self, rgb: Any, depth: Any, intrinsics: CameraIntrinsics,
    ) -> list[Detection2D]:
        if rgb is None:
            return []
        try:
            import cv2
        except ImportError:
            return self._detect_numpy(rgb)
        return self._detect_cv2(rgb, cv2)

    def _detect_cv2(self, rgb: Any, cv2: Any) -> list[Detection2D]:
        import numpy as np

        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        now = time.monotonic()
        detections: list[Detection2D] = []
        detections += self._blobs_for_color(
            hsv, cv2, np, BALL_HSV_LOWER, BALL_HSV_UPPER, "ball",
            MIN_BALL_BLOB_AREA_PX, now,
        )
        detections += self._blobs_for_color(
            hsv, cv2, np, ROBOT_HSV_LOWER, ROBOT_HSV_UPPER, "robot",
            MIN_ROBOT_BLOB_AREA_PX, now,
        )
        return detections

    def _blobs_for_color(
        self, hsv: Any, cv2: Any, np: Any,
        lower: tuple[int, int, int], upper: tuple[int, int, int],
        label: str, min_area: float, now: float,
    ) -> list[Detection2D]:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        out: list[Detection2D] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            # Cheap confidence proxy: blobs well above the minimum area score
            # higher, capped at 1.0. A learned detector should replace this
            # with an actual model score.
            confidence = min(1.0, area / (min_area * 4.0))
            if confidence < DETECTION_MIN_CONFIDENCE:
                continue
            out.append(Detection2D(
                x_px=x + w / 2.0, y_px=y + h / 2.0, w_px=float(w), h_px=float(h),
                label=label, confidence=confidence, camera_id="", timestamp=now,
            ))
        return out

    def _detect_numpy(self, rgb: Any) -> list[Detection2D]:
        """Ball-only fallback bounding-box detector without OpenCV.

        Thresholds directly on RGB channel ratios instead of HSV (no
        colorspace conversion available), and returns a single detection
        covering the bounding box of all matching pixels instead of proper
        connected components. This is deliberately cruder than
        ``_detect_cv2``; it exists so the framework keeps working if a Docker
        image ever ships without ``cv2``, not as a design to imitate.
        """
        import numpy as np

        r = rgb[:, :, 0].astype(np.int16)
        g = rgb[:, :, 1].astype(np.int16)
        b = rgb[:, :, 2].astype(np.int16)
        # Orange: red channel dominant over both green and blue.
        mask = (r > 140) & (r - b > 60) & (r - g > 30)
        ys, xs = np.nonzero(mask)
        if xs.size < MIN_BALL_BLOB_AREA_PX:
            return []
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return [Detection2D(
            x_px=(x0 + x1) / 2.0, y_px=(y0 + y1) / 2.0,
            w_px=float(x1 - x0 + 1), h_px=float(y1 - y0 + 1),
            label="ball", confidence=DETECTION_MIN_CONFIDENCE,
            camera_id="", timestamp=time.monotonic(),
        )]


class OdomAnchoredLocaliser:
    """Dead-reckon field-frame self-pose from raw wheel/gait odometry.

    Working basic default: there is no free "sim gives you a good pose"
    topic -- ``/robot{N}/odom`` (``nav_msgs/Odometry``) is the real signal a
    robot has, and it is relative: it boots at an arbitrary origin, not the
    field frame, and drifts. This class calibrates that origin against a
    fixed, pre-measured field-frame anchor (see ``ODOM_FIELD_ANCHOR`` in
    ``param.py``, captured once by comparing ``/robot{N}/odom`` against the
    sim's ground-truth topic at INITIAL-state spawn -- never at runtime) and
    reports ``odom + anchor`` thereafter.

    This calibration happens to reduce to a pure translation for this sim:
    ``/robot{N}/odom`` was observed to boot at position (0, 0) with its yaw
    already equal to field-frame theta (no rotation offset), so
    ``field_theta`` is just ``odom_yaw`` directly, and ``field_x/y`` is
    ``odom_x/y + anchor``. Odom drifts significantly even while standing
    still (bipedal balance sway), so this is a crude dead-reckoning
    estimate that degrades over a match -- the real upgrade path is fusing
    ``/imu/data`` and vision-derived field-line detections into an actual
    filter (EKF/particle filter) that periodically corrects the drift
    instead of trusting odom forever.
    """

    def __init__(self, node: Any, topic: str, anchor_x: float, anchor_y: float) -> None:
        self._anchor_x = anchor_x
        self._anchor_y = anchor_y
        self._lock = threading.Lock()
        self._pose: Pose2D | None = None
        self._last_msg_at: float | None = None

        from nav_msgs.msg import Odometry
        from rclpy.qos import (
            QoSDurabilityPolicy,
            QoSHistoryPolicy,
            QoSProfile,
            QoSReliabilityPolicy,
        )

        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.VOLATILE,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._sub = node.create_subscription(Odometry, topic, self._on_odom, qos)

    def _on_odom(self, msg: Any) -> None:
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = 2.0 * math.atan2(orientation.z, orientation.w)
        pose = Pose2D(
            x=float(position.x) + self._anchor_x,
            y=float(position.y) + self._anchor_y,
            theta=yaw,
        )
        with self._lock:
            self._pose = pose
            self._last_msg_at = time.monotonic()

    def get_pose(self) -> Pose2D | None:
        with self._lock:
            if self._pose is None or self._last_msg_at is None:
                return None
            if time.monotonic() - self._last_msg_at > LOCALISER_STALE_SEC:
                return None
            return self._pose
