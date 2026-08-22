"""Bridge Python logging to a ROS topic for rosbag capture and replay.

On the framework side, ros_source calls ``install(node)`` once the node is
ready, attaching the handler to the root logger. Without ROS, the handler is
not installed and logs continue through the platform logger.

Topic: /soccer/agent_log. Type: rcl_interfaces/msg/Log, the standard ROS log
message supported natively by Studio.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_TOPIC = "/soccer/agent_log"


def install(node) -> None:
    """Attach the ROS log publisher to the Python root logger (Docker only)."""
    try:
        handler = _RosLogHandler(node)
        # Capture all modules and inherit the root logger's level (INFO by default).
        logging.getLogger().addHandler(handler)
        _log.info("log_publisher installed, publishing to %s", _TOPIC)
    except Exception as exc:
        _log.warning("log_publisher install failed (ROS log disabled): %s", exc)


# Map Python logging levels to ROS Log constants.
_LEVEL_MAP = {
    logging.DEBUG: 10,     # Log.DEBUG
    logging.INFO: 20,      # Log.INFO
    logging.WARNING: 30,   # Log.WARN
    logging.ERROR: 40,     # Log.ERROR
    logging.CRITICAL: 50,  # Log.FATAL
}


class _RosLogHandler(logging.Handler):
    """Convert LogRecord instances to rcl_interfaces/msg/Log on /rosout."""

    def __init__(self, node) -> None:
        super().__init__()
        from rcl_interfaces.msg import Log
        from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

        # Standard /rosout QoS: TRANSIENT_LOCAL, RELIABLE, history depth 1000.
        qos = QoSProfile(
            depth=1000,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )
        self._node = node
        self._pub = node.create_publisher(Log, _TOPIC, qos)

    def emit(self, record: logging.LogRecord) -> None:
        """Publish a ROS Log message to /rosout, suppressing recursive errors."""
        try:
            from rcl_interfaces.msg import Log

            msg = Log()
            msg.stamp = self._node.get_clock().now().to_msg()
            msg.level = _LEVEL_MAP.get(record.levelno, 20)  # Default to INFO.
            msg.name = record.name
            msg.msg = record.getMessage()
            msg.file = record.pathname
            msg.function = record.funcName
            msg.line = record.lineno
            self._pub.publish(msg)
        except Exception:
            pass  # Logging here would recurse, so suppress the exception.
