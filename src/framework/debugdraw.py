"""Debug drawing layer that publishes strategy visuals as ROS MarkerArray.

Strategy-side usage without ROS dependencies:
    from .framework import debugdraw
    debugdraw.point(x, y, rgb=(1,0,0), ns="target")
    debugdraw.arrow(x0, y0, x1, y1, rgb=(0,1,0), ns="heading")
    debugdraw.line([(x0,y0),(x1,y1)], ns="path")
    debugdraw.text(x, y, "chaser", ns="label")

The agent calls ``install(node)`` after the ROS node is ready. Runtime calls
``begin_frame()``, draws, and then calls ``flush()`` each frame. Without an
installation, all drawing calls are no-ops and this module is safe to import.

Coordinates use the same team-relative field frame as Context. Marker frame_id
defaults to "world"; set the Fixed Frame to that value in Booster Studio or
RViz. Ball and robot markers use the same frame, providing a consistent top-down
view even without external TF data.
"""

from __future__ import annotations

import logging

_log = logging.getLogger(__name__)

_FRAME = "world"          # Match this value in Studio's Fixed Frame setting.
_TOPIC = "/soccer/debug"
_Z = 0.05                 # Draw slightly above the ground plane.

_impl = None              # Injected by install(); None means no-op.


def install(node) -> None:
    """Inject the real Docker-only ROS publisher; otherwise remain a no-op."""
    global _impl
    try:
        _impl = _RosDrawSink(node)
        _log.info("debugdraw installed, publishing MarkerArray on %s", _TOPIC)
    except Exception as exc:
        _impl = None
        _log.warning("debugdraw install failed (viz disabled): %s", exc)


def begin_frame() -> None:
    if _impl is not None:
        _impl.begin()


def flush() -> None:
    if _impl is not None:
        _impl.flush()


def point(x, y, rgb=(1.0, 1.0, 1.0), scale=0.12, ns="point") -> None:
    if _impl is not None:
        _impl.point(x, y, rgb, scale, ns)


def cube(x, y, rgb=(1.0, 1.0, 1.0), scale=0.12, ns="cube") -> None:
    if _impl is not None:
        _impl.cube(x, y, rgb, scale, ns)


def arrow(x0, y0, x1, y1, rgb=(1.0, 1.0, 0.0), ns="arrow") -> None:
    if _impl is not None:
        _impl.arrow(x0, y0, x1, y1, rgb, ns)


def line(points, rgb=(0.5, 0.5, 0.5), ns="line") -> None:
    """Draw a polyline through ``points`` in ``[(x, y), ...]`` form."""
    if _impl is not None and len(points) >= 2:
        _impl.line(points, rgb, ns)


def text(x, y, s, rgb=(1.0, 1.0, 1.0), ns="text") -> None:
    if _impl is not None:
        _impl.text(x, y, s, rgb, ns)


class _RosDrawSink:
    """Accumulate frame markers and flush them after first deleting old ones."""

    def __init__(self, node) -> None:
        # Delay ROS imports so development environments remain unaffected.
        from visualization_msgs.msg import MarkerArray

        self._node = node
        self._pub = node.create_publisher(MarkerArray, _TOPIC, 1)
        self._markers: list = []
        self._next_id = 0

    def begin(self) -> None:
        self._markers = []
        self._next_id = 0

    def flush(self) -> None:
        from visualization_msgs.msg import Marker, MarkerArray

        arr = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        arr.markers.extend(self._markers)
        self._pub.publish(arr)

    # -- Primitives --

    def _new(self, ns, mtype):
        from visualization_msgs.msg import Marker

        m = Marker()
        m.header.frame_id = _FRAME
        m.header.stamp = self._node.get_clock().now().to_msg()
        m.ns = ns
        m.id = self._next_id
        self._next_id += 1
        m.type = mtype
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        return m

    @staticmethod
    def _rgba(m, rgb):
        m.color.r, m.color.g, m.color.b = float(rgb[0]), float(rgb[1]), float(rgb[2])
        m.color.a = 1.0

    def point(self, x, y, rgb, scale, ns) -> None:
        from visualization_msgs.msg import Marker

        m = self._new(ns, Marker.SPHERE)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), _Z
        m.scale.x = m.scale.y = m.scale.z = float(scale)
        self._rgba(m, rgb)
        self._markers.append(m)

    def cube(self, x, y, rgb, scale, ns) -> None:
        from visualization_msgs.msg import Marker

        m = self._new(ns, Marker.CUBE)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), _Z
        m.scale.x = m.scale.y = m.scale.z = float(scale)
        self._rgba(m, rgb)
        self._markers.append(m)

    def arrow(self, x0, y0, x1, y1, rgb, ns) -> None:
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        m = self._new(ns, Marker.ARROW)
        m.points = [
            Point(x=float(x0), y=float(y0), z=_Z),
            Point(x=float(x1), y=float(y1), z=_Z),
        ]
        m.scale.x = 0.03   # Shaft diameter
        m.scale.y = 0.08   # Arrow width
        m.scale.z = 0.12   # Arrowhead length
        self._rgba(m, rgb)
        self._markers.append(m)

    def line(self, points, rgb, ns) -> None:
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker

        m = self._new(ns, Marker.LINE_STRIP)
        m.points = [Point(x=float(px), y=float(py), z=_Z) for px, py in points]
        m.scale.x = 0.02   # Line width
        self._rgba(m, rgb)
        self._markers.append(m)

    def text(self, x, y, s, rgb, ns) -> None:
        from visualization_msgs.msg import Marker

        m = self._new(ns, Marker.TEXT_VIEW_FACING)
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.3
        m.scale.z = 0.25   # Text height
        self._rgba(m, rgb)
        m.text = str(s)
        self._markers.append(m)
