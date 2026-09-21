"""Stream live per-tick state as UDP/JSON for an external visual debugger.

Thsi is separate from ``debugdraw`` and ``log_publisher`` (ROS log topic) --
this script sends one JSON packet per tick out of the container over UDP, 
so a plain Python script on the host  can render a live view. 

UDP does not require that anyone is listening, so this will always run in the 
background. By default, it targets the container Docker bridge gateway,
`172.17.0.1`, however if this is not the correct address for your set up,
use `ip route | grep default` inside the container to find the address.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time


_log = logging.getLogger(__name__)

_DEFAULT_HOST = "172.17.0.1"
_DEFAULT_PORT = 9999
_SEND_INTERVAL_SEC = 0.1  # ~10 Hz -- plenty for a live debug view

_sock: socket.socket | None = None
_addr: tuple[str, int] | None = None
_last_sent_at = 0.0


def install() -> None:
    """Open the UDP socket. Safe to call more than once."""
    global _sock, _addr
    if _sock is not None:
        return
    host = os.environ.get("SOCCER_DEBUG_STREAM_HOST", _DEFAULT_HOST)
    port = int(os.environ.get("SOCCER_DEBUG_STREAM_PORT", _DEFAULT_PORT))
    try:
        _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _addr = (host, port)
        _log.info("debug_stream installed, sending to %s:%d", host, port)
    except Exception as exc:
        _sock = None
        _log.warning("debug_stream install failed (viz disabled): %s", exc)


def send(payload: dict) -> None:
    """Send one JSON packet, throttled to _SEND_INTERVAL_SEC. No-op if not installed."""
    global _last_sent_at
    if _sock is None or _addr is None:
        return
    now = time.monotonic()
    if now - _last_sent_at < _SEND_INTERVAL_SEC:
        return
    _last_sent_at = now
    try:
        data = json.dumps(payload).encode("utf-8")
        _sock.sendto(data, _addr)
    except Exception:
        pass  # A debug side-channel must never affect the match.
