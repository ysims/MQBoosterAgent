"""Entry-point shim: ``agent.toml`` points at ``src/main.py:SoccerSimAgent``.

The actual agent lives in ``strategy/main.py`` alongside the rest of the
match strategy; this file only re-exports it so the manifest's entry path
never needs to change.

Booster's build validator (``pyagent_base.py``) statically AST-parses this
file for a top-level ``class SoccerSimAgent`` that inherits
``booster_agent_framework.AgentBase`` -- it does not follow imports. A plain
re-export doesn't satisfy that, so this file must define an actual (if
trivial) subclass rather than just aliasing the import.
"""

from __future__ import annotations

from booster_agent_framework import AgentBase

from .strategy.main import SoccerSimAgent as _SoccerSimAgentImpl


class SoccerSimAgent(_SoccerSimAgentImpl, AgentBase):
    pass


__all__ = ["SoccerSimAgent"]
