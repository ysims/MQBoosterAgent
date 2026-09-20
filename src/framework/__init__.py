"""Framework layer: platform plumbing that users rarely modify.

Includes the data contract (types), configuration (config), runtime loop
(runtime), SDK connection (robot_backend), GameController decoder
(game_codec), agent entry-point mixin (agent), and the vision + localisation
ContextSource (vision_source), which wires together the vision, odometry,
localisation, and utils packages.

Import directly from submodules as needed, for example
``from .framework.types import Context``. This module avoids eager imports so
SDK-dependent modules do not break imports on development machines without it.
"""
