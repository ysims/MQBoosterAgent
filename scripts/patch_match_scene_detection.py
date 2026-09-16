#!/usr/bin/env python3
"""Patch the 3v3 match scene to publish ball/landmark detections.

Booster Studio's ``football_match_pitch_6_K1.extensions.xml`` -- the scene
the match-runner actually loads for a 3v3 match -- omits ``detection_
extension``, which the non-match ``football_pitch_6_K1.extensions.xml``
declares. ``detection_extension`` publishes ball/goalpost/field-marker
detections as ``vision_msgs/Detection2DArray`` on
``/{robot_name}/soccer/sim/vision/detections``, computed by projecting true
object positions through each camera's real field-of-view and pose
(respecting occlusion). This is what ``VisionContextSource`` consumes for
ball detection (see ``framework/vision_source.py``'s module docstring).

This script adds ``detection_extension`` to the match scene, in place,
idempotently. It does not touch ``football_match_digua_pitch_6_K1.extensions
.xml`` (a separate scene variant; purpose unconfirmed, deliberately left
alone).

``MODEL_PATH`` is validated against a hardcoded allowlist of bundled scene
paths in Booster Studio's own ``core/config.py`` (``resolve_container_
startup()``), so there is no way to point the match runtime at an external
scene file instead -- patching the bundled file in place is the only lever.

There are TWO separate copies of this scene file to patch, since the
standalone single-robot sim (runs on the host) and the 3v3 match sim (runs
inside the Docker container the match-runner drives) each have their own:
- host: ``<studio install>/resources/app/booster-native/statics/
  virtual-robot/robocup_sim_src/mjcf/football_match_pitch_6_K1.extensions.xml``
  (--studio-root, default /usr/share/booster-studio)
- container (run this INSIDE the container, e.g. via the virtual robot
  panel's terminal): ``/usr/local/booster_robot/booster_robocup_sim/mjcf/
  football_match_pitch_6_K1.extensions.xml`` (--file)

Run once per install, and again after any Booster Studio update or
container image rebuild, since either may overwrite the scene file and
silently undo the patch:

    python3 scripts/patch_match_scene_detection.py                  # host copy
    python3 scripts/patch_match_scene_detection.py --file /usr/local/booster_robot/booster_robocup_sim/mjcf/football_match_pitch_6_K1.extensions.xml   # container copy
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_STUDIO_ROOT = Path("/usr/share/booster-studio")
SCENE_RELATIVE_PATH = Path(
    "resources/app/booster-native/statics/virtual-robot/robocup_sim_src/mjcf/"
    "football_match_pitch_6_K1.extensions.xml"
)

DETECTION_EXTENSION_BLOCK = (
    "\n"
    "  <!-- Ball/landmark detection, added by patch_match_scene_detection.py:\n"
    "       the stock match scene omits this; see the script docstring. -->\n"
    "  <extension_process\n"
    "     name=\"detection_extension\"\n"
    "     field_type=\"adult_size\"/>\n"
)

MARKER = "</extensions>"


def find_scene_file(studio_root: Path) -> Path:
    path = studio_root / SCENE_RELATIVE_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Scene file not found at {path}. Pass --studio-root if Booster "
            f"Studio is installed somewhere other than {DEFAULT_STUDIO_ROOT}."
        )
    return path


def patch(scene_path: Path) -> bool:
    """Insert the detection extension block. Return False if already patched."""
    text = scene_path.read_text(encoding="utf-8")
    if "detection_extension" in text:
        return False
    if MARKER not in text:
        raise ValueError(f"Expected closing '{MARKER}' tag not found in {scene_path}")

    backup_path = scene_path.with_suffix(scene_path.suffix + ".orig")
    if not backup_path.exists():
        shutil.copy2(scene_path, backup_path)

    patched = text.replace(MARKER, DETECTION_EXTENSION_BLOCK + MARKER, 1)
    scene_path.write_text(patched, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--studio-root", type=Path, default=DEFAULT_STUDIO_ROOT,
        help=(
            "Booster Studio install root, for the HOST copy of the scene "
            f"file (default: {DEFAULT_STUDIO_ROOT}). Ignored if --file is given."
        ),
    )
    parser.add_argument(
        "--file", type=Path, default=None,
        help=(
            "Patch this exact extensions.xml path instead of resolving it "
            "under --studio-root -- use this for the CONTAINER copy, e.g. "
            "/usr/local/booster_robot/booster_robocup_sim/mjcf/"
            "football_match_pitch_6_K1.extensions.xml"
        ),
    )
    args = parser.parse_args()

    if args.file is not None:
        scene_path = args.file
        if not scene_path.is_file():
            print(f"ERROR: scene file not found: {scene_path}", file=sys.stderr)
            return 1
    else:
        try:
            scene_path = find_scene_file(args.studio_root)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    if patch(scene_path):
        print(f"Patched: {scene_path}")
        print(f"Backup saved: {scene_path}.orig")
        print(
            "Restart the simulator (click Run again in Booster Studio) "
            "for the change to take effect."
        )
    else:
        print(f"Already patched: {scene_path} (detection_extension present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
