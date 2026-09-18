#!/bin/bash
# Runs as the virtual-robot container's CMD (after ros_entrypoint.sh sources
# the ROS environment). Booster Studio populates /usr/local/booster_robot/
# booster_robocup_sim by `docker cp`-ing it in between container create and
# container start, so by the time this script runs the scene files already
# exist -- see scripts/patch_match_scene_detection.py's docstring for why
# this can't be done at `docker build` time instead.
set -e

SCENE_FILE=/usr/local/booster_robot/booster_robocup_sim/mjcf/football_match_pitch_6_K1.extensions.xml

if [ -f "$SCENE_FILE" ]; then
    python3 /usr/local/bin/patch_match_scene_detection.py --file "$SCENE_FILE" \
        || echo "[entrypoint_wrapper] patch failed, continuing without it" >&2
else
    echo "[entrypoint_wrapper] scene file not found, skipping patch: $SCENE_FILE" >&2
fi

exec /usr/local/booster_robot/start_in_docker.sh
