# Custom sim image

Booster Studio runs the 3v3 match simulator from an image on
`booster-robotics-registry.cn-beijing.cr.aliyuncs.com` (currently
`agent-dev/agent-dev:0.7.21-alpha-humble`:

1. The registry is blocked on some university networks, so Booster Studio can never pull the image there.
2. The stock image's 3v3 match scene (`football_match_pitch_6_K1.extensions.xml`) omits `detection_extension`, so agents get no ball detections at all in a match (see `scripts/patch_match_scene_detection.py`'s docstring and `src/framework/vision_source.py`'s module docstring).

`Dockerfile.sim` builds on top of Booster's image with the scene patch baked in, avoiding manual patching inside a running container.

## One-time build (needs network access to the Beijing registry)

```bash
cd /path/to/MQAgent
docker build -f docker/Dockerfile.sim -t <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble .
docker push <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble
```

If Booster Studio ever updates and starts requesting a different base image tag (check with `docker ps -a` again), rebuild with `--build-arg BASE_IMAGE=<the new tag>` and push under a new tag of your own.
Todo: see if this can be automated.

## One-time setup

Given a docker image is available on 

```bash
docker pull <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble
docker tag <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble \
    booster-robotics-registry.cn-beijing.cr.aliyuncs.com/agent-dev/agent-dev:0.7.21-alpha-humble
```

The second command makes the local image tagged with the name Booster Studio expects. As long as that exact name:tag already exists locally, Booster Studio's own container launch should use it directly rather than pulling.

Todo: Test whether Booster Studio's launch path ever forces a `docker pull` (which would still try to reach Beijing and fail) rather than a plain `docker run` (uses a local image if the tag already exists, only pulls if missing). Test this once, tag the image as above, then try Run in Booster Studio and watch whether it succeeds or tries to hit the network. If it does force a pull, the fallback is running a local registry mirror or asking Booster directly whether the pull source is configurable.
