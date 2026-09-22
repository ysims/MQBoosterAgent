# Custom sim image

## Setup

This is the intended method for setting up the Docker image for students participating in the tournament.

First, pull the Docker image:

```bash
docker pull ysims/boostermq:0.6.5-beta
```

Second, tag the image so that Booster Studio finds and uses it:

```bash
docker tag ysims/boostermq:0.6.5-beta \
    booster-robotics-registry.cn-beijing.cr.aliyuncs.com/virtual-robot/virtual-robot:0.6.5-beta
```

Todo: Test whether Booster Studio's launch path ever forces a `docker pull` rather than a plain `docker run`. Test this once, tag the image as above, then try Run in Booster Studio and watch whether it succeeds or tries to hit the network. If it does force a pull, the fallback is running a local registry mirror or asking Booster directly whether the pull source is configurable.

## Building and Pushing to DockerHub

These instructions are intended for maintainers. Replace `<your-dockerhub-user>/mqagent-sim` below with your own repo if you're maintaining a separate build (e.g. `ysims/boostermq` is the published one students pull from above).

```bash
docker build -f docker/Dockerfile.sim -t <your-dockerhub-user>/mqagent-sim:0.6.5-beta .
docker push <your-dockerhub-user>/mqagent-sim:0.6.5-beta
```

If Booster Studio ever updates and starts requesting a different base image tag (check with `docker ps -a` for the `virtual-robot/virtual-robot` image it launches), rebuild with `--build-arg BASE_IMAGE=<the new tag>` and push under a new tag of your own.

Todo: see if this can be automated.

## Motivation

Booster Studio runs the 3v3 match simulator from a `virtual-robot/virtual-robot` image on `booster-robotics-registry.cn-beijing.cr.aliyuncs.com`.

1. The registry is blocked on some networks.
2. The stock image's 3v3 match scene does not include vision bounding box detections by default. We want the vision pipeline available for educational purposes, so the Docker image adds bounding box detections to the scene.

`Dockerfile.sim` builds on top of Booster's image, fixing the above issues. The scene file it needs to patch doesn't exist in the image itself, though -- Booster Studio copies it in after creating the container but before starting it -- so instead of patching at build time, the image overrides the container's startup command with a wrapper (`entrypoint_wrapper.sh`) that patches the scene right after Booster Studio's copy lands, then hands off to the original startup script.

