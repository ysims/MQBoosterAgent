# Custom sim image

## Setup

This is the intended method for setting up the Docker image for students participating in the tournament.

First, pull the Docker image:

```bash
docker pull <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble
```

Second, tag the image so that Booster Studio finds and uses it:

```bash
docker tag <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble \
    booster-robotics-registry.cn-beijing.cr.aliyuncs.com/agent-dev/agent-dev:0.7.21-alpha-humble
```

Todo: Test whether Booster Studio's launch path ever forces a `docker pull` rather than a plain `docker run`. Test this once, tag the image as above, then try Run in Booster Studio and watch whether it succeeds or tries to hit the network. If it does force a pull, the fallback is running a local registry mirror or asking Booster directly whether the pull source is configurable.

## Building and Pushing to DockerHub

These instructions are intended for maintainers. 

```bash
docker build -f docker/Dockerfile.sim -t <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble .
docker push <your-dockerhub-user>/mqagent-sim:0.7.21-alpha-humble
```

If Booster Studio ever updates and starts requesting a different base image tag (check with `docker ps -a` again), rebuild with `--build-arg BASE_IMAGE=<the new tag>` and push under a new tag of your own.

Todo: see if this can be automated.

## Motivation

Booster Studio runs the 3v3 match simulator from an image on `booster-robotics-registry.cn-beijing.cr.aliyuncs.com`.

1. The registry is blocked on some networks.
2. The stock image's 3v3 match scene does not include vision bounding box detections, instead expecting robots to using ground truth data. We want to include the vision pipeline in the code for educational purposes, and so the Docker image adds bounding box detections to the scene.

`Dockerfile.sim` builds on top of Booster's image, fixing the above issues.

