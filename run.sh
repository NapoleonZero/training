#!/usr/bin/env bash
# --shm-size 8G \

docker run -it \
  --rm \
  --device=nvidia.com/gpu=all \
  --privileged \
  --ipc=host \
  -p 8888:8888 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  training \
  $@
