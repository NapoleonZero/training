#!/bin/bash
# --shm-size 8G \

docker run -it \
  --name=potatorch \
  --rm \
  --gpus all \
  --runtime=nvidia \
  --privileged \
  --ipc=host \
  -p 8888:8888 \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  potatorch \
  $@
