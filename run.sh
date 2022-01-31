#!/bin/bash
# --shm-size 8G \

docker run -it \
  --rm \
  --gpus all \
  --runtime=nvidia \
  --ipc=host \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  training \
  $@
