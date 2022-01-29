#!/bin/bash

docker run -it \
  --rm \
  --gpus all \
  --runtime=nvidia \
  --shm-size 8G \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  training \
  $@
