#!/bin/bash

docker run -it \
  --rm \
  --gpus all \
  --runtime=nvidia \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  training \
  $@
