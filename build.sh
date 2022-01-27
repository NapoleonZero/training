#!/bin/bash

DOCKER_BUILDKIT=1 docker build --rm \
  -t training \
  -f Dockerfile \
  --build-arg WANDB_SECRET=$(cat ~/.wandb_secret) \
  .
