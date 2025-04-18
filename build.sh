#!/usr/bin/env bash

ARCH=gpu

options=$(getopt --longoptions cpu,gpu -n 'run' --options '' -- "$@")
[ $? -eq 0 ] || { 
  echo "Usage: build [--cpu | --gpu]"
  exit 1
}
eval set -- "$options"
while true; do
  case "$1" in
    --cpu ) ARCH=cpu; shift ;;
    --gpu ) ARCH=gpu; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

DOCKER_BUILDKIT=1 docker build --rm \
  -t training \
  -f Dockerfile.${ARCH} \
  --build-arg WANDB_SECRET=$(cat ~/.wandb_secret) \
  .
