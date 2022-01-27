#!/bin/bash

docker run -it --rm --gpus all --runtime=nvidia training $@
