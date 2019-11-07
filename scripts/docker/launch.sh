#!/bin/bash

set -x
docker run --rm -ti --name jahan_ssd --runtime=nvidia \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --net=host \
    -u $(id -u):$(id -g) \
    -v $(pwd):/workspace \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -v /datasets/coco17/raw-data:/coco \
    -v /datasets/VOCdevkit:/VOCdevkit \
    nvcr.io/nvidian/sae/jahan:ssd-cuda10.0-trt6
set +x