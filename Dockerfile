# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/tensorflow:19.10-py3

MAINTAINER Jack Han <jahan@nvidia.com>
# RUN sed -i "s/archive.ubuntu.com/mirror.kakao.com/g" /etc/apt/sources.list && \
#     sed -i "s/security.ubuntu.com/mirror.kakao.com/g" /etc/apt/sources.list

# Install requried libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    zlib1g-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Cmake
RUN apt-get remove --purge -y cmake cmake-data && \
    cd /tmp &&\
    wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4-Linux-x86_64.sh &&\
    chmod +x cmake-3.14.4-Linux-x86_64.sh &&\
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license &&\
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Download TensorRT OSS
RUN git clone -b release/6.0 https://github.com/NVIDIA/TensorRT /opt/tensorrt-oss && \
    cd /opt/tensorrt-oss && \
    git submodule update --init --recursive

# Build TensorRT OSS components
RUN mkdir -p /opt/tensorrt-oss/build && \
    cd /opt/tensorrt-oss/build && \
    cmake .. -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu -DTRT_BIN_DIR=`pwd`/out &&\
    make -j$(nproc)

### Install packages for running webcam

# Install OpenCV
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 libxext6 libxrender-dev python3-tk \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install opencv-python

# Install other dependencies
RUN pip3 install \
    pillow \
    requests \
    'pycuda>=2019.1.1' \
    ptvsd

# Export environment variable for webcam functionality
ENV QT_X11_NO_MITSHM=1

# Set environment and working directory
WORKDIR /workspace

RUN ["/bin/bash"]