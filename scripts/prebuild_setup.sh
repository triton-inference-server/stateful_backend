#!/bin/bash
# DUE TO SONAME OF ORT Binary being different in ORT release vs Triton container
# Copy the necessary files from TRT container to Triton container and
# build the backend inside Triton container and link to ORT binary
# from /opt/tritonserver/backends/onnxruntime
ORT_VERSION=1.15.0
ORT_DIR=onnxruntime-linux-x64-gpu-${ORT_VERSION}

export DEBIAN_FRONTEND=noninteractive

# change the working directory
pushd /workspace

# Download and extract ORT package
ORT_DOWNLOAD_PATH=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_DIR}.tgz
wget ${ORT_DOWNLOAD_PATH}
tar xzf ${ORT_DIR}.tgz
mv ${ORT_DIR} onnxruntime
rm -rf onnxruntime/lib # we will use the libs from Triton
rm -f ${ORT_DIR}.tgz

apt-get update && \
  apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    rapidjson-dev

# we need cmake 3.17.5 or newer
CMAKE_VERSION=3.26.4
CMAKE_DIR=cmake-${CMAKE_VERSION}-linux-x86_64
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_DIR}.tar.gz
tar xzf ${CMAKE_DIR}.tar.gz
rm -f ${CMAKE_DIR}.tar.gz
cmake_bin_path=/usr/bin/cmake #$(which cmake)
# rm -f ${cmake_bin_path}
ln -s ${PWD}/${CMAKE_DIR}/bin/cmake ${cmake_bin_path}