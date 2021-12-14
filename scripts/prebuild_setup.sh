#!/bin/bash
ORT_VERSION=1.9.0
ORT_DIR=onnxruntime-linux-x64-gpu-${ORT_VERSION}
CMAKE_UBUNTU_VERSION=20.04

export DEBIAN_FRONTEND=noninteractive

# change the working directory
pushd /workspace

# Download and extract ORT package
ORT_DOWNLOAD_PATH=https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_DIR}.tgz
wget ${ORT_DOWNLOAD_PATH}
tar xzf ${ORT_DIR}.tgz
mv ${ORT_DIR} onnxruntime
rm -f ${ORT_DIR}.tgz

apt-get update && \
  apt-get install -y --no-install-recommends \
    ca-certificates \
    software-properties-common \
    rapidjson-dev

# we need cmake 3.17.5 or newer
CMAKE_VERSION=3.22.0
CMAKE_DIR=cmake-${CMAKE_VERSION}-linux-x86_64
wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${CMAKE_DIR}.tar.gz
tar xzf ${CMAKE_DIR}.tar.gz
rm -f ${CMAKE_DIR}.tar.gz
cmake_bin_path=$(which cmake)
rm -f ${cmake_bin_path}
ln -s ${PWD}/${CMAKE_DIR}/bin/cmake ${cmake_bin_path}