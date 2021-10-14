# The MIT License (MIT)
# 
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

cmake_minimum_required(VERSION 3.17)

project(tritononnxstatefulbackend LANGUAGES C CXX CUDA)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

file (STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/NGC_VERSION" NGC_VERSION)
message("CMAKE Triton repo tag is set to: ${NGC_VERSION}")
set(TRITON_COMMON_REPO_TAG "r${NGC_VERSION}" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r${NGC_VERSION}" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r${NGC_VERSION}" CACHE STRING "Tag for triton-inference-server/backend repo")

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_stateful.ldscript libtriton_stateful.ldscript COPYONLY)

# Move the ORT Libraries to the build directory
file(COPY /workspace/onnxruntime/build/Linux/Release/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/ort-lib)

set(TRT_SAMPLE_PATH /workspace/tensorrt/samples)

add_library(
  triton-stateful-backend SHARED
  src/stateful.cc
  src/onnx_model_runner.cc
  src/cuda_kernels.cu
  ${TRT_SAMPLE_PATH}/common/logger.cpp
)

add_library(
    TritonOnnxStatefulBackend::triton-stateful-backend ALIAS triton-stateful-backend
)

target_include_directories(
  triton-stateful-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRT_SAMPLE_PATH}
    ${TRT_SAMPLE_PATH}/common
    /workspace/onnxruntime/include/onnxruntime/core/session
    /workspace/onnxruntime/include/onnxruntime/core/providers/tensorrt
    /workspace/onnxruntime/include/onnxruntime/core/providers/cuda
)

target_compile_features(triton-stateful-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-stateful-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits>
)

target_link_libraries(
  triton-stateful-backend
  PRIVATE
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
    onnxruntime
    onnxruntime_providers_tensorrt 
    onnxruntime_providers_shared
)

target_link_directories(
  triton-stateful-backend
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/ort-lib
)

set_target_properties(
  triton-stateful-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME triton_stateful
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_stateful.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_stateful.ldscript"
)

set_property(TARGET triton-stateful-backend PROPERTY CUDA_ARCHITECTURES 70 75 80)

#
# Install
#
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)
install(
  TARGETS
    triton-stateful-backend
  LIBRARY 
    DESTINATION backends/stateful
)

export(PACKAGE TritonOnnxStatefulBackend)
