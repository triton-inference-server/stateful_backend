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

import os
import sys

TRITON_REPO_VERSION = ""
root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)
version_file = os.path.join(root_dir, "NGC_VERSION")
with open(version_file) as fp:
    TRITON_REPO_VERSION = fp.read().splitlines()[0].strip()
    print("Triton repo tag is set to: ", TRITON_REPO_VERSION)
if TRITON_REPO_VERSION == "":
    print("Invalid tag for Triton repos.")
    exit(1)

STATEFUL_BACKEND_VOL_DEST = "/workspace/stateful"

TENSORRT_CONTAINER_NAME = "test_stateful_tensorrt_for_build"
TENSORRT_REPO = "nvcr.io/nvidia/tensorrt"
TENSORRT_TAG = "{0}-py3".format(TRITON_REPO_VERSION)
TENSORRT_IMAGE = TENSORRT_REPO + ":" + TENSORRT_TAG
# The following command is not run inside TRT container anymore (it is run in Triton container now).
# See script for details.
# TENSORRT_CONTAINER_PREBUILD_CMD = f"bash {STATEFUL_BACKEND_VOL_DEST}/scripts/prebuild_setup.sh"

TRITON_CONTAINER_NAME = "test_stateful_triton_for_build"
TRITON_REPO = "nvcr.io/nvidia/tritonserver"
TRITON_TAG = "{0}-py3".format(TRITON_REPO_VERSION)
TRITON_IMAGE = TRITON_REPO + ":" + TRITON_TAG
TRITON_CONTAINER_PREBUILD_CMD = f"bash {STATEFUL_BACKEND_VOL_DEST}/scripts/prebuild_setup.sh"

STATEFUL_BACKEND_CONTAINER_NAME = "test_stateful_backend"
STATEFUL_BACKEND_REPO = "triton-{0}-test-backend".format(TRITON_REPO_VERSION)
STATEFUL_BACKEND_TAG = "latest"
STATEFUL_BACKEND_IMAGE = STATEFUL_BACKEND_REPO + ":" + STATEFUL_BACKEND_TAG

STATEFUL_BACKEND_WORKDIR = "{0}/build".format(STATEFUL_BACKEND_VOL_DEST)
STATEFUL_BACKEND_BUILD_CLEANUP_CMD = f"rm -rf {STATEFUL_BACKEND_WORKDIR}"
STATEFUL_BACKEND_BUILD_CMAKE_CLEANUP_CMD = f"rm -f {STATEFUL_BACKEND_WORKDIR}/CMakeCache.txt"
STATEFUL_BACKEND_BUILD_SETUP_CMD = f"mkdir -p {STATEFUL_BACKEND_WORKDIR}"
STATEFUL_BACKEND_BUILD_CMAKE_OPTS = ' -DCMAKE_INSTALL_PREFIX:PATH=install \
    -DTRITON_BACKEND_REPO_TAG="r{0}" -DTRITON_CORE_REPO_TAG="r{0}" \
    -DTRITON_COMMON_REPO_TAG="r{0}" '.format(TRITON_REPO_VERSION)
STATEFUL_BACKEND_BUILD_MAKE_CMD = 'make -j'
STATEFUL_BACKEND_BUILD_INSTALL_CMD = 'make install'
STATEFUL_BACKEND_INSTALL_PATH = STATEFUL_BACKEND_WORKDIR + "/install/backends/stateful"


TRITONSERVER_REPO = "nvcr.io/nvidia/tritonserver"
TRITONSERVER_TAG = "{0}-py3".format(TRITON_REPO_VERSION)
TRITONCLIENT_TAG = "{0}-py3-sdk".format(TRITON_REPO_VERSION)

TRITONSERVER_IMAGE = TRITONSERVER_REPO + ":" + TRITONSERVER_TAG
TRITONCLIENT_IMAGE = TRITONSERVER_REPO + ":" + TRITONCLIENT_TAG

TRITON_GRPC_PORT = 8008 # using this to avoid possible conflict with dev environment
TRITON_PORTS = {TRITON_GRPC_PORT:TRITON_GRPC_PORT} # just use one port for now
TRITON_SHM_SIZE = "1g"
TRITON_MEMLOCK = -1
TRITON_STACK = 67108864
TRITON_VOL_DEST = STATEFUL_BACKEND_VOL_DEST

TRITON_SERVER_CONTAINER_NAME = "test_stateful_server"

TRITON_SERVER_BACKENDS_DIR = "/opt/tritonserver/backends"
TRITON_SERVER_CUSTOM_BACKEND_DIR = TRITON_SERVER_BACKENDS_DIR + "/stateful"
TRITON_SERVER_ONNXRT_BACKEND_DIR = TRITON_SERVER_BACKENDS_DIR + "/onnxruntime"
TRITON_SERVER_ONNXRT_CLEAN_CMD = "rm -rf " + TRITON_SERVER_ONNXRT_BACKEND_DIR
TRITON_SERVER_COPY_BACKEND_CMD = "cp -R {0} {1}/".format(STATEFUL_BACKEND_INSTALL_PATH, TRITON_SERVER_BACKENDS_DIR)
TRITON_SERVER_LD_LIBPATH_PREFIX = "LD_LIBRARY_PATH={0}/custom-ort/lib".format(STATEFUL_BACKEND_WORKDIR)
TRITON_SERVER_TRT_CACHE_ENABLE_ENV = "ORT_TENSORRT_ENGINE_CACHE_ENABLE=1"
TRITON_SERVER_TRT_CACHE_PATH_ENV = "ORT_TENSORRT_CACHE_PATH=/tmp"
TRITON_SERVER_ENV = [TRITON_SERVER_LD_LIBPATH_PREFIX, TRITON_SERVER_TRT_CACHE_ENABLE_ENV, TRITON_SERVER_TRT_CACHE_PATH_ENV]
TRITON_SERVER_MODEL_REPO_DIR = TRITON_VOL_DEST + "/models"
TRITON_SERVER_EXTRA_ARGS = " --model-load-thread-count 1 " #" --model-control-mode explicit --load-model accumulate_fp32 "
TRITON_SERVER_ORT_LOGGING_LEVEL = 4 # 0 means verbose
TRITON_SERVER_CMD = "tritonserver --grpc-port {0} --model-repository {1} --backend-config=stateful,ort-logging-level={2} {3}".format(TRITON_GRPC_PORT, TRITON_SERVER_MODEL_REPO_DIR, TRITON_SERVER_ORT_LOGGING_LEVEL, TRITON_SERVER_EXTRA_ARGS)
TRITON_SERVER_KILL_CMD = "pkill --signal SIGINT tritonserver"

TRITON_CLIENT_CONTAINER_NAME = "test_stateful_client"

TRITON_CLIENT_PUBKEY_FIX_CMD = "apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC"
TRITON_CLIENT_CMAKE_WGET_KEY_CMD = "wget https://apt.kitware.com/keys/kitware-archive-latest.asc"
TRITON_CLIENT_CMAKE_GPG_KEY_CMD = "gpg --dearmor kitware-archive-latest.asc"
TRITON_CLIENT_CMAKE_ADD_KEY_CMD = "mv kitware-archive-latest.asc.gpg /etc/apt/trusted.gpg.d/kitware.gpg"
TRITON_CLIENT_CMAKE_ADD_REPO_CMD = "apt-add-repository 'deb https://apt.kitware.com/ubuntu/ {0} main'"
TRITON_CLIENT_CMAKE_APT_UPDATE_CMD = "apt update"
TRITON_CLIENT_CMAKE_INSTALL_CMD = "apt install -y --no-install-recommends cmake" # " cmake-data=3.18.4-0kitware1ubuntu20.04.1 cmake=3.18.4-0kitware1ubuntu20.04.1"

TRITON_CLIENT_DEFAULT_CMAKE_INSTALL_CMD = "apt install -y --no-install-recommends cmake"

TRITON_CLIENT_WORKDIR = TRITON_VOL_DEST + "/client/build"
TRITON_PYCLIENT_DIR = TRITON_VOL_DEST + "/client"
TRITON_CLIENT_CMAKE_SETUP_CMD = "mkdir -p {0}".format(TRITON_CLIENT_WORKDIR)
TRITON_CLIENT_CMAKE_CMD = "cmake .."
TRITON_CLIENT_MAKE_CMD = "make -j"
TRITON_CLIENT_NAME = "simple_grpc_sequence_stateful_stream_infer_client"
TRITON_PYCLIENT_NAME = "mixed_io_stream_infer_client.py"
TRITON_CLIENT_RUN_CMD = "{0}/{1} -u localhost:{2}".format(TRITON_CLIENT_WORKDIR, TRITON_CLIENT_NAME, TRITON_GRPC_PORT)
TRITON_PYCLIENT_RUN_CMD = "python3 {0}/{1} -u localhost:{2}".format(TRITON_PYCLIENT_DIR, TRITON_PYCLIENT_NAME, TRITON_GRPC_PORT)
