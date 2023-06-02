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
import shlex
import subprocess
import argparse

import stateful_utils
import stateful_config
from stateful_utils import LogPrint

FLAGS = None

def is_custom_image_ready():
  return stateful_utils.is_image_ready(stateful_config.STATEFUL_BACKEND_IMAGE)

def remove_custom_image():
  # NOTE: docker module cannot differentiate different tags of the same image, so will clean manually here
  # stateful_utils.remove_image_with_containers(stateful_config.STATEFUL_BACKEND_IMAGE)
  LogPrint("Removing the old image and its containers ...")
  cnt_list_cmd = "docker ps -a"
  try:
    output = subprocess.check_output( shlex.split( cnt_list_cmd, posix=(sys.version != "nt") ) ).decode().strip()
    for ln in output.splitlines():
      ln_parts = ln.split()
      if ln_parts[1] == stateful_config.STATEFUL_BACKEND_IMAGE \
          or ln_parts[1] == stateful_config.STATEFUL_BACKEND_REPO:
        LogPrint("Stopping and removing container :", ln_parts[-1])
        stateful_utils.remove_container_by_name(ln_parts[-1])
  except Exception as ex:
    LogPrint("ERROR: Couldn't remove existing containers.", ex)
    exit(1)
  try:
    # now remove the container
    stateful_utils.remove_image_by_name(stateful_config.STATEFUL_BACKEND_IMAGE)
  except Exception as ex:
    if str(ex).find("Not such image"):
      LogPrint("WARNING:", ex)
      pass
    else:
      LogPrint("ERROR: Couldn't remove existing image", ex)
      exit(1)
  return

def build_custom_image():
  build_cmd = (f"docker build --tag {stateful_config.STATEFUL_BACKEND_IMAGE} -f docker/Dockerfile.backend " \
    + f" --build-arg BASE_IMAGE_VERSION={stateful_config.TRITON_REPO_VERSION} {FLAGS.docker_build_extra_args} .")
  LogPrint("Building the custom backend image ...", build_cmd)
  try:
    output = subprocess.check_output( shlex.split( build_cmd, posix=(sys.version != "nt") ) ).decode().strip()
  except subprocess.CalledProcessError as ex:
    print(ex.output.decode())
    LogPrint("ERROR: Couldn't build the custom backend image.", ex)
    exit(ex.returncode)
  return

BACKEND_VOLUMES = {}
BACKEND_VOL_SRC = ''
def setup_env(root_dir):
  global BACKEND_VOL_SRC, BACKEND_VOLUMES
  BACKEND_VOL_SRC = root_dir
  BACKEND_VOLUMES[BACKEND_VOL_SRC] = {
    'bind': stateful_config.TRITON_VOL_DEST, 'mode': 'rw'
  }
  return

def get_backend_build_container():
  custom_image_name = stateful_config.STATEFUL_BACKEND_IMAGE
  if FLAGS.custom_image_name != "":
    custom_image_name = FLAGS.custom_image_name
  backend_container_name = stateful_config.STATEFUL_BACKEND_CONTAINER_NAME
  if FLAGS.backend_container_name != "":
    backend_container_name = FLAGS.backend_container_name
  cnt = None
  if FLAGS.build_with_custom_image:
    LogPrint("Using the custom image ...")
    if FLAGS.force_rebuild_image:
      remove_custom_image()
    if FLAGS.with_gpus: # if with_gpu is set, remove the old container and create new
      stateful_utils.remove_container_by_name(backend_container_name)
    if not is_custom_image_ready():
      build_custom_image()
    # Get the custom container
    cnt_name = backend_container_name
    cnt_image = custom_image_name
  else:
    LogPrint("Using the stock TensorRT and Triton image ...")
    # Get the stock Triton container
    cnt_name = stateful_config.TRITON_CONTAINER_NAME
    if FLAGS.backend_container_name != "":
      cnt_name = FLAGS.backend_container_name
    cnt_image = stateful_config.TRITON_IMAGE

  cnt = stateful_utils.get_running_container(cnt_name)
  if cnt is None:
    cnt = stateful_utils.create_container(cnt_image, \
        cnt_name=cnt_name, with_gpus=FLAGS.with_gpus, \
        shm_size=stateful_config.TRITON_SHM_SIZE, memlock=stateful_config.TRITON_MEMLOCK, \
        stack_size=stateful_config.TRITON_STACK, volumes=BACKEND_VOLUMES, \
        as_root=True)
    cnt.start()
    if not FLAGS.build_with_custom_image:
      # if stock Triton container, need to prepare the container
      LogPrint(f"Running command '{stateful_config.TRITON_CONTAINER_PREBUILD_CMD}' to prepare the container ...")
      status = cnt.exec_run(stateful_config.TRITON_CONTAINER_PREBUILD_CMD)
      if status[0] != 0:
        LogPrint(status[1].decode())
      LogPrint(f"Copying TensorRT samples to Triton container ...")
      trt_cnt = stateful_utils.create_container(stateful_config.TENSORRT_IMAGE,
                                                cnt_name=stateful_config.TENSORRT_CONTAINER_NAME,
                                                auto_remove=True)
      trt_cnt.start()
      files_to_copy = ["buffers.h", "common.h", "safeCommon.h", "half.h",
                       "logger.cpp", "logger.h", "logging.h",
                       "sampleOptions.h", "sampleEntrypoints.h", "ErrorRecorder.h"]
      FILES_PATH = "/workspace/tensorrt/samples/common"
      cnt.exec_run(f"mkdir -p {FILES_PATH}", tty=True)
      for file in files_to_copy:
        cp1_cmd = f"docker cp {stateful_config.TENSORRT_CONTAINER_NAME}:{FILES_PATH}/{file} /tmp/{file}"
        cp2_cmd = f"docker cp /tmp/{file} {stateful_config.TRITON_CONTAINER_NAME}:{FILES_PATH}/{file}"
        try:
          _ = subprocess.check_output( shlex.split( cp1_cmd, posix=(sys.version != "nt") ) ).decode().strip()
          _ = subprocess.check_output( shlex.split( cp2_cmd, posix=(sys.version != "nt") ) ).decode().strip()
        except subprocess.CalledProcessError as ex:
          print(ex.output.decode())
          LogPrint(f"ERROR: Couldn't copy the required TRT file {file}.", ex)
          exit(ex.returncode)
      trt_cnt.stop()
      assert status[0] == 0
  return cnt

def build_custom_backend():
  cnt = get_backend_build_container()
  
  try:
    LogPrint("Building the custom backend ...")
    if FLAGS.cleanup_before_building:
      LogPrint("Removing the old build directory ...")
      status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_CLEANUP_CMD)
      LogPrint(status[1].decode())
      assert status[0] == 0
    else:
      LogPrint("Removing the old CMakeCache.txt ...")
      status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_CMAKE_CLEANUP_CMD)
      LogPrint(status[1].decode())
      assert status[0] == 0
    status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_SETUP_CMD)
    LogPrint(status[1].decode())
    assert status[0] == 0
    cmake_opts = stateful_config.STATEFUL_BACKEND_BUILD_CMAKE_OPTS
    if FLAGS.build_with_custom_image:
      cmake_opts += " -DWITH_CUSTOM_ORT=ON "
    cmake_cmd = "cmake " + cmake_opts + " .. "
    LogPrint(cmake_cmd)
    status = cnt.exec_run(cmake_cmd, workdir=stateful_config.STATEFUL_BACKEND_WORKDIR, tty=True)
    LogPrint(status[1].decode())
    assert status[0] == 0
    status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_MAKE_CMD, workdir=stateful_config.STATEFUL_BACKEND_WORKDIR, tty=True)
    LogPrint(status[1].decode())
    assert status[0] == 0
    status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_INSTALL_CMD, workdir=stateful_config.STATEFUL_BACKEND_WORKDIR, tty=True)
    LogPrint(status[1].decode())
    assert status[0] == 0
  except Exception as ex:
    LogPrint(ex)
    LogPrint("NOTE: Make sure to not mix the artifacts from two different build process. Try running with '--cleanup_before_building' .")
    exit(1)
  finally:
    if FLAGS.stop_containers: # autoremove is set during creation
      LogPrint("Stopping the container:", cnt.name)
      cnt.stop()
  return

def DoEverything(root_dir):
  if FLAGS.root_dir != "":
    root_dir = FLAGS.root_dir
  setup_env(root_dir=root_dir)
  LogPrint("Configuration: ")
  LogPrint("  Root directory to be used :", root_dir)
  # LogPrint("  Force rebuild image :", FLAGS.force_rebuild_image)
  # LogPrint("  Use GPUs in docker :", FLAGS.with_gpus)
  # LogPrint("  Stop/remove containers afterward :", FLAGS.stop_containers)
  # LogPrint("  Docker build extra flags :", FLAGS.docker_build_extra_flags)
  build_custom_backend()
  return

def parse_args(args=sys.argv[1:]):
  parser = argparse.ArgumentParser("Script to build the stateful backend.")
  parser.add_argument("--force_rebuild_image", 
                      action='store_true', 
                      help="Remove old images/containers and create new ones.")
  parser.add_argument("--with_gpus", 
                      action='store_true', 
                      help="Whether docker containers will be created with GPU support or not.")
  parser.add_argument("--stop_containers", 
                      action='store_true', 
                      help="Stop/remove the containers once build is done.")
  parser.add_argument("--build_with_custom_image", 
                      action='store_true', 
                      help="Whether to build and use custom docker image or not.")
  parser.add_argument("--cleanup_before_building", 
                      action='store_true', 
                      help="Whether or not to remove the old build directory before building.")

  parser.add_argument("--docker_build_extra_args", 
                      type=str, default="", 
                      help="Extra flags to pass to the 'docker build' command.")
  parser.add_argument("--root_dir", 
                      type=str, default="", 
                      help="The path to root directory where CMakeLists.txt file is present.")
  parser.add_argument("--custom_image_name", 
                      type=str, default="", 
                      help="Override the name of the new custom image.")
  parser.add_argument("--backend_container_name", 
                      type=str, default="", 
                      help="Override the container name of the custom image.")
  
  # update the global variable FLAGS
  global FLAGS
  FLAGS = parser.parse_args(args)
  LogPrint("ARGS:", " ".join(f'{arg}={val}' for arg, val in vars(FLAGS).items()))
  return

def main():
  root_dir = os.path.join(os.path.abspath(sys.path[0]), os.pardir)
  DoEverything(root_dir)
  return

if __name__ == "__main__":
  parse_args()
  main()