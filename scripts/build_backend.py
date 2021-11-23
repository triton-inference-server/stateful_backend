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
import stateful_utils
import stateful_config

def is_custom_image_ready():
  return stateful_utils.is_image_ready(stateful_config.STATEFUL_BACKEND_IMAGE)

def remove_custom_image():
  # NOTE: docker module cannot differentiate different tags of the same image, so will clean manually here
  # stateful_utils.remove_image_with_containers(stateful_config.STATEFUL_BACKEND_IMAGE)
  print("Removing the old image and its containers ...")
  cnt_list_cmd = "docker ps -a"
  try:
    output = subprocess.check_output( shlex.split( cnt_list_cmd, posix=(sys.version != "nt") ) ).decode().strip()
    for ln in output.splitlines():
      ln_parts = ln.split()
      if ln_parts[1] == stateful_config.STATEFUL_BACKEND_IMAGE \
          or ln_parts[1] == stateful_config.STATEFUL_BACKEND_REPO:
        print("Stopping and removing container :", ln_parts[-1])
        stateful_utils.remove_container_by_name(ln_parts[-1])
  except Exception as ex:
    print("ERROR: Couldn't remove existing containers.", ex)
    exit(1)
  try:
    # now remove the container
    stateful_utils.remove_image_by_name(stateful_config.STATEFUL_BACKEND_IMAGE)
  except Exception as ex:
    if str(ex).find("Not such image"):
      print("WARNING:", ex)
      pass
    else:
      print("ERROR: Couldn't remove existing image", ex)
      exit(1)
  return

def build_custom_image():
  build_cmd = ("docker build --tag {0} -f docker/Dockerfile.backend " \
    + " --build-arg BASE_IMAGE_VERSION={1} .").format(stateful_config.STATEFUL_BACKEND_IMAGE, \
      stateful_config.TRITON_REPO_VERSION)
  print("Building the custom backend image ...", build_cmd)
  try:
    output = subprocess.check_output( shlex.split( build_cmd, posix=(sys.version != "nt") ) ).decode().strip()
  except Exception as ex:
    print("ERROR: Couldn't build the custom backend image.", ex)
    exit(1)
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

def get_backend_build_container(force_rebuild_image, with_gpus):
  if force_rebuild_image:
    remove_custom_image()
  if with_gpus: # if with_gpu is set, remove the old container and create new
    stateful_utils.remove_container_by_name(stateful_config.STATEFUL_BACKEND_CONTAINER_NAME)
  if not is_custom_image_ready():
    build_custom_image()

  cnt = stateful_utils.get_running_container(stateful_config.STATEFUL_BACKEND_CONTAINER_NAME)
  if cnt is None:
    cnt = stateful_utils.create_container(stateful_config.STATEFUL_BACKEND_IMAGE, \
        cnt_name=stateful_config.STATEFUL_BACKEND_CONTAINER_NAME, with_gpus=with_gpus, \
        shm_size=stateful_config.TRITON_SHM_SIZE, memlock=stateful_config.TRITON_MEMLOCK, \
        stack_size=stateful_config.TRITON_STACK, volumes=BACKEND_VOLUMES)
    cnt.start()
  return cnt

def build_custom_backend(force_rebuild_image, with_gpus):
  cnt = get_backend_build_container(force_rebuild_image, with_gpus)
  
  print("Building the custom backend ...")
  status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_SETUP_CMD)
  print(status[1].decode())
  assert status[0] == 0
  status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_CMAKE_CMD, workdir=stateful_config.STATEFUL_BACKEND_WORKDIR)
  print(status[1].decode())
  assert status[0] == 0
  status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_MAKE_CMD, workdir=stateful_config.STATEFUL_BACKEND_WORKDIR)
  print(status[1].decode())
  assert status[0] == 0
  status = cnt.exec_run(stateful_config.STATEFUL_BACKEND_BUILD_INSTALL_CMD, workdir=stateful_config.STATEFUL_BACKEND_WORKDIR)
  print(status[1].decode())
  assert status[0] == 0

def DoEverything(root_dir, args: list):
  setup_env(root_dir=root_dir)
  force_rebuild_image = False
  with_gpus = False
  for arg in args:
    if arg == "--force_rebuild_image":
      force_rebuild_image = True
    if arg == "--with_gpus":
      with_gpus = True
  print("Configuration: ")
  print("  Force rebuild image :", force_rebuild_image)
  print("  Use GPUs in docker :", with_gpus)
  build_custom_backend(force_rebuild_image, with_gpus)
  return

def main():
  root_dir = os.path.join(os.path.abspath(sys.path[0]), os.pardir)
  DoEverything(root_dir, sys.argv)
  return

if __name__ == "__main__":
  main()