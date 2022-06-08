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
import threading
import time
import argparse

import stateful_utils
import stateful_config
import build_backend
from stateful_utils import LogPrint
from packaging import version

TEST_FLAGS = None

g_server_started = False
g_server_exited = False
g_server_thread = None
g_server_crashed = False

TRITON_VOLUMES = {}
TRITON_VOL_SRC = ''
def setup_env(root_dir):
  global TRITON_VOL_SRC, TRITON_VOLUMES
  TRITON_VOL_SRC = root_dir
  TRITON_VOLUMES[TRITON_VOL_SRC] = {
    'bind': stateful_config.TRITON_VOL_DEST, 'mode': 'rw'
  }
  return

def run_server_thread_func(cnt):
  custom_env = None
  if build_backend.FLAGS.build_with_custom_image:
    custom_env = stateful_config.TRITON_SERVER_ENV
  status = cnt.exec_run(stateful_config.TRITON_SERVER_CMD, stream=True, environment=custom_env)
  outgen = status[1]
  global g_server_started, g_server_exited, g_server_crashed
  g_server_started = g_server_exited = g_server_crashed = False
  l_server_started = False
  for ln in outgen:
    print(ln.decode(), end='')
    if ln.decode().find("Started GRPCInferenceService") >= 0:
      g_server_started = l_server_started = True
    if ln.decode().find("successfully unloaded") >= 0:
      g_server_exited = True
      break
  if not l_server_started: # server didn't start for some reason
    g_server_crashed = True
  return

def start_server(scnt):
  global g_server_started, g_server_thread
  g_server_thread = threading.Thread(target=run_server_thread_func, args=(scnt,)) # always running
  g_server_thread.start()
  print("Waiting for the server to get started ...", flush=True)
  # wait until server fully started
  while not g_server_started:
    time.sleep(1)
    if g_server_crashed:
      print("Server didn't start properly. Exiting ...", file=sys.stderr)
      exit(1)
  g_server_started = False
  return

def stop_server(scnt):
  global g_server_exited, g_server_thread
  status = scnt.exec_run(stateful_config.TRITON_SERVER_KILL_CMD)
  assert status[0] == 0
  print("Waiting for the server to exit ...", flush=True)
  while not g_server_exited:
    time.sleep(1)
  g_server_thread.join()
  g_server_exited = False
  return

def RunServer(root_dir):
  print("Starting server ...")
  # create new container if not found
  server_container_name = stateful_config.TRITON_SERVER_CONTAINER_NAME
  if TEST_FLAGS.server_container_name != "":
    server_container_name = TEST_FLAGS.server_container_name
  scnt = stateful_utils.get_running_container(server_container_name)
  if scnt is None:
    scnt = stateful_utils.create_container(stateful_config.TRITONSERVER_IMAGE, \
        cnt_name=server_container_name, \
        with_gpus=True, ports=stateful_config.TRITON_PORTS, \
        shm_size=stateful_config.TRITON_SHM_SIZE, memlock=stateful_config.TRITON_MEMLOCK, \
        stack_size=stateful_config.TRITON_STACK, volumes=TRITON_VOLUMES)
    scnt.start()
  assert scnt != None
  scnt.reload()
  assert scnt.status == "running"
  if build_backend.FLAGS.build_with_custom_image:
    status = scnt.exec_run(stateful_config.TRITON_SERVER_ONNXRT_CLEAN_CMD)
    # print(status[0], status[1].decode())
    assert status[0] == 0
  status = scnt.exec_run(stateful_config.TRITON_SERVER_COPY_BACKEND_CMD)
  # print(status[0], status[1].decode())
  assert status[0] == 0
  start_server(scnt)
  return scnt

def RunClient(root_dir):
  print("Starting client ...")
  # create new container if not found
  client_container_name = stateful_config.TRITON_CLIENT_CONTAINER_NAME
  if TEST_FLAGS.client_container_name != "":
    client_container_name = TEST_FLAGS.client_container_name
  ccnt = stateful_utils.get_running_container(client_container_name)
  new_container_created = False
  if ccnt is None:
    ccnt = stateful_utils.create_container(stateful_config.TRITONCLIENT_IMAGE, \
        cnt_name=client_container_name, volumes=TRITON_VOLUMES, as_root=True)
    ccnt.start()
    new_container_created = True
  assert ccnt != None
  ccnt.reload()
  assert ccnt.status == "running"
  print("Client container running ...", flush=True)
  # The next few setup commands are only needed for SDK container versions > 22.03
  if new_container_created and version.parse(stateful_config.TRITON_REPO_VERSION) > version.parse("22.03"):
    stateful_utils.install_default_cmake(ccnt)
    print("CMake is now installed!")

  status = ccnt.exec_run(stateful_config.TRITON_CLIENT_CMAKE_SETUP_CMD)
  # print(status[0], status[1].decode())
  assert status[0] == 0
  status = ccnt.exec_run(stateful_config.TRITON_CLIENT_CMAKE_CMD, workdir=stateful_config.TRITON_CLIENT_WORKDIR)
  print(status[0], status[1].decode())
  assert status[0] == 0
  status = ccnt.exec_run(stateful_config.TRITON_CLIENT_MAKE_CMD, workdir=stateful_config.TRITON_CLIENT_WORKDIR)
  # print(status[0], status[1].decode())
  assert status[0] == 0
  status = ccnt.exec_run(stateful_config.TRITON_CLIENT_RUN_CMD, workdir=stateful_config.TRITON_CLIENT_WORKDIR)
  print(status[1].decode())
  assert status[0] == 0
  return ccnt

def DoEverything(root_dir):
  if build_backend.FLAGS.root_dir != "":
    root_dir = build_backend.FLAGS.root_dir
  err_happened = False
  # 0. setup the environment
  setup_env(root_dir)
  # 1. Build the backend
  build_backend.DoEverything(root_dir)
  # 2. Run the server
  scnt = RunServer(root_dir)
  # 3. Run the client
  ccnt = None
  try:
    ccnt = RunClient(root_dir)
  except Exception as ex:
    err_happened = True
    print("Client error: ", ex)
  time.sleep(2) # sleep for 2 seconds
  # 4. Stop the server
  stop_server(scnt)
  if err_happened:
    print("TEST FAILED!")
    exit(1)
  print("TEST PASSED!")
  if build_backend.FLAGS.stop_containers:
    if scnt is not None:
      scnt.stop()
    if ccnt is not None:
      ccnt.stop()
  return

def parse_args(args=sys.argv[1:]):
  parser = argparse.ArgumentParser("Script to test the stateful backend.")
  parser.add_argument("--server_container_name", 
                      type=str, default="", 
                      help="Override the container name of the Triton server.")
  parser.add_argument("--client_container_name", 
                      type=str, default="", 
                      help="Override the container name of the Client.")
  
  # update the global variable FLAGS
  global TEST_FLAGS
  TEST_FLAGS, rem_args = parser.parse_known_args(args)
  LogPrint("ARGS:", " ".join(f'{arg}={val}' for arg, val in vars(TEST_FLAGS).items()))
  return rem_args

def main():
  root_dir = os.path.join(os.path.abspath(sys.path[0]), os.pardir)
  rem_args = parse_args()
  build_backend.parse_args(rem_args)
  DoEverything(root_dir)
  return

if __name__ == "__main__":
  main()