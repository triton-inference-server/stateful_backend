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

cmake_minimum_required (VERSION 3.10)

project(Clients C CXX)


if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
message(STATUS "Build configuration: " ${CMAKE_BUILD_TYPE})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

set(GRPC_CLIENT_DIR /workspace/install)
set(GRPC_CLIENT_LIB ${GRPC_CLIENT_DIR}/lib/libgrpcclient.so ${GRPC_CLIENT_DIR}/lib/libprotobuf.a)
set(GRPC_CLIENT_INC_DIR ${GRPC_CLIENT_DIR}/include)

# template for building clients
function(build_client src exe)
  add_executable(${exe} ${src})
  target_link_libraries(
    ${exe}
    PRIVATE ${GRPC_CLIENT_LIB} Threads::Threads
  )
  install(
    TARGETS ${exe}
    RUNTIME DESTINATION bin
  )
  target_include_directories(${exe} PUBLIC "/workspace/onnx-stateful/common" ${GRPC_CLIENT_INC_DIR})
endfunction(build_client)

if(WIN32)
  message("Currently, CMake will only build client library and skip example "
          "clients in Windows. Because the examples use functionalities that "
		      "are UNIX specific.")
else()

  #
  # simple_grpc_sequence_stateful_stream_infer_client
  #
  build_client(simple_grpc_sequence_stateful_stream_infer_client.cc simple_grpc_sequence_stateful_stream_infer_client)

endif() # WIN32
