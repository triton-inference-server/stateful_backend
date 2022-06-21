// The MIT License (MIT)
//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <iostream>
#include <vector>
#include "onnx_model_runner.h"

// #define VERBOSE_OUTPUT

// Define some error checking macros.
#define cudaErrCheck(stat)                     \
  {                                            \
    cudaErrCheck_((stat), __FILE__, __LINE__); \
  }
void
cudaErrCheck_(cudaError_t stat, const char* file, int line)
{
  if (stat != cudaSuccess) {
    fprintf(
        stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#ifdef VERBOSE_OUTPUT
static inline void
print_device_vector(float* vec, size_t size)
{
  std::vector<float> debugPrintFloat(size);
  cudaErrCheck(cudaMemcpy(
      &debugPrintFloat[0], vec, size * sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < size; ++i) {
    std::cout << debugPrintFloat[i] << ", ";
  }
  std::cout << std::endl;
}

static inline void
print_device_vectors(float** vec, size_t numvecs, size_t size)
{
  std::vector<float*> debugPrintFloat(numvecs);
  cudaErrCheck(cudaMemcpy(
      debugPrintFloat.data(), vec, numvecs * sizeof(float*),
      cudaMemcpyDeviceToHost));
  for (int i = 0; i < numvecs; ++i) {
    std::cout << debugPrintFloat[i] << ": ";
    print_device_vector(debugPrintFloat[i], size);
  }
}

static inline void
print_device_vector(int* vec, size_t size)
{
  std::vector<int> debugPrintInt(size);
  cudaErrCheck(cudaMemcpy(
      &debugPrintInt[0], vec, size * sizeof(int), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < size; ++i) {
    std::cout << debugPrintInt[i] << ", ";
  }
  std::cout << std::endl;
}
#endif

template<typename Ti, typename To>
__host__ __device__ To convert_if_needed(Ti val) {
  return val;  // Ti == To by default
}

template<>
__host__ __device__ __half convert_if_needed(float val) {
  return __float2half(val);
}

template<>
__host__ __device__ float convert_if_needed(__half val) {
  return __half2float(val);
}

template<typename Ti, typename To>
__host__ __device__ void restore_states_internal(
  void*** storage, void** states, int* sizesX, int* sizesY, int* storeids,
  int batchStride)
{
  int stateId = blockIdx.x;
  int reqId = blockIdx.y;
  int chunkId = storeids[reqId*2];
  int storeId = storeids[reqId*2+1];
  if (chunkId < 0 || storeId < 0)
    return;  // no empty slots
  int sizeX = sizesX[stateId];
  int sizeY = sizesY[stateId];
  Ti* pSrc = reinterpret_cast<Ti*>(storage[chunkId][stateId]) +
                                   storeId * sizeX * sizeY;
  To* pDst = reinterpret_cast<To*>(states[stateId]) + reqId * sizeY;
  for (int x = 0; x < sizeX; ++x) {
    for (int i = threadIdx.x; i < sizeY; i += blockDim.x) {
      pDst[i] = convert_if_needed<Ti, To>(pSrc[i]);
    }
    pSrc += sizeY;
    pDst += batchStride * sizeY;
  }
}

template<typename Ti, typename To>
__host__ __device__ void store_states_internal(
  void*** storage, void** states, int* sizesX, int* sizesY, int* storeids,
  int batchStride)
{
  int stateId = blockIdx.x;
  int reqId = blockIdx.y;
  int chunkId = storeids[reqId*2];
  int storeId = storeids[reqId*2+1];
  if (chunkId < 0 || storeId < 0)
    return;  // no empty slots
  int sizeX = sizesX[stateId];
  int sizeY = sizesY[stateId];
  To* pDst = reinterpret_cast<To*>(storage[chunkId][stateId]) +
                                   storeId * sizeX * sizeY;
  Ti* pSrc = reinterpret_cast<Ti*>(states[stateId]) + reqId * sizeY;
  for (int x = 0; x < sizeX; ++x) {
    for (int i = threadIdx.x; i < sizeY; i += blockDim.x) {
      pDst[i] = convert_if_needed<Ti, To>(pSrc[i]);
    }
    pDst += sizeY;
    pSrc += batchStride * sizeY;
  }
}

__global__ void
restore_states(
  void*** storage, void** states, int* sizesX, int* sizesY, int* storeids,
  int batchStride, nvinfer1::DataType* stateTypes, bool convert)
{
  int stateId = blockIdx.x;
  switch(stateTypes[stateId]) {
    case nvinfer1::DataType::kBOOL:
      restore_states_internal<bool, bool>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
    case nvinfer1::DataType::kFLOAT:
      if (convert) {
      restore_states_internal<__half, float>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      }
      else {
      restore_states_internal<float, float>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      }
      break;
    case nvinfer1::DataType::kHALF:
      restore_states_internal<__half, __half>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
    case nvinfer1::DataType::kINT32:
      restore_states_internal<int32_t, int32_t>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
    case nvinfer1::DataType::kINT8:
      restore_states_internal<int8_t, int8_t>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
  }
}

__global__ void
store_states(
  void*** storage, void** states, int* sizesX, int* sizesY, int* storeids,
  int batchStride, nvinfer1::DataType* stateTypes, bool convert)
{
  int stateId = blockIdx.x;
  switch(stateTypes[stateId]) {
    case nvinfer1::DataType::kBOOL:
      store_states_internal<bool, bool>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
    case nvinfer1::DataType::kFLOAT:
      if (convert) {
      store_states_internal<float, __half>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      }
      else {
      store_states_internal<float, float>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      }
      break;
    case nvinfer1::DataType::kHALF:
      store_states_internal<__half, __half>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
    case nvinfer1::DataType::kINT32:
      store_states_internal<int32_t, int32_t>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
    case nvinfer1::DataType::kINT8:
      store_states_internal<int8_t, int8_t>(storage, states, sizesX, sizesY,
        storeids, batchStride);
      break;
  }
}

void
launchRestoreGPUKernel(
  void*** storage, void** states, int* sizesX, int* sizesY, int numStates,
  int* storeids, int batchSize, int batchStride, cudaStream_t stream,
  nvinfer1::DataType* stateTypes, bool convert)
{
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(numStates, batchSize);
  restore_states<<<numBlocks, threadsPerBlock, 0, stream>>>(
    storage, states, sizesX, sizesY, storeids, batchStride, stateTypes,convert);
#ifdef VERBOSE_OUTPUT
  std::cout << "Restoring the instances:" << std::endl;
  print_device_vector(storeids, batchSize*2);
  print_device_vector(sizesX, numStates);
  print_device_vector(sizesY, numStates);
  print_device_vectors(states, numStates, 2);
#endif
}

void
launchStoreGPUKernel(
  void*** storage, void** states, int* sizesX, int* sizesY, int numStates,
  int* storeids, int batchSize, int batchStride, cudaStream_t stream,
  nvinfer1::DataType* stateTypes, bool convert)
{
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(numStates, batchSize);
  store_states<<<numBlocks, threadsPerBlock, 0, stream>>>(
    storage, states, sizesX, sizesY, storeids, batchStride, stateTypes,convert);
#ifdef VERBOSE_OUTPUT
  std::cout << "Storing the instances:" << std::endl;
  print_device_vector(storeids, batchSize*2);
  print_device_vector(sizesX, numStates);
  print_device_vector(sizesY, numStates);
  print_device_vectors(states, numStates, 2);
#endif
}
