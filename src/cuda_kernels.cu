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

__global__ void
storeStates_FP32(
    float*** storage, float** states, int* sizesX, int* sizesY, int* storeids,
    int batchStride)
{
  int stateId = blockIdx.x;
  int batchId = blockIdx.y;
  int storeId = storeids[batchId];
  if (storeId < 0)
    return;  // no empty slots
  int sizeX = sizesX[stateId];
  int sizeY = sizesY[stateId];
  float* pDst = storage[0][stateId] + storeId * sizeX * sizeY;
  float* pSrc = states[stateId] + batchId * sizeY;
  for (int x = 0; x < sizeX; ++x) {
    for (int i = threadIdx.x; i < sizeY; i += blockDim.x) {
      pDst[i] = pSrc[i];
    }
    pDst += sizeY;
    pSrc += batchStride * sizeY;
  }
}

__global__ void
restoreStates_FP32(
    float*** storage, float** states, int* sizesX, int* sizesY, int* storeids,
    int batchStride)
{
  int stateId = blockIdx.x;
  int batchId = blockIdx.y;
  int storeId = storeids[batchId];
  if (storeId < 0)
    return;  // no empty slots
  int sizeX = sizesX[stateId];
  int sizeY = sizesY[stateId];
  float* pSrc = storage[0][stateId] + storeId * sizeX * sizeY;
  float* pDst = states[stateId] + batchId * sizeY;
  for (int x = 0; x < sizeX; ++x) {
    for (int i = threadIdx.x; i < sizeY; i += blockDim.x) {
      pDst[i] = pSrc[i];
    }
    pSrc += sizeY;
    pDst += batchStride * sizeY;
  }
}

__global__ void
storeStates_FP16(
    __half*** storage, float** states, int* sizesX, int* sizesY, int* storeids,
    int batchStride)
{
  int stateId = blockIdx.x;
  int batchId = blockIdx.y;
  int storeId = storeids[batchId];
  if (storeId < 0)
    return;  // no empty slots
  int sizeX = sizesX[stateId];
  int sizeY = sizesY[stateId];
  __half* pDst = storage[0][stateId] + storeId * sizeX * sizeY;
  float* pSrc = states[stateId] + batchId * sizeY;
  for (int x = 0; x < sizeX; ++x) {
    for (int i = threadIdx.x; i < sizeY; i += blockDim.x) {
      pDst[i] = __float2half(pSrc[i]);
    }
    pDst += sizeY;
    pSrc += batchStride * sizeY;
  }
}

__global__ void
restoreStates_FP16(
    __half*** storage, float** states, int* sizesX, int* sizesY, int* storeids,
    int batchStride)
{
  int stateId = blockIdx.x;
  int batchId = blockIdx.y;
  int storeId = storeids[batchId];
  if (storeId < 0)
    return;  // no empty slots
  int sizeX = sizesX[stateId];
  int sizeY = sizesY[stateId];
  __half* pSrc = storage[0][stateId] + storeId * sizeX * sizeY;
  float* pDst = states[stateId] + batchId * sizeY;
  for (int x = 0; x < sizeX; ++x) {
    for (int i = threadIdx.x; i < sizeY; i += blockDim.x) {
      pDst[i] = __half2float(pSrc[i]);
    }
    pSrc += sizeY;
    pDst += batchStride * sizeY;
  }
}

void
launchRestoreGPUKernel_FP32(
    float*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream)
{
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(numStates, batchSize);
  restoreStates_FP32<<<numBlocks, threadsPerBlock, 0, stream>>>(
      storage, states, sizesX, sizesY, storeids, batchStride);
#ifdef VERBOSE_OUTPUT
  std::cout << "Restoring the instances:" << std::endl;
  print_device_vector(storeids, batchSize);
  print_device_vector(sizes, numStates);
  print_device_vectors(states, numStates, 2);
#endif
}

void
launchStoreGPUKernel_FP32(
    float*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream)
{
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(numStates, batchSize);
  storeStates_FP32<<<numBlocks, threadsPerBlock, 0, stream>>>(
      storage, states, sizesX, sizesY, storeids, batchStride);
#ifdef VERBOSE_OUTPUT
  std::cout << "Storing the instances:" << std::endl;
  print_device_vector(storeids, batchSize);
  print_device_vector(sizes, numStates);
  print_device_vectors(states, numStates, 2);
#endif
}

void
launchRestoreGPUKernel_FP16(
    __half*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream)
{
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(numStates, batchSize);
  restoreStates_FP16<<<numBlocks, threadsPerBlock, 0, stream>>>(
      storage, states, sizesX, sizesY, storeids, batchStride);
#ifdef VERBOSE_OUTPUT
  std::cout << "Restoring the instances:" << std::endl;
  print_device_vector(storeids, batchSize);
  print_device_vector(sizes, numStates);
  print_device_vectors(states, numStates, 2);
#endif
}

void
launchStoreGPUKernel_FP16(
    __half*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream)
{
  dim3 threadsPerBlock(256, 1);
  dim3 numBlocks(numStates, batchSize);
  storeStates_FP16<<<numBlocks, threadsPerBlock, 0, stream>>>(
      storage, states, sizesX, sizesY, storeids, batchStride);
#ifdef VERBOSE_OUTPUT
  std::cout << "Storing the instances:" << std::endl;
  print_device_vector(storeids, batchSize);
  print_device_vector(sizes, numStates);
  print_device_vectors(states, numStates, 2);
#endif
}
