#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <onnx_model_runner.h>


namespace {
  template<typename Ti, typename To>
  To convert_if_needed(Ti val) { return val; } // Ti == To by default

  template<>
  __half convert_if_needed(float val) { return __float2half(val); }

  template<>
  float convert_if_needed(__half val) { return __half2float(val); }

  template<typename Ti, typename To>
  void store_states_internal(
    std::vector<std::vector<void*>>& storage, std::vector<void*>& states,
    std::vector<int>& sizesX, std::vector<int>& sizesY, int i,
    std::vector<int>& storeids, int batchSize, int batchStride)
  {
    size_t sizeX = sizesX[i];
    size_t sizeY = sizesY[i];
    for (int j = 0; j < batchSize; ++j) {
      const int chunk_id = storeids[j*2];
      const int buffer_idx = storeids[j*2+1];
      if (chunk_id < 0 || buffer_idx < 0)
        continue;  // no empty slots
      Ti* srcBuffer = reinterpret_cast<Ti*>(states[i]);
      To* destBuffer = reinterpret_cast<To*>(storage[chunk_id][i]);
      for (size_t ix = 0; ix < sizeX; ++ix) {
        for (size_t iy = 0; iy < sizeY; ++iy) {
          destBuffer[buffer_idx * sizeX * sizeY + ix * sizeY + iy] =
              convert_if_needed<Ti, To>(
                srcBuffer[ix * batchStride * sizeY + j * sizeY + iy]);
        }
      }
    }
  }

  template<typename Ti, typename To>
  void restore_states_internal (
    std::vector<std::vector<void*>>& storage, std::vector<void*>& states,
    std::vector<int>& sizesX, std::vector<int>& sizesY, int i,
    std::vector<int>& storeids, int batchSize, int batchStride)
  {
    size_t sizeX = sizesX[i];
    size_t sizeY = sizesY[i];
    for (int j = 0; j < batchSize; ++j) {
      const int chunk_id = storeids[j*2];
      const int buffer_idx = storeids[j*2+1];
      if (chunk_id < 0 || buffer_idx < 0)
        continue;  // no empty slots
      Ti* srcBuffer = reinterpret_cast<Ti*>(storage[chunk_id][i]);
      To* destBuffer = reinterpret_cast<To*>(states[i]);
      for (size_t ix = 0; ix < sizeX; ++ix) {
        for (size_t iy = 0; iy < sizeY; ++iy) {
          destBuffer[ix * batchStride * sizeY + j * sizeY + iy] =
              convert_if_needed<Ti, To>(
                srcBuffer[buffer_idx * sizeX * sizeY + ix * sizeY + iy]);
        }
      }
    }
  }
} // unnamed namespace

namespace stateful
{

void store_states_cpu(
  std::vector<std::vector<void*>>& storage, std::vector<void*>& states,
  std::vector<int>& sizesX, std::vector<int>& sizesY, int numStates,
  std::vector<int>& storeids, int batchSize, int batchStride,
  std::vector<nvinfer1::DataType>& stateTypes, bool convert)
{
  for (int i = 0; i < numStates; ++i) {
    switch(stateTypes[i]) {
      case nvinfer1::DataType::kBOOL:
        store_states_internal<bool, bool>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
      case nvinfer1::DataType::kFLOAT:
        if (convert) {
          store_states_internal<float, __half>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        }
        else {
          store_states_internal<float, float>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        }
        break;
      case nvinfer1::DataType::kHALF:
        store_states_internal<__half, __half>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
      case nvinfer1::DataType::kINT32:
        store_states_internal<int32_t, int32_t>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
      case nvinfer1::DataType::kINT8:
        store_states_internal<int8_t, int8_t>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
    }
  }
}

void restore_states_cpu(
  std::vector<std::vector<void*>>& storage, std::vector<void*>& states,
  std::vector<int>& sizesX, std::vector<int>& sizesY, int numStates,
  std::vector<int>& storeids, int batchSize, int batchStride,
  std::vector<nvinfer1::DataType>& stateTypes, bool convert)
{
  for (int i = 0; i < numStates; ++i) {
    switch(stateTypes[i]) {
      case nvinfer1::DataType::kBOOL:
        restore_states_internal<bool, bool>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
      case nvinfer1::DataType::kFLOAT:
        if (convert) {
          restore_states_internal<__half, float>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        }
        else {
          restore_states_internal<float, float>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        }
        break;
      case nvinfer1::DataType::kHALF:
        restore_states_internal<__half, __half>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
      case nvinfer1::DataType::kINT32:
        restore_states_internal<int32_t, int32_t>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
      case nvinfer1::DataType::kINT8:
        restore_states_internal<int8_t, int8_t>(
            storage, states, sizesX, sizesY, i, storeids, batchSize, batchStride
          );
        break;
    }
  }
}
  
} // namespace stateful