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

#include "stateful.h"
#include "response_util.h"

namespace triton { namespace backend { namespace stateful { namespace utils {

void* GetOutputBuffer(
  TRITONBACKEND_Response** response,
  TRITONBACKEND_Output* output,
  size_t output_byte_size)
{
  // We request a buffer in CPU memory but we have to handle
  // any returned type. If we get back a buffer in GPU memory,
  // we just fail the request.
  void* output_buffer;
  TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t output_memory_type_id = 0;
  GUARDED_RESPOND_IF_ERROR_INTERNAL(
    *response,
      TRITONBACKEND_OutputBuffer(
          output, &output_buffer, output_byte_size, &output_memory_type,
          &output_memory_type_id));
  if ((*response == nullptr) ||
      (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
    GUARDED_RESPOND_IF_ERROR_INTERNAL(
      *response,
      TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "failed to create output buffer in CPU memory"));
    LOG_MESSAGE(
      TRITONSERVER_LOG_ERROR,
      "Failed to create output buffer in CPU memory, error response sent.");
    return nullptr;
  }
  return output_buffer;
}

void
SendSingleEmptyResponse(
  TRITONBACKEND_Response** response,
  std::vector<TritonTensorInfo>& output_tensors)
{
  for (auto& output_tensor : output_tensors) {
    // NOTE: 0-sized malloc behavior is undefined.
    // To minimize network traffic, we create a tensor of size 1
    std::vector<int64_t> one_shape(output_tensor.shape.size(), 1);
    TRITONBACKEND_Output* output;
    GUARDED_RESPOND_IF_ERROR_INTERNAL(
        *response,
        TRITONBACKEND_ResponseOutput(
            *response, &output, output_tensor.name.c_str(),
            static_cast<TRITONSERVER_DataType>(output_tensor.triton_type),
            one_shape.data(), one_shape.size()));
    // NOTE: Buffer creation is a must, otherwise Triton sends error to client
    GetOutputBuffer(response, output, 1); // since malloc(0) is undefined
  }
  if (*response != nullptr) {
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            *response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
        "failed sending response");
  }
}

TRITONSERVER_Error*
SendSingleResponse(
  InferenceTask& task,
  uint32_t ridx,
  void* vresponses)
{
  auto responses = reinterpret_cast<TRITONBACKEND_Response**>(vresponses);
  TRITONSERVER_Error* err = nullptr;  // success
  if (!task.err_msg.empty()) {        // error
    err = TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_INTERNAL, task.err_msg.c_str());
  }
  // If we get to this point then there hasn't been any error for the whole
  // batch and the response is complete (error or not) and we can send it.
  // This is the last (and only) response that we are sending for the request
  // so we must mark it FINAL. If there is an error when sending, all we can do
  // is log it.
  LOG_IF_ERROR(
      TRITONBACKEND_ResponseSend(
          responses[ridx], TRITONSERVER_RESPONSE_COMPLETE_FINAL, err),
      "failed sending response");

  // Now delete the Error object
  if (err != nullptr) {
    responses[ridx] = nullptr; // mark request that had error response
    TRITONSERVER_ErrorDelete(err);
  }
  return nullptr;
}

TRITONSERVER_Error*
SendResponses(
  std::vector<InferenceTask>& inference_tasks_,
  uint32_t request_count,
  void* vresponses)
{
  TRITONSERVER_Error* err = nullptr;
  // Send the response for each request
  for (uint32_t r = 0; r < request_count; ++r) {
    auto e = SendSingleResponse(inference_tasks_[r], r, vresponses);
    if (err == nullptr && e != nullptr) { // record the first error
      err = e;
    }
  }
  return err;
}

}}}}