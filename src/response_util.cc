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

#include "response_util.h"

namespace triton { namespace backend { namespace stateful { namespace utils {

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