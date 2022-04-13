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
#include "stateful.h"

namespace triton { namespace backend { namespace stateful { namespace utils {

void
PreProcessRequests(
  TRITONBACKEND_Request** requests,
  std::vector<InferenceTask>& inference_tasks_,
  const uint32_t request_count,
  std::vector<TRITONBACKEND_Response*>& responses,
  uint32_t& infer_request_count,
  uint32_t& end_request_count,
  ModelInstanceState* model_instance_state)
{
  ModelState* model_state = model_instance_state->StateForModel();
  std::vector<TRITONBACKEND_Response*> new_responses;
  const bool split_requests = !model_state->infer_end_requests_;
  if (split_requests) {
    new_responses.resize(request_count, nullptr);
  }
  infer_request_count = request_count;
  end_request_count = 0;
  uint32_t true_ridx = 0;
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    /**
     * TODO:
     *  Managing only string sequence ids in the backend for now.
     *  UINT64 sequence ids are also converted to string for convenience.
     *  Once the InferenceRequest::SequenceId class is in common repo,
     *  we can reuse that implementation.
     */
    std::string correlation_id;
    if (!model_state->is_corrId_string_) {
      uint64_t u64_correlation_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestCorrelationId(request, &u64_correlation_id));
      correlation_id = std::to_string(u64_correlation_id);
    } else {
      const char* str_correlation_id = "";
#ifdef TRITON_SUPPORTS_STRING_CORRID
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestCorrelationIdString(
              request, &str_correlation_id));
#endif
      correlation_id = std::string(str_correlation_id);
    }

    TRITONBACKEND_Input* input = nullptr;
    TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t input_memory_type_id = 0;
    uint64_t buffer_byte_size = 0;
    const void* input_buffer = nullptr;

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(
            request, model_state->start_tensor_name_.c_str(), &input));
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputBuffer(
            input, 0, &input_buffer, &buffer_byte_size, &input_memory_type,
            &input_memory_type_id));
    int input_start = *reinterpret_cast<const int*>(input_buffer);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(
            request, model_state->end_tensor_name_.c_str(), &input));
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputBuffer(
            input, 0, &input_buffer, &buffer_byte_size, &input_memory_type,
            &input_memory_type_id));
    int input_end = *reinterpret_cast<const int*>(input_buffer);

    if (split_requests && input_end == 1) {
      --infer_request_count;
      LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Not inferring EndRequest for corrId: ") +
         correlation_id + ". Clearing states ...")
          .c_str());
      // clear states for the sequence
      model_instance_state->trt_onnx_model_->clearStates(correlation_id);
      // send empty response
      SendSingleEmptyResponse(&(responses[r]), model_state->output_tensors_);
      responses[r] = nullptr;
      continue;
    }

    int i_tensor = 0;
    for (auto& input_tensor : model_state->input_tensors_) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestInput(
              request, input_tensor.name.c_str(), &input));
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_InputBuffer(
              input, 0, &input_buffer, &buffer_byte_size, &input_memory_type,
              &input_memory_type_id));
      inference_tasks_[true_ridx].mInput[i_tensor++] = input_buffer;
    }

    inference_tasks_[true_ridx].mCorrId = correlation_id;
    inference_tasks_[true_ridx].mStart = input_start;
    inference_tasks_[true_ridx].mEnd = input_end;

#ifdef VERBOSE_LOG
    std::cout << "Request Inputs: "
              << inference_tasks_[true_ridx].mStart << ", ";
    std::cout << inference_tasks_[true_ridx].mEnd << ", ";
    std::cout << inference_tasks_[true_ridx].mCorrId << ", ";
    std::cout << buffer_byte_size << ",";
    std::cout << std::endl;
#endif

    i_tensor = 0;
    for (auto& output_tensor : model_state->output_tensors_) {
      TRITONBACKEND_Output* output;
      TRITONBACKEND_Response* response = responses[r];
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, output_tensor.name.c_str(),
              TRITONSERVER_TYPE_FP32, output_tensor.shape.data(),
              output_tensor.shape.size()));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create response output, error response sent")
                .c_str());
        continue;
      }

      // Get the output buffer
      void* output_buffer = stateful::utils::GetOutputBuffer(
                              &(responses[r]), output,
                              output_tensor.type_size * output_tensor.vol);

      inference_tasks_[true_ridx].mOutput[i_tensor++] = output_buffer;
    }
    if (split_requests) {
      new_responses[true_ridx] = responses[r];
    }
    true_ridx++;
  }
  if (split_requests && infer_request_count < request_count) {
    // replace the original responses with re-ordered one
    responses = new_responses;
    end_request_count = request_count - infer_request_count;
  }
}

}}}}