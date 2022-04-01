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

#pragma once

#include "onnx_model_runner.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"

#define GUARDED_RESPOND_IF_ERROR_INTERNAL(RESPONSE, X)                  \
  do {                                                                  \
    if ((RESPONSE) != nullptr) {                                        \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSE), TRITONSERVER_RESPONSE_COMPLETE_FINAL,       \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSE) = nullptr;                                           \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  GUARDED_RESPOND_IF_ERROR_INTERNAL(RESPONSES[(IDX)], (X))

namespace triton { namespace backend { namespace stateful {

// BackendState
// All the states/configs associated with the Stateful Backend.
struct BackendState {
  BackendState()
      : m_OrtLoggingLevel(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING),
        mOrtEnv(nullptr)
  {
  }
  OrtLoggingLevel m_OrtLoggingLevel;
  std::shared_ptr<Ort::Env> mOrtEnv;
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state);
  virtual ~ModelState() = default;

  TRITONSERVER_Error* InitModelState();

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

 public:
  int64_t max_batch_size_;
  int64_t max_sequence_idle_microseconds_;
  BufferConfig buffer_config_;
  int64_t max_candidate_sequences_;
  std::vector<int64_t> pref_batch_sizes_;
  bool is_corrId_string_;

  std::string path_;
  std::string onnx_file_name_;
  std::string ort_ep_name_;
  std::string compute_precision_name_;
  std::string store_states_as_fp16_;
  std::string state_pairs_;
  std::string max_candidate_sequence_use_ratio_;
  int64_t logging_level_;
  std::string metric_logging_frequency_seconds_;
  std::string enable_trt_caching_;
  std::string trt_cache_dir_;
  std::string always_pad_to_max_batch_;
  bool infer_end_requests_;
  int64_t error_inject_rate_;

  std::vector<TritonTensorInfo> input_tensors_;
  std::vector<TritonTensorInfo> output_tensors_;

  std::string start_tensor_name_;
  std::string end_tensor_name_;
  BackendState* backend_state_;


 private:
  ModelState(TRITONBACKEND_Model* triton_model);
};


//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState() = default;

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

 public:
  std::unique_ptr<TrtOnnxModel> trt_onnx_model_;
  std::vector<InferenceTask> inference_tasks_;

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance);

  ModelState* model_state_;
};


namespace utils {

TRITONSERVER_Error*
InitTritonTensorInfo(
    common::TritonJson::Value& tensor_values,
    std::vector<TritonTensorInfo>& tensor_infos, std::string& log);

}}}}  // namespace triton::backend::stateful::utils
