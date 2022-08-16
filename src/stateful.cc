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


// #define VERBOSE_LOG

#include <memory>
#include <thread>
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"

// Backend specific header files

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include "onnx_model_runner.h"

#include <onnxruntime_cxx_api.h>
#include "stateful.h"
#include "response_util.h"
#include "request_util.h"

namespace triton { namespace backend { namespace stateful {

namespace utils {

TRITONSERVER_Error*
InitTritonTensorInfo(
    common::TritonJson::Value& tensor_values,
    std::vector<TritonTensorInfo>& tensor_infos, std::string& log)
{
  const int triton_type_offset = strlen("TYPE_");
  for (size_t i = 0; i < tensor_values.ArraySize(); ++i) {
    TritonTensorInfo triton_tensor;
    common::TritonJson::Value tensor;
    RETURN_IF_ERROR(tensor_values.IndexAsObject(i, &tensor));
    RETURN_IF_ERROR(tensor.MemberAsString("data_type", &triton_tensor.type));
    RETURN_IF_ERROR(tensor.MemberAsString("name", &triton_tensor.name));
    RETURN_IF_ERROR(backend::ParseShape(tensor, "dims", &triton_tensor.shape));
    triton_tensor.idx = i;
    log = triton_tensor.Init();
    triton_tensor.triton_type = static_cast<int>(TRITONSERVER_StringToDataType(
                        triton_tensor.type.substr(triton_type_offset).c_str()));
    tensor_infos.push_back(triton_tensor);
  }

  return nullptr;  // success
}
}

//
// Simple backend that demonstrates the TRITONBACKEND API for a
// blocking backend with state tensors for sequence models.
// A blocking backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.
// The model must store the values to initialize the state tensors
// when new sequence starts.
//

#define CHECK_IF_ERROR(X, success_)  \
  do {                               \
    TRITONSERVER_Error* err__ = (X); \
    success_ = (err__ == nullptr);   \
    TRITONSERVER_ErrorDelete(err__); \
  } while (false)

#define INT2TRITON_LOG_LEVEL(l_) static_cast<TRITONSERVER_LogLevel>(l_)
#define TRITON2INT_LOG_LEVEL(l_) static_cast<int64_t>(l_)


// enable this to test error handling client
// #define DEBUG_ERROR_INJECT

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
{
  try {
    *state = new ModelState(triton_model);
  }
  catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;  // success
}

ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model)
{
  // Obtain backend state
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));
  void* vstate;
  THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  backend_state_ = reinterpret_cast<BackendState*>(vstate);
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  // Just logging the model config for now.
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;  // success
}

inline TRITONSERVER_Error*
GetInt64Parameter(
  common::TritonJson::Value& parameters, std::string name, int64_t& i64_conf)
{
  common::TritonJson::Value buffer_config;
  std::string buffer_config_str;
  RETURN_IF_ERROR(
    parameters.MemberAsObject(name.c_str(), &buffer_config));
  RETURN_IF_ERROR(
    buffer_config.MemberAsString("string_value", &buffer_config_str));
  i64_conf = static_cast<int64_t>(std::stol(buffer_config_str));
  return nullptr;
}

inline TRITONSERVER_Error*
ParseLogLevel(common::TritonJson::Value& parameters, std::string name,
              int64_t& triton_log_level)
{
  common::TritonJson::Value logging_level;
  std::string str_logging_level;
  RETURN_IF_ERROR(parameters.MemberAsObject(name.c_str(), &logging_level));
  RETURN_IF_ERROR(
        logging_level.MemberAsString("string_value", &str_logging_level));
  if (str_logging_level.compare("NONE") == 0) {
    triton_log_level = -1;
  }
  else if (str_logging_level.compare("VERBOSE") == 0) {
    triton_log_level = static_cast<int64_t>(TRITONSERVER_LOG_VERBOSE);
  }
  else if (str_logging_level.compare("INFO") == 0) {
    triton_log_level = static_cast<int64_t>(TRITONSERVER_LOG_INFO);
  }
  return nullptr;
}

// Initialize the model specific variables shared by all instances
TRITONSERVER_Error*
ModelState::InitModelState()
{
  bool parse_succeeded;
  logging_level_ =
      TRITON2INT_LOG_LEVEL(TRITONSERVER_LOG_INFO);  // default level info

  common::TritonJson::Value parameters;
  RETURN_IF_ERROR(model_config_.MemberAsObject("parameters", &parameters));

  IGNORE_ERROR(ParseLogLevel(parameters, "logging_level", logging_level_));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Logging Level = ") + std::to_string(logging_level_))
          .c_str());

  std::string str_logs;
  common::TritonJson::Value inputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(
      utils::InitTritonTensorInfo(inputs, input_tensors_, str_logs));
  if (logging_level_ >= 0) {
    LOG_MESSAGE(INT2TRITON_LOG_LEVEL(logging_level_), str_logs.c_str());
    str_logs.clear();
  }

  common::TritonJson::Value outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));
  RETURN_IF_ERROR(
      utils::InitTritonTensorInfo(outputs, output_tensors_, str_logs));
  if (logging_level_ >= 0) {
    LOG_MESSAGE(INT2TRITON_LOG_LEVEL(logging_level_), str_logs.c_str());
    str_logs.clear();
  }

  int input_size = input_tensors_[0].shape[0];
  int output_size = output_tensors_[0].shape[0];
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Input size = ") + std::to_string(input_size) +
       std::string(", Output size = ") + std::to_string(output_size))
          .c_str());

  // set default values for configurable parameters
  max_batch_size_ = 64;
  max_sequence_idle_microseconds_ = 100000000;
  buffer_config_.max_connections = 1280;
  buffer_config_.initial_buffer_size = 400;
  buffer_config_.subsequent_buffer_size = 100;
  buffer_config_.alloc_threshold = 32;
  buffer_config_.dealloc_threshold = 150;
  max_candidate_sequences_ = buffer_config_.max_connections;
  is_corrId_string_ = false;
  infer_end_requests_ = true;
  ort_ep_name_ = "trt";
  compute_precision_name_ = "fp16";
  store_states_as_fp16_ = "0";
  max_candidate_sequence_use_ratio_ = "0.9";
  trt_cache_dir_ = "/tmp";
  batch_info_logging_level_ = TRITONSERVER_LOG_INFO;
  detailed_metrics_logging_level_ = TRITON2INT_LOG_LEVEL(
                                                  TRITONSERVER_LOG_VERBOSE);

  IGNORE_ERROR(model_config_.MemberAsInt("max_batch_size", &max_batch_size_));
  pref_batch_sizes_ = {max_batch_size_};
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Max batch size = ") + std::to_string(max_batch_size_))
          .c_str());

  common::TritonJson::Value sequence_batching;
  RETURN_IF_ERROR(
      model_config_.MemberAsObject("sequence_batching", &sequence_batching));

  common::TritonJson::Value control_inputs;
  RETURN_IF_ERROR(
      sequence_batching.MemberAsArray("control_input", &control_inputs));
  for (size_t i = 0; i < control_inputs.ArraySize(); ++i) {
    common::TritonJson::Value control_input;
    RETURN_IF_ERROR(control_inputs.IndexAsObject(i, &control_input));

    common::TritonJson::Value controls;
    RETURN_IF_ERROR(control_input.MemberAsArray("control", &controls));

    common::TritonJson::Value control;
    RETURN_IF_ERROR(controls.IndexAsObject(0, &control));

    std::string control_type;
    RETURN_IF_ERROR(control.MemberAsString("kind", &control_type));

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Control Type = ") + control_type).c_str());

    if (control_type.compare("CONTROL_SEQUENCE_START") == 0) {
      RETURN_IF_ERROR(
          control_input.MemberAsString("name", &start_tensor_name_));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Start Name = ") + start_tensor_name_).c_str());

    } else if (control_type.compare("CONTROL_SEQUENCE_END") == 0) {
      RETURN_IF_ERROR(control_input.MemberAsString("name", &end_tensor_name_));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("End Name = ") + end_tensor_name_).c_str());

    } else if (control_type.compare("CONTROL_SEQUENCE_CORRID") == 0) {
      std::string control_data_type;
      RETURN_IF_ERROR(control.MemberAsString("data_type", &control_data_type));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("CORRID Data Type = ") + control_data_type).c_str());
#ifdef TRITON_SUPPORTS_STRING_CORRID
      if (control_data_type.compare("TYPE_STRING") == 0) {
        is_corrId_string_ = true;
      }
#else
      if (control_data_type.compare("TYPE_STRING") == 0) { // INVALID TYPE
        TRITONSERVER_Error* error = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNSUPPORTED,
          "This version of Triton does not support string CORRID.");
        RETURN_IF_ERROR(error);
      }
#endif
    }
  }
  IGNORE_ERROR(sequence_batching.MemberAsInt(
      "max_sequence_idle_microseconds", &max_sequence_idle_microseconds_));

  common::TritonJson::Value sequence_batching_oldest;
  CHECK_IF_ERROR(
      sequence_batching.MemberAsObject("oldest", &sequence_batching_oldest),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(sequence_batching_oldest.MemberAsInt(
        "max_candidate_sequences", &max_candidate_sequences_));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, (std::string("Max candidate sequences = ") +
                                std::to_string(max_candidate_sequences_))
                                   .c_str());

    common::TritonJson::Value pref_batch_sizes_array;
    CHECK_IF_ERROR(
        sequence_batching_oldest.MemberAsArray(
            "preferred_batch_size", &pref_batch_sizes_array),
        parse_succeeded);
    if (parse_succeeded) {
      pref_batch_sizes_.clear();
      std::string pref_batch_sizes_str;
      for (size_t i = 0; i < pref_batch_sizes_array.ArraySize(); ++i) {
        int64_t d;
        RETURN_IF_ERROR(pref_batch_sizes_array.IndexAsInt(i, &d));
        pref_batch_sizes_.push_back(d);
        pref_batch_sizes_str += std::to_string(d) + std::string(", ");
      }
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Preferred batch sizes = ") + pref_batch_sizes_str)
              .c_str());
    }
  }

  common::TritonJson::Value onnx_file;
  RETURN_IF_ERROR(parameters.MemberAsObject("onnx_file_name", &onnx_file));
  RETURN_IF_ERROR(onnx_file.MemberAsString("string_value", &onnx_file_name_));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("onnx file name = ") + onnx_file_name_).c_str());

  buffer_config_.max_connections = max_candidate_sequences_;
  TRITONSERVER_Error* lazy_err_ = GetInt64Parameter(parameters,
     "initial_buffer_size", buffer_config_.initial_buffer_size);
  if (lazy_err_ == nullptr) { // found lazy allocation config
    RETURN_ERROR_IF_FALSE(
      buffer_config_.initial_buffer_size >= 0,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("Invalid value for `initial_buffer_size`"));
    RETURN_IF_ERROR(GetInt64Parameter(parameters, "subsequent_buffer_size",
        buffer_config_.subsequent_buffer_size));
    RETURN_ERROR_IF_FALSE(
      buffer_config_.subsequent_buffer_size >= 0,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("Invalid value for `subsequent_buffer_size`"));
    RETURN_IF_ERROR(GetInt64Parameter(parameters, "buffer_alloc_threshold",
        buffer_config_.alloc_threshold));
    RETURN_ERROR_IF_FALSE(
      buffer_config_.alloc_threshold >= 0,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("Invalid value for `buffer_alloc_threshold`"));
    RETURN_IF_ERROR(GetInt64Parameter(parameters, "buffer_dealloc_threshold",
        buffer_config_.dealloc_threshold));
    RETURN_ERROR_IF_FALSE(
      buffer_config_.dealloc_threshold >= 0,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("Invalid value for `buffer_dealloc_threshold`"));
  }
  else { // if lazy alloc config not found, mimic old behavior
    LOG_MESSAGE(TRITONSERVER_LOG_WARN, TRITONSERVER_ErrorMessage(lazy_err_));
    TRITONSERVER_ErrorDelete(lazy_err_);
    buffer_config_.initial_buffer_size = max_candidate_sequences_;
    buffer_config_.subsequent_buffer_size = max_batch_size_;
    buffer_config_.alloc_threshold = max_batch_size_;
    buffer_config_.dealloc_threshold = max_candidate_sequences_;
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Buffer Config = ") + buffer_config_.to_string()).c_str());

  common::TritonJson::Value ort_ep;
  CHECK_IF_ERROR(parameters.MemberAsObject("ort_ep", &ort_ep), parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(ort_ep.MemberAsString("string_value", &ort_ep_name_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("ort ep name = ") + ort_ep_name_).c_str());

  common::TritonJson::Value state_pairs;
  CHECK_IF_ERROR(
      parameters.MemberAsObject("state_pairs", &state_pairs), parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(state_pairs.MemberAsString("string_value", &state_pairs_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("state tensor pairs = ") + state_pairs_).c_str());

  common::TritonJson::Value compute_prec;
  CHECK_IF_ERROR(
      parameters.MemberAsObject("compute_precision", &compute_prec),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(
        compute_prec.MemberAsString("string_value", &compute_precision_name_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("compute precision name = ") + compute_precision_name_)
          .c_str());

  common::TritonJson::Value store_states;
  CHECK_IF_ERROR(
      parameters.MemberAsObject("store_states_as_fp16", &store_states),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(
        store_states.MemberAsString("string_value", &store_states_as_fp16_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("store states as FP16 = ") + store_states_as_fp16_).c_str());

  common::TritonJson::Value max_connection_ratio;
  CHECK_IF_ERROR(
      parameters.MemberAsObject(
          "max_candidate_sequence_use_ratio", &max_connection_ratio),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(max_connection_ratio.MemberAsString(
        "string_value", &max_candidate_sequence_use_ratio_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("max candidate sequence use ratio = ") +
       max_candidate_sequence_use_ratio_)
          .c_str());

  common::TritonJson::Value metric_logging_frequency_seconds;
  CHECK_IF_ERROR(
      parameters.MemberAsObject(
          "metric_logging_frequency_seconds",
          &metric_logging_frequency_seconds),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(metric_logging_frequency_seconds.MemberAsString(
        "string_value", &metric_logging_frequency_seconds_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, (std::string("Metric Logging Frequency = ") +
                              metric_logging_frequency_seconds_)
                                 .c_str());

  IGNORE_ERROR(ParseLogLevel(parameters, "detailed_metrics_logging_level",
                              detailed_metrics_logging_level_));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Detailed Metric Logging Level = ") +
       std::to_string(detailed_metrics_logging_level_)).c_str());

  int64_t i64_log_lvl = -1;
  IGNORE_ERROR(ParseLogLevel(parameters, "batch_info_logging_level",
                              i64_log_lvl));
  if (i64_log_lvl >= 0) {
    batch_info_logging_level_ = INT2TRITON_LOG_LEVEL(i64_log_lvl);
  }
  else if (logging_level_ >= 0) {
    batch_info_logging_level_ = INT2TRITON_LOG_LEVEL(logging_level_);
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Batch Info Logging Level = ") +
       std::to_string(batch_info_logging_level_)).c_str());

  common::TritonJson::Value always_pad_to_max_batch;
  CHECK_IF_ERROR(
      parameters.MemberAsObject(
          "always_pad_to_max_batch", &always_pad_to_max_batch),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(always_pad_to_max_batch.MemberAsString(
        "string_value", &always_pad_to_max_batch_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Always Pad to Max Batch = ") + always_pad_to_max_batch_)
          .c_str());

  common::TritonJson::Value infer_end_requests;
  CHECK_IF_ERROR(
      parameters.MemberAsObject(
          "infer_end_requests", &infer_end_requests),
      parse_succeeded);
  if (parse_succeeded) {
    std::string str_infer_end_requests;
    IGNORE_ERROR(infer_end_requests.MemberAsString(
        "string_value", &str_infer_end_requests));
    if (str_infer_end_requests.compare("0") == 0) {
      infer_end_requests_ = false;
    }
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Infer End-requests = ") +
       std::to_string(infer_end_requests_)).c_str());

  common::TritonJson::Value enable_trt_caching;
  CHECK_IF_ERROR(
      parameters.MemberAsObject("enable_trt_caching", &enable_trt_caching),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(enable_trt_caching.MemberAsString(
        "string_value", &enable_trt_caching_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Enable TRT Engine Caching = ") + enable_trt_caching_)
          .c_str());

  common::TritonJson::Value trt_cache_dir;
  CHECK_IF_ERROR(
      parameters.MemberAsObject("trt_cache_dir", &trt_cache_dir),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(trt_cache_dir.MemberAsString("string_value", &trt_cache_dir_));
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRT Cache Location = ") + trt_cache_dir_).c_str());

#ifdef DEBUG_ERROR_INJECT
  common::TritonJson::Value error_inject_rate;
  std::string str_error_inject_rate;
  CHECK_IF_ERROR(
      parameters.MemberAsObject("error_inject_rate", &error_inject_rate),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(error_inject_rate.MemberAsString(
        "string_value", &str_error_inject_rate));
  }
  error_inject_rate_ = 20; // 20% of requests will get random error response
  if (!str_error_inject_rate.empty()) {
    try {
      int64_t r = static_cast<int64_t>(std::stoll(str_error_inject_rate));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Parsed Error Injection Rate = ") + std::to_string(r))
              .c_str());
      if (r < 0 || r >= 100) {
        throw std::invalid_argument("error_inject_rate");
      }
      error_inject_rate_ = r;
    }
    catch (...) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_WARN, "Invalid value in 'error_inject_rate'.");
    }
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Error Injection Rate = ") +
       std::to_string(error_inject_rate_)).c_str());
  const int rseed = rand()%100;
  std::cout << "Error Injection Random Seed = " << rseed << std::endl;
  srand(rseed);
#endif  // DEBUG_ERROR_INJECT

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state)
{
}

/**
 * Casting the int value to enum.
 * static_cast behavior can be undefined if the int value is out-of-range.
 */
inline OrtLoggingLevel Int2OrtLoggingLevel(const int l) {
  switch (l)
  {
  case static_cast<int>(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE):
    return OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE;
    break;
  case static_cast<int>(OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO):
    return OrtLoggingLevel::ORT_LOGGING_LEVEL_INFO;
    break;
  case static_cast<int>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING):
    return OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
    break;
  case static_cast<int>(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR):
    return OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR;
    break;
  case static_cast<int>(OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL):
    return OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL;
    break;

  default:
    return OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  }
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendState>
        backend_state(new BackendState());
  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;
    int value_int;
    if (cmdline.Find("ort-logging-level", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      RETURN_IF_ERROR(ParseIntValue(value_str, &value_int));
    }
    backend_state->m_OrtLoggingLevel = Int2OrtLoggingLevel(value_int);
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("ORT Logging Level: ") +
       std::to_string(backend_state->m_OrtLoggingLevel)).c_str());

  // Initialize  environment...one environment per process
  // Environment maintains thread pools and other state info
  if (backend_state->mOrtEnv == nullptr) {
    backend_state->mOrtEnv.reset(
      new Ort::Env(backend_state->m_OrtLoggingLevel, "Stateful Backend"));
  }
  // save the backend state to Triton, used by ModelState/ModelInstanceState
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(backend_state.get())));

  backend_state.release(); // saved the state in Triton, so release local ptr

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  auto backend_state = reinterpret_cast<BackendState*>(vstate);
  delete backend_state;
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));
  model_state->path_ =
      std::string(clocation) + std::string("/") + std::to_string(version);


  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  RETURN_IF_ERROR(model_state->InitModelState());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

#ifdef TRITON_ENABLE_GPU
  if (kind == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(device_id);
  }
#endif  // TRITON_ENABLE_GPU

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name + " (" +
       TRITONSERVER_InstanceGroupKindString(kind) + " device " +
       std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  // ONNX Model
  if (instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU) {
    device_id = -1;
  }
  instance_state->trt_onnx_model_ =
      std::unique_ptr<TrtOnnxModel>{new TrtOnnxModel()};
  instance_state->trt_onnx_model_->SetMinMaxBatch(
      1, model_state->max_batch_size_);
  std::string full_onnx_file_name =
      model_state->path_ + std::string("/") + model_state->onnx_file_name_;

  bool useTrtEp = true;
  if (model_state->ort_ep_name_.compare("cuda") == 0)
    useTrtEp = false;

  bool useFp16 = false;
  if (model_state->compute_precision_name_.compare("fp16") == 0)
    useFp16 = true;

  bool store_states_as_fp16 = false;
  if (model_state->store_states_as_fp16_.compare("1") == 0)
    store_states_as_fp16 = true;

  float max_connection_use_ratio = 0.9;
  try {
    if (!model_state->max_candidate_sequence_use_ratio_.empty()) {
      max_connection_use_ratio = static_cast<float>(
          std::stof(model_state->max_candidate_sequence_use_ratio_));
    }
  }
  catch (...) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "Invalid value in 'max_candidate_sequence_use_ratio'.");
  }
  // create a copy of the buffer config so that we don't modify the original
  BufferConfig buffer_config = model_state->buffer_config_;
  buffer_config.max_connections = static_cast<int64_t>(
      buffer_config.max_connections * max_connection_use_ratio);
  buffer_config.initial_buffer_size =
    std::min(buffer_config.initial_buffer_size, buffer_config.max_connections);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Tentative maximum connections allowed in the backend: ") +
       std::to_string(buffer_config.max_connections)).c_str());

  bool pad_to_max_batch = false;
  if (model_state->always_pad_to_max_batch_.compare("1") == 0)
    pad_to_max_batch = true;

  bool enable_trt_caching = false;
  if (model_state->enable_trt_caching_.compare("1") == 0)
    enable_trt_caching = true;

  int64_t metricLoggingFreq = 0;  // disabled
  try {
    if (!model_state->metric_logging_frequency_seconds_.empty()) {
      metricLoggingFreq = static_cast<int64_t>(
          std::stoll(model_state->metric_logging_frequency_seconds_));
    }
  }
  catch (...) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        "Invalid value in 'metric_logging_frequency_seconds'.");
  }

  std::stringstream ss_logs;
  try {
    std::string err_msg = instance_state->trt_onnx_model_->Prepare(
        ss_logs, model_state->backend_state_->mOrtEnv, full_onnx_file_name,
        model_state->state_pairs_, buffer_config, device_id,
        model_state->pref_batch_sizes_, model_state->input_tensors_,
        model_state->output_tensors_, model_state->start_tensor_name_, useTrtEp,
        useFp16, store_states_as_fp16, pad_to_max_batch, enable_trt_caching,
        model_state->trt_cache_dir_, model_state->logging_level_,
        metricLoggingFreq, model_state->detailed_metrics_logging_level_,
        model_state->max_sequence_idle_microseconds_);

    std::string str_logs = ss_logs.str();
    if (model_state->logging_level_ >= 0 && !str_logs.empty()) {
      if (str_logs.back() == '\n')
        str_logs.pop_back();  // remove last newline since Triton adds one
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, str_logs.c_str());
    }
    RETURN_ERROR_IF_FALSE(
        err_msg.empty(), TRITONSERVER_ERROR_INVALID_ARG, err_msg);

    instance_state->inference_tasks_.resize(model_state->max_batch_size_);
  }
  catch (const std::bad_alloc& e) {
    std::cerr << ss_logs.str() << std::endl;
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "MEMORY ALLOCATION ERROR:: "
        "Please try to reduce the number of instances and/or the "
        "max_candidate_sequences.");
  }
  catch (const std::exception& e) {
    std::cerr << ss_logs.str() << std::endl;
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNKNOWN, e.what());
  }

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

#ifdef TRITON_ENABLE_GPU
  if (instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(instance_state->DeviceId());
  }
#endif  // TRITON_ENABLE_GPU

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONBACKEND_ISPEC TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
#ifdef VERBOSE_LOG
  std::cout << "Executing the model \n" << std::endl;
#endif

  uint64_t exec_start_ns = 0, exec_end_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

#ifdef TRITON_ENABLE_GPU
  if (instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    cudaSetDevice(instance_state->DeviceId());
  }
#endif  // TRITON_ENABLE_GPU

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

#ifdef VERBOSE_LOG
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());
#endif

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);


  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // Pre-process the requests
  uint32_t infer_request_count = request_count;
  uint32_t end_request_count = 0;
  utils::PreProcessRequests(requests, instance_state->inference_tasks_,
                            request_count, responses, infer_request_count,
                            end_request_count, instance_state);

#ifdef DEBUG_ERROR_INJECT
  for (uint32_t r = 0; r < infer_request_count; ++r) {
    // inject error before calling InferTasks()
    std::cerr << "Injecting error for # " << request_id << std::endl;
    // TODO: don't send error for last segment until Triton fixes bug(?)
    if (input_end == 0 && (rand() % 100) < model_state->error_inject_rate_) {
      instance_state->inference_tasks_[r].err_msg = "RANDOM ERROR";
    }
  }
#endif  // DEBUG_ERROR_INJECT

  // From here on out, process only the requests that need inference
  // until responses are sent for them.

  uint64_t comp_start_ns = 0, comp_end_ns = 0;

  if (infer_request_count > 0) {
    LOG_MESSAGE(model_state->batch_info_logging_level_,
                (std::string("Running inference of ") +
                std::to_string(infer_request_count) + " requests on instance " +
                instance_state->Name() + ".").c_str());
    const bool send_response_early = true;
    void* vp_responses = send_response_early ?
                reinterpret_cast<void*>(responses.data()) : nullptr;
    std::string err_msg;
    try {
      // Now that we set all input / output, we can do the inferencing
      std::stringstream ss_logs;
      err_msg = instance_state->trt_onnx_model_->InferTasks(
          ss_logs, instance_state->inference_tasks_, infer_request_count,
          vp_responses, comp_start_ns,
          comp_end_ns);

      std::string str_logs = ss_logs.str();
      if (model_state->logging_level_ >= 0 && !str_logs.empty()) {
        if (str_logs.back() == '\n')
          str_logs.pop_back();  // remove last newline since Triton adds one
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, str_logs.c_str());
      }
    }
    // For execptions, only set the err_msg since we need to send the responses
    // for each request
    catch (const std::bad_alloc& e) {
      err_msg = std::string(
          "MEMORY ALLOCATION ERROR during Inference:: "
          "Please try to reduce the number of instances and/or the "
          "max_candidate_sequences.");
    }
    catch (const std::exception& e) {
      err_msg += std::string("ERROR during Inference: ") + std::string(e.what());
    }

    // if err_msg is not empty the entire batch sends error response
    // NOTE: stats are not reported in this scenario.
    if (!err_msg.empty()) {
      auto error =
          TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, err_msg.c_str());
      /*
          NOTE: It seems Triton expects an error ONLY IF we don't
          send/create response for any requests.
          - Since we already created the response objects, we have
            ownership of the requests/responses.
          - If we return with an error at this point, Triton hangs at exit,
            thinking that it has the responsibility to send the responses.
          - Instead we send the same response for each request,
            release the request objects, and then return a nullptr.
          - For individual request error, we need to send appropriate response
            for the particular requests. See next section.
      */
      for (uint32_t r = 0; r < request_count; ++r) {
        // No need to send a response if we already sent one e.g.
        // empty response for end requests that didn't need inference
        if (responses[r] != nullptr) {
          LOG_IF_ERROR(
              TRITONBACKEND_ResponseSend(
                  responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
              "failed to send error response");
        }

        TRITONBACKEND_Request* request = requests[r];
        LOG_IF_ERROR(
            TRITONBACKEND_RequestRelease(
                request, TRITONSERVER_REQUEST_RELEASE_ALL),
            "failed releasing request");
      }
      return nullptr;
    }

    if (!send_response_early) {
      // Send the responses
      triton::backend::stateful::utils::SendResponses(
        instance_state->inference_tasks_, infer_request_count,
        reinterpret_cast<void*>(responses.data()));
    }
  }

  SET_TIMESTAMP(exec_end_ns);

  // post-process the original batch
  for (uint32_t r = 0; r < request_count; ++r) {

    // Done with requests...
    //
    // We could have released each request as soon as we sent the
    // corresponding response. But for clarity we just release them all
    // here. Note that is something goes wrong when releasing a request
    // all we can do is log it... there is no response left to use to
    // report an error.

    TRITONBACKEND_Request* request = requests[r];
    // Report statistics for each request.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request,
            (responses[r] != nullptr &&
             instance_state->inference_tasks_[r].err_msg.empty()) /* success */,
            exec_start_ns, comp_start_ns, comp_end_ns, exec_end_ns),
        "failed reporting request statistics");

    // clear any error for next time
    instance_state->inference_tasks_[r].err_msg.clear();

    // NOTE: Whether there was an error for individual requests or not,
    // once we send the response, we need to release the request.
    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  // Report the entire batch statistics.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), request_count, exec_start_ns,
          comp_start_ns, comp_end_ns, exec_end_ns),
      "failed reporting batch request statistics");

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::stateful
