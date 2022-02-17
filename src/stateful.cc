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

#if (TRITONBACKEND_API_VERSION_MAJOR > 1) || \
    ((TRITONBACKEND_API_VERSION_MAJOR == 1) && \
     (TRITONBACKEND_API_VERSION_MINOR >= 6))
  #define TRITON_SUPPORTS_STRING_CORRID
#endif

namespace triton { namespace backend { namespace stateful {

//
// Simple backend that demonstrates the TRITONBACKEND API for a
// blocking backend with state tensors for sequence models.
// A blocking backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.
// The model must store the values to initialize the state tensors
// when new sequence starts.
//

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

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
  bool is_corrId_string;

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
  int64_t error_inject_rate_;

  std::vector<TritonTensorInfo> input_tensors_;
  std::vector<TritonTensorInfo> output_tensors_;

  std::string start_tensor_name_;
  std::string end_tensor_name_;
  std::shared_ptr<Ort::Env> mOrtEnv;


 private:
  ModelState(TRITONBACKEND_Model* triton_model);
};

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

// Initialize the model specific variables shared by all instances
TRITONSERVER_Error*
ModelState::InitModelState()
{
  bool parse_succeeded;
  logging_level_ =
      TRITON2INT_LOG_LEVEL(TRITONSERVER_LOG_INFO);  // default level info

  common::TritonJson::Value parameters;
  RETURN_IF_ERROR(model_config_.MemberAsObject("parameters", &parameters));

  common::TritonJson::Value logging_level;
  std::string str_logging_level;
  CHECK_IF_ERROR(
      parameters.MemberAsObject("logging_level", &logging_level),
      parse_succeeded);
  if (parse_succeeded) {
    IGNORE_ERROR(
        logging_level.MemberAsString("string_value", &str_logging_level));
  }
  if (str_logging_level.compare("NONE") == 0) {
    logging_level_ = -1;
  }
  if (str_logging_level.compare("VERBOSE") == 0) {
    logging_level_ = static_cast<int64_t>(TRITONSERVER_LOG_VERBOSE);
  }
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Logging Level = ") + str_logging_level +
       " :: " + std::to_string(logging_level_))
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
  buffer_config_.consequent_buffer_size = 100;
  buffer_config_.alloc_threshold = 32;
  buffer_config_.dealloc_threshold = 150;
  max_candidate_sequences_ = buffer_config_.max_connections;
  is_corrId_string = false;
  ort_ep_name_ = "trt";
  compute_precision_name_ = "fp16";
  store_states_as_fp16_ = "0";
  max_candidate_sequence_use_ratio_ = "0.9";
  trt_cache_dir_ = "/tmp";

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
        is_corrId_string = true;
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
  RETURN_IF_ERROR(GetInt64Parameter(parameters, "initial_buffer_size",
      buffer_config_.initial_buffer_size));
  RETURN_IF_ERROR(GetInt64Parameter(parameters, "consequent_buffer_size",
      buffer_config_.consequent_buffer_size));
  RETURN_IF_ERROR(GetInt64Parameter(parameters, "buffer_alloc_threshold",
      buffer_config_.alloc_threshold));
  RETURN_IF_ERROR(GetInt64Parameter(parameters, "buffer_dealloc_threshold",
      buffer_config_.dealloc_threshold));
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

  // Initialize  environment...one environment per process
  // Environment maintains thread pools and other state info
  // TODO: Use only one environment for all model instances within a server
  auto ort_log_level = logging_level_ > 0 ? ORT_LOGGING_LEVEL_VERBOSE
                                          : ORT_LOGGING_LEVEL_WARNING;
  mOrtEnv.reset(new Ort::Env(ort_log_level, "ONNX Stateful Model"));

  return nullptr;  // success
}


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
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
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
  model_state->buffer_config_.max_connections = static_cast<int64_t>(
      model_state->buffer_config_.max_connections * max_connection_use_ratio);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Max connectionss in the backend: ") +
       std::to_string(model_state->buffer_config_.max_connections)).c_str());

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
        ss_logs, model_state->mOrtEnv, full_onnx_file_name,
        model_state->state_pairs_, model_state->buffer_config_, device_id,
        model_state->pref_batch_sizes_, model_state->input_tensors_,
        model_state->output_tensors_, model_state->start_tensor_name_, useTrtEp,
        useFp16, store_states_as_fp16, pad_to_max_batch, enable_trt_caching,
        model_state->trt_cache_dir_, model_state->logging_level_,
        metricLoggingFreq, model_state->max_sequence_idle_microseconds_);

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
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        "MEMORY ALLOCATION ERROR:: "
        "Please try to reduce the number of instances and/or the "
        "max_candidate_sequences.");
  }
  catch (const std::exception& e) {
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

  uint64_t exec_start_ns = 0;
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
    if (!model_state->is_corrId_string) {
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
      instance_state->inference_tasks_[r].mInput[i_tensor++] = input_buffer;
    }

    instance_state->inference_tasks_[r].mCorrId = correlation_id;
    instance_state->inference_tasks_[r].mStart = input_start;
    instance_state->inference_tasks_[r].mEnd = input_end;

#ifdef VERBOSE_LOG
    std::cout << "Request Inputs: "
              << instance_state->inference_tasks_[r].mStart << ", ";
    std::cout << instance_state->inference_tasks_[r].mEnd << ", ";
    std::cout << instance_state->inference_tasks_[r].mCorrId << ", ";
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

      // Step 2. Get the output buffer. We request a buffer in CPU
      // memory but we have to handle any returned type. If we get
      // back a buffer in GPU memory we just fail the request.
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      size_t output_byte_size = output_tensor.type_size * output_tensor.vol;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, output_byte_size, &output_memory_type,
              &output_memory_type_id));

      if ((response == nullptr) ||
          (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to create output buffer in CPU memory"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create output buffer in CPU memory, error response "
             "sent")
                .c_str());
        continue;
      }

      instance_state->inference_tasks_[r].mOutput[i_tensor++] = output_buffer;
    }
#ifdef DEBUG_ERROR_INJECT
    // inject error before calling InferTasks()
    std::cerr << "Injecting error for # " << request_id << std::endl;
    // TODO: don't send error for last segment until Triton fixes bug
    if (input_end == 0 && (rand() % 100) < model_state->error_inject_rate_) {
      instance_state->inference_tasks_[r].err_msg = "RANDOM ERROR";
    }
#endif  // DEBUG_ERROR_INJECT
  }

  uint64_t comp_start_ns = 0, comp_end_ns = 0;

  const bool send_response_early = true;
  void* vp_responses = send_response_early ?
              reinterpret_cast<void*>(responses.data()) : nullptr;
  std::string err_msg;
  try {
    // Now that we set all input / output, we can do the inferencing
    std::stringstream ss_logs;
    err_msg = instance_state->trt_onnx_model_->InferTasks(
        ss_logs, instance_state->inference_tasks_, request_count,
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

  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);

  // if err_msg is not empty the entire batch sends error response
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
      LOG_IF_ERROR(
          TRITONBACKEND_ResponseSend(
              responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, error),
          "failed to send error response");

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
      instance_state->inference_tasks_, request_count,
      reinterpret_cast<void*>(responses.data()));
  }

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
