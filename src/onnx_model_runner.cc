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


#include <omp.h>
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "parserOnnxConfig.h"

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <random>
#include <string>
#include "NvInfer.h"

#include "onnx_model_runner.h"
#include "triton/backend/backend_common.h"
#include "response_util.h"

#include <sys/time.h>

// #define VERBOSE_OUTPUT

#undef RETURN_IF_CUDA_ERROR
#define RETURN_IF_CUDA_ERROR(status)                                           \
  do {                                                                         \
    auto cuda_err = (status);                                                  \
    if (cuda_err != 0) {                                                       \
      char str_buf[512];                                                       \
      snprintf(                                                                \
          str_buf, 512, "%s at %d in file %s\n", cudaGetErrorString(cuda_err), \
          __LINE__, __FILE__);                                                 \
      return std::string(str_buf);                                             \
    }                                                                          \
  } while (false)

#define RETURN_IF_FALSE(cond, str)                                           \
  do {                                                                       \
    if ((cond) == false) {                                                   \
      char str_buf[512];                                                     \
      snprintf(                                                              \
          str_buf, 512, "%s at %d in file %s\n", (str), __LINE__, __FILE__); \
      return std::string(str_buf);                                           \
    }                                                                        \
  } while (false)

#undef RETURN_IF_ERROR
#define RETURN_IF_ERROR(expression)        \
  do {                                     \
    std::string ret_string = (expression); \
    if (!ret_string.empty())               \
      return ret_string;                   \
  } while (false)

#define MY_ASSERT(cond)                                              \
  do {                                                               \
    if ((cond) == false) {                                           \
      char str_buf[512];                                             \
      snprintf(str_buf, 512, "%d in file %s\n", __LINE__, __FILE__); \
      std::cerr << str_buf << std::endl;                             \
    }                                                                \
  } while (false)

// ORT helper function to check for status
static inline std::string
CheckOrtStatus(OrtStatus* status)
{
  const OrtApi* p_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (status != NULL) {
    const char* msg = p_ort->GetErrorMessage(status);
    p_ort->ReleaseStatus(status);
    return std::string(msg);
  }
  return std::string();
}

/* get time in milli secs */
static inline double
get_time_msec()
{
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)1E3 * t.tv_sec + (double)1E-3 * t.tv_usec;
}

static inline void
append_to_trtexec_string(
    std::string& trtexec_string, std::string tensor_name, nvinfer1::Dims dims)
{
  trtexec_string.append(tensor_name);
  trtexec_string.append(":");
  for (int i = 0; i < dims.nbDims; ++i) {
    trtexec_string.append(std::to_string(dims.d[i]));
    if (i < dims.nbDims - 1) {
      trtexec_string.append("x");
    }
  }
  trtexec_string.append(",");
}

static inline std::unique_ptr<samplesCommon::ManagedBufferInternal>
allocate_tensor(
    const nvinfer1::Dims& dim, const bool use_gpu, const bool alloc_fp16)
{
  std::unique_ptr<samplesCommon::ManagedBufferInternal> buffer;
  try {
    buffer.reset(new samplesCommon::ManagedBufferInternal{
        alloc_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT,
        alloc_fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT});
    buffer->resize(dim, use_gpu);
  }
  catch (const std::exception& e) {
    std::cerr << "ERROR: Couldn't allocate memory for state tensors. "
              << e.what() << '\n';
    throw;
  }

  return buffer;
}

static inline void
guarded_resizeAll(
    samplesCommon::ManagedBufferInternal& buffer, const nvinfer1::Dims& dim,
    const bool use_gpu)
{
  try {
    buffer.resizeAll(dim, use_gpu);
  }
  catch (const std::exception& e) {
    std::cerr << "ERROR: Couldn't allocate memory for input/output tensors. "
              << e.what() << '\n';
    throw;
  }
}

// Prepares the model for inference by creating an execution context and
// allocating buffers.
//
// This function sets up the runtime for inference. This involves
// allocating buffers for the inputs and outputs including state tensors.
// This only needs to be called a single time.
std::string
TrtOnnxModel::Prepare(
    std::stringstream& ss_logs, std::shared_ptr<Ort::Env> ort_env,
    std::string model_path, std::string state_pairs, int maxNbConnections,
    int gpuId, std::vector<int64_t>& pref_batch_sizes,
    const std::vector<TritonTensorInfo>& input_tensors,
    const std::vector<TritonTensorInfo>& output_tensors,
    std::string reset_tensor_name, bool useTrtEp, bool useFp16,
    bool store_states_as_fp16, bool pad_to_max_batch, bool enable_trt_caching,
    std::string trt_cache_dir, int64_t logLevel, int64_t metricLoggingFreq,
    int64_t seq_timeout_us)
{
  mLogLevel = logLevel;
#ifdef VERBOSE_COUT
  log_stream_t& verbose_ss = std::cout;
  log_stream_t& info_ss = std::cout;
#else
  log_stream_t ss_null;  // use a local stream to consume the logs
  log_stream_t& verbose_ss = (mLogLevel > 0 ? ss_logs : ss_null);
  log_stream_t& info_ss = (mLogLevel >= 0 ? ss_logs : ss_null);
#endif


  mMetricLoggingFreqSeconds = metricLoggingFreq;
  RETURN_IF_FALSE(
      maxNbConnections >= mBatchDimMax,
      "Max connections should be larger than or equal to the max batch size");
  mPreferredBatchSizes = pref_batch_sizes;
  mMaxNbConnections = maxNbConnections;
  info_ss << "Maximum connections allowed in the backend: " << mMaxNbConnections
          << std::endl;
  mSequenceTimeoutMicroseconds = seq_timeout_us;
  // setup storage buffer chunks
  mNumChunks = 10; // ((max-min)/chunk_size)+1; // TODO: 
  // populate the available indices for the buffers
  for (int i = 0; i < mMaxNbConnections; ++i)
    mStoreAvailableIds.insert(mStoreAvailableIds.end(), i);
  mInputTritonTensorInfo = input_tensors;
  mOutputTritonTensorInfo = output_tensors;
  int64_t min_pref_batch = pref_batch_sizes[0];

  mStoreStatesAsFp16 = store_states_as_fp16;
  mGpuId = gpuId;
  mUseGpu = (mGpuId >= 0);
  if (mUseGpu) {
    mDeviceBindingString = std::string("Cuda");
    mUseTrtEp = useTrtEp;
  } else {
    mDeviceBindingString = std::string("Cpu");
    mUseTrtEp = false;
  }
  mPaddBatchSize = pad_to_max_batch;  // padding usually helps for cpu and gpu

  if (mUseGpu) {
    RETURN_IF_CUDA_ERROR(cudaStreamCreate(&mCudaStreamExe));
    RETURN_IF_CUDA_ERROR(cudaStreamCreate(&mCudaStreamCpy));
  }

  mEnv = std::move(ort_env);

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetInterOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_BASIC);

  // Add execution providers if needed
  if (mUseGpu) {
    std::string ort_status;
    if (mUseTrtEp) {
      OrtTensorRTProviderOptions trt_options{
          mGpuId,
          1,
          mCudaStreamExe,
          1000,                              // trt_max_partition_iterations
          1,                                 // trt_min_subgraph_size
          1 << 30,                           // max_workspace_size
          useFp16,                           // trt_fp16_enable
          0,                                 // trt_int8_enable
          nullptr,                           // trt_int8_calibration_table_name
          0,                                 // trt_int8_calibration_table_name
          0,                                 // trt_dla_enable
          0,                                 // trt_dla_core
          0,                                 // trt_dump_subgraphs
          enable_trt_caching ? 1 : 0,        // trt_engine_cache_enable
          enable_trt_caching ? trt_cache_dir.c_str()
                             : nullptr,  // trt_engine_cache_path
          0,                             // trt_engine_decryption_enable
          nullptr,                       // trt_engine_decryption_lib_path
          0                              // trt_force_sequential_engine_build
      };
      session_options.AppendExecutionProvider_TensorRT(trt_options);
    }
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = mGpuId;
    cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
    cuda_options.gpu_mem_limit = std::numeric_limits<size_t>::max();
    cuda_options.arena_extend_strategy = 0;
    cuda_options.do_copy_in_default_stream = 1;
    cuda_options.has_user_compute_stream = 1;
    cuda_options.user_compute_stream = mCudaStreamExe;
    cuda_options.default_memory_arena_cfg = nullptr;

    session_options.AppendExecutionProvider_CUDA(cuda_options);
  }

  // Read the model description
  std::vector<std::string> input_state_names, output_state_names;

  //*************************************************************************
  // Create session and load model into memory
  verbose_ss << "Using Onnxruntime C++ API, initializing on device = " << mGpuId
             << std::endl;
  mSession.reset(new Ort::Session(*mEnv, model_path.c_str(), session_options));
  Ort::AllocatorWithDefaultOptions allocator;

  // Read state tensors
  // To add new state tensors with a specific name pattern add another line
  // below with an additional InitTensorNames call.
  auto metadata = mSession->GetModelMetadata();
  std::string producer_name(metadata.GetProducerName(allocator));
  std::string graph_name(metadata.GetGraphName(allocator));
  std::string description(metadata.GetDescription(allocator));
  verbose_ss << "Producer name = " << producer_name << std::endl;
  verbose_ss << "Graph name = " << graph_name << std::endl;
  verbose_ss << "Description = " << description << std::endl;
  if (state_pairs.empty()) {
    std::string graph_desc(metadata.GetGraphDescription(allocator));
    RETURN_IF_FALSE(
        StateTensor::InitTensorNames(
            graph_desc, "InputState", "OutputState", input_state_names,
            output_state_names) == 0,
        "Error while reading the state tensors");
    RETURN_IF_FALSE(
        StateTensor::InitTensorNames(
            graph_desc, "PastValueInput", "PastValueOutput", input_state_names,
            output_state_names) == 0,
        "Error while reading the state tensors");
  } else {
    RETURN_IF_FALSE(
        StateTensor::InitTensorNames(
            state_pairs, "", "", input_state_names, output_state_names) == 0,
        "Error while reading the state tensors");
  }

  verbose_ss << "Read " << input_state_names.size()
             << " state vectors from the description file " << std::endl;
  RETURN_IF_FALSE(
      input_state_names.size() > 0, "Could not find any state tensors");


  Ort::MemoryInfo memory_info_gpu{"Cuda", OrtDeviceAllocator, 0,
                                  OrtMemTypeDefault};

  size_t num_input_nodes = mSession->GetInputCount();
  verbose_ss << "Number of inputs = " << num_input_nodes << std::endl;
  std::string trtexec_string("trtexec --onnx=");
  trtexec_string.append(model_path);
  trtexec_string.append(" --fp16 --explicitBatch --workspace=8192");

  std::string trtexec_max_string;
  std::string trtexec_min_string;
  bool is_truncated_model = false;

  // Iterate over all input nodes
  // We assume that the tensor dimension of 1 is the batch dimension for now.
  for (size_t i = 0; i < num_input_nodes; i++) {
    char* input_name = mSession->GetInputName(i, allocator);
    std::string input_name_str(input_name);

    // print input node types
    Ort::TypeInfo type_info = mSession->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();

    // print input shapes/dims
    size_t num_dims = tensor_info.GetDimensionsCount();

    verbose_ss << "Input " << i << std::endl;
    verbose_ss << "    Name = " << input_name << std::endl;
    verbose_ss << "    type = " << type << std::endl;
    verbose_ss << "    num_dims = " << num_dims << std::endl;

    std::vector<int64_t> input_node_dims = tensor_info.GetShape();
    nvinfer1::Dims dimsMin;
    nvinfer1::Dims dimsMax;
    dimsMax.nbDims = num_dims;
    size_t dynamic_dim = INVALID_DIM;
    for (size_t j = 0; j < num_dims; j++) {
      verbose_ss << "    dim " << j << " = " << input_node_dims[j] << std::endl;
      dimsMax.d[j] = input_node_dims[j];
      if (input_node_dims[j] == -1)
        dynamic_dim = j;
    }

    size_t input_tensor_size = 1;
    TritonTensorInfo* triton_tensor = GetInputTensor(input_name_str);
    if (triton_tensor != nullptr) {
      RETURN_IF_FALSE(
          triton_tensor->batch_dim < INVALID_DIM,
          "Cannot infer the batch dimension based on the Triton model "
          "configuration");
      RETURN_IF_FALSE(
          mNumInputs < MAX_IO_NUM,
          "Max number of inputs supported by the backend is exceeded");

      for (size_t j = 0; j < num_dims; ++j) {
        input_node_dims[j] = dimsMax.d[j] = triton_tensor->shape[j];
      }
      input_node_dims[triton_tensor->batch_dim] =
          dimsMax.d[triton_tensor->batch_dim] = mBatchDimMax;

      dimsMin = dimsMax;
      dimsMin.d[triton_tensor->batch_dim] = min_pref_batch;

      input_tensor_size = samplesCommon::volume(dimsMax);
      guarded_resizeAll(mInputs[triton_tensor->idx], dimsMax, mUseGpu);
      verbose_ss << "Set dims:" << dimsMax
                 << ", tensor size = " << input_tensor_size << std::endl;

      mOrtTensors[input_name_str] = {mInputs[triton_tensor->idx].data(mUseGpu),
                                     input_node_dims,
                                     num_dims,
                                     triton_tensor->batch_dim,
                                     1,
                                     triton_tensor->type_size,
                                     type};
      mNumInputs++;
    } else if (
        samplesCommon::toLower(reset_tensor_name)
            .compare(samplesCommon::toLower(input_name_str)) == 0) {
      RETURN_IF_FALSE(
          dynamic_dim == 0,
          "Batch dimension should always be the first dimension for reset "
          "sequence tensor");
      size_t batch_dim = dynamic_dim;
      size_t type_size = 1;
      is_truncated_model = true;
      input_node_dims[0] = dimsMax.d[0] = mBatchDimMax;
      input_tensor_size = samplesCommon::volume(dimsMax);

      dimsMin = dimsMax;
      dimsMin.d[0] = min_pref_batch;

      guarded_resizeAll(mInputReset, dimsMax, mUseGpu);
      verbose_ss << "Set dims:" << dimsMax
                 << ", tensor size = " << input_tensor_size << std::endl;

      mOrtTensors[input_name_str] = {mInputReset.data(mUseGpu),
                                     input_node_dims,
                                     num_dims,
                                     batch_dim,
                                     1,
                                     type_size,
                                     type};
    } else {
      int state_idx = StateTensor::GetIdx(input_name_str, input_state_names);
      RETURN_IF_FALSE(
          state_idx >= 0, (std::string("Model configuration does not match "
                                       "ONNX model for tensor name: ") +
                           input_name_str)
                              .c_str());
      RETURN_IF_FALSE(
          dynamic_dim < INVALID_DIM,
          "No dynamic dimension in the ONNX model for a state tensor");
      RETURN_IF_FALSE(
          type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
          "Only FP32 precision is supported for the state tensors");
      size_t batch_dim = dynamic_dim;
      size_t type_size = 4;
      input_node_dims[batch_dim] = dimsMax.d[batch_dim] = mBatchDimMax;
      input_tensor_size = samplesCommon::volume(dimsMax);

      dimsMin = dimsMax;
      dimsMin.d[batch_dim] = min_pref_batch;

      mStates.emplace_back(allocate_tensor(dimsMax, mUseGpu, false));
      mStateTensors.emplace_back(
          input_state_names[state_idx], output_state_names[state_idx],
          mStates.back()->data(mUseGpu), dimsMax, nvinfer1::DataType::kFLOAT,
          batch_dim);
      verbose_ss << "Set dims:" << dimsMax
                 << ", tensor size = " << input_tensor_size << std::endl;
      mOrtTensors[input_name_str] = {mStates.back()->data(mUseGpu),
                                     input_node_dims,
                                     num_dims,
                                     batch_dim,
                                     1,
                                     type_size,
                                     type};
    }

    append_to_trtexec_string(trtexec_max_string, input_name_str, dimsMax);
    append_to_trtexec_string(trtexec_min_string, input_name_str, dimsMin);
  }

  RETURN_IF_FALSE(
      is_truncated_model,
      "The model should include ResetSequence tensor to reset the state "
      "tensors");

  size_t num_overridable_init_nodes =
      mSession->GetOverridableInitializerCount();
  verbose_ss << "Number of overridable initializers = "
             << num_overridable_init_nodes << std::endl;

  for (size_t i = 0; i < num_overridable_init_nodes; i++) {
    char* input_name = mSession->GetOverridableInitializerName(i, allocator);
    std::string input_name_str(input_name);

    // print input node types
    Ort::TypeInfo type_info = mSession->GetOverridableInitializerTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();

    // print input shapes/dims
    size_t num_dims = tensor_info.GetDimensionsCount();

    verbose_ss << "Input " << i << std::endl;
    verbose_ss << "    Name = " << input_name << std::endl;
    verbose_ss << "    type = " << type << std::endl;
    verbose_ss << "    num_dims = " << num_dims << std::endl;

    std::vector<int64_t> input_node_dims = tensor_info.GetShape();
    nvinfer1::Dims dimsMin;
    nvinfer1::Dims dimsMax;
    dimsMax.nbDims = num_dims;
    size_t dynamic_dim = INVALID_DIM;
    for (size_t j = 0; j < num_dims; j++) {
      verbose_ss << "    dim " << j << " = " << input_node_dims[j] << std::endl;
      dimsMax.d[j] = input_node_dims[j];
      if (input_node_dims[j] == -1)
        dynamic_dim = j;
    }

    // NOTE: The overridable initializer are only state tensors, however,
    // keeping the first two if branches in case things change in future.
    size_t input_tensor_size = 1;
    TritonTensorInfo* triton_tensor = GetInputTensor(input_name_str);
    if (triton_tensor != nullptr) {
      RETURN_IF_FALSE(
          triton_tensor->batch_dim < INVALID_DIM,
          "Cannot infer the batch dimension based on the Triton model "
          "configuration");
      RETURN_IF_FALSE(
          mNumInputs < MAX_IO_NUM,
          "Max number of inputs supported by the backend is exceeded");

      for (size_t j = 0; j < num_dims; ++j) {
        input_node_dims[j] = dimsMax.d[j] = triton_tensor->shape[j];
      }
      input_node_dims[triton_tensor->batch_dim] =
          dimsMax.d[triton_tensor->batch_dim] = mBatchDimMax;

      dimsMin = dimsMax;
      dimsMin.d[triton_tensor->batch_dim] = min_pref_batch;

      input_tensor_size = samplesCommon::volume(dimsMax);
      guarded_resizeAll(mInputs[triton_tensor->idx], dimsMax, mUseGpu);
      verbose_ss << "Set dims:" << dimsMax
                 << ", tensor size = " << input_tensor_size << std::endl;

      mOrtTensors[input_name_str] = {mInputs[triton_tensor->idx].data(mUseGpu),
                                     input_node_dims,
                                     num_dims,
                                     triton_tensor->batch_dim,
                                     1,
                                     triton_tensor->type_size,
                                     type};
      mNumInputs++;
    } else if (
        samplesCommon::toLower(reset_tensor_name)
            .compare(samplesCommon::toLower(input_name_str)) == 0) {
      RETURN_IF_FALSE(
          dynamic_dim == 0,
          "Batch dimension should always be the first dimension for reset "
          "sequence tensor");
      size_t batch_dim = dynamic_dim;
      size_t type_size = 1;
      is_truncated_model = true;
      input_node_dims[0] = dimsMax.d[0] = mBatchDimMax;
      input_tensor_size = samplesCommon::volume(dimsMax);

      dimsMin = dimsMax;
      dimsMin.d[0] = min_pref_batch;

      guarded_resizeAll(mInputReset, dimsMax, mUseGpu);
      verbose_ss << "Set dims:" << dimsMax
                 << ", tensor size = " << input_tensor_size << std::endl;

      mOrtTensors[input_name_str] = {mInputReset.data(mUseGpu),
                                     input_node_dims,
                                     num_dims,
                                     batch_dim,
                                     1,
                                     type_size,
                                     type};
    } else {
      int state_idx = StateTensor::GetIdx(input_name_str, input_state_names);
      RETURN_IF_FALSE(
          state_idx >= 0, (std::string("Model configuration does not match "
                                       "ONNX model for tensor name: ") +
                           input_name_str)
                              .c_str());
      RETURN_IF_FALSE(
          dynamic_dim < INVALID_DIM,
          "No dynamic dimension in the ONNX model for a state tensor");
      RETURN_IF_FALSE(
          type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
          "Only FP32 precision is supported for the state tensors");
      size_t batch_dim = dynamic_dim;
      size_t type_size = 4;
      input_node_dims[batch_dim] = dimsMax.d[batch_dim] = mBatchDimMax;
      input_tensor_size = samplesCommon::volume(dimsMax);

      dimsMin = dimsMax;
      dimsMin.d[batch_dim] = min_pref_batch;

      mStates.emplace_back(allocate_tensor(dimsMax, mUseGpu, false));
      mStateTensors.emplace_back(
          input_state_names[state_idx], output_state_names[state_idx],
          mStates.back()->data(mUseGpu), dimsMax, nvinfer1::DataType::kFLOAT,
          batch_dim);
      verbose_ss << "Set dims:" << dimsMax
                 << ", tensor size = " << input_tensor_size << std::endl;
      mOrtTensors[input_name_str] = {mStates.back()->data(mUseGpu),
                                     input_node_dims,
                                     num_dims,
                                     batch_dim,
                                     1,
                                     type_size,
                                     type};
    }

    append_to_trtexec_string(trtexec_max_string, input_name_str, dimsMax);
    append_to_trtexec_string(trtexec_min_string, input_name_str, dimsMin);
  }


  size_t num_output_nodes = mSession->GetOutputCount();
  verbose_ss << "Number of outputs = " << num_output_nodes << std::endl;
  std::vector<std::pair<std::string, void*>> matchingOutputStates{};

  for (size_t i = 0; i < num_output_nodes; i++) {
    char* output_name = mSession->GetOutputName(i, allocator);

    // print input node types
    Ort::TypeInfo type_info = mSession->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();

    // print input shapes/dims
    size_t num_dims = tensor_info.GetDimensionsCount();

    verbose_ss << "Output " << i << std::endl;
    verbose_ss << "    Name = " << output_name << std::endl;
    verbose_ss << "    type = " << type << std::endl;
    verbose_ss << "    num_dims = " << num_dims << std::endl;

    std::vector<int64_t> output_node_dims = tensor_info.GetShape();
    nvinfer1::Dims dimsMin;
    nvinfer1::Dims dimsMax;
    dimsMax.nbDims = num_dims;
    size_t dynamic_dim = INVALID_DIM;
    for (size_t j = 0; j < num_dims; j++) {
      verbose_ss << "    dim " << j << " = " << output_node_dims[j]
                 << std::endl;
      dimsMax.d[j] = output_node_dims[j];
      if (output_node_dims[j] == -1)
        dynamic_dim = j;
    }

    std::string output_name_str(output_name);
    // automatically choose one of the output names
    TritonTensorInfo* triton_tensor = GetOutputTensor(output_name_str);
    if (triton_tensor != nullptr) {
      RETURN_IF_FALSE(
          triton_tensor->batch_dim < INVALID_DIM,
          "Cannot infer the batch dimension based on the Triton model "
          "configuration");
      RETURN_IF_FALSE(
          mNumOutputs < MAX_IO_NUM,
          "Max number of outputs supported by the backend is exceeded");
      for (size_t j = 0; j < num_dims; ++j) {
        output_node_dims[j] = dimsMax.d[j] = triton_tensor->shape[j];
      }
      output_node_dims[triton_tensor->batch_dim] =
          dimsMax.d[triton_tensor->batch_dim] = mBatchDimMax;

      dimsMin = dimsMax;
      dimsMin.d[triton_tensor->batch_dim] = min_pref_batch;

      size_t output_tensor_size = samplesCommon::volume(dimsMax);
      verbose_ss << "    Set dims:" << dimsMax
                 << ", tensor size = " << output_tensor_size << std::endl;

      guarded_resizeAll(mOutputs[triton_tensor->idx], dimsMax, mUseGpu);
      verbose_ss << "Resize all finished" << std::endl;

      mOrtTensors[output_name_str] = {
          mOutputs[triton_tensor->idx].data(mUseGpu),
          output_node_dims,
          num_dims,
          triton_tensor->batch_dim,
          0,
          triton_tensor->type_size,
          type};
      verbose_ss << "Ort tensor is set  " << std::endl;
      mNumOutputs++;
    } else {
      int state_idx = StateTensor::GetIdx(output_name_str, output_state_names);
      RETURN_IF_FALSE(
          state_idx >= 0, (std::string("Model configuration does not match "
                                       "ONNX model for tensor name: ") +
                           output_name_str)
                              .c_str());
      RETURN_IF_FALSE(
          dynamic_dim < INVALID_DIM,
          "No dynamic dimension in the ONNX model for a state tensor");
      RETURN_IF_FALSE(
          type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
          "Only FP32 precision is supported for the state tensors");
      size_t batch_dim = dynamic_dim;
      size_t type_size = 4;
      output_node_dims[batch_dim] = dimsMax.d[batch_dim] = mBatchDimMax;
      size_t output_tensor_size = samplesCommon::volume(dimsMax);

      dimsMin = dimsMax;
      dimsMin.d[batch_dim] = min_pref_batch;

      mStates.emplace_back(allocate_tensor(dimsMax, mUseGpu, false));
      matchingOutputStates.emplace_back(
          output_name_str, mStates.back()->data(mUseGpu));

      verbose_ss << "    Set dims:" << dimsMax
                 << ", tensor size = " << output_tensor_size << std::endl;
      mOrtTensors[output_name_str] = {mStates.back()->data(mUseGpu),
                                      output_node_dims,
                                      num_dims,
                                      batch_dim,
                                      0,
                                      type_size,
                                      type};
    }

    // append_to_trtexec_string(trtexec_max_string, output_name_str, dimsMax);
    // append_to_trtexec_string(trtexec_min_string, output_name_str, dimsMin);
  }

  verbose_ss << "Reading output tensors is finished " << std::endl;
  verbose_ss << "String for testing with trtexec min shape:" << std::endl;
  verbose_ss << trtexec_min_string << std::endl;
  verbose_ss << "String for testing with trtexec max shape:" << std::endl;
  verbose_ss << trtexec_max_string << std::endl;

  trtexec_string.append(" --minShapes=");
  trtexec_string.append(
      trtexec_min_string.substr(0, trtexec_min_string.length() - 1));
  trtexec_string.append(" --maxShapes=");
  trtexec_string.append(
      trtexec_max_string.substr(0, trtexec_max_string.length() - 1));
  trtexec_string.append(" --optShapes=");
  trtexec_string.append(
      trtexec_max_string.substr(0, trtexec_max_string.length() - 1));
  verbose_ss << "trtexec string:" << std::endl;
  verbose_ss << trtexec_string << std::endl;

  for (auto& iTensor : mOrtTensors) {
    verbose_ss << "Tensor Name: " << iTensor.first << std::endl;
    iTensor.second.Print(verbose_ss);
    verbose_ss << std::endl;
  }

  mStoredStates.reserve(mNumChunks);
  std::vector<std::unique_ptr<samplesCommon::ManagedBufferInternal>> chunk0;
  for (auto& iTensor : mStateTensors) {
    // allocate the storage space for the storage buffer
    if (maxNbConnections != -1) {
      auto dimsMaxCon = iTensor.mDim;
      dimsMaxCon.d[iTensor.mBatchDim] = maxNbConnections;
      chunk0.emplace_back(
          allocate_tensor(dimsMaxCon, mUseGpu, mStoreStatesAsFp16));
      iTensor.mStoreBuffer.resize(mNumChunks);
      iTensor.mStoreBuffer[0] = chunk0.back()->data(mUseGpu);
    }
    // find the matching output tensor
    for (auto& jTensor : matchingOutputStates) {
      iTensor.AddOutputNameIfMatch(jTensor.first, jTensor.second);
    }
  }
  mStoredStates.push_back(std::move(chunk0));

  verbose_ss << "State Tensors " << std::endl;
  for (const auto& iTensor : mStateTensors) {
    iTensor.printStateTensors(verbose_ss);
    verbose_ss << std::endl;
  }

  try {
    if (maxNbConnections != -1) {
      mNumStates = mStateTensors.size();
      // allocate device buffers for storing/restoring the internal states
      mStorageBufferHost.resize(mNumChunks);
      mStorageBufferHost[0].resize(mNumStates); // only resize chunk0 for now
      mInputStateBufferHost.resize(mNumStates);
      mOutputStateBufferHost.resize(mNumStates);
      mBufferSizeXHost.resize(mNumStates);
      mBufferSizeYHost.resize(mNumStates);
      mStoreIdHost.resize(mBatchDimMax);
      mCorrIdToDelete.reserve(maxNbConnections);
      for (int i = 0; i < mNumStates; ++i) {
        mStorageBufferHost[0][i] = mStateTensors[i].mStoreBuffer[0];
        mInputStateBufferHost[i] =
            reinterpret_cast<float*>(mStateTensors[i].mInputBuffer);
        mOutputStateBufferHost[i] =
            reinterpret_cast<float*>(mStateTensors[i].mOutputBuffer);

        mBufferSizeXHost[i] = 1;
        mBufferSizeYHost[i] = 1;

        for (int j = 0; j < mStateTensors[i].mBatchDim; ++j)
          mBufferSizeXHost[i] *= mStateTensors[i].mDim.d[j];

        for (int j = mStateTensors[i].mBatchDim + 1;
             j < mStateTensors[i].mDim.nbDims; ++j)
          mBufferSizeYHost[i] *= mStateTensors[i].mDim.d[j];

        verbose_ss << "State tensor = " << mStateTensors[i].mInputName
                   << ", SizeX = " << mBufferSizeXHost[i]
                   << ", SizeY = " << mBufferSizeYHost[i] << std::endl;
      }

      if (mUseGpu) {
        mStorageBufferDevicePtrOnHost.resize(mNumChunks, nullptr);
        RETURN_IF_CUDA_ERROR(cudaMalloc(
            reinterpret_cast<void**>(&mStorageBufferDevice),
            mNumChunks * sizeof(void**)));
        // allocate pointers for all chunks
        for (int i=0; i<mNumChunks; ++i) {
          RETURN_IF_CUDA_ERROR(cudaMalloc(
              reinterpret_cast<void**>(&mStorageBufferDevicePtrOnHost[i]),
              mNumStates * sizeof(void*)));
        }
        RETURN_IF_CUDA_ERROR(cudaMalloc(
            reinterpret_cast<void**>(&mInputStateBufferDevice),
            mNumStates * sizeof(float*)));
        RETURN_IF_CUDA_ERROR(cudaMalloc(
            reinterpret_cast<void**>(&mOutputStateBufferDevice),
            mNumStates * sizeof(float*)));
        RETURN_IF_CUDA_ERROR(cudaMalloc(
            reinterpret_cast<void**>(&mBufferSizeXDevice),
            mNumStates * sizeof(int)));
        RETURN_IF_CUDA_ERROR(cudaMalloc(
            reinterpret_cast<void**>(&mBufferSizeYDevice),
            mNumStates * sizeof(int)));
        RETURN_IF_CUDA_ERROR(cudaMalloc(
            reinterpret_cast<void**>(&mStoreIdDevice),
            mBatchDimMax * sizeof(int)));

        // copy the storage buffer pointers
        RETURN_IF_CUDA_ERROR(cudaMemcpy(
            mStorageBufferDevice, mStorageBufferDevicePtrOnHost.data(),
            mNumChunks * sizeof(void**), cudaMemcpyHostToDevice));
        // only copy the first chunk's pointers for now
        // as we allocate new chunks, we need to copy the pointers as well
        RETURN_IF_CUDA_ERROR(cudaMemcpy(
            mStorageBufferDevicePtrOnHost[0], mStorageBufferHost[0].data(),
            mNumStates * sizeof(void*), cudaMemcpyHostToDevice));
        RETURN_IF_CUDA_ERROR(cudaMemcpy(
            mInputStateBufferDevice, mInputStateBufferHost.data(),
            mNumStates * sizeof(float*), cudaMemcpyHostToDevice));
        RETURN_IF_CUDA_ERROR(cudaMemcpy(
            mOutputStateBufferDevice, mOutputStateBufferHost.data(),
            mNumStates * sizeof(float*), cudaMemcpyHostToDevice));
        RETURN_IF_CUDA_ERROR(cudaMemcpy(
            mBufferSizeXDevice, mBufferSizeXHost.data(),
            mNumStates * sizeof(int), cudaMemcpyHostToDevice));
        RETURN_IF_CUDA_ERROR(cudaMemcpy(
            mBufferSizeYDevice, mBufferSizeYHost.data(),
            mNumStates * sizeof(int), cudaMemcpyHostToDevice));
      }
    }
  }
  catch (const std::exception& e) {
    std::cerr << "Couldn't allocate memory for buffer pointers." << std::endl;
    throw;
  }

  // make warmup runs to initialize the engines (crucial for all dims for TRT)
  {
    verbose_ss << "Warmup run for batch = " << mBatchDimMax << std::endl;
    double start = get_time_msec();
    Ort::IoBinding bindings(*mSession);
    setBindings(mBatchDimMax, bindings);
    mSession->Run(mRunOptions, bindings);
    double end = get_time_msec();
    verbose_ss << "Time required to run: " << (end - start) << " milliseconds"
                << std::endl;
  }

  // no need to generate batch=1 engine if we always padd to preferred batch
  // sizes
  if (mPaddBatchSize == false) {
    verbose_ss << "Warmup run for batch = " << mBatchDimMin << std::endl;
    double start = get_time_msec();
    Ort::IoBinding bindings(*mSession);
    setBindings(mBatchDimMin, bindings);
    mSession->Run(mRunOptions, bindings);
    double end = get_time_msec();
    verbose_ss << "Time required to run: " << (end - start) << " milliseconds"
                << std::endl;
  }

  std::sort(std::begin(mPreferredBatchSizes), std::end(mPreferredBatchSizes));
  for (int64_t pref_batch : mPreferredBatchSizes) {
    if (pref_batch == mBatchDimMax) continue; // already ran max batch
    verbose_ss << "Warmup run for batch = " << pref_batch << std::endl;
    double start = get_time_msec();
    Ort::IoBinding bindings(*mSession);
    setBindings(pref_batch, bindings);
    mSession->Run(mRunOptions, bindings);
    double end = get_time_msec();
    verbose_ss << "Time required to run: " << (end - start) << " milliseconds"
                << std::endl;
  }

  mRunOptions.SetRunLogSeverityLevel(2);
  // return empty string if we reached here
  return std::string();
}

void
TrtOnnxModel::setBindings(int batchsize, Ort::IoBinding& iobindings)
{
  Ort::MemoryInfo memory_info_gpu{mDeviceBindingString.c_str(),
                                  OrtDeviceAllocator, 0, OrtMemTypeDefault};

  for (const auto& tensor_info : mOrtTensors) {
    std::string tensor_name = tensor_info.first;
    std::vector<int64_t> tensor_shape = tensor_info.second.shape;
    void* tensor_data = tensor_info.second.data;
    size_t num_dims = tensor_info.second.num_dims;
    size_t batch_dim = tensor_info.second.batch_dim;
    int is_input = tensor_info.second.is_input;
    int type_size = tensor_info.second.type_size;
    ONNXTensorElementDataType type = tensor_info.second.type;

    tensor_shape[batch_dim] = batchsize;
    size_t tensor_size = 1;
    for (size_t i = 0; i < num_dims; ++i) tensor_size *= tensor_shape[i];

    auto ort_tensor_value = Ort::Value::CreateTensor(
        memory_info_gpu, tensor_data, tensor_size * type_size,
        tensor_shape.data(), num_dims, type);

    if (is_input)
      iobindings.BindInput(tensor_name.c_str(), ort_tensor_value);
    else
      iobindings.BindOutput(tensor_name.c_str(), ort_tensor_value);
  }
}

void launchRestoreGPUKernel_FP32(
    float*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream);
void launchStoreGPUKernel_FP32(
    float*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream);

void launchRestoreGPUKernel_FP16(
    __half*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream);
void launchStoreGPUKernel_FP16(
    __half*** storage, float** states, int* sizesX, int* sizesY, int numStates,
    int* storeids, int batchSize, int batchStride, cudaStream_t stream);

std::string
TrtOnnxModel::prepareDeviceStoreIds(
    log_stream_t& verbose_ss, std::vector<InferenceTask>& inferenceTasks,
    int batchSize)
{
  /**
   * Using the timeout as a machanism to release allocated storage
   * for correlation IDs that are in use.
   * 1. Check the last usage timestamp from the map to find the corrId
   *    to remove.
   * 2. Remove those corrIds from the map, and add the storage index
   *    to the available index list.
   */
  time_point_t now = NOW();
  // check the IDs in use for timeout
  for (auto item : mStoreIdMap) {
    int64_t time_lapsed = DURATION_MICRO(now - item.second.second).count();
    if (time_lapsed > mSequenceTimeoutMicroseconds) {
      // timeout ocurred since this sequence ID was used
      // remove its allocation of storage buffer
      mCorrIdToDelete.push_back(item.first);
    }
  }
  for (auto corrId : mCorrIdToDelete) {
    verbose_ss << "Timeout ocurred for CorrID : " << corrId << std::endl;
    mStoreAvailableIds.insert(mStoreIdMap[corrId].first);
    mStoreIdMap.erase(corrId);
  }
  mCorrIdToDelete.clear();  // resets size but not capacity

  for (int i = 0; i < batchSize; ++i) {
    auto corrId = inferenceTasks[i].mCorrId;
    auto findId = mStoreIdMap.find(corrId);
    if (findId == mStoreIdMap.end()) {
      if (mStoreAvailableIds.empty()) {
        inferenceTasks[i].err_msg = "Too many simultaneous connections";
        mStoreIdHost[i] = -1;  // no empty slots
        continue;
      }
      const int idx_to_use =
          *mStoreAvailableIds.begin();  // get the first available
      mStoreAvailableIds.erase(mStoreAvailableIds.begin());  // then remove it
      mStoreIdMap[corrId] = make_pair(idx_to_use, now);
      mStoreIdHost[i] = idx_to_use;
#ifdef VERBOSE_STORAGE_ACTIVITY
      verbose_ss << "Assigning new index for storage: " << idx_to_use
                 << std::endl;
#endif
    } else {
      findId->second.second = now;  // update timestamp
      mStoreIdHost[i] = findId->second.first;
#ifdef VERBOSE_STORAGE_ACTIVITY
      verbose_ss << "Found old index for storage: " << mStoreIdHost[i]
                 << std::endl;
#endif
    }
  }
  if (mUseGpu)
    RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        mStoreIdDevice, mStoreIdHost.data(), batchSize * sizeof(int),
        cudaMemcpyHostToDevice, mCudaStreamExe));
  return std::string();
}

void
TrtOnnxModel::storeStates_CPU_FP32(
    std::vector<InferenceTask>& inferenceTasks, int batchSize, int batchStride)
{
  for (int i = 0; i < mNumStates; ++i) {
    float* storageBuffer = static_cast<float*>(mStorageBufferHost[0][i]); // TODO: remove 0
    size_t sizeX = mBufferSizeXHost[i];
    size_t sizeY = mBufferSizeYHost[i];
    for (int j = 0; j < batchSize; ++j) {
      if (!inferenceTasks[j].err_msg.empty())
        continue;  // don't store if error
      if (mStoreIdHost[j] < 0)
        continue;  // no empty slots
      for (size_t ix = 0; ix < sizeX; ++ix) {
        for (size_t iy = 0; iy < sizeY; ++iy) {
          storageBuffer[mStoreIdHost[j] * sizeX * sizeY + ix * sizeY + iy] =
              mOutputStateBufferHost[i]
                                    [ix * batchStride * sizeY + j * sizeY + iy];
        }
      }
    }
  }
}

void
TrtOnnxModel::restoreStates_CPU_FP32(
    std::vector<InferenceTask>& inferenceTasks, int batchSize, int batchStride)
{
  for (int i = 0; i < mNumStates; ++i) {
    float* storageBuffer = static_cast<float*>(mStorageBufferHost[0][i]); // TODO: remove 0
    size_t sizeX = mBufferSizeXHost[i];
    size_t sizeY = mBufferSizeYHost[i];
    for (int j = 0; j < batchSize; ++j) {
      if (!inferenceTasks[j].err_msg.empty())
        continue;  // don't restore if error
      if (mStoreIdHost[j] < 0)
        continue;  // no empty slots
      for (size_t ix = 0; ix < sizeX; ++ix) {
        for (size_t iy = 0; iy < sizeY; ++iy) {
          mInputStateBufferHost[i][ix * batchStride * sizeY + j * sizeY + iy] =
              storageBuffer[mStoreIdHost[j] * sizeX * sizeY + ix * sizeY + iy];
        }
      }
    }
  }
}

void
TrtOnnxModel::storeStates_CPU_FP16(
    std::vector<InferenceTask>& inferenceTasks, int batchSize, int batchStride)
{
  for (int i = 0; i < mNumStates; ++i) {
    __half* storageBuffer = static_cast<__half*>(mStorageBufferHost[0][i]); // TODO: remove 0
    size_t sizeX = mBufferSizeXHost[i];
    size_t sizeY = mBufferSizeYHost[i];
    for (int j = 0; j < batchSize; ++j) {
      if (!inferenceTasks[j].err_msg.empty())
        continue;  // don't store if error
      if (mStoreIdHost[j] < 0)
        continue;  // no empty slots
      for (size_t ix = 0; ix < sizeX; ++ix) {
        for (size_t iy = 0; iy < sizeY; ++iy) {
          storageBuffer[mStoreIdHost[j] * sizeX * sizeY + ix * sizeY + iy] =
              __float2half(mOutputStateBufferHost[i]
                                                 [ix * batchStride * sizeY +
                                                  j * sizeY + iy]);
        }
      }
    }
  }
}

void
TrtOnnxModel::restoreStates_CPU_FP16(
    std::vector<InferenceTask>& inferenceTasks, int batchSize, int batchStride)
{
  for (int i = 0; i < mNumStates; ++i) {
    __half* storageBuffer = static_cast<__half*>(mStorageBufferHost[0][i]); // TODO: remove 0
    size_t sizeX = mBufferSizeXHost[i];
    size_t sizeY = mBufferSizeYHost[i];
    for (int j = 0; j < batchSize; ++j) {
      if (!inferenceTasks[j].err_msg.empty())
        continue;  // don't restore if error
      if (mStoreIdHost[j] < 0)
        continue;  // no empty slots
      for (size_t ix = 0; ix < sizeX; ++ix) {
        for (size_t iy = 0; iy < sizeY; ++iy) {
          mInputStateBufferHost[i][ix * batchStride * sizeY + j * sizeY + iy] =
              __half2float(
                  storageBuffer
                      [mStoreIdHost[j] * sizeX * sizeY + ix * sizeY + iy]);
        }
      }
    }
  }
}

void
TrtOnnxModel::storeStates(
    std::vector<InferenceTask>& inferenceTasks, int batchSize, int batchStride,
    cudaStream_t &cudaStreamToUse)
{
  if (mUseGpu) {
    if (mStoreStatesAsFp16) {
      launchStoreGPUKernel_FP16(
          reinterpret_cast<__half***>(mStorageBufferDevice),
          mOutputStateBufferDevice, mBufferSizeXDevice, mBufferSizeYDevice,
          mNumStates, mStoreIdDevice, batchSize, batchStride, cudaStreamToUse);
    } else {
      launchStoreGPUKernel_FP32(
          reinterpret_cast<float***>(mStorageBufferDevice),
          mOutputStateBufferDevice, mBufferSizeXDevice, mBufferSizeYDevice,
          mNumStates, mStoreIdDevice, batchSize, batchStride, cudaStreamToUse);
    }
  } else {
    if (mStoreStatesAsFp16) {
      storeStates_CPU_FP16(inferenceTasks, batchSize, batchStride);
    } else {
      storeStates_CPU_FP32(inferenceTasks, batchSize, batchStride);
    }
  }
}

void
TrtOnnxModel::restoreStates(
    std::vector<InferenceTask>& inferenceTasks, int batchSize, int batchStride,
    cudaStream_t &cudaStreamToUse)
{
  if (mUseGpu) {
    if (mStoreStatesAsFp16) {
      launchRestoreGPUKernel_FP16(
          reinterpret_cast<__half***>(mStorageBufferDevice),
          mInputStateBufferDevice, mBufferSizeXDevice, mBufferSizeYDevice,
          mNumStates, mStoreIdDevice, batchSize, batchStride, cudaStreamToUse);
    } else {
      launchRestoreGPUKernel_FP32(
          reinterpret_cast<float***>(mStorageBufferDevice),
          mInputStateBufferDevice, mBufferSizeXDevice, mBufferSizeYDevice,
          mNumStates, mStoreIdDevice, batchSize, batchStride, cudaStreamToUse);
    }
  } else {
    if (mStoreStatesAsFp16) {
      restoreStates_CPU_FP16(inferenceTasks, batchSize, batchStride);
    } else {
      restoreStates_CPU_FP32(inferenceTasks, batchSize, batchStride);
    }
  }
}

void
TrtOnnxModel::capture_time(double& time, int start_end, int bathsize)
{
  if (start_end == 0)
    mCaptureTime = get_time_msec();
  else
    time += get_time_msec() - mCaptureTime;
}

void
TrtOnnxModel::report_time(log_stream_t& verbose_ss, log_stream_t& info_ss)
{
  time_point_t curTimeStamp = NOW();
  if (mMetricLoggingFreqSeconds > 0  // is logging enabled
      && DURATION(curTimeStamp - lastLogTimeStamp).count() >=
             mMetricLoggingFreqSeconds) {
    verbose_ss << "Host preprocessing: " << mHostPreTime << std::endl;
    verbose_ss << "Host post-processing: " << mHostPostTime << std::endl;
    verbose_ss << "Device preprocessing: " << mDevicePreTime << std::endl;
    verbose_ss << "Device post-processing: " << mDevicePostTime << std::endl;
    verbose_ss << "Device exe time: " << mDeviceExeTime << std::endl;
    double totalTime = mHostPreTime + mHostPostTime + mDevicePreTime +
                       mDevicePostTime + mDeviceExeTime;
    verbose_ss << "Total exe time: " << totalTime << std::endl;
    verbose_ss << "Batch size sum: " << mBatchSizeSum << std::endl;
    verbose_ss << "Number of Inference calls: " << mNumInferCalls << std::endl;
    verbose_ss << std::endl;
    info_ss << "Average Batch Size: "
            << static_cast<double>(mBatchSizeSum) / mNumInferCalls
            << ", Max Batch Size: " << mMaxExecBatchSize << std::endl;
    mHostPreTime = 0.;
    mHostPostTime = 0.;
    mDevicePreTime = 0.;
    mDevicePostTime = 0.;
    mDeviceExeTime = 0.;
    mBatchSizeSum = 0;
    mMaxExecBatchSize = 0;
    mNumInferCalls = 0;
    lastLogTimeStamp = curTimeStamp;
  }
}

#ifndef NO_TRITON
#include "response_util.h"
#endif // NO_TRITON

std::string
TrtOnnxModel::InferTasks(
    std::stringstream& ss_logs, std::vector<InferenceTask>& inferenceTasks,
    int batchSize, void* responses, uint64_t& comp_start_ns, uint64_t& comp_end_ns)
{
#ifdef VERBOSE_COUT
  log_stream_t& verbose_ss = std::cout;
  log_stream_t& info_ss = std::cout;
#else
  log_stream_t ss_null;  // use a local stream to consume the logs
  log_stream_t& verbose_ss = (mLogLevel > 0 ? ss_logs : ss_null);
  log_stream_t& info_ss = (mLogLevel >= 0 ? ss_logs : ss_null);
#endif

  RETURN_IF_FALSE(
      (size_t)batchSize <= inferenceTasks.size(),
      "Number of inference tasks are more than the batch size");
  RETURN_IF_FALSE(
      batchSize <= mBatchDimMax, "Batch size exceeds the max batch dimension");

  int padded_batch_size = mBatchDimMax;
  if (mPaddBatchSize) {
    // find the preferred batch size closest to the current batch
    for (auto ibatch : mPreferredBatchSizes) {
      if (ibatch >= batchSize) {
        padded_batch_size = ibatch;
        break;
      }
    }
  }

  mNumInferCalls++;
  mMaxExecBatchSize =
      std::max(mMaxExecBatchSize, static_cast<size_t>(batchSize));
  mBatchSizeSum += batchSize;

#ifdef VERBOSE_OUTPUT
  verbose_ss << "Using padded batch size = " << padded_batch_size << std::endl;
#endif

  capture_time(mDevicePreTime, 0, batchSize);
  // copy the storage ids to device in advance
  RETURN_IF_ERROR(prepareDeviceStoreIds(verbose_ss, inferenceTasks, batchSize));
  // input dim is set, we can set the bindings
  Ort::IoBinding bindings(*mSession);
  setBindings(padded_batch_size, bindings);
  capture_time(mDevicePreTime, 1, batchSize);

  capture_time(mHostPreTime, 0, batchSize);
  // copy the input to the batch
  // generic implementation that works with more than 1 input and arbitrary
  // batch dimension
  int counter_input = 0;
  for (const auto& itensor : mInputTritonTensorInfo) {
#ifdef VERBOSE_OUTPUT
    verbose_ss << "Input tensor properties: ";
    verbose_ss << itensor.type_size << ", " << itensor.sizeX << ", "
               << itensor.sizeY << std::endl;
#endif
    for (int j = 0; j < batchSize; ++j) {
      for (size_t ix = 0; ix < itensor.sizeX; ++ix) {
        const char* pInputSrc = reinterpret_cast<const char*>(
                                    inferenceTasks[j].mInput[itensor.idx]) +
                                ix * itensor.sizeY * itensor.type_size;
        char* pInputDst =
            reinterpret_cast<char*>(mInputs[counter_input].hostBuffer.data()) +
            (ix * itensor.sizeY * padded_batch_size + j * itensor.sizeY) *
                itensor.type_size;
        memcpy(pInputDst, pInputSrc, itensor.type_size * itensor.sizeY);
      }
    }
    counter_input++;
  }
  capture_time(mHostPreTime, 1, batchSize);

  capture_time(mDevicePreTime, 0, batchSize);
  if (mUseGpu) {
    for (int i = 0; i < mNumInputs; ++i) {
      RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
          mInputs[i].deviceBuffer.data(), mInputs[i].hostBuffer.data(),
          mInputTritonTensorInfo[i].vol * padded_batch_size *
              mInputTritonTensorInfo[i].type_size,
          cudaMemcpyHostToDevice, mCudaStreamCpy));
    }
  }


#ifdef VERBOSE_OUTPUT
  verbose_ss << "Input Reset: ";
#endif
  bool* inputResetData = static_cast<bool*>(mInputReset.hostBuffer.data());
  for (int i = 0; i < padded_batch_size; ++i) {
    if (i < batchSize) {
      inputResetData[i] = (inferenceTasks[i].mStart != 0);
      if (inputResetData[i] && mLogResetSequence) {
        info_ss << "Resetting input for sequence # "
                << inferenceTasks[i].mCorrId << std::endl;
      }
    } else
      inputResetData[i] = true;
#ifdef VERBOSE_OUTPUT
    verbose_ss << inputResetData[i];
#endif
  }
#ifdef VERBOSE_OUTPUT
  verbose_ss << std::endl;
#endif

  if (mUseGpu) {
    // copy the input reset
    RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
        mInputReset.deviceBuffer.data(), mInputReset.hostBuffer.data(),
        mInputReset.deviceBuffer.nbBytes(), cudaMemcpyHostToDevice,
        mCudaStreamCpy));
  }

  // restore the states if this is not a start
  restoreStates(inferenceTasks, batchSize, padded_batch_size, mCudaStreamExe);

  if (mUseGpu) {
    RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(mCudaStreamCpy));
  }
  capture_time(mDevicePreTime, 1, batchSize);

  capture_time(mDeviceExeTime, 0, batchSize);

  SET_TIMESTAMP(comp_start_ns);

  // Execute ORT
  mSession->Run(mRunOptions, bindings);

  SET_TIMESTAMP(comp_end_ns);

  if (mUseGpu) {
    // Only sync the state store/restore execution will be more relevant when
    // there is per-session stream support in ORT
    RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(mCudaStreamExe));
  }
  capture_time(mDeviceExeTime, 1, batchSize);

  capture_time(mDevicePostTime, 0, batchSize);
  if (mUseGpu) {
    // copy the output to the inference tasks
    for (int i = 0; i < mNumOutputs; ++i) {
      size_t cpySize = padded_batch_size * mOutputTritonTensorInfo[i].vol *
                       mOutputTritonTensorInfo[i].type_size;
      verbose_ss << "Copy size = " << cpySize << std::endl;
      RETURN_IF_CUDA_ERROR(cudaMemcpyAsync(
          mOutputs[i].hostBuffer.data(), mOutputs[i].deviceBuffer.data(),
          cpySize, cudaMemcpyDeviceToHost, mCudaStreamCpy));
    }
  }

  if (mUseGpu) {
    RETURN_IF_CUDA_ERROR(cudaStreamSynchronize(mCudaStreamCpy));
  }

  // store the states if this is not an end
  storeStates(inferenceTasks, batchSize, padded_batch_size, mCudaStreamExe);

  capture_time(mDevicePostTime, 1, batchSize);

  capture_time(mHostPostTime, 0, batchSize);
  for (int j = 0; j < batchSize; ++j) {
    int counter_output = 0;
    for (const auto& itensor : mOutputTritonTensorInfo) {
#ifdef VERBOSE_OUTPUT
      verbose_ss << "Output tensor properties: ";
      verbose_ss << itensor.type_size << ", " << itensor.sizeX << ", "
                << itensor.sizeY << std::endl;
#endif
      for (size_t ix = 0; ix < itensor.sizeX; ++ix) {
        char* pOutputDst =
            reinterpret_cast<char*>(inferenceTasks[j].mOutput[itensor.idx]) +
            ix * itensor.sizeY * itensor.type_size;
        const char* pOutputSrc =
            reinterpret_cast<const char*>(
                mOutputs[counter_output].hostBuffer.data()) +
            (ix * itensor.sizeY * padded_batch_size + j * itensor.sizeY) *
                itensor.type_size;
        memcpy(pOutputDst, pOutputSrc, itensor.type_size * itensor.sizeY);
      }
      counter_output++;
    }

#ifndef NO_TRITON
    if (responses != nullptr) {
      // Send the responses early
      triton::backend::stateful::utils::SendSingleResponse(inferenceTasks[j], j, responses);
    }
#endif // NO_TRITON
  }

  // cleanup storage if EndSequence received
  for (int i = 0; i < batchSize; ++i) {
    if (inferenceTasks[i].mEnd && inferenceTasks[i].err_msg.empty()) {
      // the task with EndSequence signal finished successfully
      // release the storage buffer
      auto corrId = inferenceTasks[i].mCorrId;
      verbose_ss << "EndSequence received for CorrID : " << corrId << std::endl;
      mStoreAvailableIds.insert(mStoreIdMap[corrId].first);
      mStoreIdMap.erase(corrId);
    }
  }

  capture_time(mHostPostTime, 1, batchSize);
  report_time(verbose_ss, info_ss);
  
  return std::string();
}
