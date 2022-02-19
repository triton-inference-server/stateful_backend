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


#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>
#include <chrono>
#include <map>
#include <memory>
#include <set>
#include <unordered_set>
#include <sstream>
#include <unordered_map>
#include "buffers.h"
#include "buffers_internal.h"
#include "common.h"

using time_point_t = std::chrono::steady_clock::time_point;  // use wall clock
#define NOW std::chrono::steady_clock::now
#define DURATION_MICRO std::chrono::duration_cast<std::chrono::microseconds>
#define DURATION std::chrono::duration_cast<std::chrono::seconds>

// #define VERBOSE_COUT

#ifdef VERBOSE_COUT
using log_stream_t = std::ostream;
#else
using log_stream_t = std::stringstream;
#endif

inline std::string
GetTimeStampForLogs()
{
  std::time_t timestamp = std::time(nullptr);
  tm* tm_local = std::localtime(&timestamp);
  std::stringstream ss;
  ss << "[";
  ss << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
  ss << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
  ss << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
  ss << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
  ss << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
  ss << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
  return ss.str();
}

namespace samplesCommon {

template <typename AllocFunc, typename FreeFunc>
class GenericBufferInternal : public GenericBufferBase<AllocFunc, FreeFunc>
{
  public:
    GenericBufferInternal(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
      : GenericBufferBase<AllocFunc, FreeFunc>(type) {}

    void resize_async(size_t newSize, const cudaStream_t strm)
    {
      this->mSize = newSize;
      if (this->mCapacity < newSize) {
        cudaFreeAsync(this->mBuffer, strm);
        if (cudaMallocAsync(&(this->mBuffer), this->nbBytes(), strm)
            != cudaSuccess) {
          throw std::bad_alloc{};
        }
        this->mCapacity = newSize;
      }
      if (newSize == 0) { // size=0 => cudaFreeAsync
        cudaFreeAsync(this->mBuffer, strm);
        this->mBuffer = nullptr;
        this->mCapacity = 0;
      }
    }
    void resize_async(const nvinfer1::Dims& dims, const cudaStream_t strm)
    {
      return this->resize_async(samplesCommon::volume(dims), strm);
    }
};

using DeviceBufferInternal = GenericBufferInternal<DeviceAllocator, DeviceFree>;

class ManagedBufferInternal {
 public:
  DeviceBufferInternal deviceBuffer;
  HostBuffer hostBuffer;
  ManagedBufferInternal() :
    deviceBuffer(nvinfer1::DataType::kFLOAT),
    hostBuffer(nvinfer1::DataType::kFLOAT) {}
  ManagedBufferInternal(nvinfer1::DataType dt, nvinfer1::DataType ht)
    : deviceBuffer(dt), hostBuffer(ht) {}

  // resize host buffer and selectively resize GPU buffer if isdevice is set
  void resizeAll(const nvinfer1::Dims& dims, bool isdevice)
  {
    if (isdevice)
      deviceBuffer.resize(dims);
    hostBuffer.resize(dims);
  }
  // resize either host or device buffer
  void resize(const nvinfer1::Dims& dims, bool isdevice)
  {
    if (isdevice)
      deviceBuffer.resize(dims);
    else
      hostBuffer.resize(dims);
  }
  // return either host or device buffer
  void* data(bool isdevice)
  {
    if (isdevice)
      return deviceBuffer.data();
    else
      return hostBuffer.data();
  }
  void resize_async(
    const nvinfer1::Dims& dims, bool isdevice, const cudaStream_t strm)
  {
    if (isdevice)
      deviceBuffer.resize_async(dims, strm);
    else
      hostBuffer.resize(dims);
  }
  void resize_async(
    size_t newSize, bool isdevice, const cudaStream_t strm)
  {
    if (isdevice)
      deviceBuffer.resize_async(newSize, strm);
    else
      hostBuffer.resize(newSize);
  }
};


}  // namespace samplesCommon

static const int MAX_IO_NUM = 5;
static const int INVALID_DIM = 1000;

static uint64_t U64_ZERO = 0;

// Tensor information provided by the Triton server based
// on the config file. Batch dimension is specified by -1 value
// in the shape tensor. Only the real I/O tensors are presented
// as TritonTensorInfo classes.
class TritonTensorInfo {
 public:
  std::string name;
  std::string type;
  std::vector<int64_t> shape;
  size_t type_size;
  size_t idx;

  // calculated
  size_t vol;
  size_t batch_dim;
  size_t sizeX{1};
  size_t sizeY{1};


  std::string Init()
  {
    // need to init batch dim first to set 1 for batch dim
    InitBatchDim();
    InitVol();
    InitTypeSize();
    InitBatchStrides();

    std::string log = "Triton tensor is initialized: ";
    log += "name=" + name;
    log += "type_size=" + std::to_string(type_size);
    log += ", type=" + type;
    log += ", batch_dim=" + std::to_string(batch_dim);
    log += ", idx=" + std::to_string(idx);
    log += ", shape=";
    for (size_t i = 0; i < shape.size(); ++i)
      log += std::to_string(shape[i]) + " x ";
    log += "\n";
    log += "Size X= " + std::to_string(sizeX);
    log += ", SizeY= " + std::to_string(sizeY);
    log += "\n";
    return log;
  }

  static size_t TypeToTypeSize(const std::string& typ)
  {
    size_t sz = 0;
    if (typ.compare("TYPE_FP16") == 0) {
      sz = 2;
    } else if (typ.compare("TYPE_FP32") == 0) {
      sz = 4;
    } else if (typ.compare("TYPE_FP64") == 0) {
      sz = 8;
    } else if (typ.compare("TYPE_UINT8") == 0) {
      sz = 1;
    } else if (typ.compare("TYPE_UINT32") == 0) {
      sz = 4;
    } else if (typ.compare("TYPE_UINT64") == 0) {
      sz = 8;
    } else if (typ.compare("TYPE_INT8") == 0) {
      sz = 1;
    } else if (typ.compare("TYPE_INT32") == 0) {
      sz = 4;
    } else if (typ.compare("TYPE_INT64") == 0) {
      sz = 8;
    } else {
      assert(false);
    }
    return sz;
  }

 private:
  void InitVol()
  {
    vol = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      vol *= shape[i];
    }
  }
  void InitTypeSize() { type_size = TypeToTypeSize(type); }

  void InitBatchDim()
  {
    batch_dim = INVALID_DIM;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == -1) {
        batch_dim = (int)i;
        // set batch dim to 1 so that we can calculate volume
        shape[i] = 1;
        return;
      }
    }
  }

  void InitBatchStrides()
  {
    sizeX = 1;
    sizeY = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i < batch_dim) {
        sizeX *= shape[i];
      } else if (i > batch_dim) {
        sizeY *= shape[i];
      }
    }
  }
};

// Tensor data in the ONNX model
class OrtTensorInfo {
 public:
  void* data;
  std::vector<int64_t> shape;
  size_t num_dims;
  size_t batch_dim;
  int is_input;
  size_t type_size;
  ONNXTensorElementDataType type;


  void Print(std::ostream& os)
  {
    os << "OrtTensorInfo" << std::endl;
    os << "Num dims = " << num_dims << std::endl;
    os << "Shape = ";
    for (size_t i = 0; i < num_dims; ++i) {
      os << shape[i] << " x ";
    }
    os << std::endl;

    os << "Batch dim = " << batch_dim;
    os << std::endl;


    os << "Is input = " << is_input;
    os << std::endl;
  }
};

// Inference tasks sent by the Triton server
class InferenceTask {
 public:
  int mStart;
  int mEnd;
  std::string mCorrId;
  const void* mInput[MAX_IO_NUM];
  void* mOutput[MAX_IO_NUM];
  std::string err_msg;  // will be used to track individual error
};

// all the params for lazy allocation
class BufferConfig {
public:
  int64_t max_connections;
  int64_t initial_buffer_size;
  int64_t subsequent_buffer_size;
  int64_t alloc_threshold;
  int64_t dealloc_threshold;

  std::string to_string() {
    std::stringstream ss;
    ss << "[" << initial_buffer_size << ",";
    ss << max_connections << "," << subsequent_buffer_size << "]";
    ss << " [" << alloc_threshold << "," << dealloc_threshold << "]";
    return ss.str();
  }
};

// The TrtOnnxModel class implements loading and running a stateful ONNX model.
class TrtOnnxModel {
 public:
  TrtOnnxModel() { mUseGpu = false; }

  ~TrtOnnxModel()
  {
    // The ORT variables must be destroyed in this order
    mSession.reset(nullptr);
    mEnv.reset();

    if (mUseGpu) {
      CHECK(cudaStreamDestroy(mCudaStreamExe));
      CHECK(cudaStreamDestroy(mCudaStreamCpy));
      if (mNumStates != -1) {
        for (size_t i=0; i<mStorageBufferDevicePtrOnHost.size(); ++i) {
          CHECK(cudaFree(mStorageBufferDevicePtrOnHost[i]));
        }
        CHECK(cudaFree(mStorageBufferDevice));
        CHECK(cudaFree(mInputStateBufferDevice));
        CHECK(cudaFree(mOutputStateBufferDevice));
        CHECK(cudaFree(mBufferSizeXDevice));
        CHECK(cudaFree(mBufferSizeYDevice));
        CHECK(cudaFree(mStoreIdDevice));
      }
    }
  }

  // Set min and max batch dimension for the model
  void SetMinMaxBatch(int min, int max)
  {
    mBatchDimMin = min;
    mBatchDimMax = max;
  }

  // Prepares the model for inference by creating execution contexts and
  // allocating buffers.
  std::string Prepare(
      std::stringstream& ss_logs, std::shared_ptr<Ort::Env> ort_env,
      std::string onnx_file_name, std::string state_pairs,
      const BufferConfig& buffer_config,
      int gpuId, std::vector<int64_t>& pref_batch_sizes,
      const std::vector<TritonTensorInfo>& input_tensors,
      const std::vector<TritonTensorInfo>& output_tensors,
      std::string reset_tensor_name, bool useTrtEp = true, bool useFp16 = false,
      bool store_states_as_fp16 = false, bool pad_to_max_batch = false,
      bool enable_trt_caching = false, std::string trt_cache_dir = "/tmp",
      int64_t logLevel = 1, int64_t metricLoggingFreq = 0,
      int64_t sequnce_timeout_microseconds = INT64_MAX);

  // Runs inference for multiple tasks for Triton backend
  std::string InferTasks(
      std::stringstream& ss_logs, std::vector<InferenceTask>& inferenceTasks,
      int batchSize, void* responses=nullptr, uint64_t& comp_start_ns=U64_ZERO,
      uint64_t& comp_end_ns=U64_ZERO);

  void SetSequenceResetLogging(bool enableLogging)
  {
    mLogResetSequence = enableLogging;
  }

#ifndef BUILD_GBENCH
 private:
#endif
  TritonTensorInfo* GetInputTensor(std::string name)
  {
    for (auto& tensor : mInputTritonTensorInfo) {
      if (samplesCommon::toLower(tensor.name)
              .compare(samplesCommon::toLower(name)) == 0) {
        return &tensor;
      }
    }
    return nullptr;
  }

  TritonTensorInfo* GetOutputTensor(std::string name)
  {
    for (auto& tensor : mOutputTritonTensorInfo) {
      if (samplesCommon::toLower(tensor.name)
              .compare(samplesCommon::toLower(name)) == 0) {
        return &tensor;
      }
    }
    return nullptr;
  }


  void setBindings(int batchsize, Ort::IoBinding& iobindings);
  int GetNumSegments();

  void storeStates_CPU_FP32(
      std::vector<InferenceTask>& inferenceTasks, int batchSize,
      int batchStride);
  void restoreStates_CPU_FP32(
      std::vector<InferenceTask>& inferenceTasks, int batchSize,
      int batchStride);
  void storeStates_CPU_FP16(
      std::vector<InferenceTask>& inferenceTasks, int batchSize,
      int batchStride);
  void restoreStates_CPU_FP16(
      std::vector<InferenceTask>& inferenceTasks, int batchSize,
      int batchStride);

  void storeStates(
      std::vector<InferenceTask>& inferenceTasks, int batchSize,
      int batchStride, cudaStream_t cudaStreamToUse);
  void restoreStates(
      std::vector<InferenceTask>& inferenceTasks, int batchSize,
      int batchStride, cudaStream_t cudaStreamToUse);
  std::string prepareDeviceStoreIds(
      log_stream_t& verbose_ss, std::vector<InferenceTask>& inferenceTasks,
      int batchSize);

  std::unordered_map<std::string, std::tuple<int,int,time_point_t>> mStoreIdMap;
  std::vector<std::set<int>> mStoreAvailableIds;
  std::unordered_set<std::string> mCorrIdToDelete;


  samplesCommon::ManagedBufferInternal
      mInputs[MAX_IO_NUM];  //!< Host and device buffers for the input.
  samplesCommon::ManagedBufferInternal mInputReset{
      nvinfer1::DataType::kBOOL,
      nvinfer1::DataType::kBOOL};  //!< Host and device buffers for the input.
  samplesCommon::ManagedBufferInternal
      mOutputs[MAX_IO_NUM];  //!< Host and device buffers for the outputs
  std::vector<std::unique_ptr<samplesCommon::ManagedBufferInternal>>
      mStates{};  //!< Host and device buffers for the internal states
  std::vector<std::vector<
      std::unique_ptr<samplesCommon::ManagedBufferInternal>>>
      mStoredStates{};  //!< Host and device buffers for the internal states

  BufferConfig mBufferConfig;
  std::vector<int64_t> mPreferredBatchSizes;
  int64_t mSequenceTimeoutMicroseconds;

  int mGpuId{-1};
  bool mUseGpu{true};
  bool mUseTrtEp{true};
  bool mStoreStatesAsFp16{false};
  std::string mDeviceBindingString;
  const bool mAllocFreeAsync{true};

  int mBatchDimMin{1};
  int mBatchDimMax{2};

  bool mPaddBatchSize{false};

  cudaStream_t mCudaStreamExe;
  cudaStream_t mCudaStreamCpy;

  int mNumChunks{1}; // current number of buffer chunks
  int mMaxChunks{1}; // maximum number of buffer chunks
  std::vector<void**> mStorageBufferDevicePtrOnHost;  // dev ptrs to chunks
  void*** mStorageBufferDevice;  // either float or __half
  float** mInputStateBufferDevice{nullptr};
  float** mOutputStateBufferDevice{nullptr};
  int* mBufferSizeXDevice{nullptr};
  int* mBufferSizeYDevice{nullptr};
  int* mStoreIdDevice{nullptr};

  std::vector<std::vector<void*>> mStorageBufferHost;  // either float or __half
  std::vector<float*> mInputStateBufferHost;
  std::vector<float*> mOutputStateBufferHost;
  std::vector<int> mBufferSizeXHost;
  std::vector<int> mBufferSizeYHost;
  std::vector<int> mStoreIdHost;


  int mNumStates{-1};
  int mNumOutputs{0};
  int mNumInputs{0};

  void capture_time(double& time, int start_end, int bathsize);
  void report_time(log_stream_t& verbose_ss, log_stream_t& info_ss);
  std::string AllocateNewChunk(log_stream_t&, log_stream_t&);
  std::string DeAllocateChunk(log_stream_t&, log_stream_t&);
  std::string TryAndDeAllocateChunk(log_stream_t&, log_stream_t&);
  double mHostPreTime{0.}, mHostPostTime{0.};
  double mDevicePreTime{0.}, mDevicePostTime{0.};
  double mDeviceExeTime{0.};
  double mCaptureTime{0.};
  size_t mNumInferCalls{0};
  size_t mMaxExecBatchSize{0};
  size_t mBatchSizeSum{0};
  time_point_t lastLogTimeStamp{NOW()};
  bool mLogResetSequence{true};

  std::vector<TritonTensorInfo> mInputTritonTensorInfo;
  std::vector<TritonTensorInfo> mOutputTritonTensorInfo;
  std::unique_ptr<Ort::Session> mSession;
  std::shared_ptr<Ort::Env> mEnv;
  std::map<std::string, OrtTensorInfo> mOrtTensors;
  Ort::RunOptions mRunOptions;

  // logging
  int64_t mLogLevel;
  int64_t mMetricLoggingFreqSeconds;

 public:
  // StateTensor initializes state tensors based on a description file.
  // It also contains information about state tensors such as the dimension
  // batch dimension etc.
  class StateTensor {
   public:
    StateTensor(
        const std::string& inputName, const std::string& outputName,
        void* inputBuffer, nvinfer1::Dims dim, enum nvinfer1::DataType type,
        int batch_dim)
        : mInputName(inputName), mOutputName(outputName),
          mInputBuffer(inputBuffer), mDim(dim), mType(type),
          mBatchDim(batch_dim)
    {
    }

    std::string mInputName;
    std::string mOutputName;
    void* mInputBuffer{nullptr};
    void* mOutputBuffer{nullptr};
    std::vector<void*> mStoreBuffer;
    nvinfer1::Dims mDim;
    enum nvinfer1::DataType mType;
    int mBatchDim{-1};

    static int InitTensorNames(
        std::string desc_string, std::string input_state_name,
        std::string output_state_name,
        std::vector<std::string>& input_state_names,
        std::vector<std::string>& output_state_names)
    {
      std::stringstream graph_desc(desc_string);
      std::string line;
      // size_t iline = 0;
      // ifstream file(desc_filename);
      // if (file.is_open() == false) return 1;
      if (input_state_name.empty()) {
        // if no input state tensor name is not specified the whole string is
        // the state tensor name pairs with input states first
        while (getline(graph_desc, line)) {
          size_t next_pos = line.find("<<<", 0);
          while (next_pos != std::string::npos) {
            std::string first_string, second_string;
            size_t begin_first = next_pos + 3;
            while (line[begin_first] == ' ') begin_first++;

            size_t end_first = line.find(",", next_pos) - 1;
            size_t begin_second = end_first + 2;
            while (line[end_first] == ' ') end_first--;
            while (line[begin_second] == ' ') begin_second++;

            size_t end_second = line.find(">>>", end_first) - 1;
            while (line[end_second] == ' ') end_second--;
            next_pos = line.find("<<<", end_second);

            std::string first_name =
                line.substr(begin_first, end_first - begin_first + 1);
            std::string second_name =
                line.substr(begin_second, end_second - begin_second + 1);

            input_state_names.push_back(first_name);
            output_state_names.push_back(second_name);
          }
        }
      } else {
        while (getline(graph_desc, line)) {
          size_t input_state_pos = line.find(input_state_name);
          size_t output_state_pos = line.find(output_state_name);

          if ((input_state_pos != std::string::npos) &&
              (output_state_pos != std::string::npos)) {
            // this is the line with input / output state information

            bool input_first = false;
            if (input_state_pos < output_state_pos)
              input_first = true;
            size_t next_pos = line.find("pair", input_state_pos);
            next_pos = line.find("<<<", input_state_pos);
            while (next_pos != std::string::npos) {
              std::string first_string, second_string;
              size_t begin_first = next_pos + 3;
              while (line[begin_first] == ' ') begin_first++;

              size_t end_first = line.find(",", next_pos) - 1;
              size_t begin_second = end_first + 2;
              while (line[end_first] == ' ') end_first--;
              while (line[begin_second] == ' ') begin_second++;

              size_t end_second = line.find(">>>", end_first) - 1;
              while (line[end_second] == ' ') end_second--;

              next_pos = line.find("<<<", end_second);

              std::string first_name =
                  line.substr(begin_first, end_first - begin_first + 1);
              std::string second_name =
                  line.substr(begin_second, end_second - begin_second + 1);

              // remove anything after the first occurence of  ":" from the
              // tensor names These are tensor dimensions that could be read
              // from the ONNX model file already
              size_t col_pos = first_name.find(':');
              if (col_pos != std::string::npos)
                first_name = first_name.substr(0, col_pos);

              col_pos = second_name.find(':');
              if (col_pos != std::string::npos)
                second_name = second_name.substr(0, col_pos);

              if (input_first) {
                input_state_names.push_back(first_name);
                output_state_names.push_back(second_name);
              } else {
                input_state_names.push_back(second_name);
                output_state_names.push_back(first_name);
              }
            }
          }
        }
      }
      return 0;
    }

    static int GetIdx(const std::string& str, std::vector<std::string>& states)
    {
      for (size_t i = 0; i < states.size(); ++i)
        if (states[i].compare(str) == 0)
          return i;
      return -1;
    }

    void AddOutputNameIfMatch(const std::string& str, void* oBuffer)
    {
      if (mOutputName.compare(str) == 0) {
        assert(mOutputBuffer == nullptr);
        mOutputName = str;
        mOutputBuffer = oBuffer;
      }
    }


    size_t nbBytes()
    {
      return samplesCommon::volume(mDim) * samplesCommon::getElementSize(mType);
    }

    void printStateTensors(std::ostream& os) const
    {
      os << "I/O Tensors: " << mInputName << " / " << mOutputName;
      os << "Dims: " << mDim;
      os << "Buffers: " << mInputBuffer << ", " << mOutputBuffer << ", ";
      for (auto buffer : mStoreBuffer) {
        os << buffer;
      }
    }

   private:
    int GetTensorId(const std::string& str)
    {
      std::string::size_type startPast = str.find("PastValue");
      assert(startPast != std::string::npos);
      std::string::size_type startDash = str.find('_', startPast);
      assert(startDash != std::string::npos);
      return stoi(str.substr(startPast + 9, startDash - startPast - 9));
    }
  };
  std::vector<StateTensor> mStateTensors;
};
