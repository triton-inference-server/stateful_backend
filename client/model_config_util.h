#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#include "grpc_client.h"

namespace tc = triton::client;

#ifndef FAIL_IF_ERR
#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }
#endif

namespace stateful
{
  namespace client
  {
    namespace utils
    {
      size_t TypeSize(inference::DataType typ) {
        switch(typ)
        {
        case inference::DataType::TYPE_BOOL:
          return 1;
        case inference::DataType::TYPE_FP32:
          return 4;
        case inference::DataType::TYPE_FP16:
          return 2;
        case inference::DataType::TYPE_INT8:
          return 1;
        case inference::DataType::TYPE_INT32:
          return 4;
        }
        return 0;
      }
      std::string TypeName(inference::DataType typ) {
        switch(typ)
        {
        case inference::DataType::TYPE_BOOL:
          return "BOOL";
        case inference::DataType::TYPE_FP32:
          return "FP32";
        case inference::DataType::TYPE_FP16:
          return "FP16";
        case inference::DataType::TYPE_INT8:
          return "INT8";
        case inference::DataType::TYPE_INT32:
          return "INT32";
        }
        return "UNDEFINED";
      }
      typedef struct ModelInfo {
        bool is_corrid_string{false};
        bool infer_end_requests{true};
        std::vector<int64_t> input_vols;
        std::vector<int64_t> output_vols;
        std::vector<std::vector<int64_t>> input_dims;
        std::vector<std::vector<int64_t>> output_dims;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
        std::vector<inference::DataType> input_types;
        std::vector<inference::DataType> output_types;
        std::string to_string() {
          std::stringstream ss;
          ss << "is_corrid_string : " << is_corrid_string << std::endl;
          ss << "infer_end_requests : " << infer_end_requests << std::endl;
          ss << "Inputs: " << std::endl;
          for (size_t i=0; i<input_names.size(); ++i) {
            ss << "    " << input_names[i] << " : [" << input_dims[i][0];
            for (size_t j=1; j<input_dims[i].size(); ++j) {
              ss << ", " << input_dims[i][j];
            }
            ss << "]";
            ss << ", Volume: " << input_vols[i];
            ss << ", DataType: " << TypeName(input_types[i]);
            ss << std::endl;
          }
          ss << "Outputs: " << std::endl;
          for (size_t i=0; i<output_names.size(); ++i) {
            ss << "    " << output_names[i] << " : [" << output_dims[i][0];
            for (size_t j=1; j<output_dims[i].size(); ++j) {
              ss << ", " << output_dims[i][j];
            }
            ss << "]";
            ss << ", Volume: " << output_vols[i];
            ss << ", DataType: " << TypeName(output_types[i]);
            ss << std::endl;
          }
          return ss.str();
        }
      } ModelInfo;

      inference::ModelConfigResponse RetrieveModelConfig(
        const std::unique_ptr<tc::InferenceServerGrpcClient>& client,
        const std::string& model_name)
      {
        inference::ModelConfigResponse model_config;
        FAIL_IF_ERR(
            client->ModelConfig(&model_config, model_name),
            "unable to get model config");
        if (model_config.config().name().compare(model_name) != 0) {
          std::cerr << "error: unexpected model config: "
                    << model_config.DebugString() << std::endl;
          exit(1);
        }
        return model_config;
      }

      bool CheckIfCorrIdIsString(const inference::ModelConfig& config) {
        for (auto& control : config.sequence_batching().control_input()) {
          if (control.name().compare("CORRID") == 0) {
            for (auto& c : control.control()) {
              if (c.kind() ==
                 inference::ModelSequenceBatching_Control_Kind::
                 ModelSequenceBatching_Control_Kind_CONTROL_SEQUENCE_CORRID) {
                if (c.data_type() == inference::DataType::TYPE_STRING) {
                  return true;
                }
                break;
              }
            }
            break;
          }
        }
        return false;
      }

      bool CheckIfInferringEndRequests(const inference::ModelConfig& config) {
        auto param = config.parameters().find("infer_end_requests");
        if (param != config.parameters().end()) {
            if (param->second.string_value().compare("0") == 0) {
              return false;
            }
          }
        return true;
      }

      ModelInfo GetModelInfo(
        const std::unique_ptr<tc::InferenceServerGrpcClient>& client,
        const std::string& model_name)
      {
        ModelInfo info;
        inference::ModelConfigResponse model_config_response
          = RetrieveModelConfig(client, model_name);
        inference::ModelConfig config = model_config_response.config();
        info.is_corrid_string = CheckIfCorrIdIsString(config);
        info.infer_end_requests = CheckIfInferringEndRequests(config);
        // get the inputs
        for (auto& in : config.input()) {
          info.input_names.push_back(in.name());
          std::vector<int64_t> dim;
          int64_t vol = 1;
          for (auto& d : in.dims()) {
            if (d > 0) vol *= d;
            dim.push_back(d);
          }
          info.input_vols.push_back(vol);
          info.input_dims.push_back(dim);
          info.input_types.push_back(in.data_type());
        }
        // get the outputs
        for (auto& out : config.output()) {
          info.output_names.push_back(out.name());
          std::vector<int64_t> dim;
          int64_t vol = 1;
          for (auto& d : out.dims()) {
            if (d > 0) vol *= d;
            dim.push_back(d);
          }
          info.output_vols.push_back(vol);
          info.output_dims.push_back(dim);
          info.output_types.push_back(out.data_type());
        }
        return info;
      }

      void PrepareInputs(
        std::vector<tc::InferInput*>& inputs,
        ModelInfo& info,
        std::vector<std::shared_ptr<void>>& data)
      {
        for (size_t i=0; i<info.input_names.size(); ++i) {
          const auto& in_name = info.input_names[i];
          const auto& dims = info.input_dims[i];
          const auto& typ = info.input_types[i];
          const size_t type_size = TypeSize(typ);
          int64_t input_size = info.input_vols[i];
          // std::cout << input_size << std::endl;
          tc::InferInput* input;
          std::vector<int64_t> shape({1});
          shape.insert(shape.end(), dims.begin(), dims.end());
          for (auto& d : shape) {
            if (d < 0) d = 1; // change the batch dim
          }
          // std::cout << "Will create the input" << std::endl;
          FAIL_IF_ERR(
            tc::InferInput::Create(&input, in_name, shape,
              TypeName(typ)),
            std::string("unable to create '") + in_name + "'");
          FAIL_IF_ERR(
            input->Reset(),
            std::string("unable to reset '") + in_name + "'");
          FAIL_IF_ERR(
            input->AppendRaw(
              reinterpret_cast<uint8_t*>(data[i].get()),
              input_size * type_size),
              std::string("unable to set data for '") + in_name + "'");
          inputs.emplace_back(input);
        }
      }

      void RetrieveSingleOutputTensor(
        const std::shared_ptr<tc::InferResult> result,
        const std::string& tensor_name,
        const size_t output_size,
        const size_t type_size,
        void** output_raw_data,
        size_t& output_byte_size)
      {
        FAIL_IF_ERR(
            result->RawData(
                tensor_name, (const uint8_t**)output_raw_data,
                &output_byte_size),
            "unable to get result data for 'Output'");
        if (output_byte_size != (type_size * output_size)) {
          std::cerr << "error: received incorrect byte size for 'Output': "
                    << output_byte_size << " vs. " << output_size <<std::endl;
          exit(1);
        }
      }

      void RetrieveOutputs(
        std::shared_ptr<tc::InferResult> result,
        ModelInfo& info,
        std::vector<std::shared_ptr<void>>& output_data)
      {
        // Get pointers to the result returned...
        for (size_t i=0; i<info.output_names.size(); ++i) {
          const size_t output_size = info.output_vols[i];
          const size_t type_size = TypeSize(info.output_types[i]);
          void* output_raw_data;
          size_t output_byte_size;
          RetrieveSingleOutputTensor(result, info.output_names[i],
                                      output_size, type_size, &output_raw_data,
                                      output_byte_size);
          output_data[i] = std::shared_ptr<void>(malloc(output_byte_size), free);
          memcpy(
              (uint8_t*)output_data[i].get(),
              (const uint8_t*)output_raw_data, output_byte_size
              );

          // std::cout << this_sequence_id << "_" << segment_idx << " :: ";
          // for (auto f : output_data[i])
          //   std::cout << f << ", ";
          // std::cout << std::endl;
        }
      }

      // only supports reduce of axis=0 or axis=1 for [r, c]-sized tensors
      template<typename T, int axis>
      void op_reduce_sum(
        const size_t r, const size_t c,
        const T* data, float* sum)
      {
        if (axis == 0) {
          for (size_t i=0; i < r; ++i) {
            for(size_t j=0; j < c; ++j) {
              sum[j] += data[j+i*c];
            }
          }
        } else if (axis == 1) {
          for (size_t i=0; i < r; ++i) {
            float csum = 0;
            for(size_t j=0; j < c; ++j) {
              csum += data[j];
            }
            sum[i] += csum;
          }
        }
      }

      template<int axis>
      void op_reduce_sum(
        const size_t r, const size_t c,
        const void* vdata, float* out,
        const inference::DataType typ)
      {
        switch(typ)
        {
        case inference::DataType::TYPE_BOOL:
          op_reduce_sum<bool, axis>(r, c,
              reinterpret_cast<const bool*>(vdata), out);
          break;
        case inference::DataType::TYPE_FP32:
          op_reduce_sum<float, axis>(r, c,
              reinterpret_cast<const float*>(vdata), out);
          break;
        case inference::DataType::TYPE_FP16:
          assert(0); // FP16 in C++ clientd is not supported yet
          break;
        case inference::DataType::TYPE_INT8:
          op_reduce_sum<int8_t, axis>(r, c,
              reinterpret_cast<const int8_t*>(vdata), out);
          break;
        case inference::DataType::TYPE_INT32:
          op_reduce_sum<int32_t, axis>(r, c,
              reinterpret_cast<const int32_t*>(vdata), out);
          break;
        }
      }

      template<typename T>
      void op_add(
        const size_t r, const size_t c,
        const void* vin1, const void* vin2, void* vout)
      {
        const T* in1 = reinterpret_cast<const T*>(vin1);
        const float* in2 = reinterpret_cast<const float*>(vin2);
        T* out = reinterpret_cast<T*>(vout);
        for (size_t i=0; i<r; ++i) {
          for (size_t j=0; j<c; ++j) {
            out[i*c + j] = in1[i*c+j] + static_cast<T>(in2[j]);
          }
        }
      }

      void op_add(
        const size_t r, const size_t c,
        const void* vin1, const void* vin2, void* vout,
        const inference::DataType typ)
      {
        switch(typ)
        {
        case inference::DataType::TYPE_BOOL:
          op_add<bool>(r, c, vin1, vin2, vout);
          break;
        case inference::DataType::TYPE_FP32:
          op_add<float>(r, c, vin1, vin2, vout);
          break;
        case inference::DataType::TYPE_FP16:
          assert(0); // FP16 in C++ clientd is not supported yet
          break;
        case inference::DataType::TYPE_INT8:
          op_add<int8_t>(r, c, vin1, vin2, vout);
          break;
        case inference::DataType::TYPE_INT32:
          op_add<int32_t>(r, c, vin1, vin2, vout);
          break;
        }
      }

      template<typename T>
      int compare_floating_tensors(
        const int64_t tensor_len, const T* expected, const T* test,
        float err_tol, const std::string err_prefix="")
      {
        for (int64_t i = 0; i < tensor_len; ++i) {
          float diff = fabs(static_cast<float>(test[i] - expected[i]));
          if (diff > err_tol) {
            std::cerr << "FAILED!" << std::endl;
            std::cerr << err_prefix << " idx " << i
                      << " : expected " << expected[i]
                      << " , got " << test[i]
                      << " , tolerance = " << err_tol
                      << std::endl;
            return 1;
          }
        }
        return 0;
      }

      template<typename T>
      int compare_tensors(
        const int64_t tensor_len, const T* expected, const T* test,
        const std::string err_prefix="")
      {
        for (int64_t i = 0; i < tensor_len; ++i) {
          if (test[i] != expected[i]) {
            std::cerr << "FAILED!" << std::endl;
            std::cerr << err_prefix << " : expected " << expected[i]
                      << " , got " << test[i] << std::endl;
            return 1;
          }
        }
        return 0;
      }

      template<>
      int compare_tensors(
        const int64_t tensor_len, const float* expected, const float* test,
        const std::string err_prefix)
      {
        const float err_tol = 1e-4; // some guess for now
        return compare_floating_tensors<float>(
                tensor_len, expected, test, err_tol, err_prefix);
      }

      int compare_tensors(
        const int64_t tensor_len,
        const inference::DataType typ,
        const void* expected, const void* test,
        const std::string err_prefix="")
      {
        switch(typ)
        {
        case inference::DataType::TYPE_BOOL:
          return compare_tensors<bool>(tensor_len,
            reinterpret_cast<const bool*>(expected),
            reinterpret_cast<const bool*>(test), err_prefix);
        case inference::DataType::TYPE_FP32:
          return compare_tensors<float>(tensor_len,
            reinterpret_cast<const float*>(expected),
            reinterpret_cast<const float*>(test), err_prefix);
        case inference::DataType::TYPE_FP16:
          assert(0); // FP16 in C++ clientd is not supported yet
          return -1;
        case inference::DataType::TYPE_INT8:
          return compare_tensors<int8_t>(tensor_len,
            reinterpret_cast<const int8_t*>(expected),
            reinterpret_cast<const int8_t*>(test), err_prefix);
        case inference::DataType::TYPE_INT32:
          return compare_tensors<int32_t>(tensor_len,
            reinterpret_cast<const int32_t*>(expected),
            reinterpret_cast<const int32_t*>(test), err_prefix);
        }
        return -1;
      }

      template<typename T>
      void init_tensor(
        const size_t tensor_size,
        const int seq_idx, const int seg_idx,
        void* vdata)
      {
        T* input_values = reinterpret_cast<T*>(vdata);
        for (size_t j = 0; j < tensor_size; ++j) {
          input_values[j] = j + seg_idx * 100 + seq_idx * 1000;
        }
      }

      void init_tensor(
        const size_t tensor_size,
        const int seq_idx, const int seg_idx,
        void* vdata,
        const inference::DataType typ)
      {
        switch(typ)
        {
        case inference::DataType::TYPE_BOOL:
          init_tensor<bool>(tensor_size, seq_idx, seg_idx, vdata);
          break;
        case inference::DataType::TYPE_FP32:
          init_tensor<float>(tensor_size, seq_idx, seg_idx, vdata);
          break;
        case inference::DataType::TYPE_FP16:
          assert(0); // FP16 in C++ clientd is not supported yet
          break;
        case inference::DataType::TYPE_INT8:
          init_tensor<int8_t>(tensor_size, seq_idx, seg_idx, vdata);
          break;
        case inference::DataType::TYPE_INT32:
          init_tensor<int32_t>(tensor_size, seq_idx, seg_idx, vdata);
          break;
        }
      }
    } // namespace utils
    
  } // namespace client
  
} // namespace stateful
