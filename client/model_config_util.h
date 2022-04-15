#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

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
      typedef struct ModelInfo {
        bool is_corrid_string{false};
        bool infer_end_requests{true};
        std::vector<int64_t> input_vols;
        std::vector<int64_t> output_vols;
        std::vector<std::vector<int64_t>> input_dims;
        std::vector<std::vector<int64_t>> output_dims;
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;
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
            ss << ", Volume: " << input_vols[i] << std::endl;
          }
          ss << "Outputs: " << std::endl;
          for (size_t i=0; i<output_names.size(); ++i) {
            ss << "    " << output_names[i] << " : [" << output_dims[i][0];
            for (size_t j=1; j<output_dims[i].size(); ++j) {
              ss << ", " << output_dims[i][j];
            }
            ss << "]";
            ss << ", Volume: " << output_vols[i] << std::endl;
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
        }
        return info;
      }

      void PrepareInputs(
        std::vector<tc::InferInput*>& inputs,
        ModelInfo& info,
        std::vector<std::vector<float>>& data)
      {
        for (size_t i=0; i<info.input_names.size(); ++i) {
          auto in_name = info.input_names[i];
          auto& dims = info.input_dims[i];
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
            tc::InferInput::Create(&input, in_name, shape, "FP32"),
            std::string("unable to create '") + in_name + "'");
          FAIL_IF_ERR(
            input->Reset(),
            std::string("unable to reset '") + in_name + "'");
          FAIL_IF_ERR(
            input->AppendRaw(
              reinterpret_cast<uint8_t*>(data[i].data()),
              input_size * sizeof(float)),
              std::string("unable to set data for '") + in_name + "'");
          inputs.emplace_back(input);
        }
      }

      void RetrieveSingleOutputTensor(
        const std::shared_ptr<tc::InferResult> result,
        const std::string& tensor_name,
        const size_t output_size,
        float** output_raw_data,
        size_t& output_byte_size)
      {
        FAIL_IF_ERR(
            result->RawData(
                tensor_name, (const uint8_t**)output_raw_data,
                &output_byte_size),
            "unable to get result data for 'Output'");
        if (output_byte_size != (sizeof(float) * output_size)) {
          std::cerr << "error: received incorrect byte size for 'Output': "
                    << output_byte_size << " vs. " << output_size <<std::endl;
          exit(1);
        }
      }

      void RetrieveOutputs(
        std::shared_ptr<tc::InferResult> result,
        ModelInfo& info,
        std::vector<std::vector<float>>& output_data)
      {
        // Get pointers to the result returned...
        for (size_t i=0; i<info.output_names.size(); ++i) {
          int64_t output_size = info.output_vols[i];
          float* output_raw_data;
          size_t output_byte_size;
          RetrieveSingleOutputTensor(result, info.output_names[i],
                                      output_size, &output_raw_data,
                                      output_byte_size);
          output_data[i].resize(output_size);
          std::copy(
              output_raw_data, output_raw_data + output_size,
              std::begin(output_data[i]));

          // std::cout << this_sequence_id << "_" << segment_idx << " :: ";
          // for (auto f : output_data[i])
          //   std::cout << f << ", ";
          // std::cout << std::endl;
        }
      }
    } // namespace utils
    
  } // namespace client
  
} // namespace stateful
