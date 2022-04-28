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

#include <unistd.h>
#include <condition_variable>
#include <iostream>
#include <string>
#include <vector>
#include "grpc_client.h"

#include "model_config_util.h"

namespace tc = triton::client;

using ResultList = std::vector<std::shared_ptr<tc::InferResult>>;

// Model information for accumulate.onnx
static stateful::client::utils::ModelInfo info;
static int SEGMENT_LEN = 4;
static int STATE_DIM = 5;

// Global mutex to synchronize the threads
static std::mutex mutex_;
static std::condition_variable cv_;

namespace {

std::vector<std::vector<std::vector<std::vector<float>>>>
simulate_model(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& input_data,
    const int num_sequence, const int num_segment)
{
  /*
      'accumulate' model does the following:
        1. accumulates the input along the sequence dim
        2. adds the accumulation to the internal states (initialized to 0 for
     start of sequence)
        3. update the internal states with result of step 2
        4. add the updated states to the original input which will be the final
     output
   */
  std::vector<std::vector<std::vector<std::vector<float>>>> output(
      num_sequence,
      std::vector<std::vector<std::vector<float>>>(
        num_segment,
        std::vector<std::vector<float>>(info.output_names.size())));
  std::vector<float> states(STATE_DIM);
  for (int seq_idx = 0; seq_idx < num_sequence; ++seq_idx) {
    // initialize the states
    for (int sti = 0; sti < STATE_DIM; ++sti) states[sti] = 0.0f;

    for (int seg_idx = 0; seg_idx < num_segment; ++seg_idx) {
      output[seq_idx][seg_idx][0].resize(SEGMENT_LEN * STATE_DIM); // 1 out tnsr
      for (int j = 0; j < STATE_DIM; ++j) {
        float sum = states[j];
        // Reduce
        for (int i = 0; i < SEGMENT_LEN; ++i) {
          sum += input_data[seq_idx][seg_idx][0][j + i*STATE_DIM];
        }
        // update states
        states[j] = sum;
        // calculate output
        for (int i = 0; i < SEGMENT_LEN; ++i) {
          output[seq_idx][seg_idx][0][j + i * STATE_DIM] =
              states[j] +
               input_data[seq_idx][seg_idx][0][j + i * STATE_DIM];
        }
      }
    }
  }
  return output;
}

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service and its gRPC port>"
            << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;
  std::cerr << "\t-t <stream timeout in microseconds>" << std::endl;
  std::cerr << "\t-o <offset for sequence ID>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "For -o, the client will use sequence ID <1 + 2 * offset> "
            << "and <2 + 2 * offset>. Default offset is 0." << std::endl;

  exit(1);
}

void
StreamSend(
    const std::unique_ptr<tc::InferenceServerGrpcClient>& client,
    const std::string& model_name, std::vector<std::vector<float>>& values,
    const uint64_t sequence_id,
    bool start_of_sequence, bool end_of_sequence, const int32_t index)
{
  tc::InferOptions options(model_name);
  if (info.is_corrid_string) {
    options.sequence_id_str_ = std::to_string(sequence_id);
  }
  else {
    options.sequence_id_ = sequence_id;
  }
  options.sequence_start_ = start_of_sequence;
  options.sequence_end_ = end_of_sequence;
  options.request_id_ =
      std::to_string(sequence_id) + "_" + std::to_string(index);

  // Initialize the inputs with the data.
  std::vector<tc::InferInput*> inputs;
  stateful::client::utils::PrepareInputs(inputs, info, values);

  // Send inference request to the inference server.
  FAIL_IF_ERR(client->AsyncStreamInfer(options, inputs), "unable to run model");
}

}  // namespace

int
main(int argc, char** argv)
{
  int NUM_SEQUENCE = 2;
  int NUM_SEGMENT = 3;
  bool verbose = false;
  std::string url("localhost:8001");
  tc::Headers http_headers;
  int sequence_id_offset = 1;
  uint32_t stream_timeout = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vu:H:t:o:S:s:")) != -1) {
    switch (opt) {
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case 't':
        stream_timeout = std::stoi(optarg);
        break;
      case 'o':
        sequence_id_offset = std::stoi(optarg);
        break;
      case 'S':
        NUM_SEQUENCE = std::stoi(optarg);
        break;
      case 's':
        NUM_SEGMENT = std::stoi(optarg);
        break;
      case '?':
        Usage(argv);
        break;
    }
  }

  tc::Error err;

  // We use the custom "sequence" model which takes 1 input value. The
  // output is the accumulated value of the inputs. See
  // src/custom/sequence.
  std::string model_name = "accumulate";
  std::cout << "Using model: " << model_name << std::endl;
  std::cout << "Number of Sequences: " << NUM_SEQUENCE << std::endl;
  std::cout << "Number of Segments: " << NUM_SEGMENT << std::endl;


  std::vector<uint64_t> sequence_ids;
  for (uint64_t i = 0; i < NUM_SEQUENCE; ++i) {
    sequence_ids.push_back(i + sequence_id_offset);
  }

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
  std::unique_ptr<tc::InferenceServerGrpcClient> client;
  FAIL_IF_ERR(
      tc::InferenceServerGrpcClient::Create(&client, url, verbose),
      "unable to create grpc client");

  info = stateful::client::utils::GetModelInfo(client, model_name);
  // NOTE: Assuming first positive dim is the 'seq' dim
  for (auto& d : info.input_dims[0]) {
    if (d > 0) {
      SEGMENT_LEN = d;
      break;
    }
  }
  // std::cout << info.to_string() << std::endl;

  ResultList result_list;

  FAIL_IF_ERR(
      client->StartStream(
          [&](tc::InferResult* result) {
            {
              std::shared_ptr<tc::InferResult> result_ptr(result);
              std::lock_guard<std::mutex> lk(mutex_);
              result_list.push_back(result_ptr);
            }
            cv_.notify_all();
          },
          false /*ship_stats*/, stream_timeout, http_headers),
      "unable to establish a streaming connection to server");

  // initialize input
  std::vector<
    std::vector<
      std::vector<
        std::vector<float>>>> input_data(
          NUM_SEQUENCE,
          std::vector<std::vector<std::vector<float>>>(
            NUM_SEGMENT,
            std::vector<std::vector<float>>(info.input_names.size())));
  for (int seg_idx = 0; seg_idx < NUM_SEGMENT; ++seg_idx) {
    for (int seq_idx = 0; seq_idx < NUM_SEQUENCE; ++seq_idx) {
      for (size_t i=0; i<info.input_names.size(); ++i) {
        std::vector<float>& input_values = input_data[seq_idx][seg_idx][i];
        size_t input_size = info.input_vols[i];
        input_values.resize(input_size);
        for (size_t j = 0; j < input_size; ++j) {
          input_values[j] = j + seg_idx * 100 + seq_idx * 1000;
        }

        // std::cout << seq_idx+sequence_id_offset << "_"
        //           << seg_idx <<" (IN) :: ";
        // for (auto f : input_values)
        //   std::cout << f << ", ";
        // std::cout << std::endl;
      }
    }
  }

  // send the requests
  for (int seg_idx = 0; seg_idx < NUM_SEGMENT; ++seg_idx) {
    for (int seq_idx = 0; seq_idx < NUM_SEQUENCE; ++seq_idx) {
      StreamSend(
          client, model_name, input_data[seq_idx][seg_idx],
          seq_idx + sequence_id_offset, seg_idx == 0,
          info.infer_end_requests && seg_idx == (NUM_SEGMENT - 1), seg_idx);
    }
  }
  // If infer_end_requests=0, we need to send the end signal now
  // we can send garbage data since inference won't be run
  if (!info.infer_end_requests) {
    for (int seq_idx = 0; seq_idx < NUM_SEQUENCE; ++seq_idx) {
      StreamSend(
          client, model_name, input_data[seq_idx][0],
          seq_idx + sequence_id_offset, false,
          true, NUM_SEGMENT);
    }
  }

  size_t expected_result_count = NUM_SEGMENT * NUM_SEQUENCE;
  if (!info.infer_end_requests) {
    // we need to count the extra end signals
    expected_result_count += NUM_SEQUENCE;
  }
  // wait for all the requests to be done
  if (stream_timeout == 0) {
    // Wait until all callbacks are invoked
    {
      std::unique_lock<std::mutex> lk(mutex_);
      cv_.wait(lk, [&]() {
        if (result_list.size() >= expected_result_count) {
          return true;
        } else {
          return false;
        }
      });
    }
  } else {
    auto timeout = std::chrono::microseconds(stream_timeout);
    // Wait until all callbacks are invoked or the timeout expires
    {
      std::unique_lock<std::mutex> lk(mutex_);
      if (!cv_.wait_for(lk, timeout, [&]() {
            return (result_list.size() >= expected_result_count);
          })) {
        std::cerr << "Stream has been closed" << std::endl;
        exit(1);
      }
    }
  }

  // Extract data from the result
  std::vector<std::vector<std::vector<std::vector<float>>>> output_data(
      NUM_SEQUENCE,
      std::vector<std::vector<std::vector<float>>>(
        NUM_SEGMENT,
        std::vector<std::vector<float>>(info.output_names.size())));
  for (const auto& this_result : result_list) {
    auto err = this_result->RequestStatus();
    if (!err.IsOk()) {
      std::cerr << "The inference failed: " << err << std::endl;
      exit(1);
    }

    std::string request_id;
    FAIL_IF_ERR(
        this_result->Id(&request_id), "unable to get request id for response");
    uint64_t this_sequence_id =
        std::stoi(std::string(request_id, 0, request_id.find("_")));
    uint64_t sequence_idx = this_sequence_id - sequence_id_offset;
    uint64_t segment_idx =
        std::stoi(std::string(request_id, request_id.find("_") + 1));
    if (sequence_idx >= 0 && sequence_idx < NUM_SEQUENCE && segment_idx >= 0 &&
        segment_idx < NUM_SEGMENT) {
      stateful::client::utils::RetrieveOutputs(
        this_result, info, output_data[sequence_idx][segment_idx]);
    } else if (!info.infer_end_requests && segment_idx == NUM_SEGMENT) {
      // ignore response for end signals
    } else {
      std::cerr << "error: received incorrect sequence id in response: "
                << this_sequence_id << std::endl;
      exit(1);
    }
  }

  std::vector<std::vector<std::vector<std::vector<float>>>> expected_output =
      simulate_model(input_data, NUM_SEQUENCE, NUM_SEGMENT);

  for (int seq_idx = 0; seq_idx < NUM_SEQUENCE; ++seq_idx) {
    for (int seg_idx = 0; seg_idx < NUM_SEGMENT; ++seg_idx) {
      int64_t output_size = info.output_vols[0];
      for (int i = 0; i < output_size; ++i) {
        if (output_data[seq_idx][seg_idx][0][i] !=
            expected_output[seq_idx][seg_idx][0][i]) {
          std::cerr << "FAILED!" << std::endl;
          std::cerr << "Sequence " << sequence_ids[seq_idx] << " Segment "
                    << seg_idx << " : expected "
                    << expected_output[seq_idx][seg_idx][0][i] << " , got "
                    << output_data[seq_idx][seg_idx][0][i] << std::endl;
          return 1;
        }
      }
    }
  }
  std::cout << "PASSED" << std::endl;

  return 0;
}