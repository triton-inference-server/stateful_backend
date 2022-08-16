# Triton Stateful Backend

This repository contains the Stateful Backend for Triton Inference Server. You can learn more about backends in the [backend repo](https://github.com/triton-inference-server/backend). Ask questions or report problems in the main Triton [issues page](https://github.com/triton-inference-server/server/issues).

The backend code automatically manages the input and output states of a model. The states are associated with a sequence id and need to be tracked for inference requests associated with the sequence id. The backend currently handles ONNX models, however, it can be extended to other model types. 

![alt text](stateful_backend.png)

The example stateful model above has matching input and output tensors representing the model states. An output state tensor for a sequence id is passed as the input state during the next inference execution of the same sequence id. Therefore, we do not need to communicate the state tensors between server and client, and they can be kept on the GPU (or CPU) memory for GPU (or CPU) execution. The backend code stores the state tensors for all active sequences on the GPU (or CPU) memory and passes the stored state tensors as model input when the sequence id associated with the state tensors has an inference request.

The state tensors are provided in the model configuration file at the `state_pairs` section. For the example model in models/accumulate_fp32, the state tensor input and output pairs are specified in the `parameters` section as below:
```
   {
    key: "state_pairs"
    value: { string_value: "<<<Accumulate_In, Accumulate_Out>>>" }
   }
```
In general, each state pair must be surrounded by 3 pairs of angle brackets and the state pairs must be separated by a space `' '` e.g.
 `"<<<State_In_1, State_Out_1>>> <<<State_In_2, State_Out_2>>> ..."`.

During the model instance initialization, the stateful backend reserves GPU (or CPU) memory as large as `max_candidate_sequences * sum_of_all_state_tensor_sizes` to store  model state tensors. 

## Building the backend
Run:
```
python3 ./build.py
```
The backend binary will be produced in `build/install/backends` directory.

Alternatively, you can do the following steps to build manually:
1. Build the custom docker image which we will use to build the backend:
```
NGC_VERSION=$(head -1 ./NGC_VERSION) # read the container version to use
docker build --tag triton-${NGC_VERSION}-backend -f docker/Dockerfile.backend --build-arg BASE_IMAGE_VERSION=${NGC_VERSION} .
```

2. Create a container of the previously built image:
```
docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v${PWD}:/workspace/stateful triton-${NGC_VERSION}-backend
```

3. Inside the container, run the following:
```
mkdir -p /workspace/stateful/build && cd /workspace/stateful/build
cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
make -j
make install
```

## Using the backend with Triton
1. [Build](#building-the-backend) the backend. Run Triton server docker image, and copy the backend files to the triton [backend folder](https://github.com/triton-inference-server/backend#can-i-add-or-remove-a-backend-to-an-existing-triton-installation). Delete existing onnxruntime backend and set the LD_LIBRARY_PATH variable:
 
```
NGC_VERSION=$(head -1 ./NGC_VERSION) # read the container version to use
docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -p8005:8005 -p8006:8006 -p8007:8007 -v${PWD}:/workspace/stateful_backend nvcr.io/nvidia/tritonserver:${NGC_VERSION}-py3
rm -rf /opt/tritonserver/backends/onnxruntime # Remove existing ORT backend to avoid having two copies
cp  -R /workspace/stateful_backend/build/install/backends/stateful ./backends/ # Copy the stateful backend
export LD_LIBRARY_PATH=/workspace/stateful_backend/build/custom-ort/lib/ # Add ORT to the LD_LIBRARY_PATH
tritonserver --grpc-port 8005 --model-repository /workspace/stateful_backend/models/ # Run the triton inference server
```

 
2. Create an ONNX model that exposes input and output state tensors. The model
   should also have a mechanism to reset the initial values of state tensors for
   the beginning of the sequence. See the example model for a reference.
 

3. Create a model config file that matches the ONNX model. The model config file
   only needs to have the standard Input and Outputs excluding the state tensors
   listed. The state pairs are listed in the `parameters` section. For the example ONNX model:

```
   {
    key: "state_pairs"
    value: { string_value: "<<<Accumulate_In, Accumulate_Out>>>" }
   }
```

4. We also need a mapping between `CONTROL_SEQUENCE_START` to  `ResetSequence`
   boolean input tensor to reset the values of state tensors. If the boolean input tensor is
   set to `true` for an inference request, the input state values will be ignored and the model will use the 
   initial values of the states stored in the ONNX model file as constants. This mapping allows
   the stateful backend to reset the state tensor values for the start of a sequence. 

```
        {
          name: "ResetSequence"
          control [
            {
              kind: CONTROL_SEQUENCE_START
              int32_false_true: [ 0, 1 ]
            }
          ]
        }
```
    

5. Incorporate the model file in Triton's Model Repository

```
        model_repository
        └── accumulate_fp32
            ├── 1
            │   └── accumulate_fp32.onnx
            └── config.pbtxt

```

## Testing the backend
Run: 
```
python3 scripts/test.py
```
It will build the backend, start the tritonserver with the backend, run a simple client with the `accumulate_fp32` model.

## Example Triton model 
models/accumulate_fp32 folder contains a simple Triton model with state tensors and
reset state boolean input. The ONNX file contains a simple accumulation graph
where the input tensor are summed over the last dimension and added to a running
sum. Stateful Backend keeps track of the running sum value for all sequences and
provides the output state (the running sum) as input to the model when the
corresponding sequence has an inference request.

The model configuration file maps `CONTROL_SEQUENCE_START` signal to
`ResetSequence` model input to initialize the state values with 0 constants that are stored
in the ONNX model. The files and folder structure can be used
to serve similar stateful ONNX models.

## Additional features 
* Stateful backend can do dynamic batching along any tensor dimension. The batch dimension should be marked with -1 in the model configuration file for the input and output tensors. 
* The state tensors can only have one dynamic dimension that is assumed to be the batch dimension. 
* The ONNX model should contain the initial values for the state tensors. `CONTROL_SEQUENCE_START` control input can be mapped to a boolean model input tensor
   that signals when to reset the initial values of the states.
* `ort_ep` model config parameter to choose the ORT backend between `trt` and `cuda`
* `compute_precision` model config parameter to specify the precision (`fp32` or `fp16`). `fp16` is only supported for ORT `trt` EP.
* `always_pad_to_max_batch` model config parameter to specify whether the batch dimension should be padded to max batch size for model execution (set value to `1`)
* `store_states_as_fp16` model config paramter to specify whether the internal states are stored as FP16 to reduce memory consumption (set value to `1`).
   However, it may impact the result accuracy.
* `metric_logging_frequency_seconds` controls how frequently the backend logs the inference statistics. Default is `0` to disable such logs.
* `enable_trt_caching` and `trt_cache_dir` to control the engine caching from TRT. The default values are `0` (disabled) and `/tmp` respectively.
* `logging_level` controls the level of details in the backend's model runner logs. Possible values are `NONE`, `INFO` and `VERBOSE`. Default is `INFO`. Two additional parameters to further control some logs:
   * `batch_info_logging_level` controls whether the batch information will be logged during each execution of the model. Possible values are `INFO` and `VERBOSE`. Default is `INFO` which will put these logs into the same stream as the `logging_level`'s `INFO`.
   * `detailed_metrics_logging_level` controls whether some detailed metrics from the backend will be logged or not. Possible values are `INFO` and `VERBOSE`. Default is `VERBOSE` which will put these logs into the same stream as the `logging_level`'s `VERBOSE`. Note that metric logging must 
   be enabled for this. See `metric_logging_frequency_seconds`.
* `max_candidate_sequence_use_ratio` controls whether the number of maximum simultaneous sequences should less than what Triton uses 
   i.e. `max_candidate_sequences`. The ratio can be used to enable prompt error handling in the backend for overloaded servers. Default value for this is `0.9`.
* `infer_end_requests` controls whether to run inference on requests with `end` signal. If specified as `0`, the backend will not run inference 
   for such requests, release the state buffers for those sequences and send successful response with empty output. Default value is `1`.
* Lazy allocation of the states buffer can be enabled and configured using 4 additional **parameters**: `initial_buffer_size`, `subsequent_buffer_size`,
`buffer_alloc_threshold`, and `buffer_dealloc_threshold`.
   * By default, the lazy allocation feature is disabled and the backend allocates
      the whole buffer during the model loading for the `max_candidate_sequences` simultaneous sequences.
   * If `initial_buffer_size` is present in the model config, the other 3 configs must be present as well and
      the lazy allocation behavior is enabled. If enabled, the backend only allocates the buffer for the `initial_buffer_size` sequences during loading.
   * Once the number of free slots reach below the `buffer_alloc_threshold`, it will create a new buffer of size `subsequent_buffer_size`.
   * On the other hand, it will deallocate only the last buffer and the deallocation occurs when the following conditions are met:
      * the number of allocated buffers (including the initial buffer) is more than 1
      * the total number of free slots reaches above the `buffer_dealloc_threshold`
      * all the slots of the last buffer are free i.e. it isn't storing states for any active sequences
   * The initial buffer is never deallocated until the model gets unloaded.
   * For example, the following could be a valid combinations for the config parameters when `max_batch_size` is 16:
      ```
         parameters [
         ...
            {
               key: "initial_buffer_size"
               value: { string_value: "512" }
            },
            {
               key: "subsequent_buffer_size"
               value: { string_value: "100" }
            },
            {
               key: "buffer_alloc_threshold"
               value: { string_value: "32" }
            },
            {
               key: "buffer_dealloc_threshold"
               value: { string_value: "200" }
            },
         ...
         ]
      ```
   * Refer to the [code](https://github.com/triton-inference-server/stateful_backend/blob/201df524a08e9c5410772eb7e742573e56846c7a/src/onnx_model_runner.cc#L192) 
   for any more restrictions on these parameters.
* ORT Environment is created during the backend initialization so that multiple models can share the same environment. To control the ORT logging level, 
you can pass a command line argument when starting Triton, e.g. `tritonserver --model-repository <model dir> --backend-config=stateful,ort-logging-level=3` 
will set the ORT logging level to `ORT_LOGGING_LEVEL_ERROR`. Refer to [this](https://github.com/microsoft/onnxruntime/blob/5868413caf802bbd7f1dc0762b402988258d61d9/include/onnxruntime/core/session/onnxruntime_c_api.h#L207) for all possible values. The default logging level is `ORT_LOGGING_LEVEL_WARNING`.


## Limitations
* Stateful backend only works with ONNX models
* All tensor dimensions except for the batch dimension is fixed for a model instance
* Model state reset tensor should be a boolean tensor
* Only `oldest` sequence batching strategy is supported
* **NOTE**: The backend cannot be loaded/unloaded repeatedly (e.g. Triton's explicit loading mode) since ORT doesn't allow destroying 
and then re-creating Ort::Env objects. This should be mitigated once Triton is updated so that model unloading doesn't trigger the backend unloading. 
Also, for the same reason, beware of using this backend along with any other ORT-based backend in the same Triton instance. May cause stability issues in ORT.
