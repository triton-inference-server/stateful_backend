# Triton Stateful Backend
![alt text](stateful_backend.png)

This repository contains code for the Stateful Backend for Triton Inference Server, where the models have matching input and output tensors to keep track of the model states. The model states are input and output tensors. However, they are treated differently compared to standard input and output tensors. An output state tensor for a sequence id is passed as input state during the next inference execution of the same sequence id. In addition, we do not need to communicate the state tensors between server and client. The server keeps the state tensors for all active sequences on CPU or GPU memory to restore them when a sequence id has an inference request.

The state tensors are provided in the model configuration file for an example model with two state tensors as below:
```
<<<State0_Input, State1_Output>>> <<<State1_Input, State1_Output>>>   
```

During the model instance initialization, the stateful backend reserves CPU or GPU memory as large as `max_candidate_sequences * sum_of_all_state_tensor_sizes` to store and restore the model state tensors. 

## How to use?
1. Create an ONNX model that exposes input and output state tensors. The model
   should also has a mechanism to reset the initial values of state tensors for
   the beginning of the sequence. See the example model for a reference.
 

2. Create a model config file that matches the ONNX model. The model config file
   only needs to have the standard Input and Outputs excluding the state tensors
   listed. The state pairs are listed as below for the example ONNX model:

```
   {
    key: "state_pairs"
    value: { string_value: "<<<Accumulate_In, Accumulate_Out>>>" }
   }
```

3. Incorporate the model file in Triton's Model Repository

```
        model_repository
        └── accumulate
            ├── 1
            │   └── accumulate.onnx
            └── config.pbtxt

```

## How to build?
### How to build the backend?
Run:
```
$ python3 ./build.py
```
The backend binary will be produced in `build/install/backends` directory.

Alternatively, you can do the following steps to build manually (NOTE: we will be using 21.08 NGC containers):
1. Build the custom docker image which we will use to build the backend:
```
$ docker build --tag triton-21.08-backend -f docker/Dockerfile.backend .
```

2. Create a container of the previously built image:
```
$ docker run --gpus all -it --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 -v${PWD}:/workspace/stateful triton-21.08-backend
```

3. Inside the container, run the following:
```
$ mkdir -p /workspace/stateful/build && cd /workspace/stateful/build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_BACKEND_REPO_TAG="r21.08" -DTRITON_CORE_REPO_TAG="r21.08" -DTRITON_COMMON_REPO_TAG="r21.08" ..
$ make -j
$ make install
```

### How to build and test the backend?
Run: 
```
$ python3 scripts/test.py
```
It will build the backend, start the tritonserver with the backend, run a simple client with the accumulate model.

## Example Triton Model 
models/accumulate folder contains a simple Triton model with state tensors and
reset state boolean input. The ONNX file contains a simple accumulation graph
where the input tensor are summed over the last dimension and added to a running
sum. Stateful Backend keeps track of the running sum value for all sequences and
provides the output state (the running sum) as input to the model when the
corresponding sequence has an inference request.

The model configuration file maps `CONTROL_SEQUENCE_START` signal to
`ResetState` model input to initialize the state values with 0 constants stored
in the ONNX model. The files and folder structure can be used
to serve similar stateful ONNX models.

## Additional Features 
* Stateful backend can do dynamic batching along any tensor dimension. The batch dimension should be marked with -1 in the model configuration file for the input and output tensors. 
* The state tensors can only have one dynamic dimension that is assumed to be the batch dimension. 
* The ONNX model should contain the initial values for the state tensors. `CONTROL_SEQUENCE_START` control input can be mapped to a boolean model input tensor that signals when to reset the initial values of the states.
* `ort_ep` model config parameter to choose the ORT backend between `trt` and `cuda`
* `compute_precision` model config parameter to specify the precision for ORT `trt` EP
* `always_pad_to_max_batch` model config parameter whether the batch dimension should be padded to max batch size for model execution (set value to `1`)


## Limitations
* Stateful backend only works with ONNX models
* All tensor dimension expect from the batch dimension is fixed for a model instance
* Only float (FP32) state tensors are supported
* Model state reset tensor should be a boolean tensor
* Only oldest sequence batching strategy is supported
