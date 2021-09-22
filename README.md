# Triton Stateful Backend
![alt text](stateful_backend.png)

This repository contains code for the Stateful Backend for Triton Inference Server, where the models have matching input and output tensors for the states. The model states are I/O tensors; however, they are treated differently compared to standard I/O tensors. In particular, we do not need to communicate the state tensors between server and client. The server keeps the state tensors on CPU or GPU memory to restore them when a sequence id has an inference request again.

The state tensors are provided in the model configuration file for an example model with two state tensors as below:
```
<<<State0_Input, State1_Output>>> <<<State1_Input, State1_Output>>>   
```

During the model instance initialization, the stateful backend reserves CPU or GPU memory as large as `max_candidate_sequences * sum_of_all_state_tensor_sizes` to store and restore the model state tensors. 

## How to use?
1. Create an ONNX model that exposes input and output state tensors. The model
   should also has a mechanism to reset the initial values of state tensors for
   the beginning of the sequence. See the example model for a reference.
 

1. Create a model config file that matches the ONNX model. The model config file
   only needs to have the standard Input and Outputs excluding the state tensors
   listed. The state pairs are listed as below for the example ONNX model:

```
   {
    key: "state_pairs"
    value: { string_value: "<<<Accumulate_In, Accumulate_Out>>>" }
   }
```

1. Incorporate the model file in Triton's Model Repository

```
        model_repository
        └── accumulate
            ├── 1
            │   └── accumulate.onnx
            └── config.pbtxt

```

## How to build?

## Additional Features 
* Stateful backend can do dynamic batching along any tensor dimension. The batch dimension should be marked with -1 in the model configuration file for the input and output tensors. 
* The state tensors can only have one dynamic dimension that is assumed to be the batch dimension. 
* The ONNX model should contain the initial values for the state tensors. `CONTROL_SEQUENCE_START` control input can be mapped to a model input tensor that signals when to reset the initial values of the states.
* `ort_ep` model config parameter to choose the ORT backend between `trt` and `cuda`
* `compute_precision` model config parameter to specify the precision for ORT `trt` EP
* `always_pad_to_max_batch` model config parameter whether the batch dimension should be padded to max batch size for model execution (set value to `1`)


## Limitations
* Stateful backend only works with ONNX models
* All tensor dimension expect from the batch dimension is fixed for a model instance
