from functools import partial
import argparse
import grp
import numpy as np
import sys
import queue
import copy
import uuid
from collections import namedtuple, defaultdict

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

FLAGS = None
model_info = None

class UserData:
  def __init__(self):
    self._completed_requests = queue.Queue()

def simulate_model(inputs, input_name, output_idx):
  output_type = model_info.output_types[output_idx]
  np_dtype = serverType2npType(output_type)
  STATE_DIM = model_info.output_dims[output_idx][-1]
  SEGMENT_LEN = 4
  outputs = defaultdict(lambda: defaultdict(dict))
  for seqi in range(FLAGS.num_sequence):
    states = np.zeros((STATE_DIM), np_dtype)
    for segi in range(FLAGS.num_segment):
      # print(seqi, segi)
      input = inputs[seqi][segi][input_name]
      output = np.zeros(input.shape, np_dtype)
      # print("input\n", input)
      # print("in-states\n", states)
      reduced = np.sum(np.reshape(input, (SEGMENT_LEN, STATE_DIM)), axis=0)
      # print("reduced\n", reduced)
      states = states + reduced
      output = input + states
      outputs[seqi][segi] = output.astype(np_dtype)
      # print("out-states\n", states)
      # print("output\n", output)
  return outputs

def parse_model_config(config):
  # print(config)
  info = namedtuple("ModelInfo", ["is_corrid_string", "infer_end_requests",\
    "input_vols", "output_vols", "input_dims", "output_dims",\
    "input_names", "output_names","input_types", "output_types"])
  info.is_corrid_string = False
  try:
    for c in config["config"]["sequence_batching"]["control_input"]:
      if c["name"] == "CORRID":
        info.is_corrid_string = c["control"][0]["data_type"] == "TYPE_STRING"
        break
  except:
    print("Error in corrid type!!", flush=True)
    exit(7)
  print("is_corrid_string : ", info.is_corrid_string, flush=True)
  info.infer_end_requests = True
  try:
    info.infer_end_requests = config["config"]["parameters"]["infer_end_requests"]["string_value"] != "0"
  except:
    pass
  print("infer_end_requests : ", info.infer_end_requests, flush=True)
  try:
    info.input_names = []
    info.output_names = []
    info.input_vols = []
    info.output_vols = []
    info.input_dims = []
    info.output_dims = []
    info.input_types = []
    info.output_types = []
    print("Inputs:", flush=True)
    for input in config["config"]["input"]:
      info.input_names.append(input["name"])
      print("    ", info.input_names[-1], end=' : [', flush=True)
      vol = 1
      dims = []
      for ds in input["dims"]:
        di = int(ds)
        dims.append(di)
        print(di, end=', ', flush=True)
        if di > 0:
          vol *= di
      print("]", end='', flush=True)
      info.input_dims.append(dims)
      info.input_vols.append(vol)
      print(", Volume : ", info.input_vols[-1], end='', flush=True)
      info.input_types.append(input["data_type"])
      print(", DataType : ", info.input_types[-1], flush=True)
    print("Outputs:", flush=True)
    for output in config["config"]["output"]:
      info.output_names.append(output["name"])
      print("    ", info.output_names[-1], end=' : [', flush=True)
      vol = 1
      dims = []
      for ds in output["dims"]:
        di = int(ds)
        dims.append(di)
        print(di, end=', ', flush=True)
        if di > 0:
          vol *= di
      print("]", end='', flush=True)
      info.output_dims.append(dims)
      info.output_vols.append(vol)
      print(", Volume : ", info.output_vols[-1], end='', flush=True)
      info.output_types.append(output["data_type"])
      print(", DataType : ", info.output_types[-1], flush=True)
  except:
    print("Error in IO info!!", flush=True)
    exit(11)
  return info

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-v',
                      '--verbose',
                      action="store_true",
                      required=False,
                      default=False,
                      help='Enable verbose output')
  parser.add_argument('-u',
                      '--url',
                      type=str,
                      required=False,
                      default='localhost:8005',
                      help='Inference server URL and it gRPC port. '\
                            'Default is localhost:8001.')
  parser.add_argument('-t',
                      '--stream-timeout',
                      type=float,
                      required=False,
                      default=None,
                      help='Stream timeout in seconds. Default is None.')
  parser.add_argument('-m',
                      '--model-name',
                      type=str,
                      required=False,
                      default="mixed_io_int32_fp16",
                      help='Name of the model.')
  parser.add_argument('-o',
                      '--offset',
                      type=int,
                      required=False,
                      default=1,
                      help='Add offset to sequence ID used.')
  parser.add_argument('-S',
                      '--num-sequence',
                      type=int,
                      required=False,
                      default=2,
                      help='Number of sequences to use.')
  parser.add_argument('-s',
                      '--num-segment',
                      type=int,
                      required=False,
                      default=3,
                      help='Number of segments per sequence.')

  global FLAGS
  FLAGS = parser.parse_args()
  return

# Define the callback function. Note the last two parameters should be
# result and error. InferenceServerClient would povide the results of an
# inference as grpcclient.InferResult in result. For successful
# inference, error will be None, otherwise it will be an object of
# tritonclientutils.InferenceServerException holding the error details
def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)

def serverType2npType(stype):
  npType = None
  if stype == "TYPE_FP32":
    npType = np.float32
  elif stype == "TYPE_FP16":
    npType = np.float16
  elif stype == "TYPE_INT32":
    npType = np.int32
  elif stype == "TYPE_INT8":
    npType = np.int8
  elif stype == "TYPE_BOOL":
    npType = np.bool8
  return npType

def serverType2clientType(stype):
  return stype[5:]

def async_stream_send(triton_client, model_name, values, iseq_id,
                      start_seq, end_seq, seg_index):
  # prepare sequence_id properly
  if model_info.is_corrid_string:
    sequence_id = str(iseq_id)
  else:
    sequence_id = iseq_id
  inputs = []
  for i in range(len(model_info.input_names)):
    # print(sequence_id, seg_index, i)
    input_name = model_info.input_names[i]
    shape = list(values[input_name].shape)
    shape.insert(0, 1) # prepend an extra 1 in the tensor dim (Triton batch dim)
    input = grpcclient.InferInput(
            input_name, shape,
            serverType2clientType(model_info.input_types[i]))
    data = np.expand_dims(values[input_name], axis=0)
    input.set_data_from_numpy(data)
    inputs.append(input)
  # print(inputs)
  try:
    triton_client.async_stream_infer(model_name=model_name,
                                      inputs=inputs,
                                      request_id=f'{sequence_id}_{seg_index}',
                                      sequence_id=sequence_id,
                                      sequence_start=start_seq,
                                      sequence_end=end_seq)
  except Exception as ex:
    print(ex)
    raise ex
  return

def init_inputs():
  np.random.seed(FLAGS.num_segment*7 + FLAGS.num_sequence*11 + 13)
  inputs = defaultdict(lambda: defaultdict(dict))
  for segi in range(FLAGS.num_segment):
    for seqi in range(FLAGS.num_sequence):
      for i in range(len(model_info.input_names)):
        input_name = model_info.input_names[i]
        dims = copy.deepcopy(model_info.input_dims[i])
        for di in range(len(dims)):
          if dims[di] < 0:
            dims[di] = 1
        itype = serverType2npType(model_info.input_types[i])
        if itype in [np.int32]:
          inputs[seqi][segi][input_name] =\
            np.random.randint(10, size=dims).astype(itype)
        elif itype in [np.int8]:
          inputs[seqi][segi][input_name] =\
            np.random.randint(4, size=dims).astype(itype)
        elif itype == np.bool_:
          inputs[seqi][segi][input_name] =\
            np.random.randint(2, size=dims).astype(itype)
        else:
          inputs[seqi][segi][input_name] = np.random.rand(*dims).astype(itype)
  return inputs

def main():
  parse_args()
  print(FLAGS)
  user_data = UserData()
  # It is advisable to use client object within with..as clause
  # when sending streaming requests. This ensures the client
  # is closed when the block inside with exits.
  with grpcclient.InferenceServerClient(
          url=FLAGS.url, verbose=FLAGS.verbose) as triton_client:
    try:
      global model_info
      model_info = parse_model_config(triton_client.get_model_config(FLAGS.model_name, as_json=True))

      # Establish stream
      triton_client.start_stream(callback=partial(callback, user_data),
                                  stream_timeout=FLAGS.stream_timeout)
      # initialize inputs
      input_data = init_inputs()
      # send the requests
      for segi in range(FLAGS.num_segment):
        for seqi in range(FLAGS.num_sequence):
          is_end = model_info.infer_end_requests and segi==(FLAGS.num_segment-1)
          async_stream_send(triton_client, FLAGS.model_name, 
                            input_data[seqi][segi], seqi + FLAGS.offset,
                            segi==0, is_end, segi)
      # if infer_end_requests=0, we need to send the end signal now
      if not model_info.infer_end_requests:
        for seqi in range(FLAGS.num_sequence):
          async_stream_send(triton_client, FLAGS.model_name, 
                            input_data[seqi][0], seqi + FLAGS.offset,
                            False, True, FLAGS.num_segment)


    except InferenceServerException as error:
      print(error)
      sys.exit(1)

  # # Retrieve results...
  infer_outputs = defaultdict(lambda: defaultdict(dict))
  recv_count = 0
  while recv_count < (FLAGS.num_sequence * FLAGS.num_segment):
    data_item = user_data._completed_requests.get()
    if type(data_item) == InferenceServerException:
      print(data_item, " Result Idx: ", recv_count)
      sys.exit(1)
    else:
      try:
        data_item: grpcclient.InferResult = data_item # just for syntax
        this_id = data_item.get_response().id.split('_')
        seqi = int(this_id[0]) - FLAGS.offset
        segi = int(this_id[1])
        # print(this_id, seqi, segi)
        if seqi < FLAGS.num_sequence:
          if segi < FLAGS.num_segment:
            for output_name in model_info.output_names:
              # print("Getting ", output_name, flush=True)
              infer_outputs[seqi][segi][output_name] =\
                data_item.as_numpy(output_name)
              # if output_name.find("int") >= 0:
              #   print(infer_outputs[seqi][segi][output_name])
          elif segi == FLAGS.num_segment and\
                  not model_info.infer_end_requests:
            continue # ignore the end signal response (should be empty)
          else:
            print("Invalid segment ID for request: {}".format(this_id))
            sys.exit(1)
        else:
          print("Invalid sequence ID for request: {}".format(this_id))
          sys.exit(1)
      except ValueError as ex:
        print("ERROR : ", ex)

      recv_count = recv_count + 1

  for i in range(len(model_info.input_names)):
    # simulate model
    input_name = model_info.input_names[i]
    output_name = model_info.output_names[i]
    print("Checking ", input_name, output_name)
    expected_outputs = simulate_model(input_data, input_name, i)
    # compare results with simulated output
    for seqi in range(FLAGS.num_sequence):
      for segi in range(FLAGS.num_segment):
        if not np.allclose(infer_outputs[seqi][segi][output_name],\
                            expected_outputs[seqi][segi],\
                            atol=6e-2 if serverType2npType(\
                              model_info.input_types[i]) else 1e-4):
          print("Inequality found at ", seqi, segi, output_name)
          print("Inferred: ")
          print(infer_outputs[seqi][segi][output_name])
          print("    VS.    ")
          print("Expected: ")
          print(expected_outputs[seqi][segi])
          sys.exit(1)

  print("PASSED")

  return


if __name__ == '__main__':
  main()