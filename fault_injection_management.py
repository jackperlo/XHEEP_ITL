import numpy as np
import tensorflow as tf
import json

from constants import WEIGHTS_OUTPUT_PATH
from constants import OUTPUT_FI_FILES_PATH
from constants import INPUT_IMAGES_PATH
from constants import MODELS

def manage_fault_injection_files(model: tf.lite.Interpreter, network, model_name, target_layer, input_tensor_path=None):
  """
    Collect and save each <input, weight> involved into a convolution of the specified network
    and layer, and their indexes.
    Note: input and weight tensors must be already present to correctly run this function

    Args:
      model (tf.lite.Interpreter): the model to extract the (input,weight) pairs from.
      network (Network): the network object containing the layer names.
      model_name (str): the name of the CNN under exam
      target_layer (str): the names of the layer to save the (input,weight) pairs for.
      input_tensor_path (str): path where the input tensor of the layer under exam is saved 
                              (if None, the input_image of the model is loaded as input tensor)
  """
  save_mul_indexes(model, network, target_layer, model_name)
  if input_tensor_path is None:
    save_mul(target_layer, model_name)
  else:
    save_mul(target_layer, model_name, input_tensor_path)

def save_mul(target_layer, model_name, input_tensor_path=None):
  """
  This function save a .txt file containing the list of multiplications' factors
  involved during the convolution operation of the specified target layer

  Args:
    target_layer (str): the names of the layer to save the (input,weight) pairs for.
    model_name (str): the name of the CNN under exam
    input_tensor_path (str): path where the input tensor of the layer under exam is saved
                            (if None, the input_image of the model is loaded as input tensor)

  e.g.:
    0xFFFFFF91 0xFFFFFFB7
    0xFFFFFFB5 0xFFFFFFDF
    0xFFFFFFCD 0x00000019
  """
  output_mults_path = OUTPUT_FI_FILES_PATH+target_layer+"/"+model_name+"_"+target_layer+"_mults.txt"
  input_mul_indexes_path = OUTPUT_FI_FILES_PATH+target_layer+"/"+model_name+"_"+target_layer+"_mul_indexes.json"
  input_weight_tensor_path = OUTPUT_FI_FILES_PATH+target_layer+"/"+model_name+"_"+target_layer+"_weight_tensor.npy"
  input_in_image_tensor_path = OUTPUT_FI_FILES_PATH+target_layer+"/"+model_name+"_"+target_layer+"_input_tensor.npy"

  # loading tensors 
  if input_tensor_path is None:
    input_tensor = np.load(input_in_image_tensor_path) 
  else:
    input_tensor = np.load(input_tensor_path) 
  weight_tensor = np.load(input_weight_tensor_path)
  
  # input tensor shape and dtype checks
  if input_tensor.shape != (1,32,32,1):
    input_tensor = np.expand_dims(input_tensor, axis=0)
  if input_tensor.dtype == np.float32:
    input_tensor = input_tensor.astype(np.int8)
  
  # loading multiplication indexes
  with open(input_mul_indexes_path, 'r') as mul_indexes_file:
    mul_indexes = json.load(mul_indexes_file)
    
  def val_to_hex(num):
    return f'0x{num & 0xFFFFFFFF:08X}'

  content = ""
  tuple_data = [tuple((tuple(inner[0]), tuple(inner[1]))) for inner in mul_indexes]
  for t in tuple_data:
    i_idx0, i_idx1, i_idx2, i_idx3 = t[0]
    w_idx0, w_idx1, w_idx2, w_idx3 = t[1]
    content+=val_to_hex(input_tensor[i_idx0, i_idx1, i_idx2, i_idx3])+" "
    content+=val_to_hex(weight_tensor[w_idx0, w_idx1, w_idx2, w_idx3])+"\n"

  with open(output_mults_path, 'w') as mul_file:
    mul_file.write(content)


def save_mul_indexes(model: tf.lite.Interpreter, network, layer, model_name):
  """
  This function save a .json file which maps a list of tuples representing 
  all the multiplications performed between the corresponding input location index (as a tuple)
  and the corresponding weight location index (as another tuple).

  Args:
    model (tf.lite.Interpreter): the TensorFlow Lite interpreter object.
    network: the network object containing details about the layers.
    layer (str): the target layer of the considered model.
    model_name (str): name of the model being considered

  e.g.:
    [
      (
        (<batch_number>,<Input_height_index>,<Input_width_index>,<n_channel_in>)
        ,
        (<n_channel_out>,<Kernel_height_index>,<Kernel_height_index>,<filter_number>)
      ),
      ...
    ]
  """
  mul_indexes_path = OUTPUT_FI_FILES_PATH+layer+"/"+model_name+"_"+layer+"_mul_indexes.json"
  input_weight_pairs = dict()

  tensors = model.get_tensor_details()
  ops = model._get_ops_details()

  # get the operation index, strides and padding for by the specified layer 
  op_index, _, _ = network.get_conv_layer_details(layer)
  strides = network.get_conv_layer_strides(layer)
  padding = network.get_conv_layer_padding(layer)

  # get the index of the input tensor, kernel tensor, bias tensor and output tensor
  input_tensor_idx = ops[op_index]['inputs'][0] 
  kernel_tensor_idx = ops[op_index]['inputs'][1]
  bias_tensor_idx = ops[op_index]['inputs'][2]
  output_tensor_idx = ops[op_index]['outputs'][0]

  def check_indexes_goodness(ops, tensors, indexes):
    op_index, input_tensor_idx, kernel_tensor_idx, bias_tensor_idx, output_tensor_idx = indexes
    # check for the got layer index goodness
    if ops[op_index]['index'] != op_index:
      print('ERROR: CNN layer index not compatible with submitted target layer index')
      exit(-1)
    # check the got tensor indexes goodness
    if tensors[input_tensor_idx]['index'] != input_tensor_idx:
      print('ERROR: input tensor index of the considered convolution layer not compatible with the expected one')
      exit(-1)
    if tensors[kernel_tensor_idx]['index'] != kernel_tensor_idx:
      print('ERROR: weight tensor(kernel) index of the considered convolution layer not compatible with the expected one')
      exit(-1)
    if tensors[bias_tensor_idx]['index'] != bias_tensor_idx:
      print('ERROR: bias tensor index of the considered convolution layer not compatible with the expected one')
      exit(-1)
    if tensors[output_tensor_idx]['index'] != output_tensor_idx:
      print('ERROR: output tensor index of the considered convolution layer not compatible with the expected one')
      exit(-1)
  check_indexes_goodness(ops, tensors, [op_index, input_tensor_idx, kernel_tensor_idx, bias_tensor_idx, output_tensor_idx])
  
  # get the input and kernel shapes
  input_tensor_shape = tensors[input_tensor_idx]['shape']
  kernel_tensor_shape = tensors[kernel_tensor_idx]['shape']
  output_tensor_shape = tensors[output_tensor_idx]['shape']
  
  # manage the case in which padding must be added at the input tensor in the convolutional operation
  if padding['type'] == "same":
    input_tensor_shape[1] += (padding['values'][0]+padding['values'][2]) # top, bottom padding
    input_tensor_shape[2] += (padding['values'][1]+padding['values'][3]) # left, right padding

  # get the pairs (input, weight) for each output feature map expected
  input_weight_pairs = []
  for output_number in range(kernel_tensor_shape[0]): 
    for output_height in range(output_tensor_shape[1]):
      for output_width in range(output_tensor_shape[2]):
        for n_channel_in in range(input_tensor_shape[3]):
          for base_coord_h in range(kernel_tensor_shape[1]):
            for base_coord_w in range(kernel_tensor_shape[2]):
              input = np.array([0,base_coord_h+output_height+strides[0]-1,base_coord_w+output_width+strides[1]-1,n_channel_in], dtype=np.int8)
              weight = np.array([output_number,base_coord_h,base_coord_w,n_channel_in], dtype=np.int8)
              input_weight_pairs.append([input, weight])
    
  input_weight_pairs_serializable = [[a.tolist(), b.tolist()] for a, b in input_weight_pairs]
  with open(mul_indexes_path, 'w') as mul_indexes_file:
    json.dump(input_weight_pairs_serializable, mul_indexes_file)


def save_weight_as_tensor(model_name, target_layer):
  """
  UNUSED 
  Starting from the hex dump of a given weight tensor, saves the weight 
  tensor of the specified layer of the specified CNN as a .npy file

  Args:
      model_name (str): name of the model being considered
      target_layer (str): name of the layer being considered
  """
  input_weight_file_path = WEIGHTS_OUTPUT_PATH+"_hex/"+model_name+"_"+target_layer+"_weights.txt"
  output_weight_tensor_path = OUTPUT_FI_FILES_PATH+model_name+"_"+target_layer+"_weight_tensor"
  
  # print the int8 weight tensor to file
  with(open(input_weight_file_path, 'r')) as input_weight_file:
    hex_weights = input_weight_file.read().strip().replace('\n', '')
    hex_array = np.array([int(hex_weights[i:i+8], 16) for i in range(0, len(hex_weights), 8)], dtype=np.int8)
    tensor = hex_array[:np.prod(MODELS[model_name][3])].reshape(MODELS[model_name][3])

  np.save(output_weight_tensor_path, tensor)

def save_input_as_tensor(model_name, target_layer):
  """
  UNUSED
  Starting from the hex dump of a given input tensor, saves the input 
  tensor of the specified layer of the specified CNN as a .npy file

  Args:
      model_name (str): name of the model being considered
      target_layer (str): name of the layer being considered
  """
  input_in_image_file_path = INPUT_IMAGES_PATH+model_name+"_input_image.hex"
  output_in_image_tensor_path = OUTPUT_FI_FILES_PATH+model_name+"_"+target_layer+"_input_tensor"
  
  # print the int8 input tensor to file
  with(open(input_in_image_file_path, 'r')) as input_weight_file:
    hex_weights = input_weight_file.read().strip().replace('\n', '').replace(' ', '')
    hex_array = np.array([int(hex_weights[i:i+2], 16) for i in range(0, len(hex_weights), 2)], dtype=np.int8)
    tensor = hex_array[:np.prod(MODELS[model_name][4])].reshape(MODELS[model_name][4])

  np.save(output_in_image_tensor_path, tensor)