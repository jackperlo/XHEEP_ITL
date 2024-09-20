import tensorflow as tf
import json

from constants import INPUT_WEIGHT_PAIR_OUTPUT_PATH

def save2file_model_input_weight_pairs(model, network, target_layers, model_name):
  """
    Save the given model specified layers (input, weight) pairs for each output to file.

    Args:
      model (tf.keras.Model): the model to extract the (input,weight) pairs from.
      network (Network): the network object containing the layer names.
      target_layers (list): the names of the layers to save the (input,weight) pairs for.
      model_name (str): the name of the CNN under exam
  """
  layer_input_weight_pairs = get_layers_input_weight_pairs(model, network, target_layers)

  for layer, input_weight_pairs in layer_input_weight_pairs.items():
    file_name = INPUT_WEIGHT_PAIR_OUTPUT_PATH+"/"+model_name+"_"+layer+"_input_weight_pairs.json"
    with(open(file_name, 'w')) as output_file:
      json.dump(input_weight_pairs, output_file, indent=2)

def get_layers_input_weight_pairs(model: tf.lite.Interpreter, network, target_layers):
  """
  This function returns a dictionary that maps each target layer name to a list of dictionaries.
  Each dictionary in the list corresponds to a single output feature map, and contains a nested dictionary
  for each spatial location in the output feature map. The nested dictionary maps a tuple representing
  a spatial location and a channel index to a tuple representing the corresponding input location and
  weight index.

  Args:
    model (tf.lite.Interpreter): the TensorFlow Lite interpreter object.
    network: the network object containing details about the layers.
    target_layers (list): a list of target layer names.

  Returns:
    a dictionary mapping each target layer name to a list of dictionaries.

  e.g.:
    [
      {"<batch_number>_<output_height_index>_<output_weight_index>_<_n_channel_out>": [
          {
            "<batch_number>,<Input_height_index>,<Input_width_index>,<n_channel_in>"
            :
            "<n_channel_out>,<Kernel_height_index>,<Kernel_height_index>,<filter_number>"
          },
          ...
        ]
      }
    ]
  """
  input_weight_pairs = dict()

  tensors = model.get_tensor_details()
  ops = model._get_ops_details()

  for layer in target_layers:
    # get the operation index, strides and padding for by the specified layer 
    op_index, _, _ = network.get_conv_layer_details(layer)
    strides = network.get_conv_layer_strides(layer)
    padding = network.get_conv_layer_padding(layer)

    # get the index of the input tensor, kernel tensor, bias tensor and output tensor
    input_tensor_idx = ops[op_index]['inputs'][0] 
    kernel_tensor_idx = ops[op_index]['inputs'][1]
    bias_tensor_idx = ops[op_index]['inputs'][2]
    output_tensor_idx = ops[op_index]['outputs'][0]

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
    outputs = []
    n_mults = 0
    for output_number in range(kernel_tensor_shape[0]): 
      for output_height in range(output_tensor_shape[1]):
        for output_width in range(output_tensor_shape[2]):
          outputs.append({"1_"+str(output_height)+"_"+str(output_width)+"_"+str(output_number): []})
          for n_channel_in in range(input_tensor_shape[3]):
            for base_coord_h in range(kernel_tensor_shape[1]):
              for base_coord_w in range(kernel_tensor_shape[2]):
                n_mults+=1
                outputs[-1]["1_"+str(output_height)+"_"+str(output_width)+"_"+str(output_number)].append({
                  "1,"+str(base_coord_h+output_height+strides[0]-1)+","+str(base_coord_w+output_width+strides[1]-1)+","+str(n_channel_in) : 
                  str(output_number)+","+str(base_coord_h)+","+str(base_coord_w)+","+str(n_channel_in)})

    input_weight_pairs[layer] = outputs
    print("TOTAL MULTS: "+str(n_mults))
  
  return input_weight_pairs

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
    


