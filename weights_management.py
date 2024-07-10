import tensorflow as tf
import numpy as np

from constants import WEIGHTS_OUTPUT_PATH
from utils import int8_to_binary

def save2file_model_weights(weight_format, model, network, target_layers, model_name):
  """
  Save, on a file, all the weights of the given target_layers of the model in the given weight_format format 

  Args:
    weight_format(string): weights saving format (i.e. a value between [int8, binary, hex])
    model(tf.lite.Interpreter): the runtime instance of the pretrained model of which the weights need to be saved
    network: the instance of the class representing the class and allowing to get some of its features
    target_layers(string[]): all the layers for which the weights need to be saved
    model_name (str): name of the CNN being used 
  """
  layer_weight_list = get_layers_weights(model, network, target_layers)

  for layer, weights in layer_weight_list.items():
  
    file_name = WEIGHTS_OUTPUT_PATH+"_"+weight_format+"/"+model_name+"_"+layer+"_weights.txt"
    with(open(file_name, 'w')) as output_file:

      for weight in weights:
        if weight_format == 'int8':
          conversion_function = np.int8
        elif weight_format == 'binary':
          conversion_function = int8_to_binary
        else:
          print('ERROR: Invalid output format for model weights')
          exit(-1)

        converted_weights = conversion_function(weight)
        i=0
        for converted_weight in converted_weights:
          if i < len(converted_weights)*len(weights)-1:
            output_file.write(f'{converted_weight}\n')
          else:
            output_file.write(f'{converted_weight}')
          i+=1

def get_layers_weights(model: tf.lite.Interpreter, network, target_layers):
  """
    Extracts the weights of the specified convolutional layers from a TensorFlow Lite model.
    *Note that only convolutional layers are supported so far*

    Args:
      model: a TensorFlow Lite interpreter object.
      network: a CNN network object with layer details.
      target_layers: a list of target convolutional layer names.

    Returns:
      a dictionary mapping layer names to their corresponding weight tensors.
  """
  weights_list = dict()

  tensors = model.get_tensor_details()
  ops = model._get_ops_details()

  for layer in target_layers:
    # get the operation index, number of filters and kernel measures for the specified layer
    index, filters, kernel = network.get_conv_layer_details(layer)

    # check the got layer index goodness
    if ops[index]['index'] != index:
      print('ERROR: CNN layer index not compatible with submitted target layer index')
      exit(-1)

    # get the index of the kernel tensor; it is always is 2nd position in the input array of convolutional operations
    kernel_tensor_idx = ops[index]['inputs'][1]  

    # check the got tensor index goodness
    if tensors[kernel_tensor_idx]['index'] != kernel_tensor_idx:
      print('ERROR: tensor index(kernel) of the considered convolution layer not compatible with the expected one')
      exit(-1)

    # check the got tensor filters number goodness
    if tensors[kernel_tensor_idx]['shape'][0] != filters:
      print('ERROR: kernel tensor filters number of the considered convolution layer not compatible with the expected one')
      exit(-1)

    # check the got tensor kernel size goodness
    if tensors[kernel_tensor_idx]['shape'][1] != kernel or tensors[kernel_tensor_idx]['shape'][2] != kernel:
      print('ERROR: kernel tensor size of the considered convolution layer not compatible with the expected one')
      exit(-1)

    # add the kernel values (int8) of the target layer to the weights list
    weights_list[layer] = model.get_tensor(kernel_tensor_idx)

  return weights_list
