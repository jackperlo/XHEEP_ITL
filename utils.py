import tensorflow as tf
import numpy as np
import argparse
import os

from constants import WEIGHTS_OUTPUT_PATH
from constants import INPUT_WEIGHT_PAIR_OUTPUT_PATH
from constants import ATPG_SCRIPTS_OUTPUT_PATH
from constants import OUTPUT_HEX_MODEL_PATH
from constants import OUTPUT_FI_FILES_PATH
from constants import ATPG_PATTERNS_GATHERED_PATH

from constants import MODELS

def load_model(path):
  """
  Load a pretrained TensorFlow Lite model from the given path.

  Args:
    path(string): the path to the pretrained model.

  Returns:
    returns the interpreter to the pretrained model.
  """
  # experimental_preserve_all_tensors: allows to keep track of all the intermediate tensors, essential to navigate through layers
  interpreter = tf.lite.Interpreter(model_path=path, experimental_preserve_all_tensors=True)
  interpreter.allocate_tensors()

  return interpreter

def int8_to_binary(num):
  """
  Convert a int8(:np.ndarray) number to its binary representation according to the IEEE-754 standard.
  
  Args: 
    the number to convert.

  Returns: 
    a string containing the 32-bit binary representation.
  """
  if isinstance(num, np.ndarray):
    # If num is a NumPy array, convert to bin maintaining the sign
    binary_strings = []
    for x in num.flatten():
      binary_rep = bin(x & 0xFF)[2:]
      # Sign extension
      if (x < 0): # Negative values
        sign_extended = '1' * (32-len(binary_rep))
        sign_extended+=binary_rep
        binary_strings.append(sign_extended)
      else: # Positive values
        sign_extended = '0' * (32-len(binary_rep))
        sign_extended+=binary_rep
        binary_strings.append(sign_extended)
  else:
    print("Error: while trying to convert an int8 to binary; the int is not an instance of np.ndarray")
    exit(-1)

  return binary_strings

def int8_to_hex(num):
  """
  Convert a int8(:np.ndarray) number to its hex representation
  
  Args: 
    the number to convert.

  Returns: 
    a string containing the 32-bit hex representation.
  """
  if isinstance(num, np.ndarray):
    hex_list = []
    for val in int8_to_binary(num): 
      # Convert binary string to integer, taking into account 2's complement
      if val[0] == '1':  # Negative number
          int_val = -((1 << len(val[:-1])) - int(val[:-1], 2))
      else:  # Positive number
          int_val = int(val[:-1], 2)
      # Convert integer to hexadecimal string
      hex_val = format(int_val & 0xFFFFFFFF, '08x')  # Mask to 32 bits
      hex_list.append(hex_val)
      return hex_list
  else:
    print("Error: while trying to convert an int8 to hex; the int is not an instance of np.ndarray")
    exit(-1)

def create_output_directory(weight_format):
  """
  Create the output directories if it does not already exist.

  Args:
    weight_format(string): weights saving format.
  """
  os.makedirs(WEIGHTS_OUTPUT_PATH+"_"+weight_format+"/", exist_ok=True)
  os.makedirs(INPUT_WEIGHT_PAIR_OUTPUT_PATH, exist_ok=True)
  os.makedirs(ATPG_SCRIPTS_OUTPUT_PATH, exist_ok=True)
  os.makedirs(OUTPUT_HEX_MODEL_PATH, exist_ok=True)
  os.makedirs(OUTPUT_FI_FILES_PATH, exist_ok=True)
  os.makedirs(ATPG_PATTERNS_GATHERED_PATH, exist_ok=True)

def arg_parse():
  """
  Parse all the args passed to the main.

  Returns:
    returns all the passed arguments(if any), assigning the default ones to the arguments not provided. 
  """
  parser = argparse.ArgumentParser(description='Create a list of all the weights, the (input, weight) pairs, atpg scripts of a pre-trained model.')
  
  # argument used to specify the used pre-trained model
  parser.add_argument('--model', help='CNN model to use. Default: lenet5', choices=MODELS.keys(), default="lenet5")
  # argument used to specify the used pre-trained model layer
  parser.add_argument('--layer', help='CNN model layer of interest. Default: conv1', type=str, default="conv1")

  # arguments related to read and save weights from the run time model
  parser.add_argument('--save_weights', action='store_true', help='If this parameter is specified, then the execution will read and save the weights of the specified layer(s)')
  parser.add_argument('--weight_format', help='The output weight format. Default: binary', type=str, choices=['binary', 'int8', 'hex'], default="binary")

  # argument related to read and save (input, weight) pairs from the run time model
  parser.add_argument('--save_pairs', action='store_true', help='If this parameter is specified, then the execution will read and save the (input, weight) pairs of the specified layer(s)')

  # argument related to generate atpg_scripts
  parser.add_argument('--generate_atpg_scripts', action='store_true', help='If this parameter is specified, then the execution will generate the atpg scripts')

  # argument related to save the trained model into a hex format to be executed using TFLite 
  parser.add_argument('--save_model_hex_format', action='store_true', help='If this parameter is specified, then the model will be saved in hex format')
  
  # argument related to gather all the patterns' suitable input positions
  parser.add_argument('--gather_patterns_input_positions', action='store_true', help='If this parameter is specified, then all the patterns\' suitable input positions are gathered')
  
  # argument used to save all the files needed for the fault injection process
  parser.add_argument('--generate_FI_files', action='store_true', help='If this parameter is specified, then all the files needed for the fault injection phase are generated.')
  # argument used to specify the path of the input tensor of a given convolutional layer
  parser.add_argument('--input_tensor_path', help='If this parameter is specified, the FI generated files for the specified convolutional layer get the input tensor by the path specified in this argument', type=str)

  parsed_args = parser.parse_args()

  return parsed_args