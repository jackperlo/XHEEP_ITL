import tensorflow as tf
import numpy as np
import argparse
import os

from constants import WEIGHTS_OUTPUT_PATH
from constants import INPUT_WEIGHT_PAIR_OUTPUT_PATH
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

def create_output_directory(weight_format):
  """
  Create the output directories if it does not already exist.

  Args:
    weight_format(string): weights saving format.
  """
  os.makedirs(WEIGHTS_OUTPUT_PATH+"_"+weight_format+"/", exist_ok=True)
  os.makedirs(INPUT_WEIGHT_PAIR_OUTPUT_PATH, exist_ok=True)

def arg_parse():
  """
  Parse all the args passed to the main.

  Returns:
    returns all the passed arguments(if any), assigning the default ones to the arguments not provided. 
  """
  parser = argparse.ArgumentParser(description='Create a list of all the weights of a pre-trained model.')
  parser.add_argument('--model', help='CNN model to use. Default: lenet5', choices=MODELS.keys(), default="lenet5")
  parser.add_argument('--weight_format', help='The output weight format', type=str, choices=['binary', 'int8'], default="binary")
  parsed_args = parser.parse_args()

  return parsed_args