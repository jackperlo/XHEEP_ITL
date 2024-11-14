import re
import json
import tensorflow as tf
import random
import numpy as np

from constants import ATPG_PATTERNS_GATHERED_PATH
from constants import INPUT_IMAGES_PATH

from constants import lenet5_in_zero_point
from constants import lenet5_in_scale

def generate_custom_input_image(mode, model_name):
  """
    Generate an input image based on the model under test

    Args:
      mode (str): which mode the input image must follow
      model_name (str): name of the model being considered
  """
  if mode == "FWP":
    generate_pattern_filled_input_image(model_name)

def generate_pattern_filled_input_image(model_name):
  """
    Generate an input image which is totally filled by the ATPG gathered patterns.

    Args:
      model_name (str): name of the model being considered
  """
  patterns_file_name = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns.txt"
  patterns_all_possible_positions_file_name = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns_all_positions.json"
  FWP_input_possibilities_file_name = INPUT_IMAGES_PATH+model_name+"_FWP_input_possibilities.json"
  FWP_input_image_path = INPUT_IMAGES_PATH+model_name+"_FWP_input_image.npy"

  # open the patterns' file to get all the weight coordinates
  with open(patterns_file_name, "r") as patterns_file:
    input_image_options = dict() # e.g.: {input_1_0_0_0 => [pattern_1, pattern_2, pattern_3]}
    lines = dict()

    for line in patterns_file:
      line_split = line.split(" ")[0].split("_")
      weight_coord = ",".join(line_split[3:7])
      weight_coord = re.escape(weight_coord)
      ATPG_pattern_string = line.split(" ")[2]
      ATPG_pattern_hex = ATPG_binary_pattern_to_hex(ATPG_pattern_string)
      #print(ATPG_pattern_string+" => "+ATPG_pattern_hex)

      # for each weight coordinate check in the input/weight pairs if the weight coord is present
      with open(patterns_all_possible_positions_file_name, "r") as pattern_positions_file:
        lines = json.load(pattern_positions_file)
     
      # saving, for all the input positions of the considered weight coordinate, 
      # the patterns which are suitable/plausible to be placed in that input coordinate 
      for input_position in lines[weight_coord]:
        if input_position not in input_image_options:
          input_image_options[input_position] = []
        input_image_options[input_position].append(ATPG_pattern_hex) if ATPG_pattern_hex not in input_image_options[input_position] else input_image_options[input_position]

    with open(FWP_input_possibilities_file_name, 'w') as outfile:
      outfile.write(json.dumps(input_image_options))

    # build the FWP image assign a random pattern (suitable one) for each input position
    input_mask = tf.Variable(tf.ones((1, 32, 32, 1)))
    for x in range(32):
      for y in range(32):
        suitable_pattern_index = "1,"+str(x)+","+str(y)+",0"
        if suitable_pattern_index in input_image_options: 
          val = random.randint(0, len(input_image_options[suitable_pattern_index])-1) 
          float32_value = np.float32((np.int8(int(input_image_options[suitable_pattern_index][val], 16)) - lenet5_in_zero_point) * lenet5_in_scale)
        else:
          last_pattern_multiplied_index = "1,"+str(random.randint(0, 27))+","+str(random.randint(0, 27))+",0"
          val = random.randint(0, len(input_image_options[last_pattern_multiplied_index])-1) 
          float32_value = np.float32((np.int8(int(input_image_options[last_pattern_multiplied_index][val], 16)) - lenet5_in_zero_point) * lenet5_in_scale)
        input_mask[0, x, y, 0].assign(float32_value)

    np.save(FWP_input_image_path, input_mask)

def ATPG_binary_pattern_to_hex(binary_pattern_string):
  '''
    Convert the 32-bit binary string representing the ATPG gathered pattern
    to its 2-byte hex representation

    Args:
      binary_pattern_string (str): the 32-bit string representation of the pattern
    
    Returns:
      The hex value of the binary string value passed as argument
  '''
  decimal_value = int(binary_pattern_string, 2)
  # This keeps only the last 16 bits
  limited_value = decimal_value & 0xFF  
  # Convert to hexadecimal and format to 4 digits (2 bytes)
  hex_value = f"{limited_value:02x}"
  return hex_value