import re
import random

from constants import ATPG_PATTERNS_GATHERED_PATH
from constants import INPUT_WEIGHT_PAIR_OUTPUT_PATH
from constants import MODELS
from constants import INPUT_IMAGES_PATH
from constants import INPUT_IMAGE_ROWS
from constants import INPUT_IMAGE_COLS

def generate_image(model_name, generate_new_random_positions):
  """
    From the gathered patterns the considered weight for that pattern to be multiplied for is retrieved.
    Then, it retrieves, for each of the patterns, which are the input coordinates which are involved in the multiplication 
    with the aforementioned weight and it eventually generates the input image (chosing, if specified, some random 
    input position from the retrieved ones for each pattern).

    Args:
      model_name (str): name of the model being considered
      generate_new_random_positions (bool - default: False): argument specified by the user to request to extract, randomly, new positions in the input image for each pattern
  """
  patterns_file_name = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns.txt"
  input_weight_pairs_file_name = INPUT_WEIGHT_PAIR_OUTPUT_PATH+"/"+model_name+"_"+MODELS[model_name][2][0]+"_input_weight_pairs.txt"
  patterns_available_positions_file_name = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns_available_positions.txt"

  # open the patterns' file to get all the weight coordinates
  with open(patterns_file_name, "r") as patterns_file:
    matching_lines = dict()

    for line in patterns_file:
      line_split = line.split(" ")[0].split("_")
      weight_coord = ",".join(line_split[3:7])
      weight_coord = re.escape(weight_coord)
      row_pattern = rf'"[^"]*": "{weight_coord}"'

      # for each weight coordinate check in the input/weight pairs if the weight coord is present
      with open(input_weight_pairs_file_name, "r") as pairs_file:
        pair_lines = pairs_file.readlines()
     
      for pair_line in pair_lines:
        pair_line = pair_line.strip()
        # if a weight coordinate is present, then all the input coordinates which are multiplied for that weight coordinate are saved
        if re.search(row_pattern, pair_line):
          if weight_coord not in matching_lines:
            matching_lines[weight_coord] = []
          # ( e.g.: {weight_0_0_0_0 => [input_1_2_3_4, input_24_25_26_27, ...], ...} )
          matching_lines[weight_coord].append(pair_line.split(":")[0][1:-1])

    if matching_lines:
      if generate_new_random_positions: # specified by the user args; default: False
        generate_new_random_positions_file(matching_lines, patterns_available_positions_file_name)
      generate_hex_matrix(model_name, patterns_available_positions_file_name, patterns_file_name)
    else:
      print("No matching lines found")

def generate_new_random_positions_file(matching_lines, patterns_available_positions_file_name):
  """
    From the matching lines, which specify for each weight (and so for each related pattern) which are the available
    positions in the input image which are plausible to set the pattern into.
    At the end, a file specifying each pattern chosen position is written.

    Args:
      matching_lines (dict): dictionary containing, for each weight/pattern, ALL them possible position in the input image
        (e.g. matching_lines = {weight_0_0_0_0 => [input_1_2_3_4, input_24_25_26_27, ...], ...} )
      patterns_available_positions_file_name (str): file name of the result of the function
  """
  with open(patterns_available_positions_file_name, 'w') as outfile:
    chosen_input_positions = []
    for key, value in matching_lines.items():
      outfile.write("{'"+key+"': ")
      first_value = str(value[get_unique_random_numbers(chosen_input_positions, len(value))])
      second_value = str(value[get_unique_random_numbers(chosen_input_positions, len(value))])
      outfile.write("['"+first_value+"', '"+second_value+"']")
      outfile.write("}\n")

def get_unique_random_numbers(chosen_input_positions, upper_bound):
  """
    Chose a random index which will be used in the list of available input image positions (to choose one of them), 
    having care that those positions have not already been chosen

    Args:
      chosen_input_positions (list): list of the, so far, chosen positions for all the patterns
      upper_bound (int): length of the available positions list for the considered pattern

    Returns:
      a random integer which indicates which index of the input image available positions list of the current pattern will be chosen
  """
  num = random.randint(0, upper_bound-1)
  while num in chosen_input_positions:
    num = random.randint(0, upper_bound-1)
  chosen_input_positions.append(num)
  
  return num

def generate_hex_matrix(model_name, patterns_available_positions_file_name, gathered_patterns_file_name):
  """
    Generate the image as a ".h" C-style file, containing for each cell of the image either a pattern 
    (if the position is one of those chosen before), as a hex value, or a hardcoded 0, always as a hex value.  
 
    Args: 
      model_name (str): name of the model being considered
      patterns_available_positions_file_name (str): file name. This file contains the before chosen positions for each pattern
      gathered_patterns_file_name (str): file name. This file contains the pattern gathered from the ATPG process
  """
  computed_input_image_file_name = INPUT_IMAGES_PATH+model_name+"_input.h"
  matrix = []
  
  carriage_return = "{\n\t"
  with open(computed_input_image_file_name, 'w') as input_image_file:
    # definition of some macros and variable names of the output input_image file
    input_image_file.write("#ifndef TENSORFLOW_LITE_MICRO_LENET5_INPUT_H_\n#define TENSORFLOW_LITE_MICRO_LENET5_INPUT_H_\n\n#include <cstdint>\n\n")
    input_image_file.write(f"const unsigned int {model_name}_input_data_size = {INPUT_IMAGE_ROWS} * {INPUT_IMAGE_COLS};\n\n")
    input_image_file.write(f"const int8_t {model_name}_input_data[{model_name}_input_data_size] = {carriage_return}")

  for y in range(INPUT_IMAGE_COLS):
    row = [] 
    for x in range(INPUT_IMAGE_ROWS):
      pattern_position = False

      # check whether the current cell of the input image is one of the chosen input image positions
      with open(patterns_available_positions_file_name, "r") as pattern_positions_file:
        for line in pattern_positions_file:
          pattern_x = []
          pattern_y = []

          # chosen positions file has a format as: 
          #    weight    first_pattern_position  second_pattern_position
          # { '0,0,0,0':      ['1,6,27,0',            '1,12,2,0'] }
          # Note: there are two pattern position for each weight since there are a positive and a negative pattern
          weight_value=line.split("': ")[0].split("{'")[1].replace(',','_')
          pattern_x.append(line.split("['")[1].split(',')[1])
          pattern_x.append(line.split(", ")[1].split(',')[1])
          pattern_y.append(line.split("['")[1].split(',')[2])
          pattern_y.append(line.split(", ")[1].split(',')[2])

          # if current cell is equal to the first_pattern_position
          if x == int(pattern_x[0]) and y == int(pattern_y[0]):
            # double check with the ATPG gathered patterns file to check that the pattern 
            # is multiplied by the spcified weight and that a positive pattern has been found
            row_re = rf'positive_input_pattern_{weight_value} : ([01]+)$'
            with open(gathered_patterns_file_name, "r") as gathered_patterns_file:
              patterns = gathered_patterns_file.readlines()
  
              for pattern in patterns:
                pattern = pattern.strip()
                if re.search(row_re, pattern):
                  # if the double check was successful, then the bits specified by the gathered pattern
                  # are parsed into a hex value of 8 bits
                  row.append(format(int(pattern.split(' : ')[1], 2) & 0xFF, '#04x'))
                  pattern_position = True
                  break

          # if current cell is equal to the first_pattern_position
          if x == int(pattern_x[1]) and y == int(pattern_y[1]):
            # double check with the ATPG gathered patterns file to check that the pattern 
            # is multiplied by the spcified weight and that a negative pattern has been found
            row_re = rf'negative_input_pattern_{weight_value} : ([01]+)$'
            with open(gathered_patterns_file_name, "r") as gathered_patterns_file:
              patterns = gathered_patterns_file.readlines()

              for pattern in patterns:
                pattern = pattern.strip()
                if re.search(row_re, pattern):
                  # if the double check was successful, then the bits specified by the gathered pattern
                  # are parsed into a hex value of 8 bits
                  row.append(format(int(pattern.split(' : ')[1], 2) & 0xFF, '#04x'))
                  pattern_position = True
                  break

      # if no pattern has been place in current image position, then a hardcoded 0x00 is chosen to be set as value for the current image position
      if pattern_position == False:
        row.append(hex(0))

      # the current value is write (in append mode) into the output file
      with open(computed_input_image_file_name, 'a') as input_image_file:
        input_image_file.write(row[-1]+", ")

    with open(computed_input_image_file_name, 'a') as input_image_file:
      input_image_file.write("\n\t")
    matrix.append(row)

  # the output file is adjusted so that it can become a .h file in C-style
  with open(computed_input_image_file_name, 'r+') as input_image_file:
    contents = input_image_file.read()
    input_image_file.seek(0)
    input_image_file.write(contents[:-4]+"\n};\n\n#endif")


      