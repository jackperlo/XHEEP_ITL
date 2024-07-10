import re
import random

from constants import ATPG_PATTERNS_GATHERED_PATH
from constants import INPUT_WEIGHT_PAIR_OUTPUT_PATH
from constants import MODELS
from constants import INPUT_IMAGES_PATH
from constants import INPUT_IMAGE_ROWS
from constants import INPUT_IMAGE_COLS

def generate_image(model_name):
  patterns_file_name = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns.txt"
  input_weight_pairs_file_name = INPUT_WEIGHT_PAIR_OUTPUT_PATH+"/"+model_name+"_"+MODELS[model_name][2][0]+"_input_weight_pairs.txt"
  patterns_available_positions_file_name = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns_available_positions.txt"

  with open(patterns_file_name, "r") as patterns_file:
    matching_lines = dict()
    for line in patterns_file:
      line_split = line.split(" ")[0].split("_")
      weight_coord = ",".join(line_split[3:7])
      weight_coord = re.escape(weight_coord)
      row_pattern = rf'"[^"]*": "{weight_coord}"'

      with open(input_weight_pairs_file_name, "r") as pairs_file:
        pair_lines = pairs_file.readlines()
     
      for pair_line in pair_lines:
        pair_line = pair_line.strip()
        if re.search(row_pattern, pair_line):
          if weight_coord not in matching_lines:
            matching_lines[weight_coord] = []
          matching_lines[weight_coord].append(pair_line.split(":")[0][1:-1])

    if matching_lines:
      """ random ex
      with open(patterns_available_positions_file_name, 'w') as outfile:
        chosen_input_positions = []
        for key, value in matching_lines.items():
          outfile.write("{'"+key+"': ")
          first_value = str(value[get_unique_random_numbers(chosen_input_positions, len(value))])
          second_value = str(value[get_unique_random_numbers(chosen_input_positions, len(value))])
          outfile.write("['"+first_value+"', '"+second_value+"']")
          #outfile.write(str(value)) #to print all the input positions available
          outfile.write("}\n")
      """
      generate_hex_matrix(model_name)

    else:
      print("No matching lines found")
      return
    
def get_unique_random_numbers(chosen_input_positions, upper_bound):
  num = random.randint(0, upper_bound-1)
  while num in chosen_input_positions:
    num = random.randint(0, upper_bound-1)
  chosen_input_positions.append(num)
  
  return num

def generate_hex_matrix(model_name):
  patterns_available_positions_file_name = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns_available_positions.txt"
  gathered_patterns = ATPG_PATTERNS_GATHERED_PATH+model_name+"_patterns.txt"
  computed_input_images_file_name = INPUT_IMAGES_PATH+model_name+"_input.h"
  matrix = []
  
  carriage_return = "{\n\t"
  with open(computed_input_images_file_name, 'w') as input_image_file:
    input_image_file.write("#ifndef TENSORFLOW_LITE_MICRO_LENET5_INPUT_H_\n#define TENSORFLOW_LITE_MICRO_LENET5_INPUT_H_\n\n#include <cstdint>\n\n")
    input_image_file.write(f"const unsigned int {model_name}_input_data_size = {INPUT_IMAGE_ROWS} * {INPUT_IMAGE_COLS};\n\n")
    input_image_file.write(f"const int8_t {model_name}_input_data[{model_name}_input_data_size] = {carriage_return}")

  for y in range(INPUT_IMAGE_COLS):
    row = [] 
    for x in range(INPUT_IMAGE_ROWS):
      pattern_position = False
      with open(patterns_available_positions_file_name, "r") as pattern_positions_file:
        for line in pattern_positions_file:
          pattern_x = []
          pattern_y = []
          weight_value=line.split("': ")[0].split("{'")[1].replace(',','_')

          pattern_x.append(line.split("['")[1].split(',')[1])
          pattern_x.append(line.split(", ")[1].split(',')[1])
          pattern_y.append(line.split("['")[1].split(',')[2])
          pattern_y.append(line.split(", ")[1].split(',')[2])

          if x == int(pattern_x[0]) and y == int(pattern_y[0]):
            row_re = rf'positive_input_pattern_{weight_value} : ([01]+)$'
            with open(gathered_patterns, "r") as gathered_patterns_file:
              patterns = gathered_patterns_file.readlines()
  
              for pattern in patterns:
                pattern = pattern.strip()
                if re.search(row_re, pattern):
                  row.append(format(int(pattern.split(' : ')[1], 2) & 0xFF, '#04x'))
                  pattern_position = True
                  break

          if x == int(pattern_x[1]) and y == int(pattern_y[1]):
            row_re = rf'negative_input_pattern_{weight_value} : ([01]+)$'
            with open(gathered_patterns, "r") as gathered_patterns_file:
              patterns = gathered_patterns_file.readlines()
  
              for pattern in patterns:
                pattern = pattern.strip()
                if re.search(row_re, pattern):
                  row.append(format(int(pattern.split(' : ')[1], 2) & 0xFF, '#04x'))
                  pattern_position = True
                  break

      if pattern_position == False:
        row.append(hex(0))
      with open(computed_input_images_file_name, 'a') as input_image_file:
        input_image_file.write(row[-1]+", ")
    with open(computed_input_images_file_name, 'a') as input_image_file:
      input_image_file.write("\n\t")
    matrix.append(row)

  with open(computed_input_images_file_name, 'r+') as input_image_file:
    contents = input_image_file.read()
    input_image_file.seek(0)
    input_image_file.write(contents[:-4]+"\n};\n\n#endif")


      