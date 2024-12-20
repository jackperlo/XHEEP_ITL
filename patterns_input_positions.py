import re
import json

from constants import MODELS

def get_atpg_patterns_input_positions(model_name):
  """
    From the ATPG gathered patterns <T_p, W_i>, the considered weight W_i (which is multiplied for that pattern T_p) is retrieved.
    Given the weight W_i, all the input positions mulitplied for that W_i are collected and saved into a file .json file.

    Args:
      model_name (str): name of the model being considered
  """
  patterns_file_name = "./outputs/"+model_name+"/atpg_patterns_gathered/"+model_name+"_patterns.txt"
  input_weight_pairs_file_name = "./outputs/"+model_name+"/input_weight_pairs/"+model_name+"_"+MODELS[model_name][2][0]+"_input_weight_pairs.json"
  patterns_all_possible_positions_file_name = "./outputs/"+model_name+"/atpg_patterns_gathered/"+model_name+"_patterns_all_positions.json"

  # open the patterns file to get all the weight coordinates
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
      with open(patterns_all_possible_positions_file_name, 'w') as outfile:
        outfile.write(json.dumps(matching_lines))