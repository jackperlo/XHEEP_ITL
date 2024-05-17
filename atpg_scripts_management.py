import math

from constants import WEIGHTS_OUTPUT_PATH
from constants import MODELS
from constants import ATPG_SCRIPTS_OUTPUT_PATH

def save2files_atpg_scripts(model_name):
  """
    Save the the atpg scripts (.tcl) for each weight of the specified target layer(s) of the specified pre trained model.

    Args:
      model (str): the runtime model name 
  """
  for layer in MODELS[model_name][2]:
    with open(WEIGHTS_OUTPUT_PATH+"_binary/"+layer+"_weights.txt", 'r') as weights:
      for (line_number, weight) in enumerate(weights):
        content = write_atpg_content_for_cve2_multdiv_fast_RV32M3(weight)
        file_name = get_output_file_name(line_number)
        with open(ATPG_SCRIPTS_OUTPUT_PATH+file_name, 'w') as output_file:
          for command in content:
            output_file.write("%s\n" % command)

def write_atpg_content_for_cve2_multdiv_fast_RV32M3(weight):
  """
    Generates the content of the .tcl file based on the fixed commands and the weight values

    Args:
      weight (str): the weight, as string, whose values are going to be written as op_b_i bits
  
    Returns:
      a list of the commands to be injected into the .tcl file
  """
  #op_a_i: input constrained bits
  #op_b_i: weight constrained bits
  content = [
    "set_environment_viewer -instance_names",
    "set_messages -log tmax_unconstrained.log -replace",
    "read_netlist ./syn/out/cve2_multdiv_fast.v",
    "read_netlist ./syn/techlib/NangateOpenCellLibrary.v",
    "run_build_model cve2_multdiv_fast_RV32M3",
    "add_pi_constraints 0 op_a_i[31]",
    "add_pi_constraints 0 op_a_i[30]",
    "add_pi_constraints 0 op_a_i[29]",
    "add_pi_constraints 0 op_a_i[28]",
    "add_pi_constraints 0 op_a_i[27]",
    "add_pi_constraints 0 op_a_i[26]",
    "add_pi_constraints 0 op_a_i[25]",
    "add_pi_constraints 0 op_a_i[24]",
    "add_pi_constraints 0 op_a_i[23]",
    "add_pi_constraints 0 op_a_i[22]",
    "add_pi_constraints 0 op_a_i[21]",
    "add_pi_constraints 0 op_a_i[20]",
    "add_pi_constraints 0 op_a_i[19]",
    "add_pi_constraints 0 op_a_i[18]",
    "add_pi_constraints 0 op_a_i[17]",
    "add_pi_constraints 0 op_a_i[16]",
    "add_pi_constraints 0 op_a_i[15]",
    "add_pi_constraints 0 op_a_i[14]",
    "add_pi_constraints 0 op_a_i[13]",
    "add_pi_constraints 0 op_a_i[12]",
    "add_pi_constraints 0 op_a_i[11]",
    "add_pi_constraints 0 op_a_i[10]",
    "add_pi_constraints 0 op_a_i[9]",
    "add_pi_constraints 0 op_a_i[8]",
    "add_pi_constraints 0 op_a_i[7]"
  ]
  i=31
  for bit in weight:
    if bit != '\n': # omit the last symbol in "weight" which is the carriage return
      content.append("add_pi_constraints "+bit+" op_b_i["+str(i)+"]")
      i-=1
  content.append("add_pi_constraints 1 { signed_mode_i[0] signed_mode_i[1] }")
  content.append("add_po_masks valid_o")
  content.append("run_drc")
  content.append("read_faults ./flist.txt -maintain_detection")
  content.append("set_atpg -merge high") # -abort_limit 100 -patterns 1
  content.append("set_faults -model stuck")
  content.append("run_atpg -auto")
  content.append("write_patterns mul_patterns.txt -format stil -internal -replace")
  content.append("write_faults ./flist.txt -replace -all")
  content.append("report_summaries > ./summaries.txt")
  
  return content

def get_output_file_name(line_number):
  """
    Compute the 4-tuple used to name the atpg file relating the 4-tuple (and so the atpg of the corresponding weight)
    to the weights used in the convolution operation (specified in the [input, weight] pair files)

    Args:
      line_number (int): represents the line_number of the considered weight which will be converted into a 4-tuple
  
    Returns:
      the name of the output atpg script file (e.g. weight_0_3_4_0.tcl)
  """
  n_channels_out = 0
  height = 0
  width = 0
  n_filters = 0

  if line_number<5:
    width = line_number
  else:
    width = line_number%5
    line_number = math.floor(line_number/5)
    if line_number>=5:
      height = line_number%5
      n_channels_out = math.floor(line_number/5)
    else:
      height = line_number
      n_channels_out = 0

  return "weight_"+str(n_channels_out)+"_"+str(height)+"_"+str(width)+"_"+str(n_filters)+".tcl"

