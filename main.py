from utils import load_model
from utils import arg_parse
from utils import create_output_directory

from weights_management import save2file_model_weights
from pairs_management import save2file_model_input_weight_pairs
from atpg_scripts_management import save2files_atpg_scripts
from trained_model_management import save_model_in_hex_format_as_words
from trained_model_management import save_model_in_hex_format_as_bytes
from patterns_input_positions import get_atpg_patterns_input_positions
from fault_injection_management import manage_fault_injection_files

from constants import MODELS
from constants import PRETRAINED_MODEL_PATH

def main(args):
  network, pretrained_model_name, target_layers, _, _= MODELS[args.model]

  model = load_model(path=PRETRAINED_MODEL_PATH+pretrained_model_name)
  
  create_output_directory(args.weight_format)

  # save trained weights in the specified format
  if args.save_weights:
    print("\n~~~> saving weights mode enabled...")
    save2file_model_weights(args.weight_format, model, network, target_layers, args.model)
    print("\n~~~> model weights: SAVED")
  
  # save <input, weight> pairs of the specified convolutional layer(s) 
  if args.save_pairs:
    print("\n~~~> saving (input, weight) pairs mode enabled...")
    save2file_model_input_weight_pairs(model, network, target_layers, args.model)
    print("\n~~~> model (input, weight) pairs: SAVED\n")

  # generate atpg scripts to cover all the network signed (+, -) weights
  if args.generate_atpg_scripts:
    print("\n~~~> generating atpg scripts...")
    save2files_atpg_scripts(args.model)
    print("\n~~~> atpg scripts: SAVED\n")

  # collect the model in .hex (words/bytes) to run it into the X-HEEP platform
  if args.save_model_hex_format:
    print("\n~~~> saving "+args.model+" model in hex format (as words)...")
    save_model_in_hex_format_as_words(args.model)
    print("\n~~~> "+args.model+" model : SAVED in HEX format as words\n")

    print("\n~~~> saving "+args.model+" model in hex format (as bytes)...")
    save_model_in_hex_format_as_bytes(args.model)
    print("\n~~~> "+args.model+" model : SAVED in HEX format as bytes\n")

  # collect all the input positions which are multiplied for a given a weight which is,
  # in turn, multiplied for a given test pattern
  if args.gather_patterns_input_positions:
    print("\n~~~> gathering test patterns' input positions...")
    get_atpg_patterns_input_positions(args.model)
    print("\n~~~> "+args.model+" test patterns' input positions : GATHERED\n")

  # collect itl-validation fault injection files 
  if args.generate_FI_files:
    print("\n~~~> generating Fault Injection files...")
    if args.input_tensor_path is None:
      manage_fault_injection_files(model, network, args.model, args.layer)
    else:
      manage_fault_injection_files(model, network, args.model, args.layer, args.input_tensor_path)
    print("\n~~~> Fault Injection files: GENERATED\n")

  print("\nexecution completed!\n")
  
if __name__ == '__main__':
  main(arg_parse())