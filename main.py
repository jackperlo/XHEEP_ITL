from utilities.constants import MODELS
from utilities.constants import PRETRAINED_MODEL_PATH

from utilities.utils import load_model
from utilities.utils import arg_parse
from utilities.utils import create_output_directory
from utilities.utils import print_1ch_npy_image
from utilities.utils import print_help_menu

from lib.weights_management import save2file_model_weights
from lib.pairs_management import save2file_model_input_weight_pairs
from lib.atpg_scripts_management import save2files_atpg_scripts
from lib.hex_model_management import save_model_in_hex_format_as_words
from lib.hex_model_management import save_model_in_hex_format_as_bytes
from lib.patterns_input_positions import get_atpg_patterns_input_positions
from lib.fault_injection_management import manage_fault_injection_files
from lib.custom_input_image_management import generate_custom_input_image

def main(args):
  if args.h:
    print_help_menu()
    return

  network, pretrained_model_name, target_layers = MODELS[args.model]

  model = load_model(path=PRETRAINED_MODEL_PATH+pretrained_model_name)
  
  create_output_directory(args.weight_format, args.model)

  # save trained weights in the specified format
  if args.save_weights:
    print("\n~~~> saving weights mode enabled...")
    save2file_model_weights(args.weight_format, model, network, target_layers, args.model)
    print("\n~~~> model weights: SAVED")
  
  # save <input, weight> indexes pairs of the specified convolutional layer(s) 
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

  # generating custom input image
  if args.generate_custom_input_image:
    print("\n~~~> generate custom input image (mode: "+args.generate_custom_input_image+")...")
    generate_custom_input_image(args.generate_custom_input_image, args.model)
    print("\n~~~> custom input image: GENERATED\n")

  # collect itl-validation fault injection files 
  if args.generate_FI_files:
    print("\n~~~> generating Fault Injection files...")
    if args.input_tensor_path is None: # deprecated
      manage_fault_injection_files(model, network, args.model, args.layer)
    else: # deprecated
      manage_fault_injection_files(model, network, args.model, args.layer, args.input_tensor_path)
    print("\n~~~> Fault Injection files: GENERATED\n")

  # print a 1-channel(grey scale) tensor (.npy) as an image
  if args.print_image:
    print("\n~~~> printing image...\n")
    print_1ch_npy_image(args.print_image)
    print("\n~~~> image: PRINTED\n")

  print("\nexecution completed!\n")
  
if __name__ == '__main__':
  main(arg_parse())