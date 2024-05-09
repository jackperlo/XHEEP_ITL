from utils import load_model

from weights_management import save2file_model_weights
from pairs_management import save2file_model_input_weight_pairs
from atpg_scripts_management import save2files_atpg_scripts


from constants import MODELS
from constants import PRETRAINED_MODEL_PATH

from utils import arg_parse
from utils import create_output_directory

def main(args):
  network, pretrained_model_name, target_layers = MODELS[args.model]

  model = load_model(path=PRETRAINED_MODEL_PATH+pretrained_model_name)
  
  create_output_directory(args.weight_format)

  if args.save_weights:
    print("\n~~~> saving weights mode enabled...")
    save2file_model_weights(args.weight_format, model, network, target_layers)
    print("\n~~~> model weights: SAVED")
  
  if args.save_pairs:
    print("\n~~~> saving (input, weight) pairs mode enabled...")
    save2file_model_input_weight_pairs(model, network, target_layers)
    print("\n~~~> model (input, weight) pairs: SAVED\n")

  if args.generate_atpg_scripts:
    print("\n~~~> generating atpg scripts...")
    save2files_atpg_scripts(args.model)
    print("\n~~~> atpg scripts: SAVED\n")

  print("\nexecution completed!\n")
  

if __name__ == '__main__':
  main(arg_parse())