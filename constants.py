from models.lenet import Lenet

# path to the pretrained models folder
PRETRAINED_MODEL_PATH="./models/pretrained_models/"

# path in which the output weights will be saved
WEIGHTS_OUTPUT_PATH="./outputs/weights"

# path in which the output input_weight pairs will be saved
INPUT_WEIGHT_PAIR_OUTPUT_PATH="./outputs/input_weight_pairs"

# path in which the atpg scripts will be saved
ATPG_SCRIPTS_OUTPUT_PATH="./outputs/atpg_scripts/"

# path to the hex models folder
OUTPUT_HEX_MODEL_PATH="./outputs/hex_models/"

# path to the patterns gathered
ATPG_PATTERNS_GATHERED_PATH="./outputs/atpg_patterns_gathered/"

# path to the patterns gathered
OUTPUT_FI_FILES_PATH="./outputs/FI_files/"

# path to the computed input images
INPUT_IMAGES_PATH="./outputs/input_images/"
INPUT_IMAGE_ROWS=32
INPUT_IMAGE_COLS=32

"""
  Object containing all the supported pretrained models.

  Args: 
    each object must be composed a slice of:
      (Class Instance): the instance of the class representing the model.
      (str): the path to pretrained model.
      (str[]): a list of the target layers, composed as "<operation><operation occourence start counting from 1>".
"""
MODELS = {
  "lenet5": [
              Lenet(),
              "lenet5_mnist.tflite", 
              ["conv1", "conv2", "conv3"],
              (6,5,5,1), # conv1 weight tensor shape
              (1,32,32,1) # conv1 input tensor shape
            ]
}