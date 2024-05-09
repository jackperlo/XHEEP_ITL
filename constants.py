from models.lenet import Lenet

# path to the pretrained models folder
PRETRAINED_MODEL_PATH="./models/pretrained_models/"

# path in which the output weights will be saved
WEIGHTS_OUTPUT_PATH="./outputs/weights"

# path in which the output input_weight pairs will be saved
INPUT_WEIGHT_PAIR_OUTPUT_PATH="./outputs/input_weight_pairs"

# path in which the atpg scripts will be saved
ATPG_SCRIPTS_OUTPUT_PATH="./outputs/atpg_scripts/"

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
              "lenet5_quantized.tflite", 
              ["conv1"] 
            ]
}