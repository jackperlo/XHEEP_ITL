from models.lenet import Lenet

# path to the pretrained models folder
PRETRAINED_MODEL_PATH="./models/pretrained_models/"

# path to the patterns gathered
OUTPUT_FI_FILES_PATH="./outputs/FI_files/"

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
              "lenet5_int8_mnist.tflite", 
              ["conv1"]
            ],
  "alexnet":[],
  "resnet50":[],
  "resnet152":[],
  "vgg16":[]         
}