# ITL Pre-Processing
The idea of this repository is to work on a given pre-trained model so that two main tasks can be accomplished:
  1. Extraction of the trained weights for the specified convolutional layers
  2. Extraction of the input and weight indexes of the specified convolutional layers which will be used by the multiplier of the XHEEP architecture during the convolutional operations

Note: to date [04/2024] an interpreter < "Python 3.12" must be used due to the lack of compatibility between Tensoflow module and the latest version of the interpreter

## ANACONDA 
### 1) Create a Virtual Environment using Anaconda
  - `conda create -n <env_name>` 
  - `conda init <your_shell_name>`, then restart the shell

### 2) Activate the environment
  `conda activate <env_name>`
  - to deactivate an environment use: `conda deactivate <env_name>`
  - to list all your environments use: `conda info --envs`

### 3) Install packages in the chosen environment
  `conda install <channel_name>::<pkg_name>`

