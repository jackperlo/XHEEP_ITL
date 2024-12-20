import keras

class Alexnet(keras.Model):
  """
    Class representing the Alexnet model used.
  """
  def __init__(self):
    super().__init__()
    self.conv1 = keras.Sequential()
    self.conv1.add(keras.layers.Input(shape=(32,32,3)))
    self.conv1.add(keras.layers.UpSampling2D(size=(7,7)))
    self.conv1.add(keras.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = (1,1), activation = 'relu', input_shape = (224, 224, 3)))
    self.conv1.add(keras.layers.BatchNormalization())
    self.conv1.add(keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2), padding='same'))

    self.conv2 = keras.Sequential()
    self.conv2.add(keras.layers.Conv2D(filters = 128, kernel_size = (5,5), strides = (1,1), activation = 'relu', padding = 'same'))
    self.conv2.add(keras.layers.BatchNormalization())
    self.conv2.add(keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2), padding='same'))

    self.conv3 = keras.Sequential()
    self.conv3.add(keras.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same'))
    self.conv3.add(keras.layers.BatchNormalization())

    self.conv4 = keras.Sequential()
    self.conv4.add(keras.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same'))
    self.conv4.add(keras.layers.BatchNormalization())

    self.conv5 = keras.Sequential()
    self.conv5.add(keras.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation = 'relu', padding = 'same'))
    self.conv5.add(keras.layers.BatchNormalization())
    self.conv5.add(keras.layers.MaxPool2D(pool_size = (2,2), strides = (2,2), padding='same'))
    self.conv5.add(keras.layers.Flatten())

    self.dense1 = keras.Sequential()
    self.dense1.add(keras.layers.Dense(units = 128, activation = 'relu'))
    self.dense1.add(keras.layers.Dropout(0.5))

    self.dense2 = keras.Sequential()
    self.dense2.add(keras.layers.Dense(units = 128, activation = 'relu'))
    self.dense2.add(keras.layers.Dropout(0.5))

    self.dense3 = keras.layers.Dense(units = 10, activation = 'softmax')

  
  def call(self, inputs):
    """
      Define the forward pass of the model, applying each layer in turn to the input tensor.

      Args:
        inputs(int8[]): represents the input tensor of the model which, working on int8, will respectively be a int8.

      Returns:
        returns the output tensor of the model.
    """
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.conv5(x)
    x = self.dense1(x)
    x = self.dense2(x)
    x = self.dense3(x)

    return x
  
  def get_conv_layer_details(self, name):
    """
      Returns details about a convolutional layer by name.

      Args:
        name(str): the name of the convolutional layer.

      Returns:
        a slice containing the index of the layer, number of filters applied, and
        kernel size of the convolutional layer, or an empty dictionary
        if the layer name is not recognized.
    """
    res = []
    if name.lower() == "conv1":
      res = [12, 64, 3]
 
    return res
  
  def get_conv_layer_strides(self, name):
    """
      Returns strides about a convolutional layer by name.

      Args:
        name(str): the name of the convolutional layer.

      Returns:
        returns a tuple containing the strides of the given convolutional layer;
        the tuple represents stride in terms of (height, width).
    """
    res = ()
    if name.lower() == "conv1":
      res = (1, 1)
 
    return res
  
  def get_conv_layer_padding(self, name):
    """
      Returns padding about a convolutional layer by name.

      Args:
        name(str): the name of the convolutional layer.

      Returns:
        returns an object containing a string representing the padding mode 
        used for the given convolutional layer and the related values;
        the string could be a value between {"valid", "same"}.
    """
    res = {"type":"", "values":[0,0,0,0]}
    if name.lower() == "conv1":
      res =  {"type":"valid", "values":[0,0,0,0]}
 
    return res