import keras

from keras import layers

class Lenet(keras.Model):
  """
    Class representing the LeNet-5 model used.
  """
  def __init__(self):
    """
      Constructor of the Lenet class.
    """
    super().__init__()
    self.conv1 = self._make_conv_layer(6, 5, input_shape=(32, 32, 1))
    self.conv2 = self._make_conv_layer(16, 5)
    self.conv3 = self._make_conv_layer(120, 5, pooling=False, flatten=True)
    self.dense1 = layers.Dense(84, activation='tanh')
    self.dense2 = layers.Dense(10, activation='softmax')

  
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
    x = self.dense1(x)
    x = self.dense2(x)

    return x

  def _make_conv_layer(self, filters, kernel_size, input_shape=None, pooling=True, flatten=False):
    """
      Creates a convolutional layer with optional pooling and flattening.

      Args:
        filters(int): the number of filters in the convolutional layer.
        kernel_size(int): the size of the convolutional kernel.
        input_shape(int[]): the shape of the input tensor. If provided, this is used
          as the input shape for the first convolutional layer.
        pooling(bool): whether to add pooling to the convolutional layer. Defaults
          to True.
        flatten(bool): whether to flatten the output of the convolutional layer.
          Defaults to False.

      Returns:
        a Keras Sequential model containing the convolutional layer,
        optional pooling and flattening layers, and an activation
        function.
    """
    l = keras.Sequential()
    l.add(layers.Input(shape=(32,32,1)))
    l.add(layers.Conv2D(filters, kernel_size, activation='tanh'))
    if pooling:
      l.add(layers.AveragePooling2D(2))
      l.add(layers.Activation('sigmoid'))
    if flatten:
      l.add(layers.Flatten())
    return l
  
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
      res = [0, 6, 5]
    elif name.lower() == "conv2":
      res = [4, 16, 5]
    elif name.lower() == "conv3":
      res = [8, 120, 5]
 
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
    elif name.lower() == "conv2":
      res = (1, 1)
    elif name.lower() == "conv3":
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
    elif name.lower() == "conv2":
      res =  {"type":"valid", "values":[0,0,0,0]}
    elif name.lower() == "conv3":
      res =  {"type":"valid", "values":[0,0,0,0]}
 
    return res

      