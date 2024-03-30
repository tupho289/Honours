
#---------------- import libraries ----------------
import tensorflow as tf
from keras.layers import Dense, Dropout


#---------------- Custom activation functions for ExU hidden units ----------------
def exu(x, kernel, bias):
    """Define the calculation for ExU units.
    With ExU units, the way the inputs and weights (kernel + bias) are combined is different from
    the standard computation (e.g. x @ kernel + bias)
    """
    return tf.exp(kernel)*(x - bias)


def relu_n(x, n = 1):
    """This activation function caps the value of an ExU unit at n to avoid overfitting.
    It can be use with standard neurons."""
    return tf.clip_by_value(x, 0, n)


#---------------- Custom activation Layer to support ExU hidden units ----------------
class ActivationLayer(tf.keras.layers.Layer):
    """First hidden layer for each subnetwork in NAM"""

    # constructor
    def __init__(self, units, activation = "exu", initializer = "glorot_normal", cap = 1, **kwargs):
        """Args:
            @units: Number of hidden units in the layer.
            @activation: Activation to use. The default value corresponds to
            using the ReLU-1 activation with ExU units.
            @initializer: kernel initialization method when ExU units is not used.
            @cap: cap for ExU units when using relu_n() activation function.
        """
        super().__init__(**kwargs)
        self._units = units
        if activation == "exu":
            self._cap = cap # cap for relu_n()
            self._activation = lambda x, kernel, bias: relu_n(exu(x, kernel, bias), self._cap)
            self._kernel_initializer = tf.initializers.truncated_normal(mean = 4.0, stdev = 0.5)
        else:
            self._activation = tf.keras.activations.get(activation)
            self._kernel_initializer = initializer
        

    def build(self, input_shape):
        """Initialize the weights of the activation layer. 
        Automatically called in the first foward pass"""
        self.kernel = self.add_weight(
            name = "kernel",
            shape = [input_shape[-1], self._units],
            initializer = self._kernel_initializer)
        self.bias = self.add_weight(
            name = "bias",
            shape = [self._units],
            initializer = tf.initializers.truncated_normal(stdev = 0.5))
        super(ActivationLayer, self).build(input_shape) # to set self.built = True


    def call(self, x):
        """Perform the computation at each neuron."""
        if self._activation == "exu":
            bias_vector = tf.tile(self.bias, [tf.shape(x)[0], 1])
            return self.activation(x, self.kernel, bias_vector)
        else:
            return self.activation(x @ self.kernel + self.bias)


#---------------- Subnetwork for each shape function ----------------
class FeatureSubnet(tf.keras.layers.Layer):
    """This class defines the subnet for each individual feature and pairwise interaction effects.
    The first layer is the Activation layer can allow for ExU unit."""

    # constructor
    def __init__(self, hidden_layers, units, activation = "relu", initializer = "glorot_normal", 
                 exu = True, dropout = 0, cap = 1, **kwargs):
        """Args:
            @hidden_layers: number of hidden layers (including the Activation layer).
            @units: number of units in each hidden layer. Can either be a scalar (when all
                layers have the same number of neurons) or a 1D array.
            @activation: activation function to use for each subnet's hidden layer. When exu = True,
                this activation doesn't apply to the first hidden layer.
            @initializer: initialization method to be used for network weights.
            @exu: whether to use exu units in the first hidden layer or not. If True, first hidden
                layer use relu_n() activation, otherwise, @activation is applied.
            @dropout: dropout rate to apply during training. Value should be between 0 and 1.
            @cap: cap of relu_n() function when ExU units are used.
        """
        super().__init__(**kwargs)
        self._hidden_layers = hidden_layers
        self._activation = activation
        self._initializer = initializer
        self._exu = exu
        self._dropout = dropout
        self._cap = cap

        """initialize self.unit"""
        if isinstance(units, int):
            # If units is a scalar, make it a tuple repeated 'hidden_layers' times
            self._units = (units,) * self._hidden_layers
        elif isinstance(units, tuple):
            # If units is a tuple, check its length
            if len(units) != self._hidden_layers:
                raise ValueError("Length of 'units' tuple must be equal to 'hidden_layers'")
            self._units = units
        else:
            raise TypeError("'units' must be either an int (scalar) or a tuple")


    def build(self, input_shape):
        # add activation layer
        if self._exu:
            self.subnet = [
                ActivationLayer(self._units[0], 
                                cap = self._cap)]
        else:
            self.subnet = [
                ActivationLayer(self._units[0], 
                                activation = self._activation,
                                initializer = self._initializer)]

        # add dense layer
        for i in range(1, self._hidden_layers):
            self.subnet.append(tf.keras.layers.Dense(
                self._units[i],
                activation = self._activation,
                kernel_initializer = self._initializer))
            
        # add output layer
        self.subnet.append(tf.keras.layers.Dense(
                1,
                kernel_initializer = self._initializer))
        super(FeatureSubnet, self).build(input_shape) # to set self.built = True

    
    def call(self, x, training):
        for l in range(self.subnet) - 1:
            # apply dropout for each hidden layer
            x = tf.nn.dropout(self.subnet[l](x), 
                              rate = tf.cond(training, 
                                            lambda: self._dropout, 
                                            lambda: 0.0))
        return self.subnet[-1](x)


#---------------- Neural Additive Model ----------------
class NAM(tf.keras.Model):
    """This Neural Additive Model class will allow for both individual and pairwise interaction effect"""

    #constructor
    def __init__(self,
                num_inputs,
                interactions,
                number_hidden,
                num_units,
                activation = "relu",
                initializer = "glorot_normal",
                exu = True,
                dropout = 0.0,
                cap = 1,
                **kwargs):
       
        """Args:
            @num_inputs: Number of feature inputs in input data.
            @interactions: 2D array. Each row contains 2 elements each denoting the index of the 
                column participating in the pairwise interaction effect.
            @number_hidden: Number of layers for each subnetwork. This can either be a
                scalar (when all subnetworks have the same number of hidden layers) or an array 
                (when the number of hidden layers are specified explicitly for each subnetwork).
                If specified explicitly, values for subnets with single input should be specified
                before subnets for pairwise interaction effect. Values should match column order
                and interaction effect orders specified in @interactions.
            @num_units: Number of hidden units in first layer of each feature net. Can be a scalar,
                1D, or 2D array.
            @activation: activation function for each hidden layers. If exu = True, this argument
                doesn't apply to the first hidden layer (or activation layer) of each subnet.
            @initializer: initialization method to be used for network weights. Can be either a scalar
                or an array.
            @exu: whether to use exu units in the first hidden layer or not. If True, first hidden
                layer use relu_n() activation, otherwise, @activation is applied. Can be either a scalar
                or an array.
            @dropout: dropout rate to apply during training. Value should be between 0 and 1.
            @cap: cap of relu_n() function when ExU units are used.
        """

        super(NAM, self).__init__(**kwargs)
        self._num_inputs = num_inputs
        self._interactions = interactions
        self._dropout = dropout
        self._cap = cap