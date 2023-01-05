from platform import node
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


# define the keras model 
def nn_define_1(nodes = 64, 
                kernel_initializer=tf.keras.initializers.he_normal, 
                bias_initializer=tf.keras.initializers.Zeros()): 
    input_layer = keras.Input(shape = (1, ))
    hidden_layer_1 = keras.layers.Dense(units = nodes, 
                                    activation = 'relu', 
                                    kernel_initializer=tf.keras.initializers.he_normal, 
                                    bias_initializer=tf.keras.initializers.Zeros())
    hidden_layer_2 = keras.layers.Dense(units = nodes, 
                                    activation = 'relu', 
                                    kernel_initializer=tf.keras.initializers.he_normal, 
                                    bias_initializer=tf.keras.initializers.Zeros())
    hidden_layer_3 = keras.layers.Dense(units = nodes, 
                                    activation = 'relu', 
                                    kernel_initializer=tf.keras.initializers.he_normal, 
                                    bias_initializer=tf.keras.initializers.Zeros())
    hidden_layer_4 = keras.layers.Dense(units = nodes, 
                                    activation = 'relu', 
                                    kernel_initializer=tf.keras.initializers.he_normal, 
                                    bias_initializer=tf.keras.initializers.Zeros())
    hidden_layer_5 = keras.layers.Dense(units = nodes, 
                                    activation = 'relu', 
                                    kernel_initializer=tf.keras.initializers.he_normal, 
                                    bias_initializer=tf.keras.initializers.Zeros())
    output_layer = keras.layers.Dense(units = 1, 
                                    use_bias = False)
    model = keras.Sequential([input_layer, 
                            hidden_layer_1,
                            hidden_layer_2, 
                            hidden_layer_3, 
                            hidden_layer_4, 
                            hidden_layer_5, 
                            output_layer])
    
    return model 



def nn_define_2(input_dim = 1, 
                output_dim = 1, 
                num_layers = 1, 
                num_nodes = 2**8, 
                activation=tf.keras.activations.relu, 
                kernel_initializer=tf.keras.initializers.he_normal, 
                bias_initializer=tf.keras.initializers.Zeros()): 
    
    tf.keras.backend.set_floatx('float64')

    activation = tf.keras.activations.relu # same for two 
    # activation=tf.math.cos  # same for two , tf.cos
    kernel_initializer = tf.keras.initializers.he_normal # standard normal 
    # kernel_initializer=tf.keras.initializers.normal # standard normal 
    bias_initializer = tf.keras.initializers.Zeros() 
    # Define the network. This class corresponds to an MLP.
    # input layer
    inputs = tf.keras.Input(shape=(input_dim,), name="Input")

    # Hidden Layers
    x = Dense(units=num_nodes, activation=activation, kernel_initializer=kernel_initializer,
                  bias_initializer=bias_initializer)(inputs)
    for i in np.arange(1, num_layers):
        x = Dense(units=num_nodes, activation=activation, kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer)(x)
    # Outputs
    outputs = Dense(output_dim, activation="linear", name="predictions")(x)
    if output_dim == 3:
        outputs = Dense(output_dim, activation="linear", name="predictions")(x)

    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model