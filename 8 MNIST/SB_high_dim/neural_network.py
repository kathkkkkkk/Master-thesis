import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input 

# define model 
# Neural network 

def NN_define(input_size = 784, m_size = 2**5, output_size = 1): 
    # encoding 
    input_img = Input(shape=(input_size,))


    # middle layer 
    activation = tf.math.cos 
    m_layer = Dense(m_size, activation=activation)(input_img)
    #m_layer = Dense(m_size, activation='relu')(h2_layer)

    output_img = Dense(output_size, activation='linear')(m_layer)

    model = Model(input_img, output_img)

    print(model.summary())
    
    return model 