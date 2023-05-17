import numpy as np 
import tensorflow as tf 

from data_get import get_data_compress, get_data_orig
from result_compute import error_compute, accuracy_compute 
from nn_define import NN_define
from method_2_high_dim import method_2_w0, method_2

x_train, y_train, x_valid, y_valid, x_test, y_test  = get_data_compress(k=4)
#x_train, y_train, x_valid, y_valid, x_test, y_test  = get_data_orig() 
x_train_orig, y_train_orig, x_valid_orig, y_valid_orig, x_test_orig, y_test_orig  = get_data_orig() 


d = np.shape(x_train)[1]
epochs = 100 
batch_size = 32
K = 2**5
learning_rate = 0.001


model = NN_define(input_size = d, m_size = K, output_size = 10)

#model.compile(loss='mse', 
#            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
#            metrics=['accuracy'])
model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', 
              metrics=['accuracy'])

model.fit(x=x_train, y=y_train, 
            validation_data = (x_valid, y_valid), 
            epochs=epochs, 
            batch_size=batch_size)

d_orig = np.shape(x_train_orig)[1]

model_orig = NN_define(input_size = d_orig, m_size = K, output_size = 10)

#model_orig.compile(loss='mse', 
#            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
#            metrics=['accuracy'])
model_orig.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', 
              metrics=['accuracy'])


model_orig.fit(x=x_train_orig, y=y_train_orig, 
            validation_data = (x_valid_orig, y_valid_orig), 
            epochs=epochs, 
            batch_size=batch_size)




