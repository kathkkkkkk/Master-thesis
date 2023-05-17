import numpy as np 
import tensorflow as tf 

from data_get import get_data_orig
from result_compute import error_compute, accuracy_compute 
from nn_define import NN_define
from method_2_high_dim import method_2_w0, method_2

x_train, y_train, x_valid, y_valid, x_test, y_test  = get_data_orig()

d = np.shape(x_train)[1]
epochs = 100 
batch_size = 32
K = 2**10
learning_rate = 0.001

N_train = np.shape(x_train)[0]
pred_train = np.zeros(epochs*10*N_train).reshape(epochs, 10, N_train)
N_valid = np.shape(x_valid)[0]
pred_valid = np.zeros(epochs*10*N_valid).reshape(epochs, 10, N_valid)
N_test = np.shape(x_test)[0]
pred_test = np.zeros(epochs*10*N_test).reshape(epochs, 10, N_test)

error_train_list = []
error_valid_list = []
error_test_list = []
    
accuracy_train_list = []
accuracy_valid_list = []
accuracy_test_list = []

model = NN_define(input_size = d, m_size = K, output_size = 1)

model.compile(loss='mse', 
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
            metrics=['accuracy'])

model.fit(x=x_train, y=y_train, 
            validation_data = (x_valid, y_valid), 
            epochs=epochs, 
            batch_size=batch_size)

pred_train = model.predict(x_train)
pred_valid = model.predict(x_valid)
pred_test = model.predict(x_test) 

#error_train = error_compute(y_train, pred_train)
#accuracy_train = accuracy_compute(y_train, pred_train)

guess1, guess2 = 1, 1.1
cov_x = np.cov(x_train.T)
px_train = (np.exp(-1 / 2 * np.sum(np.multiply(x_train.T, cov_x @ x_train.T), axis=0)) * 1 / (
                                (2 * np.pi) ** (d / 2)) / np.sqrt(np.linalg.det(cov_x))).reshape(len(x_train), 1)

w0 = method_2_w0(x_train, y_train, guess1, guess2, px_train)
SB = method_2(w0, x_train, y_train, pred_train, px_train)