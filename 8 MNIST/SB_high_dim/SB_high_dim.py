import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 

from GenerateData import getDataset_highDim
from neural_network import NN_define
from method_2_high_dim import method_2_w0, method_2

N_train = 20
N_test = 10
d = 3

# import data
x_train, y_train, x_valid, y_valid, x_test, y_test = getDataset_highDim(N_train=N_train, N_test=N_test, d=d)
        
epochs = 100
batch_size = 32
K = 2**10
learning_rate = 0.001

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

cov_x = np.cov(x_train.T)
px_train = (np.exp(-1 / 2 * np.sum(np.multiply(x_train.T, cov_x @ x_train.T), axis=0)) * 1 / (
                                (2 * np.pi) ** (d / 2)) / np.sqrt(np.linalg.det(cov_x))).reshape(len(x_train), 1)
px_valid= (np.exp(-1 / 2 * np.sum(np.multiply(x_valid.T, cov_x @ x_valid.T), axis=0)) * 1 / (
                        (2 * np.pi) ** (d / 2)) / np.sqrt(np.linalg.det(cov_x))).reshape(len(x_valid), 1)


w0_2 = method_2_w0(x_train, y_train, 1, 2, px_train) 
SB_2 = method_2(w0_2, x_valid, y_valid, pred_valid, px_valid)
print(SB_2)