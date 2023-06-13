import numpy as np 
import matplotlib.pyplot as plt 

from data_get import get_data, add_noise
from adaptive_algorithm import algorithm_1 

x_train, y_train, x_valid, y_valid, x_test, y_test = get_data()
x_test = add_noise(x_test, 0, 0.05)

M = 100
d = 784
K = 2**10
lambda_ = 0.05
delta = 0.02
gamma = 3*d-2 
sd_n = 1
    
beta, w, pred_train, pred_valid, pred_test, error_train_list, error_valid_list, error_test_list, accuracy_train_list, accuracy_valid_list, accuracy_test_list = algorithm_1(
    x_train, y_train, x_valid, y_valid, x_test, y_test, K, M, lambda_, delta, gamma, sd_n)

np.savez('./am_2.npz', 
         pred_train = pred_train, 
         pred_valid = pred_valid, 
         pred_test = pred_test, 
         error_train_list = error_train_list, 
         error_valid_list = error_valid_list,
         error_test_list = error_test_list,
         accuracy_train_list = accuracy_train_list, 
         accuracy_valid_list = accuracy_valid_list, 
         accuracy_test_list = accuracy_test_list)

    
plt.figure(figsize=(10, 6))
epoch_list = np.arange(1, 101)
plt.subplot(1, 2,  1)
plt.semilogy(epoch_list, error_train_list, label = 'training_loss')
plt.semilogy(epoch_list, error_valid_list, label = 'validation_loss')
plt.legend()
plt.xlabel('epoch')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(epoch_list, accuracy_train_list*100, label = 'accuracy_train')
plt.plot(epoch_list, accuracy_valid_list*100, label = 'accuracy_valid')
plt.plot(epoch_list, accuracy_test_list*100, label = 'accuracy_test')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('%')
plt.title('Accuracy')
plt.tight_layout()
plt.savefig('./am_2.jpg')
plt.show()
