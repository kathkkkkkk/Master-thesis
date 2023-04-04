import numpy as np 
import matplotlib.pyplot as plt 
from nn_train import nn_training 

pred_train, pred_valid, pred_test, error_train_list, error_valid_list, error_test_list, accuracy_train_list, accuracy_valid_list, accuracy_test_list  = nn_training(epochs = 100, 
                batch_size = 32, 
                K = 2**10, 
                learning_rate = 0.002)

np.savez('./sgd_5.npz', 
         pred_train = pred_train, 
         pred_valid = pred_valid, 
         pred_test = pred_test, 
         error_train_list = error_train_list, 
         error_valid_list = error_valid_list,
         error_test_list = error_test_list,
         accuracy_train_list = accuracy_train_list, 
         accuracy_valid_list = accuracy_valid_list, 
         accuracy_test_list = accuracy_test_list)

epochs = 100 

plt.figure(figsize=(10, 6))
epoch_list = np.arange(1, epochs+1)
plt.subplot(1, 2,  1)
plt.semilogy(epoch_list, error_train_list, label = 'training_loss')
plt.semilogy(epoch_list, error_valid_list, label = 'validation_loss')
plt.legend()
plt.xlabel('epoch')
plt.title('Loss')

plt.subplot(1, 2, 2)
plt.plot(epoch_list, accuracy_train_list*100, label = 'accuracy_train')
plt.plot(epoch_list, accuracy_valid_list*100, label = 'accuracy_valid')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('%')
plt.title('Accuracy')
plt.tight_layout()
plt.savefig('./sgd_5.jpg')
plt.show()