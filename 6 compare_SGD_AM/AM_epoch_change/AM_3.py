from generate_data import generate_data
from AM_algorithm_1 import algorithm_1 
import matplotlib.pyplot as plt 
import numpy as np


x_train, y_train, x_validation, y_validation, x_test, y_test, x_test_sb, y_test_sb = generate_data()


# compute the SB of ARFF 
d = 1

M = 3000
K = 2**10
lambda_ = 0.5
delta = 1 # 0.05 list of parameters 
gamma = 3*d - 2

beta, w, f_est_train, pred_validation, error_train_list, error_validation_list, error_valid_min, error_test_list, error_test_end, FVU_validation, SB_M1, SB_M2 = algorithm_1(
                                            x_train, y_train, 
                                            x_test, y_test, 
                                            x_validation, y_validation,
                                            x_test_sb, y_test_sb,  
                                            K, M, lambda_, delta, gamma)

np.savez('./am_3.npz', 
         x_train = x_train, 
         y_train = y_train, 
         x_validation = x_validation, 
         y_validation = y_validation, 
         beta = beta, w = w, f_est_train = f_est_train, 
         pre_validation = pred_validation, 
         error_train_list = error_train_list, 
         error_validation_list = error_validation_list, 
         error_valid_min = error_valid_min, 
         error_test_list = error_test_list, 
         error_test_end = error_test_end, 
         FVU_validation = FVU_validation, 
         SB_M1 = SB_M1, 
         SB_M2 = SB_M2)

plt.subplot(2, 2, 1)
plt.plot(x_validation, pred_validation, '*', label = 'validation')
plt.plot(x_test_sb, y_test_sb, label = 'func')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Estimation")
plt.legend()
                      
t_vec = np.arange(0, M)
plt.subplot(2, 2, 2)
plt.semilogy(t_vec[2:], error_train_list[2:], label='training_error')
plt.semilogy(t_vec[2:], error_validation_list[2:], label='validation_error')
plt.semilogy(t_vec[2:], error_test_list[2:], label='test_error')
plt.title("Error")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t_vec[1:], FVU_validation[1:])
plt.title("FVU")

plt.subplot(2, 2, 4)
plt.plot(t_vec[1:], SB_M1[1:], color='blue', label="Method 1")
plt.plot(t_vec[1:], SB_M2[2:], color='red', label="Method 2")
plt.title("Spetral Bias")
plt.legend()

plt.tight_layout()
plt.savefig('./am_3.jpg')
plt.show()