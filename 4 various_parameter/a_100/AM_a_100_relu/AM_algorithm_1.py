import numpy as np 
from SB_method_1 import method_1_w0, method_1
from SB_method_2 import method_2_w0, method_2
from FVU_compute import FVU_compute

# minimization function 
def min_fun(x, y, w, lambda_): 
    N = np.shape(x)[0]
    K = np.shape(w)[0]
    I = np.identity(K)
    one = np.ones(N)
    x_ = np.c_[x, one]
    S = np.maximum(x_.dot(w.T), 0) 
    beta = np.linalg.inv(np.dot(S.T, S) + lambda_ * N * I).dot(S.T).dot(y)
    return beta, S


def error_compute(y, S, beta):
    N = np.shape(y)[0]
    diff = S.dot(beta) - y
    error = (np.linalg.norm(diff, ord=2))**2 / N
    return error


# Metropolis algorithm 1

def algorithm_1(x_train, y_train, x_test, y_test, x_validation, 
                y_validation, K, M, lambda_, delta, gamma):
    
    # normalize the data 
    x_mean = np.mean(x_train)
    x_std = np.std(x_train, ddof=1)
    y_mean = np.mean(y_train)
    y_std = np.std(y_train, ddof=1)
    
    x_train_norm = (x_train - x_mean)/x_std
    y_train_norm = (y_train - y_mean)/y_std
    x_test_norm = (x_test - x_mean)/x_std
    y_test_norm = (y_test - y_mean)/y_std
    x_validation_norm = (x_validation - x_mean)/x_std
    y_validation_norm = (y_validation - y_mean)/y_std
    
    d = np.shape(x_train_norm)[1] 
    
    w = np.zeros(K*(d+1)).reshape((K, d+1))
    
    error_train_list = []
    error_validation_list = []
    FVU = []
    FVU_validation = []
    SB_M1 = []
    SB_M2 = []
    
    px = 1 / x_std / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * (((x_test) / x_std) ** 2))
    px_train = 1 / x_std / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * ((x_train / x_std) ** 2))
    px_validation = 1 / x_std / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * ((x_validation / x_std) ** 2))

    
    beta, S = min_fun(x_train_norm, y_train_norm, w, lambda_)
    for i in range(M): 
        #print('w=', i)
        r_n = np.random.normal(0, 1, K*(d+1)).reshape((K, d+1))
        w_temp = w + delta * r_n
        beta_temp, S_temp = min_fun(x_train_norm, y_train_norm, w_temp, lambda_)
        for k in range(K): 
            #print('k=', k)
            r_u = np.random.uniform(0, 1)
            if (np.linalg.norm(beta_temp[k], ord=1)
                /np.linalg.norm(beta[k], ord=1))**gamma > r_u: 
                w[k] = w_temp[k]
                #print('w_k', w[k])
                beta[k] = beta_temp[k]
                #print('beta_k', beta[k])
        beta, S = min_fun(x_train_norm, y_train_norm, w, lambda_)
        
        f_est = S.dot(beta) * y_std + y_mean
        
        error_train = error_compute(y_train_norm, S, beta)
        error_train_list = np.append(error_train_list, error_train)
        
        beta_validation, S_validation = min_fun(x_validation_norm, 
                                                y_validation_norm, w, lambda_)
        f_est_validation = S_validation.dot(beta) * y_std + y_mean
        error_validation= error_compute(y_validation_norm, S_validation, beta)
        error_validation_list = np.append(error_validation_list, error_validation)
        

        # Print the training loss for every tenth epoch
        if i % 10 == 0:
            print("\nEnd of epoch  " + str(i) + ", Training error " +
                      str(error_train)) 
            print("\nEnd of epoch  " + str(i) + ", Validation error " +
                      str(error_validation))
        
        beta_test, S_test = min_fun(x_test_norm, y_test_norm, w, lambda_)
        f_est_test = S_test.dot(beta) * y_std + y_mean
        
        pred = f_est
        pred_validation = f_est_validation
        FVU = np.append(FVU, FVU_compute(y_train, pred))
        FVU_validation = np.append(FVU_validation, 
                                   FVU_compute(y_validation, 
                                               pred_validation))
    
        w0_1, var = method_1_w0(x_test, y_test, px)
        pred_test = f_est_test
        SB_1 = method_1(var, w0_1, x_test, y_test, pred_test, px)
        SB_M1 = np.append(SB_M1, SB_1)
    
        w0_2= method_2_w0(x_train, y_train, 1, 2, px_train)
        pred_validation = f_est_validation
        SB_2 = method_2(w0_2, x_validation, y_validation, pred_validation, 
                        px_validation)
        SB_M2 = np.append(SB_M1, SB_2)
        
    return beta, w, f_est, pred_validation, error_train_list, error_validation_list, FVU_validation, SB_M1, SB_M2