import numpy as np 
from result_compute import error_compute, accuracy_compute

# minimization function 
def min_fun(x, y, w, lambda_): 
    N = np.shape(x)[0]
    K = np.shape(w)[0]
    I = np.identity(K)
    one = np.ones(N)
    x_ = np.c_[x, one]
    S = np.cos(x_.dot(w.T)) 
    beta = np.linalg.inv(np.dot(S.T, S) + lambda_ * N * I).dot(S.T).dot(y)
    return beta, S

# Metropolis algorithm 1
#def algorithm_1(x_train, y_train, x_test, y_test, x_validation, y_validation, K, M, lambda_, delta, gamma):
def algorithm_1(x_train, y_train, x_valid, y_valid, x_test, y_test, K, M, lambda_, delta, gamma, sd_n):
    
    # normalize the data 
    #x_mean = np.mean(x_train)
    #x_std = np.std(x_train, ddof=1)
    #y_mean = np.mean(y_train)
    #y_std = np.std(y_train, ddof=1)
    
    #x_train_norm = (x_train - x_mean)/x_std
    #y_train_norm = (y_train - y_mean)/y_std
    #x_test_norm = (x_test - x_mean)/x_std
    #y_test_norm = (y_test - y_mean)/y_std
    #x_validation_norm = (x_validation - x_mean)/x_std
    #y_validation_norm = (y_validation - y_mean)/y_std
    
    N = np.shape(x_train)[0]
    d = np.shape(x_train)[1] 
    w = np.zeros(K*(d+1)).reshape((K, d+1)) #? 
    
    error_train_list = []
    error_valid_list = []
    error_test_list = []
    
    accuracy_train_list = []
    accuracy_valid_list = []
    accuracy_test_list = []
  
    #beta, S = min_fun(x_train_norm, y_train_norm, w, lambda_)
    beta = np.zeros(10*K).reshape(10, K)
    S = np.zeros(10*N*K).reshape(10, N, K)
    for i in np.arange(10): 
        beta[i],  S[i] = min_fun(x_train, y_train[:,i], w, lambda_)
    #print(beta)
    
    seed = 1
    np.random.seed(seed)
    
    for epoch in range(M): 
        #print('w=', i)
        r_n = np.random.normal(0, sd_n, K*(d+1)).reshape((K, d+1))
        w_temp = w + delta * r_n
        #print(w_temp)
        #beta_temp, S_temp = min_fun(x_train_norm, y_train_norm, w_temp, lambda_)
        beta_temp = np.zeros(10*K).reshape(10, K)
        S_temp = np.zeros(10*N*K).reshape(10, N, K)
        for i in np.arange(10): 
            beta_temp[i],  S_temp[i] = min_fun(x_train, y_train[:,i], w_temp, lambda_)
        #print(beta_temp)
        
        for k in range(K): 
            #print('k=', k)
            r_u = np.random.uniform(0, 1)
            #print(np.linalg.norm(beta_temp[:,k], ord=2))
            #print(np.linalg.norm(beta[:,k], ord=2))
            #print(r_u)
            #print((np.linalg.norm(beta_temp[:,k], ord=2)/np.linalg.norm(beta[:,k], ord=2))**gamma)
            
            if (np.linalg.norm(beta_temp[:,k], ord=2)
                /np.linalg.norm(beta[:,k], ord=2))**gamma > r_u: 
                #print('yes')
                w[k] = w_temp[k]
                #print('w_k', w[k])
                beta[:,k] = beta_temp[:,k]
                #print('beta_k', beta[k])
                
        #beta, S = min_fun(x_train_norm, y_train_norm, w, lambda_)
        beta = np.zeros(10*K).reshape(10, K)
        S = np.zeros(10*N*K).reshape(10, N, K)
        for i in np.arange(10): 
            beta[i],  S[i] = min_fun(x_train, y_train[:,i], w, lambda_)
        
        #f_est_train = S.dot(beta) * y_std + y_mean
        pred_train = np.zeros(10*N).reshape(N, 10)
        for i in np.arange(10): 
            pred_train[:,i] = S[i].dot(beta[i])
        error_train = error_compute(y_train, pred_train)
        error_train_list = np.append(error_train_list, error_train)
        accuracy_train = accuracy_compute(y_train, pred_train)
        accuracy_train_list = np.append(accuracy_train_list, accuracy_train)
        
        #beta_validation, S_validation = min_fun(x_valid, y_valid, w, lambda_)
        #f_est_validation = S_validation.dot(beta) * y_std + y_mean
        N_valid = np.shape(x_valid)[0]
        beta_valid = np.zeros(10*K).reshape(10, K)
        S_valid = np.zeros(10*N_valid*K).reshape(10, N_valid, K)
        for i in np.arange(10): 
            beta_valid[i],  S_valid[i] = min_fun(x_valid, y_valid[:,i], w, lambda_)
        pred_valid = np.zeros(10*N_valid).reshape(N_valid, 10)
        for i in np.arange(10): 
            pred_valid[:,i] = S_valid[i].dot(beta[i])
        error_valid= error_compute(y_valid, pred_valid)
        error_valid_list = np.append(error_valid_list, error_valid)
        accuracy_valid= accuracy_compute(y_valid, pred_valid)
        accuracy_valid_list = np.append(accuracy_valid_list, accuracy_valid)
        
        #beta_test, S_test = min_fun(x_test_norm, y_test_norm, w, lambda_)
        #f_est_test = S_test.dot(beta) * y_std + y_mean
        N_test = np.shape(x_test)[0]
        beta_test = np.zeros(10*K).reshape(10, K)
        S_test = np.zeros(10*N_test*K).reshape(10, N_test, K)
        for i in np.arange(10): 
            beta_test[i],  S_test[i] = min_fun(x_test, y_test[:,i], w, lambda_)
        pred_test = np.zeros(10*N_test).reshape(N_test, 10)
        for i in np.arange(10): 
            pred_test[:,i] = S_test[i].dot(beta[i])
        error_test = error_compute(y_test, pred_test)
        error_test_list = np.append(error_test_list, error_test)
        accuracy_test = accuracy_compute(y_test, pred_test)
        accuracy_test_list = np.append(accuracy_test_list, accuracy_test)
    
        # Print the training loss for every epoch
        print("\nEnd of epoch  " + str(epoch) + ", Training error " + str(error_train) + ",Accuracy " + str(accuracy_train)) 
        print("\nEnd of epoch  " + str(epoch) + ", Validation error " + str(error_valid) + ", Validation accuracy " + str(accuracy_valid))
        
    #error_valid_min = np.min(error_validation_list)
    
    #error_test_end = error_compute(y_test_norm, S_test, beta_test)
        
    #return beta, w, f_est_train, f_est_validation, error_train_list, error_validation_list, error_test_list,error_valid_min, error_test_end
    return beta, w, pred_train, pred_valid, pred_test, error_train_list, error_valid_list, error_test_list, accuracy_train_list, accuracy_valid_list, accuracy_test_list

