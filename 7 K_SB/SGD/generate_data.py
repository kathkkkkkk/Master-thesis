import numpy as np 
from scipy import special
from math import pi 


def generate_data(N_train = 10**4, N_validation = 2500,  N_test = 2 ** 14): 
    # try two seperate data sets 
    
    def f(x):
        a = 1 / 100 # change 
        si, ci = special.sici(x / a)
        return si * np.exp(-x ** 2 / 2)

    x_train = np.random.normal(0, 1, N_train)
    x_train = np.reshape(x_train, (N_train, 1))
    y_train = f(x_train)
    
    x_validation = np.random.normal(0, 1, N_validation) # 
    x_validation = np.reshape(x_validation, (N_validation, 1))
    y_validation = f(x_validation)
    
    x_test = np.random.normal(0, 1, N_train)
    x_test = np.reshape(x_test, (N_train, 1))
    y_test = f(x_test)
    
    x_test_sb = np.linspace(-25 * pi, 25 * pi, N_test).reshape(N_test, 1)
    x_test_sb = np.reshape(x_test_sb, (N_test, 1))
    y_test_sb = f(x_test_sb)
    
    return x_train, y_train, x_validation, y_validation, x_test, y_test, x_test_sb, y_test_sb 