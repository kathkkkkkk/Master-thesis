import numpy as np
import scipy.special
import scipy.signal

def getDataset_highDim(N_train=2 ** 12, N_test=2 ** 14, d=3): 
    
    def f(x):
        a = 1 / 100
        si, ci = scipy.special.sici(x[:,0] / a)
        return si * np.exp(- np.linalg.norm(x, axis=1) / 2)
    
    x_train = np.random.randn(N_train*d, 1).reshape((N_train, d))
    y_train = f(x_train)
    x_test = np.linspace(-25 * np.pi, 25 * np.pi, N_test*d).reshape(N_test, d)
    y_test = f(x_test)
    x_validation = np.random.randn(N_train*d, 1).reshape((N_train, d))
    y_validation = f(x_validation)

    return x_train, y_train, x_validation, y_validation, x_test, y_test
