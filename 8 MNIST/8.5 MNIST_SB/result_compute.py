import numpy as np

def error_compute(y, pred):
    #error = - np.sum(np.diagonal(y.dot(np.log(pred+1e-10).T)))
    # y pred in N*10
    #error = np.mean(np.sum((y-pred)**2, axis=1))
    N = np.shape(y)[0]
    y_index = np.argmax(y, 1)
    pred_index = np.argmax(pred, 1)
    error = np.linalg.norm((y_index-pred_index), ord=2)/N
    return error

def accuracy_compute(y, pred): 
    y_index = np.argmax(y, 1)
    pred_index = np.argmax(pred, 1)
    accuracy = np.mean((pred_index==y_index).astype('int'))
    return accuracy