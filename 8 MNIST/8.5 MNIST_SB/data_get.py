import numpy as np 
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

DATASET_SIZE = 70000
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.2
TEST_RATIO = 0.1

# change the format of data 
def data_format(y): 
    y_10 = []
    for i in np.arange(10): 
        temp = (y==i).astype(int)
        y_10 = np.append(y_10, temp)
        #print(y_10)
    y_10 = y_10.reshape(y.shape[0], -1, order='F')
    return y_10 

def get_data(): 

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    x_train_orig, x_valid_orig, y_train_orig, y_valid_orig = train_test_split(x, y, test_size=(1-TRAIN_RATIO), random_state=1)
    x_valid_orig, x_test_orig, y_valid_orig, y_test_orig = train_test_split(
        x_valid_orig, y_valid_orig, test_size=((TEST_RATIO/(VALIDATION_RATIO+TEST_RATIO))), random_state=1) 

    # normalize data 
    x_train = x_train_orig.reshape(x_train_orig.shape[0], -1).astype('float32')/255.
    y_train = data_format(y_train_orig)

    x_valid = x_valid_orig.reshape(x_valid_orig.shape[0], -1).astype('float32')/255.
    y_valid = data_format(y_valid_orig)

    x_test = x_test_orig.reshape(x_test_orig.shape[0], -1).astype('float32')/255.
    y_test = data_format(y_test_orig)

    return x_train, y_train, x_valid, y_valid, x_test, y_test 

def get_data_orig(): 
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    x_train_orig, x_valid_orig, y_train_orig, y_valid_orig = train_test_split(x, y, test_size=(1-TRAIN_RATIO), random_state=1)
    x_valid_orig, x_test_orig, y_valid_orig, y_test_orig = train_test_split(
        x_valid_orig, y_valid_orig, test_size=((TEST_RATIO/(VALIDATION_RATIO+TEST_RATIO))), random_state=1) 

    # normalize data 
    x_train = x_train_orig.reshape(x_train_orig.shape[0], -1).astype('float32')/255.
    y_train = y_train_orig

    x_valid = x_valid_orig.reshape(x_valid_orig.shape[0], -1).astype('float32')/255.
    y_valid = y_valid_orig

    x_test = x_test_orig.reshape(x_test_orig.shape[0], -1).astype('float32')/255.
    y_test = y_test_orig

    return x_train, y_train, x_valid, y_valid, x_test, y_test 