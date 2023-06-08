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


def add_noise(x_train, x_valid, x_test, mean = 0, sd = 1):
    
    np.random.seed(1)
    x_train_noise = x_train + np.random.normal(loc=mean, scale=sd, size=x_train.shape)
    x_valid_noise = x_valid + np.random.normal(loc=mean, scale=sd, size=x_valid.shape)
    x_test_noise = x_test + np.random.normal(loc=mean, scale=sd, size=x_test.shape)
    
    return x_train_noise, x_valid_noise, x_test_noise


# compress the image 
def image_compress(x_train_orig, x_valid_orig, x_test_orig, k=2):
    
    def single_image_compress(x, k=2): 
        comp_size = int(28/k)
        x_temp = x.reshape(28,28)
        x_comp = np.zeros(comp_size * comp_size).reshape(comp_size, comp_size)
        for i in range(0, comp_size): 
            for j in range(0, comp_size): 
                avg = x_temp[k*i:k*(i+1), k*j:k*(j+1)]
                x_comp[i,j] = np.mean(avg)
        x_comp = x_comp.reshape(comp_size * comp_size)
        return x_comp
    
    # compress the images 
    n_train = np.shape(x_train_orig)[0]
    x_train_comp = []
    for i in range(n_train): 
        x_train_comp = np.append(x_train_comp, single_image_compress(x_train_orig[i], k))
    x_train_comp = x_train_comp.reshape((n_train, -1))
    
    n_valid = np.shape(x_valid_orig)[0]
    x_valid_comp = []
    for i in range(n_valid): 
        x_valid_comp = np.append(x_valid_comp, single_image_compress(x_valid_orig[i], k))
    x_valid_comp = x_valid_comp.reshape((n_valid, -1))
    
    n_test = np.shape(x_test_orig)[0]
    x_test_comp = []
    for i in range(n_test): 
        x_test_comp = np.append(x_test_comp, single_image_compress(x_test_orig[i], k))
    x_test_comp = x_test_comp.reshape((n_test, -1))
    
    return x_train_comp, x_valid_comp, x_test_comp