import numpy as np 
import tensorflow as tf 

from data_get import get_data 
from result_compute import error_compute, accuracy_compute 
from nn_define import NN_define

x_train, y_train, x_valid, y_valid, x_test, y_test  = get_data()

        
def nn_training(epochs = 100, 
                batch_size = 32, 
                K = 2**10, 
                learning_rate = 0.001): 
    
    seed = 1
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    N_train = np.shape(x_train)[0]
    pred_train = np.zeros(epochs*10*N_train).reshape(epochs, 10, N_train)
    N_valid = np.shape(x_valid)[0]
    pred_valid = np.zeros(epochs*10*N_valid).reshape(epochs, 10, N_valid)
    N_test = np.shape(x_test)[0]
    pred_test = np.zeros(epochs*10*N_test).reshape(epochs, 10, N_test)

    error_train_list = []
    error_valid_list = []
    error_test_list = []
        
    accuracy_train_list = []
    accuracy_valid_list = []
    accuracy_test_list = []
    
    class CustomCallback(tf.keras.callbacks.Callback): 
    
        def __init__(self, epochs, name, i, pred_train, pred_valid, pred_test): 
            self.epochs = epochs  
            self.name = name
            self.i = i
            self.pred_train = pred_train
            self.pred_valid = pred_valid
            self.pred_test = pred_test
        
        def on_epoch_end(self, epoch, logs=None): 
                
            pred_train[epoch-1, i] = model.predict(x_train).flatten()
            pred_valid[epoch-1, i] = model.predict(x_valid).flatten()
            pred_test[epoch-1, i] = model.predict(x_test).flatten()
    
    model = NN_define(input_size = 784, m_size = K, output_size = 1)
    
    model.compile(loss='mse', 
                optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate), 
                metrics=['accuracy'])
    
    # position to set seed? 
    

    for i in range(10): 
        
        history = CustomCallback(epochs, model, i, pred_train, pred_valid, pred_test)
        model.fit(x=x_train, y=y_train[:,i], 
            validation_data = (x_valid, y_valid), 
            epochs=epochs, 
            batch_size=batch_size, 
            callbacks=[history])
        
    for m in range(epochs): 
        
        error_train = error_compute(y_train, pred_train[m].T)
        error_train_list = np.append(error_train_list, error_train)
        accuracy_train = accuracy_compute(y_train, pred_train[m].T)
        accuracy_train_list = np.append(accuracy_train_list, accuracy_train)
            
        error_valid= error_compute(y_valid, pred_valid[m].T)
        error_valid_list = np.append(error_valid_list, error_valid)
        accuracy_valid= accuracy_compute(y_valid, pred_valid[m].T)
        accuracy_valid_list = np.append(accuracy_valid_list, accuracy_valid)
            
        error_test = error_compute(y_test, pred_test[m].T)
        error_test_list = np.append(error_test_list, error_test)
        accuracy_test = accuracy_compute(y_test, pred_test[m].T)
        accuracy_test_list = np.append(accuracy_test_list, accuracy_test)
        
    return pred_train, pred_valid, pred_test, error_train_list, error_valid_list, error_test_list, accuracy_train_list, accuracy_valid_list, accuracy_test_list 