from cProfile import label
import numpy as np 
from generate_data import generate_data
from neural_network import nn_define_2
import tensorflow as tf 
import matplotlib.pyplot as plt 

from SB_method_1 import method_1_w0, method_1
from SB_method_2 import method_2_w0, method_2
from FVU_compute import FVU_compute

# generate the data 
N_train = 10**4
N_validation = 2500
N_test = 2 ** 14

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

x_train, y_train, x_validation, y_validation, x_test, y_test = generate_data(N_train, 
                                                                             N_validation, 
                                                                             N_test)

# Save training values for training error evaluation
x_train_eval = x_train
y_train_eval = y_train

# Normalize the training data
x_mean = x_train.mean(axis=0)
x_std = x_train.std(axis=0)
y_mean = y_train.mean(axis=0)
y_std = y_train.std(axis=0)
x_train = (x_train - x_mean) / x_std
y_train = (y_train - y_mean) / y_std

px = 1 / x_std / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * (((x_test) / x_std) ** 2))
px_train = 1 / x_std / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * ((x_train / x_std) ** 2))
px_validation = 1 / x_std / np.sqrt(2 * np.pi) * np.exp(-1 / 2 * ((x_validation / x_std) ** 2))


# define the neural network 
model = nn_define_2(input_dim = 1, 
                    output_dim = 1, 
                    num_layers = 1, 
                    num_nodes = 2**6, ############################
                    #activation=tf.keras.activations.relu, ####################
                    activation = tf.math.cos, 
                    kernel_initializer=tf.keras.initializers.he_normal, ##############
                    bias_initializer=tf.keras.initializers.Zeros())


batch_size = 32 ###############################
learning_rate = 0.0005 #################################
epochs = 1000
num_evals = 100
mod = int(epochs / num_evals)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1).batch(batch_size)

loss_fn=tf.keras.losses.mean_squared_error
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate) 

def nn_predict(x): # plot 
    x = (x - x_mean) / x_std
    beta = model(x) * y_std + y_mean
    return beta.numpy()
    
error_train_list = []
error_validation_list = []
FVU = []
FVU_validation = []
SB_M1 = []
SB_M2 = []

# Training step
for epoch in range(epochs):

    # Train the Network
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Record the loss computations
        with tf.GradientTape() as tape:
            pred = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, pred)
        # Compute and apply gradients
        model.grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(model.grads, model.trainable_weights))
        
    error_train = np.mean((y_train_eval - nn_predict(x_train_eval)) ** 2)
    error_train_list = np.append(error_train_list, error_train)
    
    error_valid = np.mean((y_validation - nn_predict(x_validation)) ** 2)
    error_validation_list = np.append(error_validation_list, error_valid)
    
    # Print the training loss for every tenth epoch
    if epoch % 10 == 0:
        print("\nEnd of epoch  " + str(epoch) + ", Training error " +
                      str(error_train)) 
        print("\nEnd of epoch  " + str(epoch) + ", Validation error " +
                      str(error_valid))
        
    pred = nn_predict(x_train_eval)
    pred_validation = nn_predict(x_validation)
    FVU = np.append(FVU, FVU_compute(y_train, pred))
    FVU_validation = np.append(FVU_validation, FVU_compute(y_validation, pred_validation))
    
    w0_1, var = method_1_w0(x_test, y_test, px)
    pred_test = nn_predict(x_test)
    SB_1 = method_1(var, w0_1, x_test, y_test, pred_test, px)
    SB_M1 = np.append(SB_M1, SB_1)
    
    w0_2 = method_2_w0(x_train, y_train, 1, 2, px_train) 
    pred_validation = nn_predict(x_validation)
    SB_2 = method_2(w0_2, x_validation, y_validation, pred_validation, px_validation)
    SB_M2 = np.append(SB_M1, SB_2)
    
np.savez('./sgd_26.npz', 
         x_train = x_train, 
         y_train = y_train, 
         x_validation = x_validation, 
         y_validation = y_validation,  
         pre_validation = pred_validation, 
         error_train_list = error_train_list, 
         error_validation_list = error_validation_list, 
         FVU_validation = FVU_validation, 
         SB_M1 = SB_M1, 
         SB_M2 = SB_M2)
                      
plt.subplot(2, 2, 1)
plt.plot(x_validation, pred_validation, '*', label = 'validation')
plt.plot(x_test, y_test, label = 'test')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Estimation")
plt.legend()
                      
t_vec = np.arange(0, epochs)
plt.subplot(2, 2, 2)
plt.plot(t_vec[2:], error_train_list[2:], label='training_error')
plt.plot(t_vec[2:], error_validation_list[2:], label='validation_error')
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
plt.savefig('./sgd_26.jpg')
plt.show()

