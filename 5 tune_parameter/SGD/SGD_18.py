from cProfile import label
import numpy as np 
from generate_data import generate_data
from neural_network import nn_define_2
import tensorflow as tf 
import matplotlib.pyplot as plt 

# generate the data 
N_train = 10**4
N_validation = 2500
N_test = 2 ** 14

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

x_train, y_train, x_validation, y_validation, x_test, y_test, x_test_sb, y_test_sb = generate_data(N_train, 
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
                    num_nodes = 2**10, ############################
                    #activation=tf.keras.activations.relu, ####################
                    activation = tf.math.cos, 
                    kernel_initializer=tf.keras.initializers.he_normal, ##############
                    #kernel_initializer=tf.keras.initializers.truncated_normal, 
                    bias_initializer=tf.keras.initializers.Zeros())


batch_size = 512 ###############################
learning_rate = 0.0002 #################################
epochs = 1000

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
for i in range(epochs):
#while (i < epochs) and np.abs(error_train_list[i] - 
#error_train_list[i-1]) > 10**(-3):

    # Train the Network
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # Record the loss computations
        with tf.GradientTape() as tape:
            pred = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, pred)
        # Compute and apply gradients
        model.grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(model.grads, model.trainable_weights))
        
    pred_train = nn_predict(x_train_eval)
    error_train = np.mean((y_train_eval - pred_train) ** 2)
    error_train_list = np.append(error_train_list, error_train)
    
    pred_validation = nn_predict(x_validation)
    error_valid = np.mean((y_validation - pred_validation) ** 2)
    error_validation_list = np.append(error_validation_list, error_valid)
    
    # Print the training loss for every tenth epoch
    if i % 10 == 0:
        print("\nEnd of epoch  " + str(i) + ", Training error " +
                      str(error_train)) 
        print("\nEnd of epoch  " + str(i) + ", Validation error " +
                      str(error_valid))
        
error_valid_min = np.min(error_validation_list)
    
pred_test = nn_predict(x_test)
error_test_end = np.mean((y_test - pred_test) ** 2)
    
np.savez('./sgd_18.npz', 
         x_train = x_train_eval, 
         y_train = y_train_eval, 
         x_validation = x_validation, 
         y_validation = y_validation,  
         x_test = x_test, 
         y_test = y_test, 
         x_test_sb = x_test_sb, 
         y_test_sb = y_test_sb, 
         pred_train = pred_train, 
         pre_validation = pred_validation, 
         pred_test = pred_test, 
         error_train_list = error_train_list, 
         error_validation_list = error_validation_list,
         error_valid_min = error_valid_min, 
         error_test_end = error_test_end)
                      
plt.subplot(1, 2, 1)
plt.plot(x_validation, pred_validation, '*', label = 'validation')
plt.plot(x_test_sb, y_test_sb, label = 'func')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Estimation")
plt.legend()
                      
t_vec = np.arange(0, epochs)
plt.subplot(1, 2, 2)
plt.semilogy(t_vec[2:], error_train_list[2:], label='training_error')
plt.semilogy(t_vec[2:], error_validation_list[2:], label='validation_error')
plt.title("Error")
plt.legend()

plt.tight_layout()
plt.savefig('./sgd_18.jpg')
plt.show()
