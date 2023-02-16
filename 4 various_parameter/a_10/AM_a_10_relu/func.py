import numpy as np 
from scipy import special
from math import pi 
import matplotlib.pyplot as plt 

N_test = 10**3

def f(x):
        a = 1 / 10 # change 
        si, ci = special.sici(x / a)
        return si * np.exp(-x ** 2 / 2)
    
x_test = np.linspace(- 4 , 4, N_test).reshape(N_test, 1)
x_test = np.reshape(x_test, (N_test, 1))
y_test = f(x_test)

plt.plot(x_test, y_test)