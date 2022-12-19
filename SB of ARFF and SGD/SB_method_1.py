import numpy as np 
from math import pi 


# compute cut-off frequency w0
def method_1_w0(x, y, px): 
    
    # calculate f_p
    dx = x[1] - x[0]
    E_y = np.sum(px * y * dx)
    fp = np.sqrt(px) * (y - E_y)

    # fast Fourier transform
    N = len(x)
    yf = np.fft.fft(fp, axis=0)
    
    # the minimization function 
    dw = 2*pi/N/dx 
    K = np.linspace(-N/2, N/2-1, N)
    f2 = np.abs(yf * dx)**2 * dw

    # compute the difference 
    var_f = np.sum(f2)
    n0 = range(0, int(N/2))
    e_diff = np.zeros(int(N/2))
    
    for k in n0:
        if k>=0 and k<=N/2-1: 
            if k==0: 
                e_low = f2[k]
                e_high = var_f - e_low
            else:
                e_low = f2[0] + 2*np.sum(f2[1:k])
                e_high = var_f - e_low
            e_diff[k] = np.abs(e_low-e_high)
        else: 
            print("index error")
    
    # calculate the cut-off w0
    e_diff_min = min(e_diff)
    w0 = np.argmin(e_diff) 
    
    return w0, var_f



def method_1(var_f, w0, x_test, y_test, pred, px): 
    
    N = len(x_test) 
    # compute r
    r = y_test - pred

    # calculate r_p
    dx = x_test[1] - x_test[0]
    r_mean = np.sum(px * r * dx)
    rp = np.sqrt(px) * (r - r_mean) 

    # Fourier transform 
    rf = np.fft.fft(rp, axis=0)
    rf = np.reshape(rf, (N, 1))

    # compute sum_low
    dw = 2*pi/N/dx 
    r2 = np.abs(rf * dx)**2 * dw
    w0 = int(w0)
    if w0==0: 
        e_low_sum = r2[w0]
        e_high_sum = sum(r2) - e_low_sum
    else:
        e_low_sum = r2[0] + 2*np.sum(r2[1:w0])
        e_high_sum = sum(r2) - e_low_sum

    # compute SB
    e_low = e_low_sum/var_f
    e_high = e_high_sum/var_f
    
    SB = (e_high - e_low) / (e_high + e_low)
    
    return SB