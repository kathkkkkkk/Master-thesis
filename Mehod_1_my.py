import numpy as np 
from math import pi 


def Method_1(var_f, w0, x_test, y_test, pred, px): 
    
    N = len(x_test) 
    # compute r
    r = y_test - pred

    #plt.plot(x_test, y_test, color='black', label = 'y_test')
    #plt.plot(x_test, pred, color='green', label = 'predict')
    #plt.plot(x_test, r, color='red', label = 'residual')
    #plt.title("data fitting")
    #plt.legend()
    #plt.show()

    # calculate r_p
    dx = x_test[1] - x_test[0]
    r_mean = np.sum(px * r * dx)
    rp = np.sqrt(px) * (r - r_mean) 
    # print(np.shape(rp))

    # Fourier transform 
    rf = np.fft.fft(rp, axis=0)
    rf = np.reshape(rf, (N, 1))
    #xf = fftfreq(N,dx)[:N//2]
    #plt.plot(xf, 2.0/N * np.abs(rf[0:N//2]))
    #plt.title("Fourier transform")
    #plt.show()

    # compute sum_low
    dw = 2*pi/N/dx 
    r2 = np.abs(rf * dx)**2 * dw
    # w0 = int(w0)
    if w0==0: 
        e_low_sum = r2[w0]
        e_high_sum = sum(r2) - e_low_sum
    else:
        e_low_sum = r2[0] + 2*np.sum(r2[1:w0])
        e_high_sum = sum(r2) - e_low_sum

    # compute SB
    e_low = e_low_sum/var_f
    e_high = e_high_sum/var_f

    var_r = np.var(r)
    var_f = np.var(y_test)
    FVU = var_r/var_f 
    
    SB = (e_high - e_low) / (e_high + e_low)
    
    return e_low, e_high, SB



# compute cut-off frequency w0

def Method_1_w0(x, y, px): 
    
    # calculate f_p
    dx = x[1] - x[0]
    E_y = np.sum(px * y * dx)
    fp = np.sqrt(px) * (y - E_y)

    # fast Fourier transform
    N = len(x)
    yf = np.fft.fft(fp, axis=0)
    #xf = fftfreq(N,dx)[:N//2]
    #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    #plt.title("Fourier transform")
    #plt.show()
    
    # the minimization function 
    dw = 2*pi/N/dx 
    K = np.linspace(-N/2, N/2-1, N)
    # K = np.reshape(K, (N,1))
    # yf = np.reshape(yf, (N+1,1))
    f2 = np.abs(yf * dx)**2 * dw
    #plt.plot(K,f2)
    #plt.title("the minimization function")
    #plt.show()
    
    
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

    #plt.plot(n0, e_diff)
    #plt.title("e_low - e_high")
    #plt.show()

    #plt.plot(n0, e_diff)
    #plt.title("e_low - e_high")
    #plt.xlim(0,100)
    #plt.show()
    
    # calculate the cut-off w0
    e_diff_min = min(e_diff)
    w0 = np.argmin(e_diff) 
    # print(w0)
    
    return var_f, w0