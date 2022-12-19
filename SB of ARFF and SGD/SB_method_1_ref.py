import numpy as np 


def Method_1_w0(x_test, y_test, px, split=0.5):
    # Finding w0 in one dimension for Method 1
    var = 0
    indices = []
    if np.shape(x_test)[1] == 1:
        N = len(y_test)
        dx = x_test[1] - x_test[0]
        dw = 1 / dx / (N - 1) * 2 * np.pi
        E_f = np.sum(px * y_test * dx)
        Y = np.fft.fft(np.sqrt(px) * (y_test - E_f), axis=0) * dx
        
        var = np.sum(np.abs(Y) ** 2) * dw / 2 / np.pi
        s = np.abs(Y[0]) ** 2 * dw/ 2 / np.pi
        
        i = 1
        # Want to find the terms that sum up to half of the variance in the fourier domain
        while s < var * split:
            s = s + 2 * np.abs(Y[i]) ** 2 * dw / 2 / np.pi
            i = i + 1
        # Check which index has the power closest to 1/2
        s1 = s
        t1 = s1 - 2 * np.abs(Y[i - 1]) ** 2 * dw / 2 / np.pi
        q1 = s1 / var
        q2 = t1 / var
        if q1 - 1 / 2 <  1 / 2 - q2:
            ind_low = np.append(np.arange(0, i), np.arange(-i + 1, 0))
            ind_high = np.arange(i, N - i + 1)
        elif q1 - 1 / 2 >=  1 / 2 - q2:
            ind_low = np.append(np.arange(0, i - 1), np.arange(-i, 0))
            ind_high = np.arange(i - 1, N - i)
        indices = {"ind_0": ind_low, "ind_1": ind_high}

    return var, indices, s


def Method_1(p_tot, indices, x_test, y_test, pred, px):
    E_high = 0
    E_low = 0

    # Method one in 1 dimension
    if np.shape(x_test)[1] == 1:
        dx = x_test[1] - x_test[0]
        N = len(y_test)
        dw = 1 / dx / (N - 1) * 2 * np.pi
        res = y_test - pred
        ind_low = indices["ind_0"]
        ind_high = indices["ind_1"]
        res_mean = np.sum(px * res * dx)
        Y_res = np.fft.fft(np.sqrt(px) * (res - res_mean), axis=0) * dx
        
        E_low = (1 / 2 / np.pi * dw * np.sum(np.abs(Y_res[ind_low]) ** 2)) / p_tot
        E_high = (1 / 2 / np.pi * dw * np.sum((np.abs(Y_res[ind_high])) ** 2)) / p_tot
        
    SB = (E_high - E_low) / (E_high + E_low)
    return E_low, E_high, SB