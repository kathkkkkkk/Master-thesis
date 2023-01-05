import numpy as np 

def func_min(x, y, d, w0, px): 
    
    N = len(x)
    
    dx = x[1] - x[0]
    # E_y = np.sum(px * y * dx)
    # fp = np.sqrt(px) * (y - E_y)
    fp = np.sqrt(px) * (y - np.mean(y)) / px
    #print(fp)
    
    if d==1: 
        temp = np.transpose(x)
        temp = np.tile(temp, (N, 1))
        diff = temp - np.transpose(temp)
        snc = np.sinc(w0 * diff / np.pi)
    else: 
        snc = np.ones((N, N))
        for i in np.arange(0,d): 
            temp = np.tile(x**i, (N, 1))
            diff = temp - np.transpose(temp)
            snc *= np.sinc(w0 * diff / np.pi)
    #print(snc)
    
    f = 1 / N**2 * (w0/np.pi)**d * np.transpose(fp).dot(snc).dot(fp) 
    
    return f


def method_2_w0(x, y, guess1, guess2, px): 
    # define the function includes w0

    d = 1
    y2 = y.mean(axis=1).reshape(len(y), 1)
    w0 = np.array([guess1, guess2]) 
    f = np.array([func_min(x, y, d, w0[0], px), func_min(x, y, d, w0[1], px)])
    y_var = np.var(y)
    f = f / y_var - 1/2

    while np.abs(f[-1]) > 1e-4: 
        w0_temp = w0[-1] - f[-1] * (w0[-1] - w0[-2]) / (f[-1] - f[-2])
        w0 = np.append(w0, w0_temp)
        f = np.append(f, func_min(x, y, d, w0_temp, px)/ y_var - 1/2)
        
    w0 = w0[-1]

    return w0 



def method_2(w0, x, y, pred, px):
    
    N = len(x)
    d = 1
    r = y - pred
    y_var = np.var(y)

    # calculate r_p
    rp = np.sqrt(px) * (r - np.mean(r))

    # compute sum_low
    r_sum = func_min(x, r, d, w0, px)
    
    e_low = r_sum / y_var 
    
    var_r = np.var(r)
    var_f = np.var(y)
    FVU = var_r/var_f
    e_high = FVU - e_low
    
    SB = (e_high - e_low) / (e_high + e_low) 
    # print(SB)
    
    return SB