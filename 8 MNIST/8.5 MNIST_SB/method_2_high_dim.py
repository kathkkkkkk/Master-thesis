import numpy as np 

def func_min(x, y, d, w0, px): 
    
    N = np.shape(x)[0]
    
    # dx = x[1] - x[0]
    # E_y = np.sum(px * y * dx)
    # fp = np.sqrt(px) * (y - E_y)
    y = y.reshape((N,1))
    fp = np.sqrt(px) * (y - np.mean(y)) / px
    #print(np.shape(fp))
    fp = fp.reshape((N,1))
    #print(fp)
    
    if d==1: 
        temp = np.transpose(x)
        temp = np.tile(temp, (N, 1))
        diff = temp - np.transpose(temp)
        snc = np.sinc(w0 * diff / np.pi) # pi? 
    else: 
        snc = np.ones((N, N))
        for k in np.arange(0,d): 
            temp = x[:,k].reshape((N,1))
            diff = temp.T - temp
            snc *= np.sinc(w0 * diff / np.pi)
    #print(snc)
    
    f = 1 / N**2 * (w0/np.pi)**d * np.transpose(fp).dot(snc).dot(fp) 
    
    return f


def method_2_w0(x, y, guess1, guess2, px): 
    # define the function includes w0

    d = np.shape(x)[1]
    w0 = np.array([guess1, guess2])
    f = np.array([func_min(x, y, d, w0[0], px), func_min(x, y, d, w0[1], px)])
    y_var = np.var(y)
    f = f / y_var - 1/2
    #print(f)

    while np.abs(f[-1]) > 1e-4: 
        w0_temp = w0[-1] - f[-1] * (w0[-1] - w0[-2]) / (f[-1] - f[-2])
        w0 = np.append(w0, w0_temp)
        f = np.append(f, func_min(x, y, d, w0_temp, px)/ y_var - 1/2)
        
    w0 = w0[-1]

    return w0 



def method_2(w0, x, y, pred, px):
    
    N = np.shape(x)[0]
    d = np.shape(x)[1]
    y = y.reshape((N,1))
    r = y - pred
    y_var = np.var(y)

    # calculate r_p
    rp = np.sqrt(px) * (r - np.mean(r))

    # compute sum_low
    r_sum = func_min(x, rp, d, w0, px)
    
    e_low = r_sum / y_var 
    
    var_r = np.var(r)
    var_f = np.var(y)
    FVU = var_r/var_f
    e_high = FVU - e_low
    
    SB = (e_high - e_low) / (e_high + e_low) 
    # print(SB)
    
    return SB