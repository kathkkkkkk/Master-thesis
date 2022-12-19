import numpy as np 


def fun(x_train, y_train, w0, px):

    # This function computes the quantity that needs to be minimized when finding w0 for Method 2
    d = x_train.shape[1]
    p_tot = np.mean((y_train - np.mean(y_train)) ** 2)
    f = (y_train - np.mean(y_train))
    fp = f / np.sqrt(px)

    if d == 1:
        diff = x_train - x_train.T
        snc = np.sinc(w0 * diff / np.pi)
    else:
        snc = np.ones((len(x_train), len(x_train)))
        for i in np.arange(0, d):
            diff = x_train[:, i].reshape(len(x_train), 1) - x_train[:, i].reshape(len(x_train), 1).T
            snc *= np.sinc(w0 * diff / np.pi)
    
    return 1 / len(x_train) ** 2 * fp.T @ (snc @ fp) * (w0 / np.pi) ** d / p_tot - 1 / 2




def Method_2_w0(x_train, y_train, guess1, guess2, px):
    # This function finds the cutoff frequency using the secant method, if the ouput dimension is >1, the SB is computed
    # on the average output.

    y_train2 = y_train.mean(axis=1).reshape(len(y_train), 1)
    w0 = np.array([guess1, guess2])
    fun_vals = np.array([fun(x_train, y_train2, w0[0], px), fun(x_train, y_train2, w0[1], px)])

    while np.abs(fun_vals[-1]) > 1e-4:
        w0_temp = w0[-1] - fun_vals[-1] * (w0[-1] - w0[-2]) / (fun_vals[-1] - fun_vals[-2])
        w0 = np.append(w0, w0_temp)
        fun_vals = np.append(fun_vals, fun(x_train, y_train2, w0_temp, px))

    w0 = w0[-1]
    return w0




def Method_2(w0, x, y, prediction, px, snc):
    # Given the cutoff frequency, validation data (x,y), network prediction, density px, and sinc-matrix snc,
    # this function computes the spectral bias with Method 2.

    N, d = x.shape
    var = np.mean((y - np.mean(y)) ** 2)

    r = (y - prediction - np.mean(y - prediction))
    rp = r / np.sqrt(px)
    E_low = 1 / N ** 2 * rp.T @ (snc @ rp) * (w0 / np.pi) ** d / var
    E_high = np.mean(r ** 2) / var - E_low
    SB = (E_high - E_low) / (E_high + E_low)

    return E_low, E_high, SB