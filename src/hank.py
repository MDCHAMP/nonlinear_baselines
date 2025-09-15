import numpy as np


def get_lags(lags, shape, t0=True):
    off = int(not t0)
    N, D = shape
    # parse lags
    if isinstance(lags, (int, np.integer)):
        new_lags = [tuple([i for i in range(off, lags + off)]) for _ in range(D)]
    elif isinstance(lags, tuple):
        new_lags = [lags for _ in range(D)]
    elif isinstance(lags, list):
        new_lags = []
        for lag in lags:
            if isinstance(lags, list):
                raise ValueError("Incorrect lag format provided, see docs.")
            lag_single_dim = get_lags(lag, (N, 1), t0)[0]
            new_lags.append(lag_single_dim)
    else:
        raise ValueError("Incorrect lag format provided, see docs.")
    # make sure that lags are valid
    assert len(new_lags) == D  # correct number of dims
    for lag in new_lags:
        arr = np.array(lag)
        assert np.all(arr < N)  # does not exceed data length
        if not t0:
            assert np.all(arr > 0)  # has t0?
    return new_lags


def hank(X, lags, include_tt=True):
    if len(X.shape) == 1:
        X = X[:, None]
    N = X.shape[0]
    lags = get_lags(lags, X.shape, include_tt)
    Dh = sum([len(ll) for ll in lags])
    off = N - max([max(ll) if len(ll) > 0 else 0 for ll in lags])
    H = np.zeros((off, Dh))
    for i in range(H.shape[0]):
        idx = 0
        for d, lag in enumerate(lags):
            L = len(lag)
            if L == 0:
                continue
            H[i, idx : idx + L] = X[i - np.array(lag) - off, d]
            idx += L
    return H, slice(-off, None, None)


def NARXify(X, Y, Xlags, Ylags):
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    assert X.shape[0] == Y.shape[0]
    Hy, slcy = hank(Y, Ylags, include_tt=False)
    Hx, slcx = hank(X, Xlags, include_tt=True)
    if Hx.shape[0] <= Hy.shape[0]:
        return np.hstack([Hy[slcx], Hx]), slcx  
    elif Hx.shape[0] > Hy.shape[0]:
        return np.hstack([Hy, Hx[slcy]]), slcy

def marvin_np(f, x, y, lagsx, lagsy):
    t0 = np.maximum(np.max(lagsx), np.max(lagsy))
    hy = y[:t0]
    hx = x[:t0]
    y_mpo = np.zeros_like(y) * np.nan
    y_mpo[:t0] = hy
    for t in range(t0, y_mpo.shape[0]):
        hx = x[t - t0 : t + 1][::-1]  # t->t-t0
        hy = y_mpo[t - t0 : t + 1][::-1]  # t->t-t0
        # note that in hy[0] is nan by construction
        y_mpo[t] = f(np.concat([hy[lagsy], hx[lagsx]])) # note that this is different to other internal ways of building the hankel internally i.e in MDC.jacks.hank
    return y_mpo

def predict_np(Xp, Yp, nx, ny, F, theta, mode="MPO"):
    xlags_arr = np.arange(nx)
    ylags_arr = np.arange(1, ny+1)
    Hp, slc_p = NARXify(Xp, Yp, nx, ny)
    if mode == "OSA":
        Yhat = F(Hp, theta)
    elif mode == "MPO":
        Yhat = marvin_np(lambda h: F(h.T, theta), Xp, Yp, xlags_arr, ylags_arr)
    return Yhat[slc_p].reshape(-slc_p.start, -1), slc_p


# def marvin(H, Y, ylags, pred):
#     Y = Y.reshape(Y.shape[0], -1)
#     ylags = get_lags(ylags, Y.shape, False)
#     YMPO = Y.copy()
#     Htt = H.copy()
#     off = max([max(ll) if len(ll) > 0 else 0 for ll in ylags])
#     for i in range(len(H)):
#         idx = 0
#         for d, lag in enumerate(ylags):
#             L = len(lag)
#             if L == 0:
#                 continue
#             Htt[i, idx : idx + L] = YMPO[i - np.array(lag) + off, d].T
#             idx += L
#         YMPO[i + off, :] = pred(Htt[i, :][None, :])
#     return YMPO


# def predict(Xp, Yp, x_lags, y_lags, F, theta, mode="MPO"):
#     Hp, slc_p = NARXify(Xp, Yp, x_lags, y_lags)
#     if mode == "OSA":
#         Yhat = F(Hp, theta)
#     elif mode == "MPO":
#         Yhat = marvin(Hp, Yp, y_lags, lambda h: F(h, theta))
#     return Yhat[slc_p].reshape(-slc_p.start, -1), slc_p




# %%

# if __name__ == '__main__':
    
#     from MDC import fr, np, plt, sns, ss, pd
#     from MDC.linalg import stable_least_squares
#     import nonlinear_benchmarks as nlb

#     train, test = nlb.Cascaded_Tanks()
#     nlags = (2, 3)
#     Ht, slc_p = NARXify(*train, *nlags)
#     alpha = stable_least_squares(Ht, train.y[slc_p])
    
#     ympo1, slc = predict(*test, *nlags, lambda  th, h: (th@h), alpha, mode="MPO")
#     ympo2, slc = predict_np(*test, *nlags, lambda  th, h: (th@h), alpha, mode="MPO")

#     plt.plot(ympo1)
#     plt.plot(ympo2)
#     assert np.allclose(ympo1, ympo2)