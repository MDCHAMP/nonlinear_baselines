# %% imports
import os
from copy import deepcopy as dc

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MM
import optax
from flax import linen as nn
from flax.linen.module import nowrap

from src.jacks import jax, jnp, jr
from src import polys
from src.hank import NARXify, predict
from src.jacks import jeep, opt
from src.linalg import stable_least_squares as SLS
from src.metrics import AIC, rmse

from dataload import benchmarks_tvt

# %% Utils

def concat_dataset(datasets):
    y = np.concatenate([np.atleast_2d(d.y) for d in datasets], axis=-1).T
    u = np.concatenate([np.atleast_2d(d.u) for d in datasets], axis=-1).T
    return y, u


def scale_data(data, scaler=SS, scaler_params={}):
    """Scale the data acording to scaler"""
    trains, vals, tests, opts = dc(data)

    y_train, u_train = concat_dataset(trains)  # use all training data to fit scaler
    SSy = scaler(**scaler_params).fit(y_train)
    SSu = scaler(**scaler_params).fit(u_train)

    for sig in [*trains, *vals, *tests]:
        sig.u = SSu.transform(sig.u.reshape(sig.u.shape[0], -1))
        sig.y = SSy.transform(sig.y.reshape(sig.y.shape[0], -1))

    def yinv(y):
        return SSy.inverse_transform(np.atleast_2d(y)).squeeze()

    data_scaled = trains, vals, tests, opts
    return data_scaled, yinv


def batch_Hank(trains, nx, ny, n_batch=1):
    """divide the training data into batches for minibatch training"""

    n_trains = len(trains)
    n_batch = max(n_batch, n_trains) // n_trains
    H_batches = []
    Y_batches = []
    for dat in trains:
        H, slc = NARXify(dat.u.squeeze(), dat.y.squeeze(), nx, ny)
        Y = np.atleast_2d(dat.y[slc])
        batch_len = int(jnp.floor(len(H) / n_batch))

        Hbatch = H[: n_batch * batch_len].reshape(n_batch, batch_len, H.shape[-1])
        Ybatch = Y[: n_batch * batch_len].reshape(n_batch, batch_len, Y.shape[-1])

        H_batches.append(Hbatch)
        Y_batches.append(Ybatch)

    Hbatch = np.concatenate(H_batches, axis=0)
    Ybatch = np.concatenate(Y_batches, axis=0)
    return Hbatch, Ybatch, slc


def batch_SLS(Xbatch, Ybatch):
    """compute a stable LSQ over batched data"""
    batches = [SLS(X, y) for X, y in zip(Xbatch, Ybatch)]
    return np.mean(batches, 0).squeeze()


def multi_predict_AR(datas, lags, F, theta):
    preds = []
    for tgt in datas:
        YMPO, slc = predict(*tgt, *lags, F, theta, "MPO")
        preds.append(YMPO[slc])
    return preds


def evaluate(data, preds, inv, k=0, eval_type="test", metric=rmse):
    _, vals, tests, opts = data
    if eval_type == "val":
        targets = vals
    elif eval_type == "test":
        targets = tests
    scores = []
    for (
        tgt,
        pred,
    ) in zip(targets, preds):
        if eval_type == "val":
            neval = pred.shape[0] # compare everything after the lags
        elif eval_type == 'test':
            neval = tgt.y.shape[0] - opts['max_ny'] # use the cuttoff for the test set
        score = metric(inv(tgt.y)[-neval:], inv(pred)[-neval:], k)
        scores.append(score)
    return np.array(scores)


# %% ARX


def F_ARX(H, alpha):
    return H @ alpha


def Batch_ARX_MPO(trains, vals, lags_x, lags_y, n_batch):
    lags = lags_x, lags_y
    H, Y, _ = batch_Hank(trains, *lags, n_batch=n_batch)
    alpha = batch_SLS(H, Y)
    return multi_predict_AR(vals, lags, F_ARX, alpha)


def ARX_lag_scan(data, inv, n_batch=1, max_lag=None):
    if 'LAG_OVERRIDE_NX' in os.environ: # override ARX lags and skip xval
        nx = int(os.environ['LAG_OVERRIDE_NX'])
        ny = int(os.environ['LAG_OVERRIDE_NY'])
        return nx,ny

    trains, vals, _, opts = data
    if max_lag is None:
        max_lag = opts['max_lag']
    best = 10e10
    for nx in range(1, max_lag + 1):  # use the same as y for now
        for ny in range(1, max_lag + 1):
            try:
                H, Y, _ = batch_Hank(trains, nx, ny, n_batch=n_batch)
                alpha = batch_SLS(H, Y)
                preds = multi_predict_AR(vals, (nx, ny), F_ARX, alpha)

                scores = evaluate(data, preds, inv, H.shape[-1], "val", AIC)
                score = np.mean(scores)
                # print(nx, ny, score)
                if score < best:
                    best = score
                    lags = nx, ny
            except ValueError as e:
                continue
    return lags


# %% P-NARX


def basis(H, order):
    basis = H[..., None] ** np.arange(1, order + 1)
    basis = basis @ polys.legendre(order)
    return basis.reshape(*H.shape[:-1], -1)


def F_PNARX(H, theta):
    return basis(H, theta["order"]) @ theta["alpha"]



# %% GP-NARX



# %% NN models


class RNNCell(nn.RNNCellBase):
    features: int

    @nn.compact
    def __call__(self, carry, x):
        h = carry
        hidden_features = h.shape[-1]

        dense_h = nn.Dense(hidden_features)
        dense_i = nn.Dense(hidden_features)

        new_h = nn.tanh(dense_i(x) + dense_h(h))
        return new_h, new_h

    @nowrap
    def initialize_carry(self, rng, input_shape):
        batch_dims = input_shape[:-1]
        _, k2 = jax.random.split(rng)
        mem_shape = batch_dims + (self.features,)
        h = nn.initializers.zeros_init()(k2, mem_shape, jnp.float32)
        return h

    @property
    def num_feature_axes(self) -> int:
        return 1


def MLP(nh):
    class network(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(nh)(x)
            x = nn.tanh(x)
            x = nn.Dense(1)(x)
            return x

    return network()


def RNN(celltype=RNNCell):
    def model(nh):
        class network(nn.Module):
            @nn.compact
            def __call__(self, x):
                x = nn.RNN(celltype(nh))(x)
                x = nn.Dense(1)(x)
                return x

        return network()
    return model

def multi_predict_NN(datas, lags, model, theta):
    preds = []
    for tgt in datas:
        H, _ = NARXify(*tgt, *lags)  # full unroll so no batching
        Ypred = model.apply(theta, H.astype(jnp.float32))
        preds.append(Ypred)
    return preds


def multi_train_NN(trains, model, lags, n_batch, opts):
    H, Y, _ = batch_Hank(trains, *lags, n_batch=n_batch)
    H = jnp.array(H).astype(jnp.float32)
    Y = jnp.array(Y).astype(jnp.float32)

    def NN_loss(theta):
        return rmse(Y, model.apply(theta, H))

    return opt.optaximiser(NN_loss, **opts), H

