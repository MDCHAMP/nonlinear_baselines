# %% imports

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


# data = benchmarks_tvt["SB"]
# data, inv = scale_data(data)
# trains, vals, tests, opts = data
# n_batch = 1
# lags = 10, 10  # ARX_lag_scan(data, inv, n_batch, 3)
# preds = Batch_ARX_MPO(trains, tests, *lags, n_batch)
# scores = evaluate(data, preds, inv) * 1000
# print(scores)

# %% P-NARX


def basis(H, order):
    basis = H[..., None] ** np.arange(1, order + 1)
    basis = basis @ polys.legendre(order)
    return basis.reshape(*H.shape[:-1], -1)


def F_PNARX(H, theta):
    return basis(H, theta["order"]) @ theta["alpha"]


# data = benchmarks_tvt["SB"]
# data, inv = scale_data(data)
# trains, vals, tests, opts = data

# lags = 10, 10#ARX_lag_scan(data, inv, n_batch, 10)
# n_batch = 100

# H, Y, _ = batch_Hank(trains, *lags, n_batch=n_batch)

# best = 10e10  # xval for model order
# for order in range(2,10):
#     phi = basis(H, order)
#     alpha = batch_SLS(phi, Y)
#     theta = {'alpha':alpha, 'order':order}
#     preds = multi_predict_AR(vals, lags, F_PNARX, theta)
#     scores = evaluate(data, preds, inv, phi.shape[-1], "val", AIC)
#     print(scores, order)
#     if scores.mean() < best:
#         best = scores.mean()
#         best_order = order

# best_alpha = batch_SLS(basis(H, best_order), Y)
# theta = {'alpha':best_alpha, 'order':best_order}
# preds = multi_predict_AR(tests, lags, F_PNARX, theta)
# evaluate(data, preds, inv)*1000

# %% GP-NARX

# data = benchmarks_tvt["SB"]
# data, inv = scale_data(data)
# trains, vals, tests, opts = data

# lags = 10, 10  # ARX_lag_scan(data, inv, n_batch, 10)
# n_inits = 10
# n_inducing = 200
# opt_ions = {
#     "num_iters": 500,
#     "thresh": None,
#     "optimizer": optax.adam(learning_rate=1e-2),
#     "vb": False,
#     "jit": True,
# }

# key = jr.key(0)
# k0, k1, k2, k3 = jax.random.split(key, 4)

# H, Y, _ = batch_Hank(trains, *lags, n_batch=1)
# H = H[0]  # no batching
# Y = Y[0]

# # Use FITC approximation for > 200 training examples
# if n_inducing < H.shape[0]:  # use sparse GP
#     idx = np.arange(0, H.shape[0], H.shape[0] // n_inducing)
#     train_GP, F_GPNARX, _, nlml = jeep.FITC(H, Y, H[idx], jeep.SE)
# else:
#     train_GP, F_GPNARX, _, nlml = jeep.GP(H, Y, jeep.SE)

# theta0 = {
#     "sf_se": jnp.zeros((n_inits, 1)),
#     "sn": jax.random.uniform(k2, minval=-5, maxval=0, shape=(n_inits, 1)),
#     "ll": jax.random.uniform(k3, minval=-3, maxval=2, shape=(n_inits, 1)),
# }
# thetas, hists = jax.pmap(opt.optaximiser(nlml, **opt_ions))(theta0)

# plt.plot(hists.T)

# # select best model on validation set
# best = 10e10
# val_scores = []
# for i in range(n_inits):
#     theta = jax.tree.map(lambda a: a[i], thetas)
#     preds = multi_predict_AR(vals, lags, F_GPNARX, train_GP(theta))
#     val_scores.append(evaluate(data, preds, inv, 3, "val", AIC).mean())
# print(val_scores)

# # evaluate
# best_theta = jax.tree.map(lambda a: a[np.argmin(np.array(val_scores))], thetas)
# preds = multi_predict_AR(tests, lags, F_GPNARX, train_GP(theta))
# scores = evaluate(data, preds, inv)
# print(scores)

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


# %% MLP-NARX


# data = benchmarks_tvt["SB"]
# data, inv = scale_data(data)
# trains, vals, tests, opts = data

# lags = 10, 10  # ARX_lag_scan(data, inv, n_batch, 10)
# n_inits = 10
# n_batch = 10
# nh = 10
# opt_ions = {
#     "num_iters": 100,
#     "thresh": None,
#     "optimizer": optax.adam(learning_rate=1e-2),
#     "vb": False,
#     "jit": True,
# }

# # wrap in xval loop

# model = MLP(nh)
# opter, Htrain = multi_train_NN(trains, model, lags, n_batch, opt_ions)
# keys = jr.split(jr.key(0), n_inits)
# theta0s = jax.vmap(model.init)(keys, Htrain)
# thetas, histories = jax.pmap(opter)(theta0s)
# best_theta = jax.tree.map(lambda a: a[jnp.argmin(histories[:, -1])], thetas)


# def F_MLP(h, theta):
#     return model.apply(theta, h)


# val_preds = multi_predict_AR(vals, lags, F_MLP, best_theta)
# val_scores = evaluate(data, val_preds, inv, nh, "val", AIC)
# print(val_scores)

# preds = multi_predict_AR(tests, lags, F_MLP, best_theta)
# scores = evaluate(data, preds, inv) * 1000


# %% RNN-type models

# nh = 50
# n_batch = 10
# lags = 10, 0
# n_inits = 10
# model_type = RNN
# opt_ions = {
#     "num_iters": 100,
#     "thresh": None,
#     "optimizer": optax.adam(learning_rate=1e-3),
#     "vb": False,
#     "jit": True,
# }

# data = benchmarks_tvt["SB"]
# data, inv = scale_data(data)  # MM scaling for NNs
# trains, vals, tests, opts = data

# # wrap in xval loop

# model = model_type(nh)
# opter, Htrain = multi_train(trains, model, lags, n_batch, opt_ions)
# keys = jr.split(jr.key(0), n_inits)
# theta0s = jax.vmap(model.init)(keys, Htrain)
# thetas, histories = jax.pmap(opter)(theta0s)
# best_theta = jax.tree.map(lambda a: a[jnp.argmin(histories[:, -1])], thetas)

# val_preds = multi_predict_NN(vals, lags, model, best_theta)
# val_scores = evaluate(data, val_preds, inv, nh, "val", AIC)
# print(val_scores)

# preds = multi_predict_NN(tests, lags, model, best_theta)
# scores = evaluate(data, preds, inv) * 1000

# plt.plot(histories.T)
# print(scores)
