import numpy as np
from functools import partial
from copy import deepcopy as dc
from src.jacks import jax, jnp, jr
from flax import linen as nn
from models import (
    scale_data,
    MM,
    Batch_ARX_MPO,
    evaluate,
    ARX_lag_scan,
    batch_Hank,
    batch_SLS,
    basis,
    F_PNARX,
    multi_predict_AR,
    AIC,
    optax,
    jeep,
    opt,
    MLP,
    multi_train_NN,
    multi_predict_NN,
    RNN,
    RNNCell,
)


def ARX_trainer(data, key):
    data, inv = scale_data(data)
    trains, vals, tests, opts = data
    n_batch = opts["n_batch"]
    lags = ARX_lag_scan(data, inv, n_batch, opts["max_lag"])
    preds = Batch_ARX_MPO(trains, tests, *lags, n_batch)
    scores = evaluate(data, preds, inv) * opts["rmse_scaling"]
    return preds, scores, {"lags": lags}


def PNARX_trainer(data, key):
    data, inv = scale_data(data)
    trains, vals, tests, opts = data
    n_batch = opts["n_batch"]
    lags = ARX_lag_scan(data, inv, n_batch, opts["max_lag"])
    H, Y, _ = batch_Hank(trains, *lags, n_batch=n_batch)
    best = 10e10  # xval for model order
    for order in range(2, 10):  # consider up to 10th order
        try:
            phi = basis(H, order)
            alpha = batch_SLS(phi, Y)
            theta = {"alpha": alpha, "order": order}
            preds = multi_predict_AR(vals, lags, F_PNARX, theta)
            scores = evaluate(data, preds, inv, phi.shape[-1], "val", AIC)
            if scores.mean() < best:
                best = scores.mean()
                best_order = order
        except ValueError:
            continue
    best_alpha = batch_SLS(basis(H, best_order), Y)
    theta = {"alpha": best_alpha, "order": best_order}
    preds = multi_predict_AR(tests, lags, F_PNARX, theta)
    scores = evaluate(data, preds, inv) * opts["rmse_scaling"]
    return preds, scores, {"lags": lags, "order": best_order}


def GPNARX_trainer(data, key):
    data, inv = scale_data(data)
    trains, vals, tests, opts = data
    n_batch = opts["n_batch"]
    lags = ARX_lag_scan(data, inv, n_batch, opts["max_lag"])
    n_inits = 10  # 10 random intialisations
    opt_ions = {
        "num_iters": opts["GP_opt_iters"],
        "thresh": None,
        "optimizer": optax.adam(learning_rate=1e-2),
        "vb": False,
        "jit": True,
    }
    H, Y, _ = batch_Hank(trains, *lags, n_batch=1)
    k1, k2 = jax.random.split(key)
    idx = slice(None, 1000, None)
    # no batching, only use 1k inital points as before - no FITC
    H = H.reshape(-1, H.shape[-1])[idx]
    Y = Y.reshape(-1, Y.shape[-1])[idx]
    train_GP, F_GPNARX, _, nlml = jeep.GP(H, Y, jeep.SE)
    theta0 = {
        "sf_se": jnp.zeros((n_inits, 1)),
        "sn": jax.random.uniform(k1, minval=-5, maxval=0, shape=(n_inits, 1)),
        "ll": jax.random.uniform(k2, minval=-3, maxval=2, shape=(n_inits, 1)),
    }
    thetas, hists = jax.pmap(opt.optaximiser(nlml, **opt_ions))(theta0)
    # select best model on validation set
    val_scores = []
    for i in range(n_inits):
        theta = jax.tree.map(lambda a: a[i], thetas)
        preds = multi_predict_AR(vals, lags, F_GPNARX, train_GP(theta))
        val_scores.append(evaluate(data, preds, inv, 3, "val", AIC).mean())
    # evaluate on test set
    best_theta = jax.tree.map(lambda a: a[np.argmin(np.array(val_scores))], thetas)
    preds = multi_predict_AR(tests, lags, F_GPNARX, train_GP(best_theta))
    scores = evaluate(data, preds, inv) * opts["rmse_scaling"]
    return preds, scores, {"lags": lags, **best_theta, "key": key}


def MLPNARX_trainer(data, key):
    data, inv = scale_data(data, scaler=MM, scaler_params={"feature_range": (-1, 1)})
    trains, vals, tests, opts = data
    n_batch = opts["n_batch"]
    lags = ARX_lag_scan(data, inv, n_batch, opts["max_lag"])
    n_inits = 10
    nhs = [2, 4, 8, 16, 32]
    opt_ions = {
        "num_iters": opts["NN_opt_iters"],
        "thresh": None,
        "optimizer": optax.adam(learning_rate=1e-2),
        "vb": False,
        "jit": True,
    }
    # wrap in xval loop
    best_xval_score = 10e10
    for nh in nhs:
        try:
            model = MLP(nh)
            opter, Htrain = multi_train_NN(trains, model, lags, n_batch, opt_ions)
            keys = jr.split(jr.key(0), n_inits)
            theta0s = jax.vmap(model.init, in_axes=(0, None))(keys, Htrain)
            thetas, histories = jax.pmap(opter)(theta0s)

            def F_MLP(h, theta):
                return model.apply(theta, h)

            best_theta = jax.tree.map(lambda a: a[jnp.argmin(histories[:, -1])], thetas)
            val_preds = multi_predict_AR(vals, lags, F_MLP, best_theta)
            val_score = evaluate(data, val_preds, inv, nh, "val", AIC).mean()
            if val_score < best_xval_score:
                best_xval_score = val_score
                best_xval = model, nh, best_theta
        except ValueError:
            continue
    model, nh, final_theta = best_xval

    def F_MLP(h, theta):
        return model.apply(theta, h)

    preds = multi_predict_AR(tests, lags, F_MLP, final_theta)
    scores = evaluate(data, preds, inv) * opts["rmse_scaling"]
    return preds, scores, {"lags": lags, **final_theta, "key": key, "nh": nh}


def RNN_trainer(model_type, data, key):
    data, inv = scale_data(data, scaler=MM, scaler_params={"feature_range": (-1, 1)})
    trains, vals, tests, opts = data
    n_batch = opts["n_batch"]
    # lags = ARX_lag_scan(data, inv, n_batch, opts["max_lag"])
    n_inits = 10
    # nhs = np.array([2, 4, 8, 16, 32])
    # nxs = np.array([2, 4, 8, 16, 32])
    nhs = np.array([2, 4, 8]) # limited for ParWH system by memmory allocation limit
    nxs = np.array([2, 4, 8])
    
    nxs = nxs[nxs < opts['max_ny']]
    opt_ions = {
        "num_iters": opts['NN_opt_iters'],
        "thresh": None,
        "optimizer": optax.adam(learning_rate=1e-2),
        "vb": False,
        "jit": True,
    }
    # wrap in xval loop
    best_xval_score = 10e10
    for nh in nhs:
        for nx in nxs:
            lags = nx, 0
            model = model_type(nh)
            opter, Htrain = multi_train_NN(trains, model, lags, n_batch, opt_ions)
            keys = jr.split(key, n_inits)
            theta0s = jax.vmap(model.init, in_axes=(0, None))(keys, Htrain)
            thetas, histories = jax.pmap(opter)(theta0s)
            best_theta = jax.tree.map(lambda a: a[jnp.argmin(histories[:, -1])], thetas)

            val_preds = multi_predict_NN(vals, lags, model, best_theta)
            val_score = evaluate(data, val_preds, inv, nh, "val", AIC).mean()
            print(nh, nx, val_score)
            if val_score < best_xval_score:
                best_xval_score = val_score
                best_xval = dc(model), dc(nh), dc(nx), dc(best_theta)

    model, nh, nx, final_theta = best_xval
    preds = multi_predict_NN(tests, (nx, 0), model, final_theta)
    scores = evaluate(data, preds, inv) * opts["rmse_scaling"]
    return preds, scores, {**final_theta, "key": key, "nh": nh, "nx": nx}


all_trainers = {
    "ARX": ARX_trainer,
    "PNARX": PNARX_trainer,
    "GPNARX": GPNARX_trainer,
    "MLPNARX": MLPNARX_trainer,
    "MLPFIR": partial(RNN_trainer, MLP),
    "RNN": partial(RNN_trainer, RNN(RNNCell)),
    "LSTM": partial(RNN_trainer, RNN(nn.LSTMCell)),
    "OLSTM": partial(RNN_trainer, RNN(nn.OptimizedLSTMCell)),
    "GRU": partial(RNN_trainer, RNN(nn.GRUCell)),
}
