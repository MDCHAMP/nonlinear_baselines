import pytest

import numpy as np
from copy import deepcopy as dc
from sklearn.preprocessing import StandardScaler as SS
from sklearn.preprocessing import MinMaxScaler as MM

from dataload import benchmarks_tvt
from src.linalg import stable_least_squares as SLS
from models import scale_data, batch_Hank, batch_SLS

@pytest.mark.parametrize('dataset', benchmarks_tvt)

def test_scaler(dataset):

    data = benchmarks_tvt[dataset]

    data2, inv = scale_data(data, MM, {'feature_range':(-1,1)})
    for a, b in zip(data[2], data2[2]):
        assert np.allclose(a.y, inv(b.y))

    data3, inv = scale_data(data, SS)
    for a, b in zip(data[2], data3[2]):
        assert np.allclose(a.y, inv(b.y))


@pytest.mark.parametrize('n', [1, 10])
@pytest.mark.parametrize('dataset', benchmarks_tvt)
# @pytest.mark.parametrize('nx', [0, 5])
# @pytest.mark.parametrize('ny', [0, 5])
def test_batcher(n, dataset):
        data = dc(benchmarks_tvt[dataset])
        data, inv = scale_data(data)
        trains, vals, tests, _ = data
        Htrain_batched, ytrain_batched, slc = batch_Hank(trains, nx=2, ny=3, n_batch=n)

        assert Htrain_batched.shape[0] == max(len(trains),n)
        assert ytrain_batched.shape[0] == max(len(trains),n)

def test_batch_SLS():
    x = np.linspace(-10,10, 100*100)[:, None]
    X = x ** np.arange(4)
    a = np.array([1, -5, 20, 10])
    Y = X@a
    a = SLS(X, Y).shape
    b = batch_SLS(X.reshape(100, 100, -1), Y.reshape(100, 100, -1)).shape
    assert np.allclose(a, b)