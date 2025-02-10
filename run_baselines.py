import os
import numpy as np
import time
import jax
import jax.random as jr

from src.tricks import flatten

from trainers import all_trainers as base
from dataload import benchmarks_tvt as bench

# %%

baseline = "LSTM"
benchmark = "CT"

# baseline = os.environ['BASELINE']
# benchmark = os.environ['BENCHMARK']

if __name__ == "__main__":
    t0 = time.perf_counter()
    pred, score, meta = base[baseline](bench[benchmark], jr.key(0))
    t1 = time.perf_counter()
    np.savez_compressed(
        f"results/{baseline}_{benchmark}.npz",
        pred=pred,
        score=score,
        time=t1-t0,
        meta=flatten(meta),
    )

# res = np.load(f"results/{baseline}_{benchmark}.npz", allow_pickle=1)
# print(res['meta'])