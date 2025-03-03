import os
import numpy as np
import time
import jax.random as jr

from src.tricks import flatten

from trainers import all_trainers as base
from dataload import benchmarks_tvt as bench

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# %%
# Read form env variable or pass a string as 
baseline = os.environ['BASELINE']
benchmark = os.environ['BENCHMARK']

if __name__ == "__main__":
    t0 = time.perf_counter()
    pred, score, meta = base[baseline](bench[benchmark], jr.key(0)) # all paper results are produced with key=jr.key(0) PRNG Key initial state.
    t1 = time.perf_counter()
    np.savez_compressed(
        f"results/{baseline}_{benchmark}.npz",
        **{f'test_{i}':p for i, p in enumerate(pred)},
        score=score,
        time=t1-t0,
        meta=flatten(meta),
    )
    print(f'{baseline} {benchmark} complete in {t1-t0:4g}s. {score=}')
    print(flatten(meta))
# load results as
# res = np.load(f"results/{baseline}_{benchmark}.npz", allow_pickle=1)
# print(res['meta'])
