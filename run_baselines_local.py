import subprocess
import os
from trainers import all_trainers as base
from dataload import benchmarks_tvt as bench

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Model subsets
AR = ["ARX", "PNARX", "MLPNARX", "GPNARX"]
# RNN = ['RNN', 'LSTM', 'OLSTM', 'GRU']

for baseline in AR:
    for benchmark in bench:
        os.environ["BASELINE"] = baseline
        os.environ["BENCHMARK"] = benchmark



        process = subprocess.Popen(
            ["nohup", "python", "run_baselines.py"],
            stdout=open(f"./log/{benchmark}_{baseline}", 'w+'),
            stderr=open(f"./log/{benchmark}_{baseline}", 'w+'),
            stdin=subprocess.DEVNULL,
            preexec_fn=os.setpgrp,  # ensures it won't die if SSH closes
        )

        print(benchmark, baseline)

# watch the queue
# nice watch -n 1 "squeue -u $USER_NAME"
