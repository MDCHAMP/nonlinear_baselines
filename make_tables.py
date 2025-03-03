import os
import numpy as np

from trainers import all_trainers as base
from dataload import benchmarks_tvt as bench


# %%



# %%

print("ORDER")
res_dir = "results"
top = "                  "
nx = "Polynomial order  "
for b in bench:
    fname = f"{res_dir}/PNARX_{b}.npz"
    if os.path.isfile(fname):
        res = np.load(fname, allow_pickle=1)
        order = res["meta"].sum()["order"]
    else:
        order = "-"
    top += f"& {b:<7}"
    nx += f"& {order:<7}"
print(top, "\\\\")
print(nx, "\\\\")

# %%

print("LAGS")
res_dir = "results"
top = "                  "
nx = "$N_x$ (AR)        "
ny = "$N_y$ (AR)        "
nx2 = "$N_x$ (MLP NARX)  "
ny2 = "$N_y$ (MLP NARX)  "
for b in bench:
    fname = f"{res_dir}/ARX_{b}.npz"
    res = np.load(fname, allow_pickle=1)
    lags = res["meta"].sum()["lags"]

    top += f"& {b:<7}"
    nx += f"& {lags[0]:<7}"
    ny += f"& {lags[1]:<7}"

for b in bench:
    fname = f"{res_dir}/MLPNARX_{b}.npz"
    res = np.load(fname, allow_pickle=1)
    lags = res["meta"].sum()["lags"]

    nx2 += f"& {lags[0]:<7}"
    ny2 += f"& {lags[1]:<7}"

print(top, "\\\\")
print(nx, "\\\\")
print(ny, "\\\\")
print(nx2, "\\\\")
print(ny2, "\\\\")

# for m in ["MLPFIR", "RNN", "LSTM", "OLSTM", "GRU"]:
#     nx3 = f"$N_x$ ({m})    "
#     nx3 = f"{nx3:<18}"
#     for b in bench:
#         fname = f"{res_dir}/{m}_{b}.npz"
#         if os.path.isfile(fname):
#             res = np.load(fname, allow_pickle=1)
#             lags = res["meta"].sum()["nx"]
#         else:
#             lags = "-"
#         nx3 += f"& {lags:<7}"

#     print(nx3, "\\\\")

print("RNN nh")
res_dir = "results"
print(top, "\\\\")
for m in ["MLPFIR", "RNN", "LSTM", "OLSTM", "GRU"]:
    Hidden = f"$n_h$ ({m})"
    Hidden = f"{Hidden:<18}"

    for b in bench:
        fname = f"{res_dir}/{m}_{b}.npz"
        if os.path.isfile(fname):
            res = np.load(fname, allow_pickle=1)
            hid = res["meta"].sum()["nh"]
        else:
            hid = "hid"
        Hidden += f"& {hid:<7}"

    # print(look, "\\\\")
    print(Hidden, "\\\\")

print("RNN nx")
res_dir = "results"
print(top, "\\\\")
for m in ["MLPFIR", "RNN", "LSTM", "OLSTM", "GRU"]:
    look = f"$n_x$ ({m})"
    look = f"{look:<18}"

    for b in bench:
        fname = f"{res_dir}/{m}_{b}.npz"
        if os.path.isfile(fname):
            res = np.load(fname, allow_pickle=1)
            lags = res["meta"].sum()["nx"]
        else:
            lags = "-"
            hid = "hid"
        look += f"& {lags:<7}"

    print(look, "\\\\")
    # print(Hidden, "\\\\")

# %% 
print("TIME (nearest minute)")
res_dir = "results"
top = f" {' ' * 10}"
for b in bench:
    top += f"& {b:<10}"
print(top + "\\\\")
for baseline in base:
    scores = ""
    for benchmark in bench:
        fname = f"{res_dir}/{baseline}_{benchmark}.npz"
        if os.path.isfile(fname):
            res = np.load(fname, allow_pickle=1)
            time = res['time'] /60
            scores += f"& {time:<10.0f}"
        else:
            res = []
            scores += f"& {'-':<10}"
    scores += "\\\\"
    print(f"{baseline:<10} {scores}")
# %%
# scores

print("")
print("RMSE")
res_dir = "results"
top = f"{' ' * 10} "
for b in bench:
    if b in ["F16", "ParWH"]:
        continue
    for i in range(len(bench[b][2])):
        if len(bench[b][2]) > 1:
            tmp = b + f" {i+1}" 
        else:
             tmp = b 
        top += f"& {tmp:<9}"
top += "\\\\"
print(top)
for baseline in base:
    scores = ""
    for benchmark in bench:
        if benchmark in ["F16", "ParWH"]:
            continue
        fname = f"{res_dir}/{baseline}_{benchmark}.npz"
        if os.path.isfile(fname):
            res = np.load(fname, allow_pickle=1)
            for s in res["score"]:
                scores += f"& {s:<9.3g}"
        else:
            res = []
            for i in range(len(bench[benchmark][2])):
                scores += f"& {'-':<9}"
    scores += "\\\\"
    print(f"{baseline:<10} {scores}")


print("")
print("RMSE")
res_dir = "results"
top = f"{' ' * 10} "
for b in bench:
    if b not in ["ParWH"]:
        continue
    for i in range(len(bench[b][2])):
        if len(bench[b][2]) > 1:
            tmp = b + f" {i+1}" 
        else:
             tmp = b 
        top += f"& {tmp:<9}"
top += "\\\\"
print(top)
for baseline in base:
    scores = ""
    for benchmark in bench:
        if benchmark not in ["ParWH"]:
            continue
        fname = f"{res_dir}/{baseline}_{benchmark}.npz"
        if os.path.isfile(fname):
            res = np.load(fname, allow_pickle=1)
            for s in res["score"]:
                scores += f"& {s:<9.3g}"
        else:
            res = []
            for i in range(len(bench[benchmark][2])):
                scores += f"& {'-':<9}"
    scores += "\\\\"
    print(f"{baseline:<10} {scores}")


print("")
print("RMSE")
res_dir = "results"
top = f"{' ' * 10} "
for b in bench:
    if b not in ["F16"]:
        continue
    for i in range(len(bench[b][2])):
        if len(bench[b][2]) > 1:
            tmp = b + f" {i+1}" 
        else:
             tmp = b 
        top += f"& {tmp:<9}"
top += "\\\\"
print(top)
for baseline in base:
    scores = ""
    for benchmark in bench:
        if benchmark not in ["F16"]:
            continue
        fname = f"{res_dir}/{baseline}_{benchmark}.npz"
        if os.path.isfile(fname):
            res = np.load(fname, allow_pickle=1)
            for s in res["score"]:
                scores += f"& {s:<9.3g}"
        else:
            res = []
            for i in range(len(bench[benchmark][2])):
                scores += f"& {'-':<9}"
    scores += "\\\\"
    print(f"{baseline:<10} {scores}")



# Failed tests

# RNN ParWH - OOM
# OLSTM ParWH - OOM
# LSTM ParWH - OOM
# GRU ParWH - OOM
# LSTM F16 - OOM
# PNARX ParWH - all model orders error out
