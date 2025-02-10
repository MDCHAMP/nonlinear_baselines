# %%
import nonlinear_benchmarks as nlb
import matplotlib.pyplot as plt


def plot_split(name, data):
    return
    print(name)
    trains, vals, tests, opts = data
    print(f'{len(trains)} training sets, {trains[0].y.shape}')
    print(f'{len(vals)} validation sets {vals[0].y.shape}')
    print(f'{len(tests)} testing sets {[a.y.shape for a in tests]}')
    print(f'{opts=}')
    for tr in trains:
        plt.plot(tr.y)
    for v in vals:
        plt.plot(v.y)
    for te in tests:
        plt.plot(te.y)

# %% CascadedTanks

train, test = nlb.Cascaded_Tanks()

opts = {
    'max_ny': test.state_initialization_window_length,
    'max_lag':20,
    'n_batch':1,
    'NN_opt_iters':20_000,
    'GP_opt_iters':1_000,
    'rmse_scaling':1.0
}

cut = 700
CT = [train[:cut]], [train[cut:]], [test], opts
plot_split("CT", CT)

# %% Silverbox

train_val, tests = nlb.Silverbox()
cut = len(train_val) // 2

opts = {
    'max_ny': tests[0].state_initialization_window_length,
    'max_lag':20,
    'n_batch':1,
    'NN_opt_iters':10_000,
    'GP_opt_iters':1_000,
    'rmse_scaling':1000.0
}

SB = [train_val[:cut]], [train_val[cut:]], tests, opts
plot_split("SB", SB)


# %% EMPS

train_val, test = nlb.EMPS()

opts = {
    'max_ny': test.state_initialization_window_length,
    'max_lag':20,
    'n_batch':1,
    'NN_opt_iters':10_000,
    'GP_opt_iters':1_000,
    'rmse_scaling':1000.0
}

cut = len(train_val) // 2
EMPS = [train_val[:cut]], [train_val[cut:]], [test,], opts
plot_split("EMPS", EMPS)


# %% CED

train_vals, tests = nlb.CED()
cut = len(train_vals[0]) // 2
trains = []
vals = []
for dat in train_vals:
    trains.append(dat[:cut])
    vals.append(dat[cut:])

opts = {
    'max_ny': tests[0].state_initialization_window_length,
    'max_lag':10,
    'n_batch':1,
    'NN_opt_iters':20_000,
    'GP_opt_iters':1_000,
    'rmse_scaling':1.0
}

CED = trains, vals, tests, opts
plot_split("CED", CED)

# %% WH

train_val, test = nlb.WienerHammerBenchMark()

opts = {
    'max_ny': test.state_initialization_window_length,
    'max_lag':20,
    'n_batch':50,
    'NN_opt_iters':10_000,
    'GP_opt_iters':1_000,
    'rmse_scaling':1000.0
}

cut = len(train_val) // 2
WH = [train_val[:cut]], [train_val[cut:]], (test,), opts
plot_split("WH", WH)

# %% ParWH

train_vals, tests = nlb.ParWH()
cut = len(train_vals[0]) // 2

opts = {
    'max_ny': 50, # not given in nlb atm
    'max_lag':20,
    'n_batch':800,
    'NN_opt_iters':10_000,
    'GP_opt_iters':1_000,
    'rmse_scaling':1000.0
}

trains = []
vals = []
for dat in train_vals:
    trains.append(dat[:cut])
    vals.append(dat[cut:])
max_ny = tests[0].state_initialization_window_length
ParWH = trains, vals, tests, opts
plot_split("ParWH", ParWH)

# %% F16

train_vals, tests = nlb.F16()
cut = len(train_vals[0]) // 2
trains = []
vals = []
for dat in train_vals:
    trains.append(dat[:cut])
    vals.append(dat[cut:])

opts = {
    'max_ny': 50, # not given in nlb atm
    'max_lag':20,
    'n_batch':64,
    'NN_opt_iters':10_000,
    'GP_opt_iters':1_000,
    'rmse_scaling':1.0
}


max_ny = tests[0].state_initialization_window_length
F16 = trains, vals, tests, opts
plot_split("F16", F16)


# %% all data

benchmarks_tvt = {
    "SB": SB,
    "EMPS": EMPS,
    "WH": WH,
    "CT": CT,
    "CED": CED,
    "ParWH":ParWH,
    "F16":F16
}