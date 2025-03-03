Code to reproduce the baseline system identification results on several of the nonlinear benchmark datasets available at https://www.nonlinearbenchmark.org/

Accompanying paper explaining identificaiton approaches is available at [preprint link tbc].

All datasets are available to download from [the python dataloader](https://github.com/MaartenSchoukens/nonlinear_benchmarks).

Results can be run locally by installing requirements and then from `run_baselines.py`.

For users with HPC access a SLURM compatible submission script is available in  `run_baselines_slurm.py`

Note that some NN models take many hours to train on large datasets.

