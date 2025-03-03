from simple_slurm import Slurm

from trainers import all_trainers as base
from dataload import benchmarks_tvt as bench


# Model subsets
# AR = ['ARX', 'PNARX', 'MLPNARX', 'GPNARX']
# RNN = ['RNN', 'LSTM', 'OLSTM', 'GRU']

for baseline in base:
    for benchmark in bench:

        # if not wanting or overwrite existing files 
        # if os.path.isfile(f'results/{baseline}_{benchmark}.npz'):
            # continue

        job = Slurm(
        job_name=f'{baseline[:4]}{benchmark}',
        cpus_per_task=10,
        mem="24G", # double memory reqs for RNNs + ParWH
        output=f".output/{baseline}_{benchmark}",
        error=f".output/{baseline}_{benchmark}",
        mail_user="max.champneys@sheffield.ac.uk",
        mail_type='FAIL',
        time='4-00:00:00' # 4 days max runtime
        )
        
        job.add_cmd(f'export BASELINE="{baseline}"')
        job.add_cmd(f'export BENCHMARK="{benchmark}"')
        
        if benchmark=='CED': # See paper for rationale
            job.add_cmd(f'export LAG_OVERRIDE_NX="10"')
            job.add_cmd(f'export LAG_OVERRIDE_NY="10"')
    
        print(benchmark, baseline)
        jid = job.sbatch('./bench_env/bin/python run_baselines.py')

# watch the queue
# nice watch -n 1 "squeue -u $USER_NAME"