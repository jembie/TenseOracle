#!/bin/bash

#SBATCH --time=[time]                             # walltime
#SBATCH --nodes=1                                   # number of nodes
#SBATCH --ntasks=1                                  # limit to one node
#SBATCH --cpus-per-task=4                           # number of processor cores (i.e. threads)
#SBATCH --gres=gpu:1
#SBATCH --mem=128GB
#SBATCH --job-name=[branch]
#SBATCH --output=[branch]%j.out
#SBATCH --array=0-24
#SBATCH --exclude=i8019,i8007,i8037,i8024,i8023 # Möglicherweise obsolete

module purge
module load release/23.04  GCC/11.3.0 Python/3.10.4 OpenMPI/4.1.4 CUDA/11.7.0 PyTorch/1.12.1-CUDA-11.7.0
source /lustre/ssd/ws/toma076c-SHK/venv/bin/activate

strategy_name=[strategy]
filter_strategy_name=[filter_strategy]
comet_api_key=
comet_workspace=active-learning-filters
random_seed=$((42 + ${SLURM_ARRAY_TASK_ID}))
task_config="./Configs/Tasks/${CONFIG_NAME}.json"

cd [job_directory]
nvidia-smi

python [job_directory]/main.py --task_config ${task_config} --filter_strategy_name ${filter_strategy_name} --strategy_name ${strategy_name} --comet_api_key ${comet_api_key} --comet_workspace ${comet_workspace} --random_seed ${random_seed}

