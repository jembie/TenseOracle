#!/bin/bash

#SBATCH --time=[time]                             # walltime
#SBATCH --nodes=1                                   # number of nodes
#SBATCH --ntasks=1                                  # limit to one node
#SBATCH --cpus-per-task=4                           # number of processor cores (i.e. threads)
#SBATCH --partition=gpu2  #,alpha
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH -A p_ml_il
#SBATCH --job-name=[branch]
#SBATCH --output=[branch]%j.out
#SBATCH --array=0-2

module purge
#module load modenv/hiera
#GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 PyTorch/1.8.1
#SBATCH --error=AL%j.err

#source /sw/installed/Python/3.10.4-GCCcore-11.3.0/bin/python
module load Python/3.10.4-GCCcore-11.3.0 modenv/hiera  GCC/11.3.0  OpenMPI/4.1.4 PyTorch/1.12.0-CUDA-11.7.0 CUDA/11.7.0
source /beegfs/.global0/ws/jipo020b-AL/venvs/LazyOracle/bin/activate

# Wait to avoid interference
sleep $(($SLURM_ARRAY_TASK_ID * 10))

strategy_name=[strategy]
filter_strategy_name=[filter_strategy]
slurm_id=${SLURM_ARRAY_TASK_ID} #TODO Set to array id
telegram_user_id=817154082
telegram_bot_token=6020310637:AAGmoymdbNe4olnpAQ8tGK9lwMFPTRS0aNU
comet_api_key=Z7zhSkfHKCsYmlZcaG8x5ssqJ
comet_project_name=[comet_project]
comet_workspace=jimmy-ml
time_limit=$(squeue -j ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} -o "%L" --noheader)
random_seed=$((42 + ${SLURM_ARRAY_TASK_ID}))
tag=[branch]
config="./Configs/standard${CONFIG_ID}.json"

cd [job_directory]

nvidia-smi

python [job_directory]/main.py --profiling --config_file ${config} --filter_strategy_name ${filter_strategy_name} --strategy_name ${strategy_name} --slurm_id ${slurm_id} --telegram_user_id ${telegram_user_id} --telegram_bot_token ${telegram_bot_token} --comet_api_key ${comet_api_key} --comet_project_name ${comet_project_name} --comet_workspace ${comet_workspace} --time_limit ${time_limit} --random_seed ${random_seed} --tag ${tag}

nvidia-smi