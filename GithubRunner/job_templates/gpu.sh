#!/bin/bash

#SBATCH --time=[time]                             # walltime
#SBATCH --nodes=1                                   # number of nodes
#SBATCH --ntasks=1                                  # limit to one node
#SBATCH --cpus-per-task=4                           # number of processor cores (i.e. threads)
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --mem=160GB
#SBATCH -A p_ml_il
#SBATCH --job-name=[branch]
#SBATCH --output=[branch]%j.out


module purge
source /beegfs/.global0/ws/jipo020b-exper/kernel/Seed3.9.5/bin/activate
module load Python/3.9.5 modenv/hiera GCC/10.3.0 OpenMPI/4.1.1 CUDA/11.3.1 TensorFlow/2.6.0-CUDA-11.3.1

python --version


#strategy_name=[strategy]
slurm_id=${MY_SLURM_ID} #TODO Set to array id
#telegram_user_id=817154082
#telegram_bot_token=6020310637:AAGmoymdbNe4olnpAQ8tGK9lwMFPTRS0aNU
comet_api_key=Z7zhSkfHKCsYmlZcaG8x5ssqJ
comet_project_name=[comet_project]
comet_workspace=jimmy-ml
#time_limit=$(squeue -j ${SLURM_JOB_ID} -o "%L" --noheader)
random_seed=$((42 + ${MY_SLURM_ID}))
tag=[branch]
base_path="/beegfs/.global0/ws/jipo020b-exper"

cd [job_directory]

python [job_directory]/test.py --base_path ${base_path} --slurm_id ${slurm_id} --comet_api_key ${comet_api_key} --comet_project_name ${comet_project_name} --comet_workspace ${comet_workspace} --random_seed ${random_seed} --tag ${tag}

