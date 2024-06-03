#!/bin/bash
#SBATCH --nodes=1              # request 2 nodes
#SBATCH --cpus-per-task=2      # use 6 threads per task
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --time=50:00:00        # run for 1 hour
#SBATCH --mem=128G
#SBATCH --job-name=Standard_50h

module --force purge

module load release/23.04 GCCcore/11.3.0 Python/3.10.4
source /home/toma076c/shk_outliers/bin/activate

nvidia-smi

srun python3 main.py \
    --task_config ./Configs/Tasks/dbpedia.json \
    --experiment_config ./Configs/debug.json \
    --filter_strategy_name LocalOutlierFactorFilter \
    --comet_api_key  \
    --comet_workspace active-learning-filters
