#!/bin/bash
#SBATCH --nodes=1              # request 2 nodes
#SBATCH --cpus-per-task=6      # use 6 threads per task
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --time=70:00:00        # run for 1 hour
#SBATCH --mem=10G
#SBATCH --job-name=Standard_50h
#SBATCH --array=0-9
#SBATCH --exclude=i8008


module --force purge

module load release/23.04 GCCcore/11.3.0 Python/3.10.4
source /data/horse/ws/toma076c-outlier-detection/venv/bin/activate

nvidia-smi
hostname

srun python3 main.py \
    --task_config ./Configs/Tasks/rotten_tomatoes.json \
    --experiment_config ./Configs/debug.json \
    --filter_strategy_name LocalOutlierFactorFilter HDBScanFilter IsolationForestFilter SimpleSS SimpleDSM SemanticAE \
    --comet_api_key  \
    --comet_workspace active-learning-filters
