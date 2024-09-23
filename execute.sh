#!/bin/bash

JSON_PATH="./Configs/Tasks"
comet_workspace="outlier-detection"

# Ensure output directories exist
mkdir -p ./slurm-runs/${comet_workspace}
mkdir -p ./generated-slurm-scripts/${comet_workspace}


for json_file in "$JSON_PATH"/*.json; do
  # Get the base filename without the path and extension
  config=$(basename "${json_file}" .json)

  # These variables are meant to be expanded during script creation
  output_file="${comet_workspace}-${config}-%a-%A-%j.out"

  # Generate the SLURM script with the replaced values
  slurm_script=$(cat << EOF
#!/bin/bash
#SBATCH --nodes=1              # request 1 node
#SBATCH --cpus-per-task=6     # use 12 threads per task
#SBATCH --gres=gpu:1           # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --time=100:00:00        # run for 70 hours
#SBATCH --mem=10G
#SBATCH --account=p_ml_il
#SBATCH --job-name=${comet_workspace}-${config}
#SBATCH --output=./slurm-runs/${comet_workspace}/${output_file}
#SBATCH --exclude=i8008,i8009,i8011,i8014,i8021,i8023
#SBATCH --array=0-19

module --force purge

module load release/23.04 GCC/11.3.0 Python/3.10.4
source /data/horse/ws/toma076c-outlier-detection/venv/bin/activate

nvidia-smi
hostname

# Calculate the random seed within the SLURM script
random_seed=\$((42 + \${SLURM_ARRAY_TASK_ID}))

echo "Seed: \${random_seed}"

srun python3 main.py \
    --task_config ${json_file} \
    --experiment_config ./Configs/standard.json \
    --filter_strategy_name HDBScanFilter LocalOutlierFactorFilter IsolationForestFilter SimpleDSM SemanticAE SimpleSS \
    --comet_api_key  \
    --comet_workspace ${comet_workspace} \
    --random_seed \${random_seed}
EOF
    )

  # Save the generated SLURM script to a file
  script_file="./generated-slurm-scripts/${comet_workspace}/${config}.sh"
  touch "$script_file"
  echo "$slurm_script" > "$script_file"

  sbatch "$script_file"

  # Optionally, remove the file after submission
  # rm "$script_file"
done
