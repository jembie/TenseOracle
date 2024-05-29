#!/bin/bash

#SBATCH --time=00:05:00                             # walltime
#SBATCH --nodes=1                                   # number of nodes
#SBATCH --ntasks=1                                  # limit to one node
#SBATCH --cpus-per-task=1                           # number of processor cores (i.e. threads)
#SBATCH -A p_ml_il
#SBATCH --job-name=commitJob

# CLI Parameters to provide:
# Template File Address
# All the Parameters the template misses
# A github address and branch to get the code from

module load release/23.04  GCCcore/11.3.0 Python/3.10.4

# Replace these variables with your own values
USERNAME="jembie" # USername von GH
ACCESS_TOKEN=<Github_token>
REPO_URL="TenseOracle"

# Needs to be set from the outside
BRANCH_NAME=${BRANCH}
JOB_TEMPLATE=${TEMPLATE}
time_quick=${TIME_QUICK}
time_slow=${TIME_SLOW}
strategy=${STRATEGY}
filter_strategy=${FILTER_STRATEGY}


CACHE_ADR="/beegfs/.global0/ws/jipo020b-AL/TestRunner/Cache/${BRANCH_NAME}_${SLURM_JOB_ID}" # there where the logs are saved
TEMP_ADR="/beegfs/.global0/ws/jipo020b-AL/TestRunner/TMP/${BRANCH_NAME}_${SLURM_JOB_ID}"  # There where the branch is downloaded to
mkdir -p ${CACHE_ADR}
mkdir -p ${TEMP_ADR}


# Clone the repository using the provided credentials
git clone -b ${BRANCH_NAME} --single-branch "https://${ACCESS_TOKEN}@github.com/${USERNAME}/${REPO_URL}.git" ${TEMP_ADR}

# Make Job-File from Template pass cache and tmp
python /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/create_job.py --job_directory "${TEMP_ADR}" --template_address ${JOB_TEMPLATE} --time_quick ${time_quick} --time_slow ${time_slow} --strategy ${strategy} --filter_strategy ${filter_strategy} --branch ${BRANCH_NAME}

# Run Job
sbatch --export=ALL,CONFIG_NAME="ag_news" "${TEMP_ADR}/job_slow.sh"
#sbatch --export=ALL,CONFIG_NAME="bank77" "${TEMP_ADR}/job_quick.sh"
sbatch --export=ALL,CONFIG_NAME="dbpedia" "${TEMP_ADR}/job_slow.sh"
sbatch --export=ALL,CONFIG_NAME="fnc_one" "${TEMP_ADR}/job_slow.sh"
sbatch --export=ALL,CONFIG_NAME="imdb" "${TEMP_ADR}/job_slow.sh"
#sbatch --export=ALL,CONFIG_NAME="mnli" "${TEMP_ADR}/job_slow.sh"
sbatch --export=ALL,CONFIG_NAME="qnli" "${TEMP_ADR}/job_slow.sh"
sbatch --export=ALL,CONFIG_NAME="rotten_tomatoes" "${TEMP_ADR}/job_quick.sh"
sbatch --export=ALL,CONFIG_NAME="sst2" "${TEMP_ADR}/job_slow.sh"
#sbatch --export=ALL,CONFIG_NAME="trec" "${TEMP_ADR}/job_quick.sh"
sbatch --export=ALL,CONFIG_NAME="wiki_talk" "${TEMP_ADR}/job_slow.sh"
#sbatch --export=ALL,CONFIG_NAME="yelp" "${TEMP_ADR}/job_slow.sh"
