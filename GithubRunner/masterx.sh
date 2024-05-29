#!/bin/bash

#SBATCH --time=00:05:00                             # walltime
#SBATCH --nodes=1                                   # number of nodes
#SBATCH --ntasks=1                                  # limit to one node
#SBATCH --cpus-per-task=1                           # number of processor cores (i.e. threads)
#SBATCH -A p_ml_il
#SBATCH --job-name=commitJob


sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=SingleStepEntropy_SimplePseudo,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=SingleStepEntropy,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=AutoFilter_Chen_Like,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=AutoFilter_LSTM,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=AutoFilter_LSTM_SIMPLE,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=LoserFilter_SSL_Variety,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=LoserFilter_Plain,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=LoserFilter_Optimized_Pseudo_Labels,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=TeachingFilter,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=TeachingFilter_Smooth,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh

sbatch --export=ALL,TEMPLATE=/beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/job_templates/standard.sh,TIME_SLOW=50:00:00,TIME_QUICK=30:00:00,STRATEGY=PredictionEntropy,FILTER_STRATEGY=TeachingFilter_WOW,BRANCH=master /beegfs/.global0/ws/jipo020b-AL/GithubRunner_TO/start_jobx.sh