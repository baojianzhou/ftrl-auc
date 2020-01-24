#!/bin/bash
#SBATCH --job-name=ftrl-fast
#SBATCH --nodelist=ceashpc-07
#SBATCH --partition=ceashpc
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --array=00-00
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/ftrl-auc/logs/re_04_avazu_ftrl_fast_%A_%02a.out
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_high_dim.py run_huge ftrl_fast 04_avazu $SLURM_ARRAY_TASK_ID
