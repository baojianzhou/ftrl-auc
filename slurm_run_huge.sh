#!/bin/bash
#SBATCH --job-name=ftrl-fast
#SBATCH --nodelist=ceashpc-10
#SBATCH --partition=ceashpc
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --array=00-09
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/ftrl-auc/logs/re_07_url_ftrl_fast_%A_%02a.out
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_high_dim.py run_huge ftrl_fast 07_url $SLURM_ARRAY_TASK_ID
