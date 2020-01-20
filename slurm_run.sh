#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodelist=ceashpc-09
#SBATCH --partition=ceashpc
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/ftrl-auc/logs/re_03_real_sim_fsauc.out
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_high_dim.py run fsauc 03_real_sim 10
