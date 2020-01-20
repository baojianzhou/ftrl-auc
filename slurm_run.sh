#!/bin/bash
#SBATCH --job-name=solam
#SBATCH --nodelist=ceashpc-07
#SBATCH --partition=ceashpc
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/ftrl-auc/logs/re_05_rcv1b_solam.out
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_high_dim.py run solam 05_rcv1_bin 10
