#!/bin/bash
#SBATCH --job-name=solam
#SBATCH --nodelist=ceashpc-04
#SBATCH --partition=ceashpc
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=10
#SBATCH --output=/network/rit/lab/ceashpc/bz383376/git/ftrl-auc/logs/re_08_farmads_solam.out
# /network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python test_on_high_dim.py run solam 08_farmads 10
/network/rit/lab/ceashpc/bz383376/opt/env-python2.7.14/bin/python data_preprocess.py
