# -*- coding: utf-8 -*-

import os
import sys
import time
import numpy as np
import pickle as pkl
import multiprocessing
from os.path import join
from os.path import exists
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_spam_sparse
        from sparse_module import c_algo_solam_sparse
        from sparse_module import c_algo_sht_am_sparse
        from sparse_module import c_algo_fsauc_sparse
        from sparse_module import c_algo_opauc_sparse
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/'


def cv_ftrl_proximal():
    pass


def main():
    pass


if __name__ == '__main__':
    main()
