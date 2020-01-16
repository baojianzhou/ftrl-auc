# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
from os.path import join
from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_ftrl_proximal
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')


def cv_ftrl_proximal():
    import matplotlib.pyplot as plt
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/00_sentiment/processed_acl/books/'
    data = pkl.load(open(data_path + 'data_00_sentiment.pkl'))
    for run_id in range(10):
        para_l1 = 0.05 / float(data['n'])
        print(para_l1)
        verbose, record_aucs = 0, 1

        for para_l2, para_beta, para_gamma in product([0.0], [1.], np.arange(0.0003, 1.9, 0.14)):
            global_paras = np.asarray([verbose, record_aucs], dtype=float)
            wt, aucs, rts = c_algo_ftrl_proximal(
                data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'],
                data['x_tr_lens'], data['y_tr'], data['rand_perm_%d' % run_id],
                1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
            plt.plot(rts, aucs)
            print(np.count_nonzero(wt), aucs[-1])
        plt.show()
        plt.close()


def main():
    cv_ftrl_proximal()


if __name__ == '__main__':
    main()
