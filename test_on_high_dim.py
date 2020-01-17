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


def get_data_by_ind(data, tr_ind, sub_tr_ind):
    sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens = [], [], [], []
    prev_posi = 0
    for index in tr_ind[sub_tr_ind]:
        cur_len = data['x_tr_lens'][index]
        cur_posi = data['x_tr_poss'][index]
        sub_x_vals.extend(data['x_tr_vals'][cur_posi:cur_posi + cur_len])
        sub_x_inds.extend(data['x_tr_inds'][cur_posi:cur_posi + cur_len])
        sub_x_lens.append(cur_len)
        sub_x_poss.append(prev_posi)
        prev_posi += cur_len
    sub_x_vals = np.asarray(sub_x_vals, dtype=float)
    sub_x_inds = np.asarray(sub_x_inds, dtype=np.int32)
    sub_x_poss = np.asarray(sub_x_poss, dtype=np.int32)
    sub_x_lens = np.asarray(sub_x_lens, dtype=np.int32)
    sub_y_tr = np.asarray(data['y_tr'][tr_ind[sub_tr_ind]], dtype=float)
    return sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_tr


def pred_auc(data, tr_index, sub_te_ind, wt):
    if np.isnan(wt).any() or np.isinf(wt).any():  # not a valid score function.
        return 0.0
    sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_te = get_data_by_ind(data, tr_index, sub_te_ind)
    y_pred_wt = np.zeros_like(sub_te_ind, dtype=float)
    for i in range(len(sub_te_ind)):
        cur_posi = sub_x_poss[i]
        cur_len = sub_x_lens[i]
        cur_x = sub_x_vals[cur_posi:cur_posi + cur_len]
        cur_ind = sub_x_inds[cur_posi:cur_posi + cur_len]
        y_pred_wt[i] = np.sum([cur_x[_] * wt[cur_ind[_]] for _ in range(cur_len)])
    return roc_auc_score(y_true=sub_y_te, y_score=y_pred_wt)


def data_process_01_webspam_whole():
    start_time = time.time()
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/01_webspam/'
    data = dict()
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, feature_index, features = 0, 0, dict()
    with open(data_path + 'webspam_wc_normalized_trigram.svm') as f:
        for ind, each_line in enumerate(f.readlines()):
            items = str(each_line).lstrip().rstrip().split(' ')
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) for _ in items[1:]]
            for item in cur_indices:
                if item not in features:
                    features[item] = feature_index
                    feature_index += 1
            data['y_tr'].append(float(items[0]))
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend([features[_] for _ in cur_indices])
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(items) == 1:
                print(each_line)
    print(min(features.keys()), max(features.keys()))
    print(min(features.values()), max(features.values()))
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(features.keys())
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['name'] = '01_webspam'
    data['tr_indices'] = np.arange(280000)
    data['te_indices'] = np.arange(280000, 350000)
    print('total run_time: %.2f' % (time.time() - start_time))
    sys.stdout.flush()
    return data


def data_process_01_webspam_small():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/01_webspam/'
    data = dict()
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, feature_index, features = 0, 0, dict()
    with open(data_path + 'webspam_wc_normalized_trigram_small.svm') as f:
        for ind, each_line in enumerate(f.readlines()):
            items = str(each_line).lstrip().rstrip().split(' ')
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) for _ in items[1:]]
            for item in cur_indices:
                if item not in features:
                    features[item] = feature_index
                    feature_index += 1
            data['y_tr'].append(float(items[0]))
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend([features[_] for _ in cur_indices])
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(items) == 1:
                print(each_line)
    print(min(features.keys()), max(features.keys()))
    print(min(features.values()), max(features.values()))
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(features.keys())
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['name'] = '01_webspam'
    for run_id in range(10):
        rand_perm = np.random.permutation(data['n'])
        data['rand_perm_%d' % run_id] = rand_perm
    pkl.dump(data, open(data_path + '01_webspam_10000.pkl', 'wb'))
    return data


def run_single_small(para):
    (sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_tr, x_tr_indices,
     is_sparse, p, global_paras, para_l1, para_l2, para_beta, para_gamma) = para
    wt, aucs, rts = c_algo_ftrl_proximal(sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_tr, x_tr_indices,
                                         is_sparse, p, global_paras, para_l1, para_l2, para_beta, para_gamma)
    return para_gamma, para_l1, wt, aucs, rts


def cv_ftrl_01_webspam_small():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/01_webspam/'
    verbose, record_aucs = 0, 0
    data = pkl.load(open(data_path + '01_webspam_10000.pkl'))
    all_indices = np.arange(data['n'])
    x_tr_indices = all_indices[:8000]
    x_te_indices = all_indices[8000:]
    __ = get_data_by_ind(data, all_indices, x_tr_indices)
    sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_tr = __
    para_space = []
    for para_l2, para_beta, para_gamma, para_l1 in product(
            [0.0], [1.], [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1],
            [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2]):
        global_paras = np.asarray([verbose, record_aucs], dtype=float)
        para_space.append((sub_x_vals, sub_x_inds, sub_x_poss, sub_x_lens, sub_y_tr, x_tr_indices,
                           1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma))
    pool = multiprocessing.Pool(processes=27)
    ms_res = pool.map(run_single_small, para_space)
    pool.close()
    pool.join()
    for para_gamma, para_l1, wt, aucs, rts in ms_res:
        print('para_gamma: %.4f para_l1: %.4f nonzero-ratio: %.4f predicted-auc: %.4f' %
              (para_gamma, para_l1, np.count_nonzero(wt) / float(data['p']),
               pred_auc(data, all_indices, x_te_indices, wt)))
    pkl.dump(ms_res, open(data_path + 're_small.pkl', 'wb'))


def cv_ftrl_01_webspam_whole():
    verbose, record_aucs = 0, 0
    data = data_process_01_webspam_whole()
    all_indices = np.arange(data['n'])
    x_tr_indices = all_indices[:280000]
    x_te_indices = all_indices[280000:]
    for para_l2, para_beta, para_gamma, para_l1 in product(
            [0.0], [1.], [1.0], np.asarray([0.05, 0.5, 1.0, 5., 10., 50., 100., 1000.]) / 280000.):
        global_paras = np.asarray([verbose, record_aucs], dtype=float)
        run_time = time.time()
        wt, aucs, rts = c_algo_ftrl_proximal(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'],
            data['x_tr_lens'], np.asarray(data['y_tr'], dtype=float), x_tr_indices,
            1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
        print('run_time: %.4f nonzero-ratio: %.4f predicted-auc: %.4f' %
              (time.time() - run_time, np.count_nonzero(wt) / float(data['p']),
               pred_auc(data, all_indices, x_te_indices, wt)))
        sys.stdout.flush()


def cv_ftrl_02_new20b():
    pass


def main():
    # data_process_01_webspam_small()
    cv_ftrl_01_webspam_small()


if __name__ == '__main__':
    main()
