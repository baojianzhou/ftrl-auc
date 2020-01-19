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
from data_preprocess import data_process_01_webspam_whole
from data_preprocess import data_process_01_webspam_small
import matplotlib.pyplot as plt

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_ftrl_auc
        from sparse_module import c_algo_ftrl_auc_fast
        from sparse_module import c_algo_ftrl_proximal
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

root_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/'


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


def run_ftrl_proximal(para):
    data, trial_i, global_paras, para_l1, para_l2, para_beta, para_gamma = para
    print('test')
    wt, aucs, rts = c_algo_ftrl_proximal(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
    return para_gamma, para_l1, wt, aucs, rts


def run_ftrl_auc(para):
    data, trial_i, global_paras, para_l1, para_l2, para_beta, para_gamma = para
    para_gamma = 10.
    wt, aucs, rts = c_algo_ftrl_auc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
    return para_gamma, para_l1, wt, aucs, rts


def run_ftrl_auc_fast(para):
    data, trial_i, global_paras, para_l1, para_l2, para_beta, para_gamma = para
    wt, aucs, rts = c_algo_ftrl_auc_fast(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
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
    ms_res = pool.map(run_ftrl_proximal, para_space)
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
    data = pkl.load(open(root_path + '02_news20b/processed_02_news20b.pkl'))
    print(data['n'], data['num_posi'], data['num_nega'], data['p'], data['k'])
    para_space = []
    for trial_i, para_l2, para_beta, para_gamma, para_l1 in product(
            range(10), [0.0], [1.], [.3, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1],
            [6e-1, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2]):
        verbose, record_aucs = 0, 1
        global_paras = np.asarray([verbose, record_aucs], dtype=float)
        para_space.append((data, trial_i, global_paras, para_l1, para_l2, para_beta, para_gamma))

    import matplotlib.pyplot as plt
    para_gamma, para_l1, wt, aucs, rts = run_ftrl_proximal(para_space[0])
    print(aucs[-1], np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
    plt.plot(rts[:1000], aucs[:1000], label='Proximal')
    para_gamma, para_l1, wt, aucs, rts = run_ftrl_auc_fast(para_space[0])
    print(aucs[-1], np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
    plt.plot(rts[:1000], aucs[:1000], label='AUC-FAST')
    plt.legend()
    plt.show()


def test_on_03_real_sim():
    data = pkl.load(open(root_path + '03_real_sim/processed_03_real_sim.pkl'))
    print('n: %d num_posi: %d num_nega: %d p: %d k: %d' %
          (data['n'], data['num_posi'], data['num_nega'], data['p'], data['k']))
    verbose, eval_step, record_aucs = 1, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    trial_i, para_l1, para_l2, para_beta, para_gamma = 0, .5, 0.0, 1., 0.5
    for para_l1, para_gamma in product([0.1, .5, 1.], [.5, 1., 5.]):
        para = (data, 0, global_paras, para_l1, para_l2, para_beta, para_gamma)
        para_gamma, para_l1, wt, aucs, rts = run_ftrl_auc_fast(para)
        print(np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
        plt.plot(rts, aucs)
        plt.savefig('/home/baojian/%.1f_%.1f.png' % (para_l1, para_gamma))
        plt.close()


def test_on_04_webspam_u():
    data = pkl.load(open(root_path + '04_webspam_u/processed_04_webspam_u.pkl'))
    para_space = []
    for trial_i, para_l2, para_beta, para_gamma, para_l1 in product(
            range(10), [0.0], [1.], [.3, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1],
            [5e-1, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2]):
        verbose, record_aucs = 0, 1
        global_paras = np.asarray([verbose, record_aucs], dtype=float)
        para_space.append((data, trial_i, global_paras, para_l1, para_l2, para_beta, para_gamma))

    import matplotlib.pyplot as plt
    para_gamma, para_l1, wt, aucs, rts = run_ftrl_proximal(para_space[0])
    print(aucs[-1], np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
    plt.plot(rts[:1000], aucs[:1000], label='Proximal')
    para_gamma, para_l1, wt, aucs, rts = run_ftrl_auc_fast(para_space[0])
    print(aucs[-1], np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
    plt.plot(rts[:1000], aucs[:1000], label='AUC-FAST')
    plt.legend()
    plt.show()
    exit()
    para_gamma, para_l1, wt, aucs, rts = run_ftrl_auc(para_space[0])
    print(aucs[-1], rts[-1], np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
    plt.plot(rts[:1000], aucs[:1000], label='AUC')
    plt.legend()
    plt.show()
    exit()
    pool = multiprocessing.Pool(processes=1)
    ms_res = pool.map(run_ftrl_proximal, para_space)
    pool.close()
    pool.join()


def test_on_05_rcv1_bin():
    data = pkl.load(open(root_path + '05_rcv1_bin/processed_05_rcv1_bin.pkl'))
    print(data['n'], data['num_posi'], data['num_nega'], data['p'], data['k'])
    verbose, record_aucs = 0, 1
    global_paras = np.asarray([verbose, record_aucs], dtype=float)
    para_space = [(data, 0, global_paras, 0.5, 0.0, 1., 0.3)]
    para_gamma, para_l1, wt, aucs, rts = run_ftrl_auc_fast(para_space[0])
    print(np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
    return
    exit()
    para_gamma, para_l1, wt, aucs, rts = run_ftrl_proximal(para_space[0])
    print(aucs[-1], np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))


def show_figure():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/01_webspam/'
    import matplotlib.pyplot as plt
    ms_res = pkl.load(open(data_path + 're_small.pkl'))
    data = pkl.load(open(data_path + '01_webspam_10000.pkl'))
    all_indices = np.arange(data['n'])
    x_tr_indices = all_indices[:8000]
    x_te_indices = all_indices[8000:]
    results = dict()
    for para_gamma, para_l1, wt, aucs, rts in ms_res:
        auc = pred_auc(data, all_indices, x_te_indices, wt)
        print(para_gamma, para_l1, auc)
        results[(para_gamma, para_l1)] = auc
    pkl.dump(results, open(data_path + 're_small_aucs.pkl', 'wb'))


def draw_graph():
    data_path = '01_webspam/'
    # data_process_01_webspam_small()
    import matplotlib.pyplot as plt
    results = pkl.load(open(data_path + 're_small_aucs.pkl'))
    for para_gamma in [1e-1, 5e-1, 1e0, 5e0, 1e1]:
        aucs = []
        for para_l1 in [1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0, 1e1, 5e1, 1e2, 5e2]:
            aucs.append(results[(para_gamma, para_l1)])
        plt.plot(aucs, label=para_gamma)
    plt.legend()
    plt.show()
    show_figure()
    # cv_ftrl_01_webspam_small()
    pass


def main():
    test_on_04_webspam_u()


def test_on_06_pcmac():
    data = pkl.load(open(root_path + '06_pcmac/processed_06_pcmac.pkl'))
    print(data['n'], data['num_posi'], data['num_nega'], data['p'], data['k'])
    verbose, eval_step, record_aucs = 1, 1, 1
    global_paras = np.asarray([verbose, record_aucs], dtype=float)
    trial_i, para_l1, para_l2, para_beta, para_gamma = 0, .5, 0.0, 1., 0.5
    for para_l1, para_gamma in product([0.1, .5, 1.], [.5, 1., 5.]):
        para = (data, 0, global_paras, para_l1, para_l2, para_beta, para_gamma)
        para_gamma, para_l1, wt, aucs, rts = run_ftrl_auc_fast(para)
        print(np.count_nonzero(wt) / float(data['p']), np.linalg.norm(wt))
        plt.plot(rts, aucs)
        plt.savefig('/home/baojian/%.1f_%.1f.png' % (para_l1, para_gamma))
        plt.close()


if __name__ == '__main__':
    test_on_03_real_sim()
