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
from data_preprocess import data_process_01_webspam
from data_preprocess import data_process_04_avazu
from data_preprocess import data_process_07_url
from data_preprocess import data_process_08_farmads
from data_preprocess import data_process_09_kdd2010

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_spam
        from sparse_module import c_algo_spauc
        from sparse_module import c_algo_solam
        from sparse_module import c_algo_fsauc
        from sparse_module import c_algo_spauc
        from sparse_module import c_algo_ftrl_auc
        from sparse_module import c_algo_ftrl_proximal
        from sparse_module import c_algo_rda_l1
        from sparse_module import c_algo_adagrad
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

root_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/'


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


def cv_ftrl_01_webspam_whole():
    verbose, record_aucs = 0, 0
    data = data_process_01_webspam()
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


def cv_ftrl_auc(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_gamma, para_l1 in product([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0],
                                       [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]):
        para_l2, para_beta = 0.0, 1.0
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, metrics = c_algo_ftrl_auc(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
        cv_res[(trial_i, para_gamma, para_l1)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_gamma, para_l1, para_l2, para_beta)
    para_l2, para_beta, verbose, eval_step, record_aucs = 0.0, 1.0, 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_gamma, para_l1, para_l2, para_beta = para
    wt, aucs, rts, metrics = c_algo_ftrl_auc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
    print(para_gamma, para_l1, metrics[1])
    sys.stdout.flush()
    return trial_i, (para_gamma, para_l1), cv_res, wt, aucs, rts, metrics


def cv_fsauc(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_r, para_g in product(10. ** np.arange(-1, 6, 1, dtype=float), 2. ** np.arange(-10, 11, 1, dtype=float)):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, metrics = c_algo_fsauc(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_r, para_g)
        print(rts)
        print(aucs)
        cv_res[(trial_i, para_r, para_g)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_r, para_g)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_r, para_g = para
    wt, aucs, rts, metrics = c_algo_fsauc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_r, para_g)
    return trial_i, (para_r, para_g), cv_res, wt, aucs, rts, metrics


def cv_spauc(input_para):
    data, trial_i = input_para
    mu_arr = 10. ** np.asarray([-7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5])
    l1_arr = 10. ** np.asarray([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    best_auc, para, cv_res = None, None, dict()
    for para_mu, para_l1 in product(mu_arr, l1_arr):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, metrics = c_algo_spauc(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_mu, para_l1)
        cv_res[(trial_i, para_mu, para_l1)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_mu, para_l1)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_mu, para_l1 = para
    wt, aucs, rts, metrics = c_algo_spauc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_mu, para_l1)
    return trial_i, (para_mu, para_l1), cv_res, wt, aucs, rts, metrics


def cv_solam(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_r in product(np.arange(1, 101, 9, dtype=float), 10. ** np.arange(-1, 6, 1, dtype=float)):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, metrics = c_algo_solam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_xi, para_r)
        cv_res[(trial_i, para_xi, para_r)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_r)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_r = para
    wt, aucs, rts, metrics = c_algo_solam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_xi, para_r)
    return trial_i, (para_xi, para_r), cv_res, wt, aucs, rts, metrics


def cv_spam_l1(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_l1 in product(10. ** np.arange(-5, 4, 1, dtype=float),
                                    10. ** np.arange(-5, 4, 1, dtype=float)):
        verbose, eval_step, record_aucs = 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, metrics = c_algo_spam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_xi, para_l1, 0.0)
        cv_res[(trial_i, para_xi, para_l1)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_l1)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_l1 = para
    wt, aucs, rts, metrics = c_algo_spam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_xi, para_l1, 0.0)
    return trial_i, (para_xi, para_l1), cv_res, wt, aucs, rts, metrics


def cv_spam_l2(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_l2 in product(10. ** np.arange(-5, 4, 1, dtype=float),
                                    10. ** np.arange(-5, 4, 1, dtype=float)):
        verbose, eval_step, record_aucs = 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, metrics = c_algo_spam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_xi, 0.0, para_l2)
        cv_res[(trial_i, para_xi, para_l2)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_l2)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_l2 = para
    wt, aucs, rts, metrics = c_algo_spam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_xi, 0.0, para_l2)
    return trial_i, (para_xi, para_l2), cv_res, wt, aucs, rts, metrics


def cv_spam_l1l2(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_l1, para_l2 in product(10. ** np.arange(-5, 4, 1, dtype=float),
                                             10. ** np.arange(-5, 4, 1, dtype=float),
                                             10. ** np.arange(-5, 4, 1, dtype=float)):
        verbose, eval_step, record_aucs = 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, metrics = c_algo_spam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_xi, para_l1, para_l2)
        cv_res[(trial_i, para_xi, para_l1, para_l2)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_l1, para_l2)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_l1, para_l2 = para
    wt, aucs, rts, metrics = c_algo_spam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_xi, para_l1, para_l2)
    return trial_i, (para_xi, para_l1, para_l2), cv_res, wt, aucs, rts, metrics


def cv_ftrl_proximal(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_gamma, para_l1 in product([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0],
                                       [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]):
        para_l2, para_beta, verbose, eval_step, record_aucs = 0.0, 1.0, 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, metrics = c_algo_ftrl_proximal(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
        cv_res[(trial_i, para_l1, para_l2, para_beta, para_gamma)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_l1, para_l2, para_beta, para_gamma)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_l1, para_l2, para_beta, para_gamma = para
    wt, aucs, rts, metrics = c_algo_ftrl_proximal(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
    print(para_gamma, para_l1, metrics[1])
    sys.stdout.flush()
    return trial_i, (para_l1, para_l2, para_beta, para_gamma), cv_res, wt, aucs, rts, metrics


def cv_rda_l1(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    # lambda: to control the sparsity
    lambda_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e3, 5e3, 1e4]
    # rho: to control the sparsity-enhancing parameter.
    rho_list = [0.0, 5e-3]
    for para_lambda, para_gamma, para_rho in product(lambda_list, gamma_list, rho_list):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, metrics = c_algo_rda_l1(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_lambda, para_gamma, para_rho)
        cv_res[(trial_i, para_lambda, para_gamma, para_rho)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_lambda, para_gamma, para_rho)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_lambda, para_gamma, para_rho = para
    wt, aucs, rts, metrics = c_algo_rda_l1(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_lambda, para_gamma, para_rho)
    return trial_i, (para_lambda, para_gamma, para_rho), cv_res, wt, aucs, rts, metrics


def cv_adagrad(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_gamma, para_l1 in product([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0],
                                       [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]):
        para_l2, para_beta, verbose, eval_step, record_aucs = 0.0, 1.0, 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, metrics = c_algo_adagrad(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
        cv_res[(trial_i, para_l1, para_l2, para_beta, para_gamma)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_l1, para_l2, para_beta, para_gamma)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_l1, para_l2, para_beta, para_gamma = para
    wt, aucs, rts, metrics = c_algo_adagrad(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        1, data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
    return trial_i, para_l1, para_l2, para_beta, para_gamma, cv_res, wt, aucs, rts, metrics


def run_high_dimensional(method, dataset, num_cpus):
    if dataset == '01_webspam':
        data = data_process_01_webspam()
    elif dataset == '07_url':
        data = data_process_07_url()
    elif dataset == '08_farmads':
        data = data_process_08_farmads()
    else:
        f_name = root_path + '%s/processed_%s.pkl' % (dataset, dataset)
        data = pkl.load(open(f_name))
    para_space = [(data, trial_i) for trial_i in range(10)]
    pool = multiprocessing.Pool(processes=num_cpus)
    if method == 'ftrl_fast':
        ms_res = pool.map(cv_ftrl_auc, para_space)
    elif method == 'spauc':
        ms_res = pool.map(cv_spauc, para_space)
    elif method == 'fsauc':
        ms_res = pool.map(cv_fsauc, para_space)
    elif method == 'solam':
        ms_res = pool.map(cv_solam, para_space)
    elif method == 'spam_l1':
        ms_res = pool.map(cv_spam_l1, para_space)
    elif method == 'spam_l2':
        ms_res = pool.map(cv_spam_l2, para_space)
    elif method == 'spam_l1l2':
        ms_res = pool.map(cv_spam_l1l2, para_space)
    elif method == 'ftrl_proximal':
        ms_res = pool.map(cv_ftrl_proximal, para_space)
    elif method == 'rda_l1':
        ms_res = pool.map(cv_rda_l1, para_space)
    elif method == 'adagrad':
        ms_res = pool.map(cv_adagrad, para_space)
    else:
        ms_res = None
    pool.close()
    pool.join()
    f_name = root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)
    pkl.dump(ms_res, open(f_name, 'wb'))


def run_huge_dimensional(method, dataset, task_id):
    if dataset == '07_url':
        data = data_process_07_url()
    elif dataset == '01_webspam':
        data = data_process_01_webspam()
    elif dataset == '04_avazu':
        data = data_process_04_avazu()
    elif dataset == '09_kdd2010':
        data = data_process_09_kdd2010()
    else:
        f_name = root_path + '%s/processed_%s.pkl' % (dataset, dataset)
        data = pkl.load(open(f_name))
    trial_i = int(task_id)
    if method == 'ftrl_fast':
        ms_res = cv_ftrl_auc((data, trial_i))
    elif method == 'ftrl_proximal':
        ms_res = cv_ftrl_proximal((data, trial_i))
    else:
        ms_res = None
    f_name = root_path + '%s/re_%s_%s_%d.pkl' % \
             (dataset, dataset, method, task_id)
    pkl.dump(ms_res, open(f_name, 'wb'))


def result_statistics(dataset):
    aucs = []
    list_methods = ['ftrl_fast', 'spam_l1', 'spam_l2', 'spam_l1l2', 'solam', 'spauc', 'fsauc', 'ftrl_proximal']
    for method in list_methods:
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)))
        te_auc = []
        for item in results:
            metrics = item[-1]
            te_auc.append(metrics[1])
        a = ("%0.5f" % float(np.mean(np.asarray(te_auc)))).lstrip('0')
        b = ("%0.5f" % float(np.std(np.asarray(te_auc)))).lstrip('0')
        aucs.append('$\pm$'.join([a, b]))
    print(' & '.join(aucs))
    run_times = []
    for method in list_methods:
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)))
        run_time = []
        for item in results:
            metrics = item[-1]
            run_time.append(metrics[5])
        a = ("%0.3f" % float(np.mean(np.asarray(run_time))))
        b = ("%0.3f" % float(np.std(np.asarray(run_time))))
        run_times.append('$\pm$'.join([a, b]))
    print(' & '.join(run_times))
    sparse_ratios = []
    for method in list_methods:
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)))
        sparse_ratio = []
        for item in results:
            metrics = item[-1]
            sparse_ratio.append(metrics[3])
        a = ("%0.4f" % float(np.mean(np.asarray(sparse_ratio)))).lstrip('0')
        b = ("%0.4f" % float(np.std(np.asarray(sparse_ratio)))).lstrip('0')
        sparse_ratios.append('$\pm$'.join([a, b]))
    print(' & '.join(sparse_ratios))


def result_statistics_huge(dataset='07_url'):
    aucs, num_trials = [], 1
    list_methods = ['ftrl_fast', 'ftrl_proximal']
    for method in list_methods:
        te_auc = []
        for _ in range(num_trials):
            item = pkl.load(open(root_path + '%s/re_%s_%s_%d.pkl'
                                 % (dataset, dataset, method, _)))
            metrics = item[-1]
            te_auc.append(metrics[1])
        a = ("%0.5f" % float(np.mean(np.asarray(te_auc)))).lstrip('0')
        b = ("%0.5f" % float(np.std(np.asarray(te_auc)))).lstrip('0')
        aucs.append('$\pm$'.join([a, b]))
    print(' & '.join(aucs))
    run_times = []
    for method in list_methods:
        run_time = []
        for _ in range(num_trials):
            item = pkl.load(open(root_path + '%s/re_%s_%s_%d.pkl'
                                 % (dataset, dataset, method, _)))
            metrics = item[-1]
            run_time.append(metrics[5])
        a = ("%0.3f" % float(np.mean(np.asarray(run_time))))
        b = ("%0.3f" % float(np.std(np.asarray(run_time))))
        run_times.append('$\pm$'.join([a, b]))
    print(' & '.join(run_times))
    sparse_ratios = []
    for method in list_methods:
        sparse_ratio = []
        for _ in range(num_trials):
            item = pkl.load(open(root_path + '%s/re_%s_%s_%d.pkl'
                                 % (dataset, dataset, method, _)))
            metrics = item[-1]
            sparse_ratio.append(metrics[3])
        a = ("%0.4f" % float(np.mean(np.asarray(sparse_ratio)))).lstrip('0')
        b = ("%0.4f" % float(np.std(np.asarray(sparse_ratio)))).lstrip('0')
        sparse_ratios.append('$\pm$'.join([a, b]))
    print(' & '.join(sparse_ratios))


def result_curves():
    import matplotlib.pyplot as plt
    label_method = ['FTRL-AUC', 'SPAM-L1', 'SPAM-L2', 'SPAM-L1L2', 'FSAUC', 'SOLAM']
    fig, ax = plt.subplots(1, 2)
    for ind, method in enumerate(['ftrl_auc_fast', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'solam']):
        results = pkl.load(open(root_path + '03_real_sim/re_03_real_sim_%s.pkl' % method))
        rts_matrix, aucs_matrix = None, None
        for item in results:
            rts = item[-2]
            aucs = item[-3]
            if rts_matrix is None:
                rts_matrix = np.zeros_like(rts)
                aucs_matrix = np.zeros_like(aucs)
            rts_matrix += rts
            aucs_matrix += aucs
        rts_matrix /= float(len(results))
        aucs_matrix /= float(len(results))
        ax[0].plot(rts_matrix, aucs_matrix, label=label_method[ind])
        ax[1].plot(aucs_matrix[:100], label=label_method[ind])
    plt.legend()
    plt.show()


def result_sparsity(dataset='03_real_sim'):
    import matplotlib.pyplot as plt
    label_method = ['FTRL-AUC', 'SPAM-L1', 'SPAM-L2', 'SPAM-L1L2', 'FSAUC', 'SOLAM']
    fig, ax = plt.subplots(1, 2)
    for ind, method in enumerate(['ftrl_fast', 'spam_l1', 'spam_l2', 'spam_l1l2', 'fsauc', 'solam']):
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)))
        rts_matrix, aucs_matrix = None, None
        for item in results:
            rts = item[-2]
            aucs = item[-3]
            if rts_matrix is None:
                rts_matrix = np.zeros_like(rts)
                aucs_matrix = np.zeros_like(aucs)
            rts_matrix += rts
            aucs_matrix += aucs
        rts_matrix /= float(len(results))
        aucs_matrix /= float(len(results))
        ax[0].plot(rts_matrix, aucs_matrix, label=label_method[ind])
        ax[1].plot(aucs_matrix[:100], label=label_method[ind])
    plt.legend()
    plt.show()


def result_curves_huge(dataset='07_url'):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 16
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 10, 4

    label_method = ['FTRL-AUC', 'FTRL-Proximal']
    fig, ax = plt.subplots(1, 2)
    num_trials = 1
    for ind, method in enumerate(['ftrl_fast', 'ftrl_proximal']):
        rts_matrix, aucs_matrix = None, None
        for _ in range(num_trials):
            item = pkl.load(open(root_path + '%s/re_%s_%s_%d.pkl' %
                                 (dataset, dataset, method, _)))
            rts = item[-2]
            aucs = item[-3]
            if rts_matrix is None:
                rts_matrix = np.zeros_like(rts)
                aucs_matrix = np.zeros_like(aucs)
            rts_matrix += rts
            aucs_matrix += aucs
        rts_matrix /= float(num_trials)
        aucs_matrix /= float(num_trials)
        print(len(rts_matrix), len(aucs_matrix))
        ax[0].plot(rts_matrix[:200], aucs_matrix[:200], label=label_method[ind])
        ax[1].plot(aucs_matrix[:200], label=label_method[ind])
    ax[0].set_ylabel('AUC')
    ax[1].set_ylabel('AUC')
    ax[0].set_xlabel('Run Time(seconds)')
    ax[1].set_xlabel('Iteration * $\displaystyle 10^{4}$')
    ax[0].legend()
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/kdd20-oda-auc/figs/avazu-auc.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def show_parameter_select():
    pass


def show_auc_curves(dataset):
    import matplotlib.pyplot as plt
    from matplotlib import rc
    from pylab import rcParams
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times"
    plt.rcParams["font.size"] = 16
    rc('text', usetex=True)
    rcParams['figure.figsize'] = 4, 4
    list_methods = ['ftrl_fast', 'spam_l1', 'spam_l2', 'spam_l1l2', 'solam', 'spauc', 'fsauc', 'ftrl_proximal']
    label_list = ['FTRL-AUC', 'SPAM-L1', 'SPAM-L2', 'SPAM-L1L2', 'SOLAM', 'SPAUC', 'FSAUC', 'FTRL-Proximal']
    num_trials = 10
    for ind, method in enumerate(list_methods):
        print(method)
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)))
        aucs = np.mean(np.asarray([results[trial_i][4] for trial_i in range(num_trials)]), axis=0)
        rts = np.mean(np.asarray([results[trial_i][5] for trial_i in range(num_trials)]), axis=0)
        xx = range(0, 1100, 50)
        xx.extend(range(1100, 11100, 500))
        plt.plot(xx[:len(aucs)], aucs, '-', label=label_list[ind])
    plt.legend()
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/kdd20-oda-auc/figs/curves-%s.pdf' % dataset
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run_high_dimensional(method=sys.argv[2],
                             dataset=sys.argv[3],
                             num_cpus=int(sys.argv[4]))
    elif sys.argv[1] == 'run_huge':
        run_huge_dimensional(method=sys.argv[2],
                             dataset=sys.argv[3],
                             task_id=int(sys.argv[4]))
    elif sys.argv[1] == 'show_auc':
        result_statistics(dataset=sys.argv[2])
    elif sys.argv[1] == 'show_auc_curves':
        show_auc_curves(dataset=sys.argv[2])
    elif sys.argv[1] == 'show_sparsity':
        result_sparsity(dataset=sys.argv[2])
    elif sys.argv[1] == 'show_auc_huge':
        result_statistics_huge(dataset=sys.argv[2])
    elif sys.argv[1] == 'show_curves_huge':
        result_curves_huge(dataset=sys.argv[2])
    elif sys.argv[1] == 'show_para':
        show_parameter_select()
