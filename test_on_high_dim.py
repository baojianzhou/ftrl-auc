# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
from os.path import join
from itertools import product

import numpy as np
from data_preprocess import data_process_01_webspam
from data_preprocess import data_process_02_news20b
from data_preprocess import data_process_03_realsim
from data_preprocess import data_process_04_avazu
from data_preprocess import data_process_05_rcv1_bin
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
        from sparse_module import c_algo_ftrl_auc_hybrid
        from sparse_module import c_algo_ftrl_proximal
        from sparse_module import c_algo_rda_l1
        from sparse_module import c_algo_adagrad
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

root_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/'


def cv_ftrl_auc(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_gamma, para_l1 in product(
            [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0],
            [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]):
        para_l2, para_beta = 0.0, 1.0
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
        cv_res[(trial_i, para_gamma, para_l1)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_gamma, para_l1, para_l2, para_beta)
    para_l2, para_beta, verbose, eval_step, record_aucs = 0.0, 1.0, 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_gamma, para_l1, para_l2, para_beta = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
    print(para_gamma, para_l1, metrics[1])
    sys.stdout.flush()
    return trial_i, (para_gamma, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_ftrl_auc_hybrid(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_gamma, para_l1 in product(
            [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0],
            [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]):
        para_l2, para_beta, para_k = 0.0, 1.0, 500
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc_hybrid(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma, para_k)
        cv_res[(trial_i, para_gamma, para_l1)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_gamma, para_l1, para_l2, para_beta)
    para_l2, para_beta, verbose, eval_step, record_aucs, para_k = 0.0, 1.0, 0, 100, 1, 500
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_gamma, para_l1, para_l2, para_beta = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc_hybrid(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma, para_k)
    print(para_gamma, para_l1, metrics[1])
    sys.stdout.flush()
    return trial_i, (para_gamma, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_fsauc(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_r, para_g in product(10. ** np.arange(-1, 6, 1, dtype=float),
                                  2. ** np.arange(-10, 11, 1, dtype=float)):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_fsauc(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_r, para_g)
        cv_res[(trial_i, para_r, para_g)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_r, para_g)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_r, para_g = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_fsauc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_r, para_g)
    return trial_i, (para_r, para_g), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_spauc(input_para):
    data, trial_i = input_para
    mu_arr = 10. ** np.asarray([-7.0, -6.5, -6.0, -5.5, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5])
    l1_arr = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
    best_auc, para, cv_res = None, None, dict()
    for para_mu, para_l1 in product(mu_arr, l1_arr):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_spauc(
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
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_spauc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_mu, para_l1)
    return trial_i, (para_mu, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_solam(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_r in product(np.arange(1, 101, 9, dtype=float), 10. ** np.arange(-1, 6, 1, dtype=float)):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_solam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_xi, para_r)
        cv_res[(trial_i, para_xi, para_r)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_r)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_r = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_solam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i]
        , data['p'], global_paras, para_xi, para_r)
    return trial_i, (para_xi, para_r), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_spam_l1(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_l1 in product(10. ** np.arange(-3, 4, 1, dtype=float),
                                    [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]):
        verbose, eval_step, record_aucs = 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_spam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_xi, para_l1, 0.0)
        cv_res[(trial_i, para_xi, para_l1)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_l1)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_l1 = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_spam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_xi, para_l1, 0.0)
    return trial_i, (para_xi, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_spam_l2(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_l2 in product(10. ** np.arange(-3, 4, 1, dtype=float),
                                    [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]):
        verbose, eval_step, record_aucs = 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_spam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_xi, 0.0, para_l2)
        cv_res[(trial_i, para_xi, para_l2)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_l2)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_l2 = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_spam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_xi, 0.0, para_l2)
    return trial_i, (para_xi, para_l2), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_spam_l1l2(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_xi, para_l1, para_l2 in product(
            10. ** np.arange(-3, 4, 1, dtype=float),
            [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0],
            [1e-4, 1e-3, 1e-2, 1e-1, 1e0]):
        verbose, eval_step, record_aucs = 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_spam(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_xi, para_l1, para_l2)
        cv_res[(trial_i, para_xi, para_l1, para_l2)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_xi, para_l1, para_l2)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_xi, para_l1, para_l2 = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_spam(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_xi, para_l1, para_l2)
    return trial_i, (para_xi, para_l1, para_l2), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_ftrl_proximal(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    for para_gamma, para_l1 in product([1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0],
                                       [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]):
        para_l2, para_beta, verbose, eval_step, record_aucs = 0.0, 1.0, 0, data['n'], 0
        global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_proximal(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
        cv_res[(trial_i, para_l1, para_l2, para_beta, para_gamma)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_l1, para_l2, para_beta, para_gamma)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_l1, para_l2, para_beta, para_gamma = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_proximal(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_l1, para_l2, para_beta, para_gamma)
    print(para_gamma, para_l1, metrics[1])
    sys.stdout.flush()
    return trial_i, (para_l1, para_l2, para_beta, para_gamma), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_rda_l1(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
    # gamma: to control the learning rate. (it cannot be too small)
    gamma_list = [1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
    # rho: to control the sparsity-enhancing parameter.
    rho_list = [0.0, 5e-3]
    for para_lambda, para_gamma, para_rho in product(lambda_list, gamma_list, rho_list):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_rda_l1(
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
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_rda_l1(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_lambda, para_gamma, para_rho)
    return trial_i, (para_lambda, para_gamma, para_rho), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_adagrad(input_para):
    data, trial_i = input_para
    best_auc, para, cv_res = None, None, dict()
    # lambda: to control the sparsity
    lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
    # eta: to control the learning rate. (it cannot be too small)
    eta_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3]
    epsilon_list = [1e-8]
    for para_lambda, para_eta, para_epsilon in product(lambda_list, eta_list, epsilon_list):
        global_paras = np.asarray([0, data['n'], 0], dtype=float)
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_adagrad(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], global_paras, para_lambda, para_eta, para_epsilon)
        cv_res[(trial_i, para_lambda, para_eta, para_epsilon)] = metrics
        va_auc = metrics[0]
        if best_auc is None or best_auc < va_auc:
            best_auc, para = va_auc, (para_lambda, para_eta, para_epsilon)
    verbose, eval_step, record_aucs = 0, 100, 1
    global_paras = np.asarray([verbose, eval_step, record_aucs], dtype=float)
    para_lambda, para_eta, para_epsilon = para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_adagrad(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], global_paras, para_lambda, para_eta, para_epsilon)
    return trial_i, (para_lambda, para_eta, para_epsilon), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def run_high_dimensional(method, dataset, num_cpus):
    if dataset == '01_webspam':
        data = data_process_01_webspam()
    elif dataset == '02_news20b':
        data = data_process_02_news20b()
    elif dataset == '03_real_sim':
        data = data_process_03_realsim()
    elif dataset == '05_rcv1_bin':
        data = data_process_05_rcv1_bin()
    elif dataset == '07_url':
        data = data_process_07_url()
    elif dataset == '08_farmads':
        data = data_process_08_farmads()
    else:
        f_name = root_path + '%s/processed_%s.pkl' % (dataset, dataset)
        data = pkl.load(open(f_name))
    para_space = [(data, trial_i) for trial_i in range(10)]
    pool = multiprocessing.Pool(processes=num_cpus)
    if method == 'ftrl_auc':
        ms_res = pool.map(cv_ftrl_auc, para_space)
    elif method == 'ftrl_auc_hybrid':
        ms_res = pool.map(cv_ftrl_auc_hybrid, para_space)
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
    if method == 'ftrl_auc':
        ms_res = cv_ftrl_auc((data, trial_i))
    elif method == 'ftrl_proximal':
        ms_res = cv_ftrl_proximal((data, trial_i))
    else:
        ms_res = None
    f_name = root_path + '%s/re_%s_%s_%d.pkl' % (dataset, dataset, method, task_id)
    pkl.dump(ms_res, open(f_name, 'wb'))


def result_statistics(dataset):
    aucs = []
    list_methods = ['ftrl_auc', 'spam_l1', 'spam_l2', 'spam_l1l2', 'solam', 'spauc', 'fsauc']
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


def show_parameter_select(dataset):
    import matplotlib.pyplot as plt
    from pylab import rcParams
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = '\usepackage{libertine}'
    plt.rcParams["font.size"] = 14
    rcParams['figure.figsize'] = 8, 4
    list_methods = ['ftrl_auc', 'spam_l1', 'spam_l1l2', 'spauc']
    label_list = [r'FTRL-AUC', r'\textsc{SPAM}-$\displaystyle \ell^1$',
                  r'SPAM-$\displaystyle \ell^1/\ell^2$', r'SPAUC']
    marker_list = ['s', 'D', 'o', '>', '>', '<', 'v', '^']
    color_list = ['r', 'b', 'g', 'm', 'y', 'c', 'm', 'black']
    fig, ax = plt.subplots(1, 2)
    ax[0].grid(which='y', color='lightgray', linewidth=0.3, linestyle='dashed', axis='both')
    ax[1].grid(which='x', color='lightgray', linewidth=0.3, linestyle='dashed', axis='both')
    num_trials = 10
    for ind, method in enumerate(list_methods):
        print(method)
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)))
        if method == 'ftrl_auc':
            para_l1_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(para_l1_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_l1_list)))
            for result in results:
                trial_i, (para_gamma, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
                for ind_l1, para_l1 in enumerate(para_l1_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_gamma, para_l1)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_gamma, para_l1)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            ax[0].plot(para_l1_list, xx, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
            ax[1].plot(xx, yy, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        elif method == 'spam_l1':
            para_l1_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(para_l1_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_l1_list)))
            for result in results:
                trial_i, (para_xi, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
                for ind_l1, para_l1 in enumerate(para_l1_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_l1)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_l1)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            ax[0].plot(para_l1_list, xx, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
            ax[1].plot(xx, yy, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        elif method == 'spam_l2':
            para_l2_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(para_l2_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_l2_list)))
            for result in results:
                trial_i, (para_xi, para_l2), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
                for ind_l1, para_l2 in enumerate(para_l2_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_l2)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_l2)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            plt.plot(xx, yy, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                     markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        elif method == 'spam_l1l2':
            para_l1_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(para_l1_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_l1_list)))
            for result in results:
                trial_i, (para_xi, para_l1, para_l2), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
                for ind_l1, para_l1 in enumerate(para_l1_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_l1, para_l2)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_l1, para_l2)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            ax[0].plot(para_l1_list, xx, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
            ax[1].plot(xx, yy, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        elif method == 'fsauc':
            para_r_list = 10. ** np.arange(-1, 6, 1, dtype=float)
            auc_matrix = np.zeros(shape=(num_trials, len(para_r_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_r_list)))
            for result in results:
                trial_i, (para_r, para_g), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
                for ind_l1, para_r in enumerate(para_r_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_r, para_g)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_r, para_g)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            plt.plot(xx, yy, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                     markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        elif method == 'solam':
            para_r_list = 10. ** np.arange(-1, 6, 1, dtype=float)
            auc_matrix = np.zeros(shape=(num_trials, len(para_r_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_r_list)))
            for result in results:
                trial_i, (para_xi, para_r), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
                for ind_l1, para_r in enumerate(para_r_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_r)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_xi, para_r)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            plt.plot(xx, yy, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                     markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        elif method == 'spauc':
            para_l1_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(para_l1_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_l1_list)))
            for result in results:
                trial_i, (para_mu, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
                for ind_l1, para_l1 in enumerate(para_l1_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_mu, para_l1)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_mu, para_l1)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            ax[0].plot(para_l1_list, xx, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
            ax[1].plot(xx, yy, marker=marker_list[ind], markersize=4.0, markerfacecolor='w',
                       markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        elif method == 'rda_l1':
            lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(lambda_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(lambda_list)))
            for result in results:
                trial_i, (para_lambda, para_gamma, para_rho), cv_res, wt, aucs, rts, metrics = result
                for ind_l1, para_lambda in enumerate(lambda_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_lambda, para_gamma, para_rho)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_lambda, para_gamma, para_rho)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            plt.plot(xx, yy, marker='D', label='RDA-L1')
        elif method == 'ftrl_proximal':
            para_l1_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(para_l1_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(para_l1_list)))
            for result in results:
                trial_i, (para_l1, para_l2, para_beta, para_gamma), cv_res, wt, aucs, rts, metrics = result
                for ind_l1, para_l1 in enumerate(para_l1_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_l1, para_l2, para_beta, para_gamma)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_l1, para_l2, para_beta, para_gamma)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            plt.plot(xx, yy, marker='o', label='FTRL-Proximal')
        elif method == 'adagrad':
            lambda_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
            auc_matrix = np.zeros(shape=(num_trials, len(lambda_list)))
            sparse_ratio_mat = np.zeros(shape=(num_trials, len(lambda_list)))
            for result in results:
                trial_i, (para_lambda, para_eta, para_epsilon), cv_res, wt, aucs, rts, metrics = result
                for ind_l1, para_lambda in enumerate(lambda_list):
                    auc_matrix[trial_i][ind_l1] = cv_res[(trial_i, para_lambda, para_eta, para_epsilon)][1]
                    sparse_ratio_mat[trial_i][ind_l1] = cv_res[(trial_i, para_lambda, para_eta, para_epsilon)][3]
            xx = np.mean(auc_matrix, axis=0)
            yy = np.mean(sparse_ratio_mat, axis=0)
            plt.plot(xx, yy, marker='o', label='AdaGrad')
    ax[0].set_ylabel('AUC')
    ax[0].set_xlabel('$\displaystyle \lambda $')
    ax[1].set_ylabel('Sparse-Ratio')
    ax[1].set_xlabel('AUC')
    ax[0].set_xscale('log')
    ax[1].set_yscale('log')
    plt.subplots_adjust(wspace=0.27, hspace=0.2)
    ax[1].legend(fancybox=True, loc='upper left', framealpha=1.0, frameon=False, borderpad=0.1,
                 labelspacing=0.2, handletextpad=0.1, markerfirst=True)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/kdd20-oda-auc/figs/para-select-%s.pdf' % dataset
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def show_auc_curves(dataset):
    import matplotlib.pyplot as plt
    from pylab import rcParams
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = '\usepackage{libertine}'
    plt.rcParams["font.size"] = 14
    rcParams['figure.figsize'] = 8, 4
    list_methods = ['ftrl_auc', 'spam_l1', 'spam_l2', 'spam_l1l2', 'solam', 'spauc', 'fsauc']
    label_list = [r'FTRL-AUC', r'\textsc{SPAM}-$\displaystyle \ell^1$',
                  r'SPAM-$\displaystyle \ell^2$', r'SPAM-$\displaystyle \ell^1/\ell^2$',
                  r'SOLAM', r'SPAUC', r'FSAUC']
    marker_list = ['s', 'D', 'o', 'H', '>', '<', 'v', '^']
    color_list = ['r', 'b', 'g', 'gray', 'y', 'c', 'm', 'black']
    fig, ax = plt.subplots(1, 2, sharey=True)
    ax[0].grid(b=True, which='both', color='lightgray', linewidth=0.3, linestyle='dashed', axis='both')
    ax[1].grid(b=True, which='both', color='lightgray', linewidth=0.3, linestyle='dashed', axis='both')
    num_trials = 10
    for ind, method in enumerate(list_methods):
        print(method)
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, method)))
        aucs = np.mean(np.asarray([results[trial_i][4] for trial_i in range(num_trials)]), axis=0)
        rts = np.mean(np.asarray([results[trial_i][5] for trial_i in range(num_trials)]), axis=0)
        iters = np.mean(np.asarray([results[trial_i][6] for trial_i in range(num_trials)]), axis=0)
        online_aucs = np.mean(np.asarray([results[trial_i][7] for trial_i in range(num_trials)]), axis=0)
        ax[0].plot(rts, online_aucs, marker=marker_list[ind], markersize=3.0, markerfacecolor='w',
                   markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
        ax[1].plot(iters, online_aucs, marker=marker_list[ind], markersize=3.0, markerfacecolor='w',
                   markeredgewidth=.7, linewidth=0.5, label=label_list[ind], color=color_list[ind])
    ax[0].set_ylabel('AUC')
    ax[0].set_xlabel('Run Time')
    ax[1].set_xlabel('Samples Seen')
    for i in range(2):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    plt.subplots_adjust(wspace=0.05, hspace=0.2)
    ax[1].legend(fancybox=True, loc='lower right', framealpha=1.0,
                 bbox_to_anchor=(1.0, 0.0), frameon=False, borderpad=0.1,
                 labelspacing=0.2, handletextpad=0.1, markerfirst=True)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/kdd20-oda-auc/figs/curves-oline-%s.pdf' % dataset
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
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
    elif sys.argv[1] == 'show_para_select':
        show_parameter_select(dataset=sys.argv[2])
