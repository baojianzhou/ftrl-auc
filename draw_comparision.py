# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
from os.path import join
from itertools import product

import numpy as np
from data_preprocess import data_process_02_news20b
from data_preprocess import data_process_03_realsim
from data_preprocess import data_process_04_avazu
from data_preprocess import data_process_05_rcv1_bin
from data_preprocess import data_process_07_url
from data_preprocess import data_process_06_pcmac
from data_preprocess import data_process_08_farmads
from data_preprocess import data_process_10_imdb
from data_preprocess import data_process_11_reviews

try:
    sys.path.append(os.getcwd())
    import sparse_module

    try:
        from sparse_module import c_algo_ftrl_auc
        from sparse_module import c_algo_ftrl_auc_non_lazy
    except ImportError:
        print('cannot find some function(s) in sparse_module')
        exit(0)
except ImportError:
    print('cannot find the module: sparse_module')

root_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/'


def get_from_spam_l2(dataset, num_trials):
    if os.path.exists(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, 'spam_l2')):
        results = pkl.load(open(root_path + '%s/re_%s_%s.pkl' % (dataset, dataset, 'spam_l2')))
    else:
        results = []
    para_xi_list = np.zeros(num_trials)
    para_l2_list = np.zeros(num_trials)
    for result in results:
        trial_i, (para_xi, para_l2), cv_res, wt, aucs, rts, iters, online_aucs, metrics = result
        para_xi_list[trial_i] = para_xi
        para_l2_list[trial_i] = para_l2
    return para_xi_list, para_l2_list


def cv_ftrl_auc(input_para):
    data, gamma_list, para_l1_list, trial_i = input_para
    best_auc, best_para, cv_res = None, None, dict()
    para_l2, para_beta = 0.0, 1.
    for para_gamma, para_l1 in product(gamma_list, para_l1_list):
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], np.asarray([0, data['n'], 0], dtype=float), para_l1, para_l2, para_beta, para_gamma)
        cv_res[(trial_i, para_gamma, para_l1)] = metrics
        if best_auc is None or best_auc < metrics[0]:  # va_auc
            best_auc, best_para = metrics[0], (para_gamma, para_l1, para_l2, para_beta)
    para_gamma, para_l1, para_l2, para_beta = best_para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], np.asarray([0, 100, 1], dtype=float), para_l1, para_l2, para_beta, para_gamma)
    print(para_gamma, para_l1, metrics[1])
    sys.stdout.flush()
    return trial_i, (para_gamma, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def cv_ftrl_auc_non_lazy(input_para):
    data, gamma_list, para_l1_list, trial_i = input_para
    best_auc, best_para, cv_res = None, None, dict()
    para_l2, para_beta = 0.0, 1.
    for para_gamma, para_l1 in product(gamma_list, para_l1_list):
        wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc_non_lazy(
            data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
            data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
            data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
            data['p'], np.asarray([0, data['n'], 0], dtype=float), para_l1, para_l2, para_beta, para_gamma)
        cv_res[(trial_i, para_gamma, para_l1)] = metrics
        if best_auc is None or best_auc < metrics[0]:  # va_auc
            best_auc, best_para = metrics[0], (para_gamma, para_l1, para_l2, para_beta)
    para_gamma, para_l1, para_l2, para_beta = best_para
    wt, aucs, rts, iters, online_aucs, metrics = c_algo_ftrl_auc_non_lazy(
        data['x_tr_vals'], data['x_tr_inds'], data['x_tr_poss'], data['x_tr_lens'], data['y_tr'],
        data['trial_%d_all_indices' % trial_i], data['trial_%d_tr_indices' % trial_i],
        data['trial_%d_va_indices' % trial_i], data['trial_%d_te_indices' % trial_i],
        data['p'], np.asarray([0, 100, 1], dtype=float), para_l1, para_l2, para_beta, para_gamma)
    print(para_gamma, para_l1, metrics[1])
    sys.stdout.flush()
    return trial_i, (para_gamma, para_l1), cv_res, wt, aucs, rts, iters, online_aucs, metrics


def run_high_dimensional(method, dataset, num_cpus):
    num_trials = 10
    if dataset == '02_news20b':
        data = data_process_02_news20b()
    elif dataset == '03_real_sim':
        data = data_process_03_realsim()
    elif dataset == '04_avazu':
        data = data_process_04_avazu()
    elif dataset == '05_rcv1_bin':
        data = data_process_05_rcv1_bin()
    elif dataset == '06_pcmac':
        data = data_process_06_pcmac()
    elif dataset == '07_url':
        data = data_process_07_url()
    elif dataset == '08_farmads':
        data = data_process_08_farmads()
    elif dataset == '10_imdb':
        data = data_process_10_imdb()
    elif dataset == '11_reviews':
        data = data_process_11_reviews()
    else:
        data = pkl.load(open(root_path + '%s/processed_%s.pkl' % (dataset, dataset)))
    pool = multiprocessing.Pool(processes=num_cpus)
    if method == 'ftrl_auc':
        para_gamma_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]
        para_l1_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                        1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
        para_space = [(data, para_gamma_list, para_l1_list, trial_i) for trial_i in range(num_trials)]
        ms_res = pool.map(cv_ftrl_auc, para_space)
        print(np.mean(np.asarray([_[-1][1] for _ in ms_res])))
    elif method == 'ftrl_auc_non_lazy':
        para_gamma_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e0, 5e0]
        para_l1_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 5e-2,
                        1e-1, 3e-1, 5e-1, 7e-1, 1e0, 3e0, 5e0]
        para_space = [(data, para_gamma_list, para_l1_list, trial_i) for trial_i in range(num_trials)]
        ms_res = pool.map(cv_ftrl_auc_non_lazy, para_space)
    else:
        ms_res = None
    pool.close()
    pool.join()
    pkl.dump(ms_res, open(root_path + '%s/re_%s_%s_show.pkl' % (dataset, dataset, method), 'wb'))


def show_case_1():
    import matplotlib.pyplot as plt
    # get_single_test(dataset='03_real_sim')
    # get_single_test(dataset='10_imdb')
    # get_single_test(dataset='08_farm-ads')
    from pylab import rcParams
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = '\usepackage{libertine}'
    plt.rcParams["font.size"] = 16
    rcParams['figure.figsize'] = 10, 3
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    iters, mean_lazy, std_lazy, mean_non_lazy, std_non_lazy, aver_time1, aver_time2 = \
        pkl.load(open(root_path + 'test_lazy_10_imdb.pkl'))
    ax[0].plot(iters[1:], mean_lazy[1:], c='r', marker='s', markersize=2.5, label='With Lazy Rule', alpha=0.8)
    ax[0].fill_between(iters[1:], mean_lazy[1:] - std_lazy[1:], mean_lazy[1:] + std_lazy[1:],
                       color='r', alpha=0.3)
    ax[0].plot(iters[1:], mean_non_lazy[1:], c='g', marker='D', markersize=2.5, label=r"Without Lazy Rule", alpha=0.8)
    ax[0].fill_between(iters[1:], mean_non_lazy[1:] - std_non_lazy[1:],
                       mean_non_lazy[1:] + std_non_lazy[1:], color='g', alpha=0.3)
    # ax[0].set_xscale('log')
    print(np.mean(aver_time1), np.std(aver_time1), np.mean(aver_time2), np.std(aver_time2))
    iters, mean_lazy, std_lazy, mean_non_lazy, std_non_lazy, aver_time1, aver_time2 = \
        pkl.load(open(root_path + 'test_lazy_03_real_sim.pkl'))
    print(np.mean(aver_time1), np.std(aver_time1), np.mean(aver_time2), np.std(aver_time2))
    ax[1].plot(iters[1:], mean_lazy[1:], c='r', marker='s', markersize=2.5, label='With lazy update rule', alpha=0.8)
    ax[1].fill_between(iters[1:], mean_lazy[1:] - std_lazy[1:], mean_lazy[1:] + std_lazy[1:],
                       color='r', alpha=0.3)
    ax[1].plot(iters[1:], mean_non_lazy[1:], c='g', marker='D', markersize=2.5, label='Without lazy update rule',
               alpha=0.8)
    ax[1].fill_between(iters[1:], mean_non_lazy[1:] - std_non_lazy[1:],
                       mean_non_lazy[1:] + std_non_lazy[1:], color='g', alpha=0.3)
    iters, mean_lazy, std_lazy, mean_non_lazy, std_non_lazy, aver_time1, aver_time2 = \
        pkl.load(open(root_path + 'test_lazy_08_farmads.pkl'))
    ax[2].plot(iters[1:], mean_lazy[1:], c='r', marker='s', markersize=2.5, label='With lazy update rule', alpha=0.8)
    ax[2].fill_between(iters[1:], mean_lazy[1:] - std_lazy[1:], mean_lazy[1:] + std_lazy[1:],
                       color='r', alpha=0.3)
    ax[2].plot(iters[1:], mean_non_lazy[1:], c='g', marker='D', markersize=2.5, label='Without lazy update rule',
               alpha=0.8)
    ax[2].fill_between(iters[1:], mean_non_lazy[1:] - std_non_lazy[1:],
                       mean_non_lazy[1:] + std_non_lazy[1:], color='g', alpha=0.3)
    # ax[1].set_xscale('log')
    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    ax[0].set_ylabel('AUC')
    ax[0].set_title('imdb')
    ax[0].set_xticks([0, 10000, 20000, 30000])
    ax[0].set_xticklabels([0, 10000, 20000, 30000])
    ax[0].set_ylim([0.65, 0.95])
    ax[0].set_yticks([0.7, 0.8, 0.9])
    ax[0].set_yticklabels([0.7, 0.8, 0.9])
    ax[0].set_xlabel('Samples Seen')

    ax[1].set_title('real-sim')
    ax[1].set_xticks([0, 15000, 30000, 45000])
    ax[1].set_xticklabels([0, 15000, 30000, 45000])
    ax[1].set_ylim([0.76, 0.99])
    ax[1].set_yticks([0.8, 0.85, 0.90, 0.95])
    ax[1].set_yticklabels([0.80, 0.85, 0.90, 0.95])
    ax[1].set_xlabel(r'Samples Seen')

    ax[2].set_title('farm-ads')
    ax[2].set_xticks([0, 800, 1600, 2400])
    ax[2].set_xticklabels([0, 800, 1600, 2400])
    ax[2].set_ylim([0.62, 0.98])
    ax[2].set_yticks([0.7, 0.8, 0.9])
    ax[2].set_yticklabels([0.7, 0.8, 0.9])
    ax[2].set_xlabel(r"Samples Seen")

    ax[1].legend(fancybox=True, loc='lower right', framealpha=0.0, frameon=None, borderpad=0.1, ncol=2,
                 bbox_to_anchor=(1.8, -0.45), labelspacing=0.2, handletextpad=0.1, markerfirst=True)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/kdd20-oda-auc/figs/' \
             'compare-figure.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def show_case_2():
    import matplotlib.pyplot as plt
    # get_single_test(dataset='02_news20b')
    # get_single_test(dataset='05_rcv1_bin')
    # get_single_test(dataset='11_reviews')
    from pylab import rcParams
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = '\usepackage{libertine}'
    # plt.rcParams['text.latex.preamble'] = '\usepackage{bm}'
    plt.rcParams["font.size"] = 16
    rcParams['figure.figsize'] = 10, 3
    fig, ax = plt.subplots(1, 3)
    for i in range(3):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
    iters, mean_lazy, std_lazy, mean_non_lazy, std_non_lazy, aver_time1, aver_time2 = \
        pkl.load(open(root_path + 'test_lazy_02_news20b.pkl'))
    ax[0].plot(iters[1:], mean_lazy[1:], c='r', marker='s', markersize=2.5, label='With lazy update rule', alpha=0.8)
    ax[0].fill_between(iters[1:], mean_lazy[1:] - std_lazy[1:], mean_lazy[1:] + std_lazy[1:],
                       color='r', alpha=0.3)
    ax[0].plot(iters[1:], mean_non_lazy[1:], c='g', marker='D', markersize=2.5, label=r"Without lazy update rule",
               alpha=0.8)
    ax[0].fill_between(iters[1:], mean_non_lazy[1:] - std_non_lazy[1:],
                       mean_non_lazy[1:] + std_non_lazy[1:], color='g', alpha=0.3)
    # ax[0].set_xscale('log')
    print(np.mean(aver_time1), np.std(aver_time1), np.mean(aver_time2), np.std(aver_time2))
    iters, mean_lazy, std_lazy, mean_non_lazy, std_non_lazy, aver_time1, aver_time2 = \
        pkl.load(open(root_path + 'test_lazy_05_rcv1_bin.pkl'))
    print(np.mean(aver_time1), np.std(aver_time1), np.mean(aver_time2), np.std(aver_time2))
    ax[1].plot(iters[1:], mean_lazy[1:], c='r', marker='s', markersize=2.5, label='With lazy update rule', alpha=0.8)
    ax[1].fill_between(iters[1:], mean_lazy[1:] - std_lazy[1:], mean_lazy[1:] + std_lazy[1:],
                       color='r', alpha=0.3)
    ax[1].plot(iters[1:], mean_non_lazy[1:], c='g', marker='D', markersize=2.5, label='Without lazy update rule',
               alpha=0.8)
    ax[1].fill_between(iters[1:], mean_non_lazy[1:] - std_non_lazy[1:],
                       mean_non_lazy[1:] + std_non_lazy[1:], color='g', alpha=0.3)
    iters, mean_lazy, std_lazy, mean_non_lazy, std_non_lazy, aver_time1, aver_time2 = \
        pkl.load(open(root_path + 'test_lazy_11_reviews.pkl'))
    ax[2].plot(iters[1:], mean_lazy[1:], c='r', marker='s', markersize=2.5, label='With lazy update rule', alpha=0.8)
    ax[2].fill_between(iters[1:], mean_lazy[1:] - std_lazy[1:], mean_lazy[1:] + std_lazy[1:],
                       color='r', alpha=0.3)
    ax[2].plot(iters[1:], mean_non_lazy[1:], c='g', marker='D', markersize=2.5, label='Without lazy update rule',
               alpha=0.8)
    ax[2].fill_between(iters[1:], mean_non_lazy[1:] - std_non_lazy[1:],
                       mean_non_lazy[1:] + std_non_lazy[1:], color='g', alpha=0.3)
    # ax[1].set_xscale('log')
    plt.subplots_adjust(wspace=0.15, hspace=0.2)
    ax[0].set_ylabel('AUC')
    ax[0].set_title('news20b')
    ax[0].set_xticks([0, 4000, 8000, 12000])
    ax[0].set_xticklabels([0, 4000, 8000, 12000])
    ax[0].set_ylim([0.75, 1.0])
    ax[0].set_yticks([0.8, 0.85, 0.9, 0.95])
    ax[0].set_yticklabels([0.8, 0.85, 0.9, 0.95])
    ax[0].set_xlabel('Samples Seen')

    ax[1].set_title('rcv1b')
    ax[1].set_xticks([1000, 10000, 100000])
    ax[1].set_xticklabels([1000, 10000, 100000])
    ax[1].set_ylim([0.70, 1.01])
    ax[1].set_yticks([0.75, .85, 0.95])
    ax[1].set_yticklabels([0.75, .85, 0.95])
    ax[1].set_xscale('log')
    ax[1].set_xlabel("Samples Seen")

    ax[2].set_title('reviews')
    ax[2].set_xticks([0, 1500, 3000, 4500])
    ax[2].set_xticklabels([0, 1500, 3000, 4500])
    ax[2].set_ylim([0.65, 0.95])
    ax[2].set_yticks([0.7, 0.8, 0.9])
    ax[2].set_yticklabels([0.7, 0.8, 0.9])
    ax[2].set_xlabel("Samples Seen")

    ax[1].legend(fancybox=True, loc='lower right', framealpha=0.0, frameon=None, borderpad=0.1, ncol=2,
                 bbox_to_anchor=(1.5, -0.45), labelspacing=0.2, handletextpad=0.1, markerfirst=True)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/kdd20-oda-auc/figs/' \
             'compare-figure-2.pdf'
    plt.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


def get_single_test(dataset):
    auc_matrix_lazy, auc_matrix_non_lazy, iters = [], [], None
    aver_time1, aver_time2 = [], []
    if dataset == '02_news20b':
        data = data_process_02_news20b()
    elif dataset == '03_real_sim':
        data = data_process_03_realsim()
    elif dataset == '04_avazu':
        data = data_process_04_avazu()
    elif dataset == '05_rcv1_bin':
        data = data_process_05_rcv1_bin()
    elif dataset == '06_pcmac':
        data = data_process_06_pcmac()
    elif dataset == '07_url':
        data = data_process_07_url()
    elif dataset == '08_farmads':
        data = data_process_08_farmads()
    elif dataset == '10_imdb':
        data = data_process_10_imdb()
    elif dataset == '11_reviews':
        data = data_process_11_reviews()
    else:
        data = pkl.load(open(root_path + '%s/processed_%s.pkl' % (dataset, dataset)))
    for trial_i in range(10):
        para_gamma_list, para_l1_list = [1.0], [0.5]
        ms_res_lazy = cv_ftrl_auc((data, para_gamma_list, para_l1_list, trial_i))
        aucs, rts, iters = ms_res_lazy[4], ms_res_lazy[5], ms_res_lazy[6]
        auc_matrix_lazy.append(aucs)
        aver_time1.append(rts[-1])
        ms_res_non_lazy = cv_ftrl_auc_non_lazy((data, para_gamma_list, para_l1_list, trial_i))
        aucs, rts, iters = ms_res_non_lazy[4], ms_res_non_lazy[5], ms_res_non_lazy[6]
        auc_matrix_non_lazy.append(aucs)
        aver_time2.append(rts[-1])
    mean_lazy, std_lazy = np.mean(auc_matrix_lazy, axis=0), np.std(auc_matrix_lazy, axis=0)
    mean_non_lazy, std_non_lazy = np.mean(auc_matrix_non_lazy, axis=0), np.std(auc_matrix_non_lazy, axis=0)
    pkl.dump([iters, mean_lazy, std_lazy, mean_non_lazy, std_non_lazy, aver_time1, aver_time2],
             open(root_path + 'test_lazy_%s.pkl' % dataset, 'wb'))


def main():
    # get_single_test(dataset='05_rcv1_bin')
    # get_single_test(dataset='11_reviews')
    show_case_1()
    show_case_2()


if __name__ == '__main__':
    main()
