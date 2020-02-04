#ifndef SPARSE_AUC_AUC_OPT_METHODS_H
#define SPARSE_AUC_AUC_OPT_METHODS_H

#include <time.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
// These are the third part library needed.
#include <cblas.h>
#include "loss.h"

#define PI 3.14159265358979323846
#define sign(x) (x > 0) - (x < 0)
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))
#define swap(a, b) { register double temp=(a);(a)=(b);(b)=temp; }
#define is_posi(x) ( x > 0.0 ? 1.0 : 0.0)
#define is_nega(x) ( x < 0.0 ? 1.0 : 0.0)
#define expit(x) ( x > 0.0 ? 1./(1. + exp(-x)) : 1.- 1./(1.+exp(x)))

typedef struct {
    double *wt;
    double *scores;
    double *aucs;
    double *rts;
    int auc_len; // how many auc evaluated.
    double run_time;
    double eval_time;
    double total_time;
    double te_auc;
    double va_auc;
    int nonzero_wt;
    double sparse_ratio;
} AlgoResults;


typedef struct {
    int verbose;
    int eval_step;
    int record_aucs;
} GlobalParas;

AlgoResults *make_algo_results(int data_p, int total_num_eval);

bool free_algo_results(AlgoResults *re);

typedef struct {
    const double *x_vals;
    const int *x_inds;
    const int *x_poss;
    const int *x_lens;
    const double *y;
    const int *indices; // the permutation of given data
    const int *tr_indices;
    const int *va_indices;
    const int *te_indices;
    bool is_sparse;
    int n; // total numbers
    int n_tr;
    int n_va;
    int n_te;
    int p;
} Data;

typedef struct {
    double val;
    int index;
} data_pair;

/**
 * SOLAM: Stochastic Online AUC Maximization
 * ---
 * BibTEX:
 * @inproceedings{ying2016stochastic,
 * title={Stochastic online AUC maximization},
 * author={Ying, Yiming and Wen, Longyin and Lyu, Siwei},
 * booktitle={Advances in neural information processing systems},
 * pages={451--459},
 * year={2016}
 * }
 * @param data
 * @param paras
 * @param re
 * @param para_xi
 * @param para_r
 * @author --- (Email: ---)
 * @return
 */
bool _algo_solam(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_xi,
                 double para_r);

/**
 * This function implements the algorithm proposed in the following paper.
 * Stochastic Proximal Algorithms for AUC Maximization.
 * ---
 * @inproceedings{natole2018stochastic,
 * title={Stochastic proximal algorithms for AUC maximization},
 * author={Natole, Michael and Ying, Yiming and Lyu, Siwei},
 * booktitle={International Conference on Machine Learning},
 * pages={3707--3716},
 * year={2018}}
 * ---
 * Do not use the function directly. Instead, call it by Python Wrapper.
 * @param data
 * @param paras
 * @param re
 * @param para_xi
 * @param para_l1
 * @param para_l2
 * @author --- (Email: ---)
 */
void _algo_spam(Data *data,
                GlobalParas *paras,
                AlgoResults *re,
                double para_xi,
                double para_l1,
                double para_l2);

/**
 * Stochastic Hard Thresholding for AUC maximization.
 * @param data
 * @param paras
 * @param re
 * @param para_s
 * @param para_b
 * @param para_c
 * @param para_l2_reg
 */
void _algo_spauc(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_mu,
                 double para_l1);


/**
 * This function implements the algorithm, FSAUC, proposed in the following paper:
 * ---
 * @inproceedings{liu2018fast,
 * title={Fast stochastic AUC maximization with O (1/n_tr)-convergence rate},
 * author={Liu, Mingrui and Zhang, Xiaoxuan and Chen, Zaiyi and Wang, Xiaoyu and Yang, Tianbao},
 * booktitle={International Conference on Machine Learning},
 * pages={3195--3203},
 * year={2018}}
 * ---
 * @param data
 * @param paras
 * @param re
 * @param para_r
 * @param para_g
 */
void _algo_fsauc(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_r,
                 double para_g);

void _algo_ftrl_auc(Data *data,
                    GlobalParas *paras,
                    AlgoResults *re,
                    double para_l1,
                    double para_l2,
                    double para_beta,
                    double para_gamma);


void _algo_ftrl_auc_fast(Data *data,
                         GlobalParas *paras,
                         AlgoResults *re,
                         double para_l1,
                         double para_l2,
                         double para_beta,
                         double para_gamma);

void _algo_ftrl_proximal(Data *data,
                         GlobalParas *paras,
                         AlgoResults *re,
                         double para_l1,
                         double para_l2,
                         double para_beta,
                         double para_gamma);

#endif //SPARSE_AUC_AUC_OPT_METHODS_H
