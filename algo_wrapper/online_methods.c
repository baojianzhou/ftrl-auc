#include "online_methods.h"

/**
 * The Box–Muller method uses two independent random
 * numbers U and V distributed uniformly on (0,1).
 * Then the two random variables X and Y.
 * @param n
 * @param samples
 */
void std_normal(int n, double *samples) {
    double epsilon = 2.22507e-308, x, y;
    for (int i = 0; i < n; i++) {
        do {
            x = rand() / (RAND_MAX * 1.);
            y = rand() / (RAND_MAX * 1.);
        } while (x <= epsilon);
        samples[i] = sqrt(-2.0 * log(x)) * cos(2.0 * PI * y);
    }
}

static inline int __comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

void _arg_sort_descend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
}

AlgoResults *make_algo_results(int data_p, int total_num_eval) {
    AlgoResults *re = malloc(sizeof(AlgoResults));
    re->wt = calloc((size_t) data_p, sizeof(double));
    re->te_aucs = calloc((size_t) total_num_eval, sizeof(double));
    re->online_aucs = calloc((size_t) total_num_eval, sizeof(double));
    re->pred_scores = calloc((size_t) total_num_eval, sizeof(double));
    re->true_labels = calloc((size_t) total_num_eval, sizeof(double));
    re->rts = calloc((size_t) total_num_eval, sizeof(double));
    re->iters = calloc((size_t) total_num_eval, sizeof(int));
    re->scores = calloc((size_t) total_num_eval, sizeof(double));
    re->auc_len = 0;
    return re;
}

bool free_algo_results(AlgoResults *re) {
    free(re->rts);
    free(re->pred_scores);
    free(re->true_labels);
    free(re->online_aucs);
    free(re->iters);
    free(re->te_aucs);
    free(re->wt);
    free(re->scores);
    free(re);
    return true;
}


/**
 * Calculate the AUC score.
 * We assume true labels contain only +1,-1
 * We also assume scores are real numbers.
 * @param true_labels
 * @param scores
 * @param len
 * @return AUC score.
 */
double _auc_score(const double *true_labels, const double *scores, int len) {
    double *fpr = malloc(sizeof(double) * (len + 1));
    double *tpr = malloc(sizeof(double) * (len + 1));
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < len; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * len);
    _arg_sort_descend(scores, sorted_indices, len);
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < len; i++) {
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    cblas_dscal(len, 1. / num_posi, tpr, 1);
    cblas_dscal(len, 1. / num_nega, fpr, 1);
    // AUC score
    double auc = 0.0;
    double prev = 0.0;
    for (int i = 0; i < len; i++) {
        auc += (tpr[i] * (fpr[i] - prev));
        prev = fpr[i];
    }
    free(sorted_indices);
    free(fpr);
    free(tpr);
    return auc;
}

/**
 * Please find the algorithm in the following paper:
 * ---
 * @article{floyd1975algorithm,
 * title={Algorithm 489: the algorithm SELECT—for finding the ith
 *        smallest of n_tr elements [M1]},
 * author={Floyd, Robert W and Rivest, Ronald L},
 * journal={Communications of the ACM},
 * volume={18}, number={3}, pages={173},
 * year={1975},
 * publisher={ACM}}
 * @param array
 * @param l
 * @param r
 * @param k
 */
void _floyd_rivest_select(double *array, int l, int r, int k) {
    register int n, i, j, s, sd, ll, rr;
    register double z, t;
    while (r > l) {
        if (r - l > 600) {
            /**
             * use select() recursively on a sample of size s to get an
             * estimate for the (k-l+1)-th smallest element into array[k],
             * biased slightly so that the (k-l+1)-th element is expected to
             * lie in the smaller set after partitioning.
             */
            n = r - l + 1;
            i = k - l + 1;
            z = log(n);
            s = (int) (0.5 * exp(2 * z / 3));
            sd = (int) (0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2));
            ll = max(l, k - i * s / n + sd);
            rr = min(r, k + (n - i) * s / n + sd);
            _floyd_rivest_select(array, ll, rr, k);
        }
        t = array[k];
        /**
         * the following code partitions x[l:r] about t, it is similar to partition
         * but will run faster on most machines since subscript range checking on i
         * and j has been eliminated.
         */
        i = l;
        j = r;
        swap(array[l], array[k])
        if (array[r] < t) {
            swap(array[r], array[l])
        }
        while (i < j) {
            swap(array[i], array[j])
            do i++; while (array[i] > t);
            do j--; while (array[j] < t);
        }
        if (array[l] == t) {
            swap(array[l], array[j])
        } else {
            j++;
            swap(array[j], array[r])
        }
        /**
         * New adjust l, r so they surround the subset containing the
         * (k-l+1)-th smallest element.
         */
        if (j <= k) {
            l = j + 1;
        }
        if (k <= j) {
            r = j - 1;
        }
    }
}


/**
 * Given the unsorted array, we threshold this array by using Floyd-Rivest algorithm.
 * @param arr the unsorted array.
 * @param n, the number of elements in this array.
 * @param k, the number of k largest elements will be kept.
 * @return 0, successfully project arr to a k-sparse vector.
 */
int _hard_thresholding(double *arr, int n, int k) {
    double *temp_arr = malloc(sizeof(double) * n), kth_largest;
    for (int i = 0; i < n; i++) {
        temp_arr[i] = fabs(arr[i]);
    }
    _floyd_rivest_select(temp_arr, 0, n - 1, k - 1);
    kth_largest = temp_arr[k - 1];
    for (int i = 0; i < n; i++) {
        if (fabs(arr[i]) <= kth_largest) { arr[i] = 0.0; }
    }
    free(temp_arr);
    return 0;
}

static void _l1ballproj_condat(double *y, double *x, int length, const double a) {
    // This code is implemented by Laurent Condat, PhD, CNRS research fellow in France.
    if (a <= 0.0) {
        if (a == 0.0) memset(x, 0, length * sizeof(double));
        return;
    }
    double *aux = (x == y ? (double *) malloc(length * sizeof(double)) : x);
    int aux_len = 1;
    int aux_len_hold = -1;
    double tau = (*aux = (*y >= 0.0 ? *y : -*y)) - a;
    int i = 1;
    for (; i < length; i++) {
        if (y[i] > 0.0) {
            if (y[i] > tau) {
                if ((tau += ((aux[aux_len] = y[i]) - tau) / (aux_len - aux_len_hold)) <=
                    y[i] - a) {
                    tau = y[i] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        } else if (y[i] != 0.0) {
            if (-y[i] > tau) {
                if ((tau += ((aux[aux_len] = -y[i]) - tau) / (aux_len - aux_len_hold))
                    <= aux[aux_len] - a) {
                    tau = aux[aux_len] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        }
    }
    if (tau <= 0) {    /* y is in the l1 ball => x=y */
        if (x != y) memcpy(x, y, length * sizeof(double));
        else free(aux);
    } else {
        double *aux0 = aux;
        if (aux_len_hold >= 0) {
            aux_len -= ++aux_len_hold;
            aux += aux_len_hold;
            while (--aux_len_hold >= 0)
                if (aux0[aux_len_hold] > tau)
                    tau += ((*(--aux) = aux0[aux_len_hold]) - tau) / (++aux_len);
        }
        do {
            aux_len_hold = aux_len - 1;
            for (i = aux_len = 0; i <= aux_len_hold; i++)
                if (aux[i] > tau)
                    aux[aux_len++] = aux[i];
                else
                    tau += (tau - aux[i]) / (aux_len_hold - i + aux_len);
        } while (aux_len <= aux_len_hold);
        for (i = 0; i < length; i++)
            x[i] = (y[i] - tau > 0.0 ? y[i] - tau : (y[i] + tau < 0.0 ? y[i] + tau : 0.0));
        if (x == y) free(aux0);
    }
}

void _get_posi_nega_x(double *posi_x, double *nega_x, double *posi_t, double *nega_t, double *prob_p, Data *data) {
    if (data->is_sparse) {
        for (int i = 0; i < data->n_tr; i++) {
            const int *xt_inds = data->x_inds + data->x_poss[i];
            const double *xt_vals = data->x_vals + data->x_poss[i];
            if (data->y[i] > 0) {
                (*posi_t)++;
                for (int kk = 0; kk < data->x_lens[i]; kk++)
                    posi_x[xt_inds[kk]] += xt_vals[kk];
            } else {
                (*nega_t)++;
                for (int kk = 0; kk < data->x_lens[i]; kk++)
                    nega_x[xt_inds[kk]] += xt_vals[kk];
            }
        }
    } else {
        for (int i = 0; i < data->n_tr; i++) {
            const double *xt = (data->x_vals + i * data->p);
            if (data->y[i] > 0) {
                (*posi_t)++;
                cblas_daxpy(data->p, 1., xt, 1, posi_x, 1);
            } else {
                (*nega_t)++;
                cblas_daxpy(data->p, 1., xt, 1, nega_x, 1);
            }
        }
    }
    *prob_p = (*posi_t) / (data->n_tr * 1.0);
    cblas_dscal(data->p, 1. / (*posi_t), posi_x, 1);
    cblas_dscal(data->p, 1. / (*nega_t), nega_x, 1);
}

void _evaluate_aucs(Data *data, double *y_pred, AlgoResults *re, double start_time) {
    double t_eval = clock();
    if (data->is_sparse) {
        memset(y_pred, 0, sizeof(double) * data->n_tr);
        for (int q = 0; q < data->n_tr; q++) {
            const int *xt_inds = data->x_inds + data->x_poss[q];
            const double *xt_vals = data->x_vals + data->x_poss[q];
            for (int tt = 0; tt < data->x_lens[q]; tt++)
                y_pred[q] += re->wt[xt_inds[tt]] * xt_vals[tt];
        }
    } else {
        cblas_dgemv(CblasRowMajor, CblasNoTrans, data->n_tr, data->p, 1.,
                    data->x_vals, data->p, re->wt, 1, 0.0, y_pred, 1);
    }
    re->te_aucs[re->auc_len] = _auc_score(data->y, y_pred, data->n_tr);
    re->rts[re->auc_len++] = clock() - start_time - (clock() - t_eval);
}


double _eval_auc(Data *data, AlgoResults *re, bool is_va) {
    double *true_labels;
    int num_samples;
    if (is_va) {
        num_samples = data->n_va;
    } else {
        num_samples = data->n_te;
    }
    true_labels = malloc(sizeof(double) * num_samples);
    const int *xt_inds;
    const double *xt_vals;
    double xtw;
    for (int jj = 0; jj < num_samples; jj++) {
        int cur_index;
        if (is_va) {
            cur_index = data->va_indices[jj];
        } else {
            cur_index = data->te_indices[jj];
        }
        true_labels[jj] = data->y[cur_index];
        xt_inds = data->x_inds + data->x_poss[cur_index];
        xt_vals = data->x_vals + data->x_poss[cur_index];
        xtw = 0.0;
        for (int kk = 0; kk < data->x_lens[cur_index]; kk++) {
            xtw += (re->wt[xt_inds[kk]] * xt_vals[kk]);
        }
        re->scores[jj] = xtw;
    }
    double auc_score = _auc_score(true_labels, re->scores, num_samples);
    free(true_labels);
    return auc_score;
}

void _cal_sparse_ratio(AlgoResults *re, int d) {
    re->nonzero_wt = 0;
    for (int i = 0; i < d; i++) {
        if (re->wt[i] != 0.0) {
            re->nonzero_wt += 1;
        }
    }
    re->sparse_ratio = (double) re->nonzero_wt / (double) d;
}

int _check_step_size(int tt) {
    if (tt < 1100) {
        return tt % 50;
    } else if (tt < 11100) {
        return tt % 500;
    } else if (tt < 111100) {
        return tt % 5000;
    } else if (tt < 1111100) {
        return tt % 50000;
    } else if (tt < 11111100) {
        return tt % 500000;
    } else if (tt < 111111100) {
        return tt % 5000000;
    } else {
        return tt % 50000000;
    }
}

void _algo_spauc(Data *data,
                 GlobalParas *paras,
                 AlgoResults *re,
                 double para_mu,
                 double para_l1) {

    double start_time = clock();
    openblas_set_num_threads(1);
    double *grad_wt = malloc(sizeof(double) * data->p); // gradient
    double *posi_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=1]
    double *nega_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=-1]
    double *y_pred = calloc((size_t) data->n, sizeof(double));
    double a_wt = 0.0, b_wt = 0.0;
    double posi_t = 0.0, nega_t = 0.0;
    double prob_p = 0.0;
    double eta_t;
    double total_time, run_time, eval_time = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        int ind = data->tr_indices[tt];
        eta_t = 2. / (para_mu * tt + 1.0); // current learning rate
        const int *xt_inds;
        const double *xt_vals;
        double xtw = 0.0;
        // receive zt=(xt,yt)
        xt_inds = data->x_inds + data->x_poss[ind];
        xt_vals = data->x_vals + data->x_poss[ind];
        if (data->y[ind] > 0) {
            posi_t++;
            for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                posi_x[xt_inds[ii]] += xt_vals[ii];
            }
            prob_p = (tt * prob_p + 1.) / (tt + 1.0);
            // update a(wt)
            a_wt = cblas_ddot(data->p, re->wt, 1, posi_x, 1) / posi_t;
        } else {
            nega_t++;
            for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                nega_x[xt_inds[ii]] += xt_vals[ii];
            }
            prob_p = (tt * prob_p) / (tt + 1.0);
            // update b(wt)
            b_wt = cblas_ddot(data->p, re->wt, 1, nega_x, 1) / nega_t;
        }
        // make online prediction
        re->pred_scores[tt] = xtw;
        re->true_labels[tt] = data->y[ind];
        double wei_x, wei_posi, wei_nega;
        if (data->y[ind] > 0) {
            wei_x = 2. * (1. - prob_p) * (xtw - a_wt);
            wei_posi = 2. * (1. - prob_p) * (a_wt - xtw - prob_p * (1. + b_wt - a_wt));
            wei_nega = 2. * prob_p * (1. - prob_p) * (1. + b_wt - a_wt);
        } else {
            wei_x = 2. * prob_p * (xtw - b_wt);
            wei_posi = 2. * prob_p * (1. - prob_p) * (-1. - b_wt + a_wt);
            wei_nega = 2. * prob_p * (b_wt - xtw - (1.0 - prob_p) * (-1. - b_wt + a_wt));
        }
        memset(grad_wt, 0, sizeof(double) * (data->p));
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            grad_wt[xt_inds[ii]] = wei_x * xt_vals[ii];
        }
        if (nega_t > 0) {
            wei_nega /= nega_t;
        } else {
            wei_nega = 0.0;
        }
        if (posi_t > 0) {
            wei_posi /= posi_t;
        } else {
            wei_posi = 0.0;
        }
        // gradient descent
        cblas_daxpy(data->p, wei_nega, nega_x, 1, grad_wt, 1);
        cblas_daxpy(data->p, wei_posi, posi_x, 1, grad_wt, 1);
        double tmp;
        for (int ii = 0; ii < data->p; ii++) {
            tmp = re->wt[ii] - eta_t * grad_wt[ii];
            double tmp_sign = (double) sign(tmp);
            re->wt[ii] = tmp_sign * fmax(0.0, fabs(tmp) - eta_t * para_l1);
        }
        // evaluate the AUC score
        if (paras->record_aucs == 1) {
            if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                double start_eval = clock();
                re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                double end_eval = clock();
                // this may not be very accurate.
                eval_time += end_eval - start_eval;
                run_time = (end_eval - start_time) - eval_time;
                re->iters[re->auc_len] = tt;
                re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                if (paras->verbose > 0) {
                    printf("tt: %d auc: %.4f n_va:%d\n",
                           tt, re->te_aucs[re->auc_len - 1], data->n_va);
                }
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    if (paras->verbose > 0) {
        printf("\n-------------------------------------------------------\n");
        printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
               data->p, data->n_tr, data->n_va, data->n_te);
        printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
               run_time, eval_time, total_time);
        printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
        printf("para_mu: %.4f para_l1: %.4f sparse_ratio: %.4f\n",
               para_mu, para_l1, re->sparse_ratio);
        printf("\n-------------------------------------------------------\n");
    }
    printf("para_mu: %.4e para_l1: %.4e sparse_ratio: %.4e ",
           para_mu, para_l1, re->sparse_ratio);
    printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    free(y_pred);
    free(nega_x);
    free(posi_x);
    free(grad_wt);
}


bool _algo_solam(
        Data *data,
        GlobalParas *paras,
        AlgoResults *re,
        double para_xi,
        double para_r) {

    double start_time = clock();
    openblas_set_num_threads(1);
    double gamma_bar;
    double gamma_bar_prev = 0.0;
    double alpha_bar;
    double alpha_bar_prev = 0.0;
    double gamma;
    double p_hat = 0.;
    double *v, *v_prev;
    double *v_bar, *v_bar_prev;
    double alpha, alpha_prev;
    double *grad_v;
    double is_p_yt, is_n_yt;
    double vt_dot, wei_posi, wei_nega, xtw;
    double weight, grad_alpha, norm_v;
    double total_time, run_time, eval_time = 0.0;
    v = malloc(sizeof(double) * (data->p + 2));
    v_prev = malloc(sizeof(double) * (data->p + 2));
    for (int i = 0; i < data->p; i++) {
        v_prev[i] = sqrt((para_r * para_r) / data->p);
    }
    v_prev[data->p] = para_r, v_prev[data->p + 1] = para_r;
    alpha_prev = 2. * para_r;
    grad_v = malloc(sizeof(double) * (data->p + 2));
    v_bar = malloc(sizeof(double) * (data->p + 2));
    v_bar_prev = calloc((data->p + 2), sizeof(double));
    for (int tt = 0; tt < data->n_tr; tt++) {
        // example x_i arrives and then we make prediction.
        // the index of the current training example.
        int ind = data->tr_indices[tt];
        // receive a training sample.
        const double *xt_vals = data->x_vals + data->x_poss[ind];
        const int *xt_inds = data->x_inds + data->x_poss[ind]; // current sample
        is_p_yt = is_posi(data->y[ind]);
        is_n_yt = is_nega(data->y[ind]);
        p_hat = (tt * p_hat + is_p_yt) / (tt + 1.); // update p_hat
        gamma = para_xi / sqrt(tt + 1.); // current learning rate
        vt_dot = 0.0, xtw = 0.0;
        // calculate the gradient w
        memset(grad_v, 0, sizeof(double) * (data->p + 2));
        for (int kk = 0; kk < data->x_lens[ind]; kk++) {
            grad_v[xt_inds[kk]] = xt_vals[kk];
            vt_dot += (v_prev[xt_inds[kk]] * xt_vals[kk]);
            xtw += re->wt[xt_inds[kk]] * xt_vals[kk];
        }
        // make online prediction
        re->pred_scores[tt] = xtw;
        re->true_labels[tt] = data->y[ind];
        wei_posi = 2. * (1. - p_hat) * (vt_dot - v_prev[data->p] - (1. + alpha_prev));
        wei_nega = 2. * p_hat * ((vt_dot - v_prev[data->p + 1]) + (1. + alpha_prev));
        weight = wei_posi * is_p_yt + wei_nega * is_n_yt;
        cblas_dscal(data->p, weight, grad_v, 1);
        grad_v[data->p] = -2. * (1. - p_hat) * (vt_dot - v_prev[data->p]) * is_p_yt; //grad of a
        grad_v[data->p + 1] = -2. * p_hat * (vt_dot - v_prev[data->p + 1]) * is_n_yt; //grad of b
        cblas_dscal(data->p + 2, -gamma, grad_v, 1); // gradient descent step of vt
        cblas_daxpy(data->p + 2, 1.0, v_prev, 1, grad_v, 1);
        memcpy(v, grad_v, sizeof(double) * (data->p + 2));
        wei_posi = -2. * (1. - p_hat) * vt_dot; // calculate the gradient of dual alpha
        wei_nega = 2. * p_hat * vt_dot;
        grad_alpha = wei_posi * is_p_yt + wei_nega * is_n_yt;
        grad_alpha += -2. * p_hat * (1. - p_hat) * alpha_prev;
        alpha = alpha_prev + gamma * grad_alpha; // gradient descent step of alpha
        norm_v = sqrt(cblas_ddot(data->p, v, 1, v, 1)); // projection w
        if (norm_v > para_r) { cblas_dscal(data->p, para_r / norm_v, v, 1); }
        v[data->p] = (v[data->p] > para_r) ? para_r : v[data->p]; // projection a
        v[data->p + 1] = (v[data->p + 1] > para_r) ? para_r : v[data->p + 1]; // projection b
        // projection alpha
        alpha = (fabs(alpha) > 2. * para_r) ? (2. * alpha * para_r) / fabs(alpha) : alpha;
        gamma_bar = gamma_bar_prev + gamma; // update gamma_
        memcpy(v_bar, v_prev, sizeof(double) * (data->p + 2)); // update v_bar
        cblas_dscal(data->p + 2, gamma / gamma_bar, v_bar, 1);
        cblas_daxpy(data->p + 2, gamma_bar_prev / gamma_bar, v_bar_prev, 1, v_bar, 1);
        // update alpha_bar
        alpha_bar = (gamma_bar_prev * alpha_bar_prev + gamma * alpha_prev) / gamma_bar;
        alpha_prev = alpha, alpha_bar_prev = alpha_bar, gamma_bar_prev = gamma_bar;
        memcpy(v_bar_prev, v_bar, sizeof(double) * (data->p + 2));
        memcpy(v_prev, v, sizeof(double) * (data->p + 2));
        // to calculate AUC score, v_var is the current values.
        memcpy(re->wt, v_bar, sizeof(double) * (data->p));
        if (paras->record_aucs == 1) {
            if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                double start_eval = clock();
                re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                double end_eval = clock();
                // this may not be very accurate.
                eval_time += end_eval - start_eval;
                run_time = (end_eval - start_time) - eval_time;
                re->iters[re->auc_len] = tt;
                re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                if (paras->verbose > 0) {
                    printf("tt: %d auc: %.4f n_va:%d\n",
                           tt, re->te_aucs[re->auc_len - 1], data->n_va);
                }
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    if (paras->verbose > 0) {
        printf("\n-------------------------------------------------------\n");
        printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
               data->p, data->n_tr, data->n_va, data->n_te);
        printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
               run_time, eval_time, total_time);
        printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
        printf("para_xi: %.4f para_r: %.4f sparse_ratio: %.4f\n",
               para_xi, para_r, re->sparse_ratio);
        printf("\n-------------------------------------------------------\n");
    }
    printf("para_xi: %.4e para_r: %.4e sparse_ratio: %.4e ",
           para_xi, para_r, re->sparse_ratio);
    printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    free(v_bar_prev);
    free(v_bar);
    free(grad_v);
    free(v_prev);
    free(v);
    return true;
}

void _algo_spam(Data *data,
                GlobalParas *paras,
                AlgoResults *re,
                double para_xi,
                double para_l1,
                double para_l2) {

    double start_time = clock();
    openblas_set_num_threads(1);
    double *grad_wt = malloc(sizeof(double) * data->p); // gradient
    double *posi_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=1]
    double *nega_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=-1]
    double *y_pred = calloc((size_t) data->n, sizeof(double));
    double a_wt = 0.0, b_wt = 0.0;
    double posi_t = 0.0, nega_t = 0.0;
    double prob_p = 0.0;
    double eta_t;
    double total_time, run_time, eval_time = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        int ind = data->tr_indices[tt];
        eta_t = para_xi / sqrt(tt + 1.0); // current learning rate
        const int *xt_inds;
        const double *xt_vals;
        double xtw = 0.0;
        // receive zt=(xt,yt)
        xt_inds = data->x_inds + data->x_poss[ind];
        xt_vals = data->x_vals + data->x_poss[ind];
        if (data->y[ind] > 0) {
            posi_t++;
            for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                posi_x[xt_inds[ii]] += xt_vals[ii];
            }
            prob_p = (tt * prob_p + 1.) / (tt + 1.0);
            // update a(wt)
            a_wt = cblas_ddot(data->p, re->wt, 1, posi_x, 1) / posi_t;
        } else {
            nega_t++;
            for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                nega_x[xt_inds[ii]] += xt_vals[ii];
            }
            prob_p = (tt * prob_p) / (tt + 1.0);
            // update b(wt)
            b_wt = cblas_ddot(data->p, re->wt, 1, nega_x, 1) / nega_t;
        }
        // make online prediction
        re->pred_scores[tt] = xtw;
        re->true_labels[tt] = data->y[ind];
        double wei_x, wei_posi, wei_nega;
        if (data->y[ind] > 0) {
            wei_x = 2. * (1. - prob_p) * (xtw - 1. - b_wt);
            wei_posi = 2. * (1. - prob_p) * a_wt;
            wei_nega = 2. * (1. - prob_p) * (-xtw);
        } else {
            wei_x = 2. * (prob_p) * (xtw + 1. - a_wt);
            wei_posi = 2. * (prob_p) * (-xtw);
            wei_nega = 2. * (prob_p) * b_wt;
        }
        memset(grad_wt, 0, sizeof(double) * (data->p));
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            grad_wt[xt_inds[ii]] = wei_x * xt_vals[ii];
        }
        if (nega_t > 0) {
            wei_nega /= nega_t;
        } else {
            wei_nega = 0.0;
        }
        if (posi_t > 0) {
            wei_posi /= posi_t;
        } else {
            wei_posi = 0.0;
        }
        // gradient descent
        cblas_daxpy(data->p, wei_nega, nega_x, 1, grad_wt, 1);
        cblas_daxpy(data->p, wei_posi, posi_x, 1, grad_wt, 1);
        cblas_daxpy(data->p, -eta_t, grad_wt, 1, re->wt, 1);
        if (para_l1 <= 0.0 && para_l2 > 0.0) {
            // l2-regularization
            cblas_dscal(data->p, 1. / (eta_t * para_l2 + 1.), re->wt, 1);
        } else {
            // elastic-net
            double tmp_demon = (eta_t * para_l2 + 1.);
            for (int ii = 0; ii < data->p; ii++) {
                double tmp_sign = (double) sign(re->wt[ii]) / tmp_demon;
                re->wt[ii] = tmp_sign * fmax(0.0, fabs(re->wt[ii]) - eta_t * para_l1);
            }
        }
        // evaluate the AUC score
        if (paras->record_aucs == 1) {
            if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                double start_eval = clock();
                re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                double end_eval = clock();
                // this may not be very accurate.
                eval_time += end_eval - start_eval;
                run_time = (end_eval - start_time) - eval_time;
                re->iters[re->auc_len] = tt;
                re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                if (paras->verbose > 0) {
                    printf("tt: %d auc: %.4f n_va:%d\n",
                           tt, re->te_aucs[re->auc_len - 1], data->n_va);
                }
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    if(paras->verbose >0){
        printf("\n-------------------------------------------------------\n");
        printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
               data->p, data->n_tr, data->n_va, data->n_te);
        printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
               run_time, eval_time, total_time);
        printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
        printf("para_xi: %.4f para_l1: %.4f para_l2: %.4f sparse_ratio: %.4f\n",
               para_xi, para_l1, para_l2, re->sparse_ratio);
        printf("\n-------------------------------------------------------\n");
    }
    printf("para_xi: %.4e para_l1: %.4e para_l2: %.4e sparse_ratio: %.4e ",
           para_xi, para_l1, para_l2, re->sparse_ratio);
    printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    free(y_pred);
    free(nega_x);
    free(posi_x);
    free(grad_wt);
}


void _algo_fsauc(Data *data, GlobalParas *paras, AlgoResults *re, double para_r, double para_g) {

    double start_time = clock();
    openblas_set_num_threads(1);

    double delta = 0.1;
    double eta = para_g;
    double R = para_r;
    double n_ids = data->n_tr;
    double alpha_1 = 0.0;
    double alpha;
    double *v_1 = calloc(((unsigned) data->p + 2), sizeof(double));
    double *sx_pos = calloc((size_t) data->p, sizeof(double));
    double *sx_neg = calloc((size_t) data->p, sizeof(double));
    double *m_pos = calloc((size_t) data->p, sizeof(double));
    double *m_neg = calloc((size_t) data->p, sizeof(double));
    int m = (int) floor(0.5 * log2(2 * n_ids / log2(n_ids))) - 1;
    int n_0 = (int) floor(n_ids / m);
    para_r = 2. * sqrt(3.) * R;
    double p_hat = 0.0;
    double beta = 9.0;
    double D = 2. * sqrt(2.) * para_r;
    double sp = 0.0;
    double *gd = calloc((unsigned) data->p + 2, sizeof(double));
    double gd_alpha;
    double *v_ave = calloc((unsigned) data->p + 2, sizeof(double));
    double *v_sum = malloc(sizeof(double) * (data->p + 2));
    double *v = malloc(sizeof(double) * (data->p + 2));
    double *vd = malloc(sizeof(double) * (data->p + 2)), ad;
    double *tmp_proj = malloc(sizeof(double) * data->p), beta_new;
    double *y_pred = calloc((size_t) data->n_tr, sizeof(double));
    int tt = 0;
    double total_time, run_time, eval_time = 0.0;
    for (int k = 0; k < m + 1; k++) {
        memset(v_sum, 0, sizeof(double) * (data->p + 2));
        memcpy(v, v_1, sizeof(double) * (data->p + 2));
        alpha = alpha_1;
        for (int kk = 0; kk < n_0; kk++) {
            if ((k * n_0 + kk) >= data->n_tr) {
                break;
            }
            int ind = data->tr_indices[k * n_0 + kk];
            double is_posi_y = is_posi(data->y[ind]);
            double is_nega_y = is_nega(data->y[ind]);
            const int *xt_inds;
            const double *xt_vals;
            double xtw = 0.0, weight;
            sp = sp + is_posi_y;
            p_hat = sp / (tt + 1.);
            xt_inds = data->x_inds + data->x_poss[ind];
            xt_vals = data->x_vals + data->x_poss[ind];
            for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                xtw += (v[xt_inds[ii]] * xt_vals[ii]);
            }
            if (data->y[ind] > 0) {
                for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                    sx_pos[xt_inds[ii]] += xt_vals[ii];
                }
            } else {
                for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                    sx_neg[xt_inds[ii]] += xt_vals[ii];
                }
            }
            // make online prediction
            re->pred_scores[tt] = xtw;
            re->true_labels[tt] = data->y[ind];
            weight = (1. - p_hat) * (xtw - v[data->p] - 1. - alpha) * is_posi_y;
            weight += p_hat * (xtw - v[data->p + 1] + 1. + alpha) * is_nega_y;
            memset(gd, 0, sizeof(double) * data->p);
            for (int jj = 0; jj < data->x_lens[ind]; jj++) {
                gd[xt_inds[jj]] = weight * xt_vals[jj];
            }
            gd[data->p] = (p_hat - 1.) * (xtw - v[data->p]) * is_posi_y;
            gd[data->p + 1] = p_hat * (v[data->p + 1] - xtw) * is_nega_y;
            gd_alpha = (p_hat - 1.) * (xtw + p_hat * alpha) * is_posi_y +
                       p_hat * (xtw + (p_hat - 1.) * alpha) * is_nega_y;
            cblas_daxpy(data->p + 2, -eta, gd, 1, v, 1);
            alpha = alpha + eta * gd_alpha;
            _l1ballproj_condat(v, tmp_proj, data->p, R); //projection to l1-ball
            memcpy(v, tmp_proj, sizeof(double) * data->p);
            if (fabs(v[data->p]) > R) {
                v[data->p] = v[data->p] * (R / fabs(v[data->p]));
            }
            if (fabs(v[data->p + 1]) > R) {
                v[data->p + 1] = v[data->p + 1] * (R / fabs(v[data->p + 1]));
            }
            if (fabs(alpha) > 2. * R) {
                alpha = alpha * (2. * R / fabs(alpha));
            }
            memcpy(vd, v, sizeof(double) * (data->p + 2));
            cblas_daxpy(data->p + 2, -1., v_1, 1, vd, 1);
            double norm_vd = sqrt(cblas_ddot(data->p + 2, vd, 1, vd, 1));
            if (norm_vd > para_r) {
                cblas_dscal(data->p + 2, para_r / norm_vd, vd, 1);
            }
            memcpy(v, vd, sizeof(double) * (data->p + 2));
            cblas_daxpy(data->p + 2, 1., v_1, 1, v, 1);
            ad = alpha - alpha_1;
            if (fabs(ad) > D) {
                ad = ad * (D / fabs(ad));
            }
            alpha = alpha_1 + ad;
            cblas_daxpy(data->p + 2, 1., v, 1, v_sum, 1);
            memcpy(v_ave, v_sum, sizeof(double) * (data->p + 2));
            cblas_dscal(data->p + 2, 1. / (kk + 1.), v_ave, 1);
            // to calculate AUC score
            if (paras->record_aucs == 1) {
                if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                    double start_eval = clock();
                    memcpy(re->wt, v_ave, sizeof(double) * data->p);
                    re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                    re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                    double end_eval = clock();
                    // this may not be very accurate.
                    eval_time += end_eval - start_eval;
                    run_time = (end_eval - start_time) - eval_time;
                    re->iters[re->auc_len] = tt;
                    re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                    if (paras->verbose > 0) {
                        printf("tt: %d auc: %.4f n_va:%d\n",
                               tt, re->te_aucs[re->auc_len - 1], data->n_va);
                    }
                }
            }
            tt++;
        }
        para_r = para_r / 2.;
        double tmp1 = 12. * sqrt(2.) * (2. + sqrt(2. * log(12. / delta))) * R;
        double tmp2 = fmin(p_hat, 1. - p_hat) * n_0 - sqrt(2. * n_0 * log(12. / delta));
        if (tmp2 > 0) { D = 2. * sqrt(2.) * para_r + tmp1 / sqrt(tmp2); } else { D = 1e7; }
        tmp1 = 288. * (pow(2. + sqrt(2. * log(12 / delta)), 2.));
        tmp2 = fmin(p_hat, 1. - p_hat) - sqrt(2. * log(12 / delta) / n_0);
        if (tmp2 > 0) { beta_new = 9. + tmp1 / tmp2; } else { beta_new = 1e7; }
        eta = fmin(sqrt(beta_new / beta) * eta / 2, eta);
        beta = beta_new;
        if (sp > 0.0) {
            memcpy(m_pos, sx_pos, sizeof(double) * data->p);
            cblas_dscal(data->p, 1. / sp, m_pos, 1);
        }
        if (sp < tt) {
            memcpy(m_neg, sx_neg, sizeof(double) * data->p);
            cblas_dscal(data->p, 1. / (tt - sp), m_neg, 1);
        }
        memcpy(v_1, v_ave, sizeof(double) * (data->p + 2));
        memcpy(tmp_proj, m_neg, sizeof(double) * data->p);
        cblas_daxpy(data->p, -1., m_pos, 1, tmp_proj, 1);
        alpha_1 = cblas_ddot(data->p, v_ave, 1, tmp_proj, 1);
    }
    memcpy(re->wt, v_ave, sizeof(double) * data->p);
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    if (paras->verbose > 0) {
        printf("\n-------------------------------------------------------\n");
        printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
               data->p, data->n_tr, data->n_va, data->n_te);
        printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
               run_time, eval_time, total_time);
        printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
        printf("para_g: %.4f para_r: %.4f sparse_ratio: %.4f\n",
               para_g, para_r, re->sparse_ratio);
        printf("\n-------------------------------------------------------\n");
    }

    printf("para_g: %.4e para_r: %.4e sparse_ratio: %.4e ",
           para_g, para_r, re->sparse_ratio);
    printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    free(y_pred);
    free(tmp_proj);
    free(vd);
    free(v);
    free(v_sum);
    free(v_ave);
    free(gd);
    free(m_neg);
    free(m_pos);
    free(sx_neg);
    free(sx_pos);
    free(v_1);
}


void _algo_ftrl_auc(Data *data,
                    GlobalParas *paras,
                    AlgoResults *re,
                    double para_l1,
                    double para_l2,
                    double para_beta,
                    double para_gamma) {

    clock_t start_time = clock();
    openblas_set_num_threads(1);
    double t_posi = 0.0, t_nega = 0.0;
    double utw = 0.0, vtw = 0.0;
    double *x_posi = calloc(data->p, sizeof(double));
    double *x_nega = calloc(data->p, sizeof(double));
    double *gt = calloc(data->p, sizeof(double));
    double *zt = calloc(data->p, sizeof(double));
    double *gt_square = calloc(data->p, sizeof(double));
    double prob_p = 0.0;
    double total_time, run_time, eval_time = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        // example x_i arrives and then we make prediction.
        // the index of the current training example.
        int ind = data->tr_indices[tt];
        // receive a training sample.
        bool is_posi_y = is_posi(data->y[ind]);
        prob_p = (tt * prob_p + is_posi_y) / (tt + 1.);
        const int *xt_inds = data->x_inds + data->x_poss[ind];
        const double *xt_vals = data->x_vals + data->x_poss[ind];
        double xtw = 0.0, ni, pow_gt, weight, lr;
        // calculate the gradient
        if (is_posi_y) {
            for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                x_posi[xt_inds[ii]] += xt_vals[ii];
            }
            t_posi += 1.;
            utw = (t_posi - 1.) * utw / t_posi + xtw / t_posi;
            weight = 2. * (1.0 - prob_p) * (xtw - vtw - 1.0);
        } else {
            for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                x_nega[xt_inds[ii]] += xt_vals[ii];
            }
            t_nega += 1.;
            vtw = (t_nega - 1.) * vtw / t_nega + xtw / t_nega;
            weight = 2. * prob_p * (xtw - utw + 1.0);
        }
        // make online prediction
        re->pred_scores[tt] = xtw;
        re->true_labels[tt] = data->y[ind];
        // lazy update the model and make prediction.
        // to make a prediction of AUC score
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            ni = gt_square[xt_inds[ii]];
            re->wt[xt_inds[ii]] =
                    fabs(zt[xt_inds[ii]]) <= para_l1 ?
                    0.0 : -(zt[xt_inds[ii]] - sign(zt[xt_inds[ii]]) * para_l1)
                          / ((para_beta + sqrt(ni)) / para_gamma + para_l2);
        }
        // update the learning rate and gradient
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            gt[xt_inds[ii]] = weight * xt_vals[ii];
            ni = gt_square[xt_inds[ii]];
            pow_gt = pow(gt[xt_inds[ii]], 2.);
            lr = (sqrt(ni + pow_gt) - sqrt(ni)) / para_gamma;
            zt[xt_inds[ii]] += gt[xt_inds[ii]] - lr * re->wt[xt_inds[ii]];
            gt_square[xt_inds[ii]] += pow_gt;
        }
        if (paras->record_aucs == 1) {
            if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                double start_eval = clock();
                re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                double end_eval = clock();
                // this may not be very accurate.
                eval_time += end_eval - start_eval;
                run_time = (end_eval - start_time) - eval_time;
                re->iters[re->auc_len] = tt;
                re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                if (paras->verbose > 0) {
                    printf("tt: %d auc: %.4f n_va:%d\n",
                           tt, re->te_aucs[re->auc_len - 1], data->n_va);
                }
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    if (paras->verbose > 0) {
        printf("\n-------------------------------------------------------\n");
        printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
               data->p, data->n_tr, data->n_va, data->n_te);
        printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
               run_time, eval_time, total_time);
        printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
        printf("para_l1: %.4f para_gamma: %.4f sparse_ratio: %.4f\n",
               para_l1, para_gamma, re->sparse_ratio);
        printf("\n-------------------------------------------------------\n");
    }
    printf("para_l1: %.4e para_gamma: %.4e sparse_ratio: %.4e ",
           para_l1, para_gamma, re->sparse_ratio);
    printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    free(x_nega);
    free(x_posi);
    free(gt);
    free(zt);
    free(gt_square);
}


void _algo_ftrl_proximal(Data *data,
                         GlobalParas *paras,
                         AlgoResults *re,
                         double para_l1,
                         double para_l2,
                         double para_beta,
                         double para_gamma) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    double *gt = calloc(data->p + 1, sizeof(double));
    double *zt = calloc(data->p + 1, sizeof(double));
    double *gt_square = calloc(data->p + 1, sizeof(double));
    double total_time, run_time, eval_time = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        // 1. example x_i arrives and then we make prediction.
        int ind = data->tr_indices[tt]; // the index of the current training example.
        double weight;
        double xtw = 0.0, ni, pow_gt;
        // receive a training sample.
        const int *xt_inds = data->x_inds + data->x_poss[ind];
        const double *xt_vals = data->x_vals + data->x_poss[ind];
        // lazy update the model
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            ni = gt_square[xt_inds[ii]];
            re->wt[xt_inds[ii]] = fabs(zt[xt_inds[ii]]) <= para_l1 ? 0.0 :
                                  -(zt[xt_inds[ii]] - sign(zt[xt_inds[ii]]) * para_l1)
                                  / ((para_beta + sqrt(ni)) / para_gamma + para_l2);
            xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
        }
        // make online prediction
        re->pred_scores[tt] = xtw;
        re->true_labels[tt] = data->y[ind];
        // make a prediction
        double z0 = (data->y[ind]) * (xtw + re->wt[data->p]), lr;
        // calculate the gradient of w
        weight = -(data->y[ind]) * expit(-z0);
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            gt[xt_inds[ii]] = weight * xt_vals[ii];
            ni = gt_square[xt_inds[ii]];
            pow_gt = pow(gt[xt_inds[ii]], 2.);
            lr = (sqrt(ni + pow_gt) - sqrt(ni)) / para_gamma;
            zt[xt_inds[ii]] += gt[xt_inds[ii]] - lr * re->wt[xt_inds[ii]];
            gt_square[xt_inds[ii]] += pow_gt;
        }
        // calculate the gradient of intercept
        gt[data->p] = weight;
        ni = gt_square[data->p];
        pow_gt = pow(gt[data->p], 2.);
        zt[data->p] += gt[data->p] - (sqrt(ni + pow_gt) - sqrt(ni)) / para_gamma;
        gt_square[data->p] += pow_gt;

        if (paras->record_aucs == 1) {
            if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                double start_eval = clock();
                re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                double end_eval = clock();
                // this may not be very accurate.
                eval_time += end_eval - start_eval;
                run_time = (end_eval - start_time) - eval_time;
                re->iters[re->auc_len] = tt;
                re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                if (paras->verbose > 0) {
                    printf("tt: %d auc: %.4f n_va:%d\n",
                           tt, re->te_aucs[re->auc_len - 1], data->n_va);
                }
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    printf("\n-------------------------------------------------------\n");
    printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
           data->p, data->n_tr, data->n_va, data->n_te);
    printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
           run_time, eval_time, total_time);
    printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    printf("para_l1: %.4f para_gamma: %.4f sparse_ratio: %.4f\n",
           para_l1, para_gamma, re->sparse_ratio);
    printf("\n-------------------------------------------------------\n");
    free(gt);
    free(zt);
    free(gt_square);
}

void _algo_rda_l1(Data *data,
                  GlobalParas *paras,
                  AlgoResults *re,
                  double para_lambda,
                  double para_gamma,
                  double para_rho) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    double *gt = calloc(data->p + 1, sizeof(double));
    double *gt_bar = calloc(data->p + 1, sizeof(double));
    double total_time, run_time, eval_time = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        // 1. example x_i arrives and then we make prediction.
        int ind = data->tr_indices[tt]; // the index of the current training example.
        double weight;
        double xtw = 0.0;
        // receive a training sample.
        const int *xt_inds = data->x_inds + data->x_poss[ind];
        const double *xt_vals = data->x_vals + data->x_poss[ind];
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
        }
        // make a prediction
        double z0 = (data->y[ind]) * (xtw + re->wt[data->p]);
        // calculate the gradient of w
        weight = -(data->y[ind]) * expit(-z0);
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            gt[xt_inds[ii]] = weight * xt_vals[ii];
        }
        // make online prediction
        re->pred_scores[tt] = xtw;
        re->true_labels[tt] = data->y[ind];
        // calculate the gradient of intercept
        gt[data->p] = weight;
        // 3.   compute the dual average.
        cblas_dscal(data->p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(data->p + 1, 1. / (tt + 1.), gt, 1, gt_bar, 1);

        // 4.   update the model: enhanced l1-rda method. Equation (30)
        double wei = -sqrt(tt + 1.) / para_gamma;
        double lambda_t_rda = para_lambda + (para_gamma * para_rho) / sqrt(tt + 1.);
        for (int i = 0; i < data->p; i++) {
            if (fabs(gt_bar[i]) <= lambda_t_rda) {
                re->wt[i] = 0.0; //thresholding entries
            } else {
                re->wt[i] = wei * (gt_bar[i] - lambda_t_rda * sign(gt_bar[i]));
            }
        }
        // Notice: the bias term do not need to add regularization.
        re->wt[data->p] = wei * gt_bar[data->p];
        if (paras->record_aucs == 1) {
            if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                double start_eval = clock();
                re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                double end_eval = clock();
                // this may not be very accurate.
                eval_time += end_eval - start_eval;
                run_time = (end_eval - start_time) - eval_time;
                re->iters[re->auc_len] = tt;
                re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                if (paras->verbose > 0) {
                    printf("tt: %d auc: %.4f n_va:%d\n",
                           tt, re->te_aucs[re->auc_len - 1], data->n_va);
                }
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    if (paras->verbose > 0) {
        printf("\n-------------------------------------------------------\n");
        printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
               data->p, data->n_tr, data->n_va, data->n_te);
        printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
               run_time, eval_time, total_time);
        printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
        printf("para_lambda: %.4f para_gamma: %.4f para_rho: %.4f sparse_ratio: %.4f\n",
               para_lambda, para_gamma, para_rho, re->sparse_ratio);
        printf("\n-------------------------------------------------------\n");
    }
    printf("para_lambda: %.4e para_gamma: %.4e para_rho: %.4e",
           para_lambda, para_gamma, para_rho);
    printf(" va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    free(gt);
    free(gt_bar);
}


void _algo_adagrad(Data *data,
                   GlobalParas *paras,
                   AlgoResults *re,
                   double para_lambda,
                   double para_eta,
                   double para_epsilon) {
    clock_t start_time = clock();
    openblas_set_num_threads(1);
    double *gt = calloc(data->p + 1, sizeof(double));
    double *gt_bar = calloc(data->p + 1, sizeof(double));
    double *gt_square = calloc(data->p + 1, sizeof(double));
    double total_time, run_time, eval_time = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        // 1. example x_i arrives and then we make prediction.
        int ind = data->tr_indices[tt]; // the index of the current training example.
        double weight;
        double xtw = 0.0;
        // receive a training sample.
        const int *xt_inds = data->x_inds + data->x_poss[ind];
        const double *xt_vals = data->x_vals + data->x_poss[ind];
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
        }
        // make online prediction
        re->pred_scores[tt] = xtw;
        re->true_labels[tt] = data->y[ind];
        // make a prediction
        double z0 = (data->y[ind]) * (xtw + re->wt[data->p]);
        // calculate the gradient of w
        weight = -(data->y[ind]) * expit(-z0);
        for (int ii = 0; ii < data->x_lens[ind]; ii++) {
            gt[xt_inds[ii]] = weight * xt_vals[ii];
            gt_square[xt_inds[ii]] += gt[xt_inds[ii]] * gt[xt_inds[ii]];
        }
        // calculate the gradient of intercept
        gt[data->p] = weight;
        gt_square[data->p] += weight * weight;
        // 3.   compute the dual average.
        cblas_dscal(data->p + 1, tt / (tt + 1.), gt_bar, 1);
        cblas_daxpy(data->p + 1, 1. / (tt + 1.), gt, 1, gt_bar, 1);
        // 4.   update the model: enhanced l1-rda method. Equation (30)
        for (int i = 0; i < data->p; i++) {
            double wei = sign(-gt_bar[i]) * para_eta * (tt + 1.) / (para_epsilon + sqrt(gt_square[i]));
            double truncate = fmax(0.0, fabs(gt_bar[i]) - para_lambda);
            re->wt[i] = wei * truncate;
        }
        double wei = sign(-gt_bar[data->p]) * para_eta * (tt + 1.) /
                                            (para_epsilon + sqrt(gt_square[data->p]));
        double truncate = fmax(0.0, fabs(gt_bar[data->p]));
        re->wt[data->p] = wei * truncate;
        if (paras->record_aucs == 1) {
            if ((_check_step_size(tt) == 0) || (tt == (data->n_tr - 1))) {
                double start_eval = clock();
                re->online_aucs[re->auc_len] = _auc_score(re->true_labels, re->pred_scores, tt + 1);
                re->te_aucs[re->auc_len] = _eval_auc(data, re, false);
                double end_eval = clock();
                // this may not be very accurate.
                eval_time += end_eval - start_eval;
                run_time = (end_eval - start_time) - eval_time;
                re->iters[re->auc_len] = tt;
                re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
                if (paras->verbose > 0) {
                    printf("tt: %d auc: %.4f n_va:%d\n",
                           tt, re->te_aucs[re->auc_len - 1], data->n_va);
                }
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    re->run_time = run_time;
    re->eval_time = eval_time;
    re->total_time = total_time;
    _cal_sparse_ratio(re, data->p);
    re->va_auc = _eval_auc(data, re, true);
    re->te_auc = _eval_auc(data, re, false);
    if (paras->verbose > 0) {
        printf("\n-------------------------------------------------------\n");
        printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
               data->p, data->n_tr, data->n_va, data->n_te);
        printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
               run_time, eval_time, total_time);
        printf("va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
        printf("para_lambda: %.4f para_gamma: %.4f para_rho: %.4f sparse_ratio: %.4f\n",
               para_lambda, para_eta, para_epsilon, re->sparse_ratio);
        printf("\n-------------------------------------------------------\n");
    }
    printf("para_lambda: %.4e para_gamma: %.4e para_rho: %.4e",
           para_lambda, para_eta, para_epsilon);
    printf(" va_auc: %.4f te_auc: %.4f\n", re->va_auc, re->te_auc);
    free(gt);
    free(gt_bar);
    free(gt_square);
}