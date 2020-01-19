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
    re->wt_prev = calloc((size_t) data_p, sizeof(double));
    re->aucs = calloc((size_t) total_num_eval, sizeof(double));
    re->rts = calloc((size_t) total_num_eval, sizeof(double));
    re->scores = calloc((size_t) total_num_eval, sizeof(double));
    re->auc_len = 0;
    return re;
}

bool free_algo_results(AlgoResults *re) {
    free(re->rts);
    free(re->aucs);
    free(re->wt);
    free(re->wt_prev);
    free(re->scores);
    free(re);
    return true;
}

GraphStat *make_graph_stat(int p, int m) {
    GraphStat *stat = malloc(sizeof(GraphStat));
    stat->num_pcst = 0;
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->run_time = 0;
    stat->costs = malloc(sizeof(double) * m);
    stat->prizes = malloc(sizeof(double) * p);
    return stat;
}

bool free_graph_stat(GraphStat *graph_stat) {
    free(graph_stat->re_nodes->array);
    free(graph_stat->re_nodes);
    free(graph_stat->re_edges->array);
    free(graph_stat->re_edges);
    free(graph_stat->costs);
    free(graph_stat->prizes);
    free(graph_stat);
    return true;
}


bool head_tail_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat) {

    // malloc: cur_costs, sorted_prizes, and sorted_indices
    // free: cur_costs, sorted_prizes, and sorted_indices
    double *cur_costs = malloc(sizeof(double) * m);
    double *sorted_prizes = malloc(sizeof(double) * n);
    int *sorted_indices = malloc(sizeof(int) * n);
    for (int ii = 0; ii < m; ii++) {
        cur_costs[ii] = costs[ii];
    }
    for (int ii = 0; ii < n; ii++) {
        sorted_prizes[ii] = prizes[ii];
    }
    int guess_pos = n - sparsity_high;
    _arg_sort_descend(sorted_prizes, sorted_indices, n);
    double lambda_low = 0.0;
    double lambda_high = fabs(2.0 * sorted_prizes[sorted_indices[guess_pos]]);
    if (lambda_high == 0.0) {
        guess_pos = n - sparsity_low;
        lambda_high = fabs(2.0 * sorted_prizes[sorted_indices[guess_pos]]);
        if (lambda_high != 0.0) {
        } else {
            lambda_high = fabs(prizes[0]);
            for (int ii = 1; ii < n; ii++) {
                lambda_high = fmax(lambda_high, fabs(prizes[ii]));
            }
            lambda_high *= 2.0;
        }
    }
    stat->num_iter = 0;
    lambda_high /= 2.0;
    int cur_k;
    do {
        stat->num_iter += 1;
        lambda_high *= 2.0;
        if (lambda_high <= 0.0) { printf("lambda_high: %.6e\n", lambda_high); }
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_high * costs[ii];
        }
        PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                               1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) printf("increase:   l_high: %e  k: %d\n", lambda_high, cur_k);
    } while (cur_k > sparsity_high && stat->num_iter < max_num_iter);

    if (stat->num_iter < max_num_iter && cur_k >= sparsity_low) {
        if (verbose >= 1) printf("Found good lambda in exponential increase phase, returning.\n");
        free(cur_costs);
        free(sorted_prizes);
        free(sorted_indices);
        return true;
    }
    double lambda_mid;
    while (stat->num_iter < max_num_iter) {
        stat->num_iter += 1;
        lambda_mid = (lambda_low + lambda_high) / 2.0;
        if (lambda_mid <= 0.0) { printf("lambda_mid: %.6e\n", lambda_mid); }
        for (int ii = 0; ii < m; ii++) { cur_costs[ii] = lambda_mid * costs[ii]; }

        PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters, 1e-10,
                               pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (sparsity_low <= cur_k && cur_k <= sparsity_high) {
            free(cur_costs);
            free(sorted_prizes);
            free(sorted_indices);
            return true;
        }
        if (cur_k > sparsity_high) {
            lambda_low = lambda_mid;
        } else {
            lambda_high = lambda_mid;
        }
    }
    if (lambda_high <= 0.0) { printf("lambda_high: %.6e\n", lambda_high); }
    for (int ii = 0; ii < m; ++ii) { cur_costs[ii] = lambda_high * costs[ii]; }
    PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                           1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges);
    free_pcst(pcst);
    free(cur_costs);
    free(sorted_prizes);
    free(sorted_indices);
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
    re->aucs[re->auc_len] = _auc_score(data->y, y_pred, data->n_tr);
    re->rts[re->auc_len++] = clock() - start_time - (clock() - t_eval);
}


double eval_auc(Data *data, AlgoResults *re, bool is_va) {
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

double get_sparse_ratio(const double *x, int d) {
    double sparse_ratio = 0.0;
    for (int i = 0; i < d; i++) {
        if (x[i] != 0.0) {
            sparse_ratio += 1.;
        }
    }
    return sparse_ratio / (double) d;
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
    double *y_pred, *grad_v;
    double is_p_yt, is_n_yt;
    double vt_dot, wei_posi, wei_nega;
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
    y_pred = calloc((size_t) data->n, sizeof(double));
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
        vt_dot = 0.0;
        // calculate the gradient w
        memset(grad_v, 0, sizeof(double) * (data->p + 2));
        for (int kk = 0; kk < data->x_lens[ind]; kk++) {
            grad_v[xt_inds[kk]] = xt_vals[kk];
            vt_dot += (v_prev[xt_inds[kk]] * xt_vals[kk]);
        }
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
        if (tt % paras->eval_step == 0) {
            printf("%.4f\n", cblas_ddot(data->p, re->wt, 1, re->wt, 1));
            double start_eval = clock();
            re->aucs[re->auc_len] = eval_auc(data, re, true);
            double end_eval = clock();
            // this may not be very accurate.
            eval_time += end_eval - start_eval;
            run_time = (end_eval - start_time) - eval_time;
            re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
            if (paras->verbose > 0) {
                printf("tt: %d auc: %.4f n_va:%d\n",
                       tt, re->aucs[re->auc_len - 1], data->n_va);
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    printf("\n-------------------------------------------------------\n");
    printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
           data->p, data->n_tr, data->n_va, data->n_te);
    printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
           run_time, eval_time, total_time);
    printf("va_auc: %.4f te_auc: %.4f\n",
           eval_auc(data, re, true), eval_auc(data, re, false));
    printf("para_xi: %.4f para_r: %.4f sparse_ratio: %.4f\n",
           para_xi, para_r, get_sparse_ratio(re->wt, data->p));
    printf("\n-------------------------------------------------------\n");
    free(y_pred);
    free(v_bar_prev);
    free(v_bar);
    free(grad_v);
    free(v_prev);
    free(v);
    return true;
}

void _algo_spam(Data *data, GlobalParas *paras, AlgoResults *re,
                double para_xi, double para_l1_reg, double para_l2_reg) {

    double start_time = clock();
    openblas_set_num_threads(1);
    double *grad_wt = malloc(sizeof(double) * data->p); // gradient
    double *posi_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=1]
    double *nega_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=-1]
    double *y_pred = calloc((size_t) data->n, sizeof(double));
    double a_wt, b_wt;
    double alpha_wt;
    double posi_t = 0.0, nega_t = 0.0;
    double prob_p;
    double eta_t;
    _get_posi_nega_x(posi_x, nega_x, &posi_t, &nega_t, &prob_p, data);

    for (int t = 1; t <= data->n_tr; t++) {
        eta_t = para_xi / sqrt(t); // current learning rate
        a_wt = cblas_ddot(data->p, re->wt, 1, posi_x, 1); // update a(wt)
        b_wt = cblas_ddot(data->p, re->wt, 1, nega_x, 1); // para_b(wt)
        alpha_wt = b_wt - a_wt; // alpha(wt)
        const double *xt;
        const int *xt_inds;
        const double *xt_vals;
        double xtw = 0.0, weight;
        if (data->is_sparse) {
            // receive zt=(xt,yt)
            xt_inds = data->x_inds + data->x_poss[(t - 1) % data->n_tr];
            xt_vals = data->x_vals + data->x_poss[(t - 1) % data->n_tr];
            for (int tt = 0; tt < data->x_lens[(t - 1) % data->n_tr]; tt++) {
                xtw += (re->wt[xt_inds[tt]] * xt_vals[tt]);
            }
            weight = data->y[(t - 1) % data->n_tr] > 0 ?
                     2. * (1.0 - prob_p) * (xtw - a_wt) -
                     2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                     2.0 * prob_p * (xtw - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            // gradient descent
            for (int kk = 0; kk < data->x_lens[(t - 1) % data->n_tr]; kk++) {
                re->wt[xt_inds[kk]] += -eta_t * weight * xt_vals[kk];
            }
        } else {
            xt = data->x_vals + ((t - 1) % data->n_tr) * data->p;
            xtw = cblas_ddot(data->p, re->wt, 1, xt, 1);
            weight = data->y[(t - 1) % data->n_tr] > 0 ?
                     2. * (1.0 - prob_p) * (xtw - a_wt) -
                     2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                     2.0 * prob_p * (xtw - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            // gradient descent
            cblas_daxpy(data->p, -eta_t * weight, xt, 1, re->wt, 1);
        }
        if (para_l1_reg <= 0.0 && para_l2_reg > 0.0) {
            // l2-regularization
            cblas_dscal(data->p, 1. / (eta_t * para_l2_reg + 1.), re->wt, 1);
        } else {
            // elastic-net
            double tmp_demon = (eta_t * para_l2_reg + 1.);
            for (int k = 0; k < data->p; k++) {
                double tmp_sign = (double) sign(re->wt[k]) / tmp_demon;
                re->wt[k] = tmp_sign * fmax(0.0, fabs(re->wt[k]) - eta_t * para_l1_reg);
            }
        }
        // evaluate the AUC score
        if (paras->record_aucs == 1) {
            _evaluate_aucs(data, y_pred, re, start_time);
        }
        // at the end of each epoch, we check the early stop condition.
        re->total_iterations++;
        memcpy(re->wt_prev, re->wt, sizeof(double) * (data->p));
    }
    cblas_dscal(re->auc_len, 1. / CLOCKS_PER_SEC, re->rts, 1);
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
    int t = 0;
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
    for (int k = 0; k < m; k++) {
        memset(v_sum, 0, sizeof(double) * (data->p + 2));
        memcpy(v, v_1, sizeof(double) * (data->p + 2));
        alpha = alpha_1;
        for (int kk = 0; kk < n_0; kk++) {
            int ind = (k * n_0 + kk) % data->n_tr;
            double is_posi_y = is_posi(data->y[ind]);
            double is_nega_y = is_nega(data->y[ind]);
            const int *xt_inds;
            const double *xt_vals, *xt;
            double xtw = 0.0, weight;
            sp = sp + is_posi_y;
            p_hat = sp / (t + 1.);
            if (data->is_sparse) {
                xt_inds = data->x_inds + data->x_poss[ind];
                xt_vals = data->x_vals + data->x_poss[ind];
                for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                    xtw += (v[xt_inds[tt]] * xt_vals[tt]);
                }
                if (data->y[ind] > 0) {
                    for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                        sx_pos[xt_inds[tt]] += xt_vals[tt];
                    }
                } else {
                    for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                        sx_neg[xt_inds[tt]] += xt_vals[tt];
                    }
                }
                weight = (1. - p_hat) * (xtw - v[data->p] - 1. - alpha) * is_posi_y;
                weight += p_hat * (xtw - v[data->p + 1] + 1. + alpha) * is_nega_y;
                memset(gd, 0, sizeof(double) * data->p);
                for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                    gd[xt_inds[tt]] = weight * xt_vals[tt];
                }
            } else {
                xt = data->x_vals + ind * data->p;
                xtw = cblas_ddot(data->p, xt, 1, v, 1);
                cblas_daxpy(data->p, is_posi_y, xt, 1, sx_pos, 1);
                cblas_daxpy(data->p, is_nega_y, xt, 1, sx_neg, 1);
                weight = (1. - p_hat) * (xtw - v[data->p] - 1. - alpha) * is_posi_y;
                weight += p_hat * (xtw - v[data->p + 1] + 1. + alpha) * is_nega_y;
                memcpy(gd, xt, sizeof(double) * (data->p));
                cblas_dscal(data->p, weight, gd, 1);
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
            t++;
            // to calculate AUC score
            if (paras->record_aucs == 1) {
                memcpy(re->wt, v_ave, sizeof(double) * (data->p));
                _evaluate_aucs(data, y_pred, re, start_time);
            }
            // at the end of each epoch, we check the early stop condition.
            re->total_iterations++;
            memcpy(re->wt_prev, v_ave, sizeof(double) * (data->p));
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
        if (sp < t) {
            memcpy(m_neg, sx_neg, sizeof(double) * data->p);
            cblas_dscal(data->p, 1. / (t - sp), m_neg, 1);
        }
        memcpy(v_1, v_ave, sizeof(double) * (data->p + 2));
        memcpy(tmp_proj, m_neg, sizeof(double) * data->p);
        cblas_daxpy(data->p, -1., m_pos, 1, tmp_proj, 1);
        alpha_1 = cblas_ddot(data->p, v_ave, 1, tmp_proj, 1);
    }
    memcpy(re->wt, v_ave, sizeof(double) * data->p);
    cblas_dscal(re->auc_len, 1. / CLOCKS_PER_SEC, re->rts, 1);
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

void _algo_sht_am(Data *data, GlobalParas *paras, AlgoResults *re,
                  int version, int operator_id, int para_s, int para_b, double para_c, double para_l2_reg) {

    srand((int) lrand48());
    double start_time = clock();
    openblas_set_num_threads(1);

    double *posi_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=1]
    double *nega_x = calloc((size_t) data->p, sizeof(double)); // E[x|y=-1]
    double *y_pred = calloc((size_t) data->n_tr, sizeof(double)); // y_predication
    double *grad_wt = calloc((size_t) data->p, sizeof(double)); // gradient
    double *var = calloc((size_t) data->p, sizeof(double));
    double *tmp = calloc((size_t) data->p, sizeof(double));

    double posi_t = 0.0;
    double nega_t = 0.0;
    double prob_p;
    int min_b_ind = 0;
    int max_b_ind = data->n_tr / para_b;
    _get_posi_nega_x(posi_x, nega_x, &posi_t, &nega_t, &prob_p, data);
    memcpy(var, nega_x, sizeof(double) * data->p);
    cblas_daxpy(data->p, -1.0, posi_x, 1, var, 1);
    cblas_dscal(data->p, 2.0 * prob_p * (1.0 - prob_p), var, 1);
    for (int t = 1; t <= data->n_tr; t++) { // for each block
        // block bi is in [min_b_ind,max_b_ind-1]
        int bi = (int) (lrand48() % (max_b_ind - min_b_ind));
        double utw = cblas_ddot(data->p, re->wt, 1, posi_x, 1);
        double vtw = cblas_ddot(data->p, re->wt, 1, nega_x, 1);
        // the gradient of a block training samples
        memset(grad_wt, 0, sizeof(double) * data->p);
        // take care of the last block
        int cur_b_size = (bi == (max_b_ind - 1) ? para_b + (data->n_tr % para_b) : para_b);
        // for each block of training samples
        for (int kk = 0; kk < cur_b_size; kk++) {
            int ind = bi * para_b + kk; // initial position of block bi
            double weight, wei_x, wei_posi, wei_nega;
            if (data->is_sparse) {
                double xtw = 0.0;
                const int *xt_inds = NULL;
                const double *xt_vals = NULL;
                xt_inds = data->x_inds + data->x_poss[ind];
                xt_vals = data->x_vals + data->x_poss[ind];
                for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                    xtw += (re->wt[xt_inds[tt]] * xt_vals[tt]);
                }
                switch (version) {
                    case 0:
                        memcpy(tmp, var, sizeof(double) * data->p);
                        cblas_dscal(data->p, 1 + vtw - utw, tmp, 1);
                        if (data->y[ind] > 0) {
                            double part_wei = 2. * (1 - prob_p) * (xtw - utw);
                            for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                                tmp[xt_inds[tt]] += part_wei * xt_vals[tt];
                            }
                            cblas_daxpy(data->p, -part_wei, posi_x, 1, tmp, 1);
                        } else {
                            double part_wei = 2. * prob_p * (xtw - vtw);
                            for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                                tmp[xt_inds[tt]] += part_wei * xt_vals[tt];
                            }
                            cblas_daxpy(data->p, -part_wei, nega_x, 1, tmp, 1);
                        }
                        // calculate the gradient
                        cblas_daxpy(data->p, 1., tmp, 1, grad_wt, 1);
                        break;
                    case 1:
                        weight = data->y[ind] > 0 ? 2. * (1.0 - prob_p) * (xtw - utw) -
                                                    2. * (1.0 + (vtw - utw)) * (1.0 - prob_p) :
                                 2.0 * prob_p * (xtw - vtw) + 2.0 * (1.0 + (vtw - utw)) * prob_p;
                        // calculate the gradient
                        for (int tt = 0; tt < data->x_lens[ind]; tt++)
                            grad_wt[xt_inds[tt]] += (weight * xt_vals[tt]); // calculate the gradient for xi
                        break;
                    case 2:
                        if (data->y[ind] > 0) {
                            wei_x = 2. * (1. - prob_p) * (xtw - vtw - 1.0);
                            wei_posi = 2. * (1. - prob_p) * utw - 2. * prob_p * (1. - prob_p) * (utw - vtw);
                            wei_nega = -2. * (1. - prob_p) * xtw + 2. * prob_p * (1. - prob_p) * (utw - vtw);
                        } else {
                            wei_x = 2. * prob_p * (xtw - utw + 1.0);
                            wei_posi = -2. * prob_p * xtw + 2. * prob_p * (1. - prob_p) * (vtw - utw);
                            wei_nega = 2. * prob_p * vtw - 2. * prob_p * (1. - prob_p) * (vtw - utw);
                        }
                        // calculate the gradient
                        for (int tt = 0; tt < data->x_lens[ind]; tt++) {
                            grad_wt[xt_inds[tt]] += (wei_x * xt_vals[tt]);
                        }
                        cblas_daxpy(data->p, wei_posi, posi_x, 1, grad_wt, 1);
                        cblas_daxpy(data->p, wei_nega, nega_x, 1, grad_wt, 1);
                    default:
                        break;
                }

            } else {
                const double *cur_xt = NULL;
                cur_xt = data->x_vals + ind * data->p;
                double xtw = cblas_ddot(data->p, re->wt, 1, cur_xt, 1);
                switch (version) {
                    case 0:
                        /**
                         * if (data->y[ind] > 0) {
                                wei_x = 2. * (1. - prob_p) * (xtw - utw);
                                wei_posi = 2. * (1. - prob_p) * (-xtw + utw + (-1. - vtw + utw) * prob_p);
                                wei_nega = 2. * (1. - prob_p) * prob_p * ((1. + vtw - utw));
                            } else {
                                wei_x = 2. * prob_p * (vtw - xtw);
                                wei_posi = 2. * prob_p * (1. - prob_p) * (-1. - vtw + utw);
                                wei_nega = 2. * prob_p * (xtw - vtw + (1. + vtw - utw) * (1. - prob_p));
                            }
                            // calculate the gradient
                            cblas_daxpy(data->p, wei_x, cur_xt, 1, grad_wt, 1);
                            cblas_daxpy(data->p, wei_posi, posi_x, 1, grad_wt, 1);
                            cblas_daxpy(data->p, wei_nega, nega_x, 1, grad_wt, 1);
                         */
                        memcpy(tmp, var, sizeof(double) * data->p);
                        cblas_dscal(data->p, 1 + vtw - utw, tmp, 1);
                        if (data->y[ind] > 0) {
                            double part_wei = 2. * (1 - prob_p) * (xtw - utw);
                            cblas_daxpy(data->p, part_wei, cur_xt, 1, tmp, 1);
                            cblas_daxpy(data->p, -part_wei, posi_x, 1, tmp, 1);
                        } else {
                            double part_wei = 2. * prob_p * (xtw - vtw);
                            cblas_daxpy(data->p, part_wei, cur_xt, 1, tmp, 1);
                            cblas_daxpy(data->p, -part_wei, nega_x, 1, tmp, 1);
                        }
                        // calculate the gradient
                        cblas_daxpy(data->p, 1., tmp, 1, grad_wt, 1);
                        break;
                    case 1:
                        weight = data->y[ind] > 0 ? 2. * (1.0 - prob_p) * (xtw - utw) -
                                                    2. * (1.0 + (vtw - utw)) * (1.0 - prob_p) :
                                 2.0 * prob_p * (xtw - vtw) + 2.0 * (1.0 + (vtw - utw)) * prob_p;
                        cblas_daxpy(data->p, weight, cur_xt, 1, grad_wt, 1); // calculate the gradient
                        break;
                    case 2:
                        if (data->y[ind] > 0) {
                            wei_x = 2. * (1. - prob_p) * (xtw - vtw - 1.0);
                            wei_posi = 2. * (1. - prob_p) * utw - 2. * prob_p * (1. - prob_p) * (utw - vtw);
                            wei_nega = -2. * (1. - prob_p) * xtw + 2. * prob_p * (1. - prob_p) * (utw - vtw);
                        } else {
                            wei_x = 2. * prob_p * (xtw - utw + 1.0);
                            wei_posi = -2. * prob_p * xtw + 2. * prob_p * (1. - prob_p) * (vtw - utw);
                            wei_nega = 2. * prob_p * vtw - 2. * prob_p * (1. - prob_p) * (vtw - utw);
                        }
                        // calculate the gradient
                        cblas_daxpy(data->p, wei_x, cur_xt, 1, grad_wt, 1);
                        cblas_daxpy(data->p, wei_posi, posi_x, 1, grad_wt, 1);
                        cblas_daxpy(data->p, wei_nega, nega_x, 1, grad_wt, 1);
                        break;
                    default:
                        break;
                }
            }
        }
        // wt = wt - eta * grad(wt)
        cblas_daxpy(data->p, -para_c / cur_b_size, grad_wt, 1, re->wt, 1);
        // ell_2 reg. we do not need it in our case.
        if (para_l2_reg != 0.0) {
            cblas_dscal(data->p, 1. / (para_c * para_l2_reg + 1.), re->wt, 1);
        }
        if (operator_id == 0) {
            // k-sparse projection step.
            _hard_thresholding(re->wt, data->p, para_s);
        } else if (operator_id == 1) {
            // to do graph projection.
            double total_prizes = 0.0;
            for (int kk = 0; kk < data->p; kk++) {
                data->proj_prizes[kk] = re->wt[kk] * re->wt[kk];
                total_prizes += data->proj_prizes[kk];
            }
            if (total_prizes >= 1e6) {
                printf("not good, large prizes detected.");
            }
            int s_low = para_s, s_high = para_s + 2;
            head_tail_binsearch(data->edges, data->weights, data->proj_prizes,
                                data->p, data->m, data->g, -1, s_low, s_high,
                                20, GWPruning, 0, data->graph_stat);
            memcpy(grad_wt, re->wt, sizeof(double) * data->p);
            memset(re->wt, 0, sizeof(double) * data->p);
            for (int kk = 0; kk < data->graph_stat->re_nodes->size; kk++) {
                int cur_node = data->graph_stat->re_nodes->array[kk];
                re->wt[cur_node] = grad_wt[cur_node];
            }
        }
        if (paras->record_aucs == 1) { // to evaluate AUC score
            _evaluate_aucs(data, y_pred, re, start_time);
        }
        // at the end of each epoch, we check the early stop condition.
        re->total_iterations++;
        memcpy(re->wt_prev, re->wt, sizeof(double) * (data->p));
    }
    cblas_dscal(re->auc_len, 1. / CLOCKS_PER_SEC, re->rts, 1);
    free(var);
    free(tmp);
    free(y_pred);
    free(nega_x);
    free(posi_x);
    free(grad_wt);
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
    double *x_posi = calloc(data->p, sizeof(double));
    double *x_nega = calloc(data->p, sizeof(double));
    double t_posi = 0.0, t_nega = 0.0;
    double *gt = malloc(sizeof(double) * (data->p));
    double *zt = calloc(data->p, sizeof(double));
    double *gt_square = calloc(data->p, sizeof(double));
    double *true_labels = malloc(sizeof(double) * (data->n_tr + data->n_va + data->n_te));
    double prob_p = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        // 1. example x_i arrives and then we make prediction.
        int ind = data->indices[tt]; // the index of the current training example.
        bool is_posi_y = is_posi(data->y[ind]);
        prob_p = is_posi_y ? (tt * prob_p + 1.) / (tt + 1.) : (tt * prob_p) / (tt + 1.);
        if (data->is_sparse) {
            double xtw = 0.0, ni, pow_gt, wei_x, wei_posi, wei_nega, utw = 0.0, vtw = 0.0;
            // receive a training sample.
            const int *xt_inds = data->x_inds + data->x_poss[ind];
            const double *xt_vals = data->x_vals + data->x_poss[ind];
            // calculate the gradient
            memset(gt, 0, sizeof(double) * (data->p));
            if (is_posi_y) {
                for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                    xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                    x_posi[xt_inds[ii]] += xt_vals[ii];
                }
                t_posi += 1.;
                utw = cblas_ddot(data->p, re->wt, 1, x_posi, 1) / t_posi;
                wei_x = 2. * (1. - prob_p) * (xtw - utw);
                wei_nega = 2. * prob_p * (1. - prob_p) * (utw - vtw - 1.0);
                wei_posi = -wei_nega - wei_x;
            } else {
                for (int ii = 0; ii < data->x_lens[ind]; ii++) {
                    xtw += (re->wt[xt_inds[ii]] * xt_vals[ii]);
                    x_nega[xt_inds[ii]] += xt_vals[ii];
                }
                t_nega += 1.;
                vtw = cblas_ddot(data->p, re->wt, 1, x_nega, 1) / t_nega;
                wei_x = 2. * prob_p * (xtw - vtw);
                wei_posi = 2. * prob_p * (1. - prob_p) * (utw - vtw - 1.0);
                wei_nega = -wei_posi - wei_x;
            }
            for (int kk = 0; kk < data->x_lens[ind]; kk++) {
                gt[xt_inds[kk]] += (wei_x * xt_vals[kk]);
            }
            if (t_posi > 0.0) {
                cblas_daxpy(data->p, wei_posi / t_posi, x_posi, 1, gt, 1);
            }
            if (t_nega > 0.0) {
                cblas_daxpy(data->p, wei_nega / t_nega, x_nega, 1, gt, 1);
            }
            // lazy update the model and make prediction.
            for (int ii = 0; ii < data->p; ii++) {
                ni = gt_square[ii];
                re->wt[ii] = fabs(zt[ii]) <= para_l1 ? 0.0 : -(zt[ii] - sign(zt[ii]) * para_l1)
                                                             / ((para_beta + sqrt(ni)) / para_gamma + para_l2);
            }
            double lr;
            for (int ii = 0; ii < data->p; ii++) {
                ni = gt_square[ii];
                pow_gt = pow(gt[ii], 2.);
                lr = (sqrt(ni + pow_gt) - sqrt(ni)) / para_gamma;
                zt[ii] += gt[ii] - lr * re->wt[ii];
                gt_square[ii] += pow_gt;
            }
            if (tt % 50 == 0 && paras->record_aucs == 1) {
                double t_eval = clock();
                for (int jj = 0; jj < data->n_va; jj++) {
                    int cur_index = data->va_indices[jj];
                    true_labels[jj] = data->y[cur_index];
                    xtw = 0.0;
                    for (int kk = 0; kk < data->x_lens[cur_index]; kk++) {
                        xt_inds = data->x_inds + data->x_poss[cur_index];
                        xt_vals = data->x_vals + data->x_poss[cur_index];
                        xtw += (re->wt[xt_inds[kk]] * xt_vals[kk]);
                    }
                    re->scores[jj] = xtw;
                }
                re->aucs[re->auc_len] = _auc_score(true_labels, re->scores, data->n_va);
                re->rts[re->auc_len++] = (double) clock() - start_time - (clock() - t_eval);
            }
        } else {

        }
    }
    cblas_dscal(re->auc_len, 1. / CLOCKS_PER_SEC, re->rts, 1);
    free(x_nega);
    free(x_posi);
    free(true_labels);
    free(gt);
    free(zt);
    free(gt_square);
}


void _algo_ftrl_auc_fast(Data *data,
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
    double *true_labels = calloc(data->n, sizeof(double));
    double prob_p = 0.0;
    double total_time, run_time, eval_time = 0.0;
    for (int tt = 0; tt < data->n_tr; tt++) {
        // example x_i arrives and then we make prediction.
        // the index of the current training example.
        int ind = data->indices[tt];
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
        if (tt % paras->eval_step == 0) {
            double start_eval = clock();
            re->aucs[re->auc_len] = eval_auc(data, re, true);
            double end_eval = clock();
            // this may not be very accurate.
            eval_time += end_eval - start_eval;
            run_time = (end_eval - start_time) - eval_time;
            re->rts[re->auc_len++] = run_time / CLOCKS_PER_SEC;
            if (paras->verbose > 0) {
                printf("tt: %d auc: %.4f n_va:%d\n",
                       tt, re->aucs[re->auc_len - 1], data->n_va);
            }
        }
    }
    total_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    eval_time /= CLOCKS_PER_SEC;
    run_time = total_time - eval_time;
    printf("\n-------------------------------------------------------\n");
    printf("p: %d num_tr: %d num_va: %d num_te: %d\n",
           data->p, data->n_tr, data->n_va, data->n_te);
    printf("run_time: %.4f eval_time: %.4f total_time: %.4f\n",
           run_time, eval_time, total_time);
    printf("va_auc: %.4f te_auc: %.4f\n",
           eval_auc(data, re, true), eval_auc(data, re, false));
    printf("para_l1: %.4f para_gamma: %.4f sparse_ratio: %.4f\n",
           para_l1, para_gamma, get_sparse_ratio(re->wt, data->p));
    printf("\n-------------------------------------------------------\n");
    free(x_nega);
    free(x_posi);
    free(true_labels);
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
    double *true_labels = malloc(sizeof(double) * (data->n));
    double exe_time = 0.0;
    printf("test6\n");
    printf("n: %d p: %d\n", data->n, data->p);
    for (int tt = 0; tt < data->n_tr; tt++) {
        // 1. example x_i arrives and then we make prediction.
        int ind = data->indices[tt]; // the index of the current training example.
        double weight;
        memset(gt, 0, sizeof(double) * (data->p + 1));
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
        double t_eval = clock();
        if (tt % 10000 == 0 && paras->record_aucs == 1) {
            for (int jj = 0; jj < data->n_va; jj++) {
                int cur_index = data->va_indices[jj];
                true_labels[jj] = data->y[cur_index];
                xtw = 0.0;
                for (int kk = 0; kk < data->x_lens[cur_index]; kk++) {
                    xt_inds = data->x_inds + data->x_poss[cur_index];
                    xt_vals = data->x_vals + data->x_poss[cur_index];
                    xtw += (re->wt[xt_inds[kk]] * xt_vals[kk]);
                }
                re->scores[jj] = xtw;
            }
            re->aucs[re->auc_len] = _auc_score(true_labels, re->scores, data->n_va);
            re->rts[re->auc_len++] = (double) clock() - start_time - (clock() - t_eval);
        }
        exe_time += (clock() - t_eval);
    }
    printf("run time: %.4f\n", (((double) clock() - start_time) - exe_time) / CLOCKS_PER_SEC);
    printf("eval time: %.4f\n", exe_time / CLOCKS_PER_SEC);
    cblas_dscal(re->auc_len, 1. / CLOCKS_PER_SEC, re->rts, 1);
    free(true_labels);
    free(gt);
    free(zt);
    free(gt_square);
}