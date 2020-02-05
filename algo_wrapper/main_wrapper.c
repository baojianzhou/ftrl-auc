#include <Python.h>
#include <numpy/arrayobject.h>
#include "online_methods.h"

static PyObject *test(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    int verbose = 0;
    double sum = 0.0;
    PyArrayObject *x_tr_;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x_tr_)) { return NULL; }
    int n = (int) (x_tr_->dimensions[0]);     // number of samples
    int p = (int) (x_tr_->dimensions[1]);     // number of features
    double *x_tr = PyArray_DATA(x_tr_);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            if (verbose > 0) {
                printf("%.2f ", x_tr[i * p + j]);
            }

            sum += x_tr[i * p + j];
        }
        if (verbose > 0) {
            printf("\n");
        }
    }
    PyObject *results = PyFloat_FromDouble(sum);
    return results;
}

PyObject *get_results(int data_p, AlgoResults *re) {
    PyObject *results = PyTuple_New(6);
    PyObject *wt = PyList_New(data_p);
    PyObject *auc = PyList_New(re->auc_len);
    PyObject *rts = PyList_New(re->auc_len);
    PyObject *iters = PyList_New(re->auc_len);
    PyObject *online_aucs = PyList_New(re->auc_len);
    PyObject *metrics = PyList_New(7);
    for (int i = 0; i < data_p; i++) {
        PyList_SetItem(wt, i, PyFloat_FromDouble(re->wt[i]));
    }
    for (int i = 0; i < re->auc_len; i++) {
        PyList_SetItem(auc, i, PyFloat_FromDouble(re->te_aucs[i]));
        PyList_SetItem(rts, i, PyFloat_FromDouble(re->rts[i]));
        PyList_SetItem(iters, i, PyFloat_FromDouble(re->iters[i]));
        PyList_SetItem(online_aucs, i, PyFloat_FromDouble(re->online_aucs[i]));
    }
    PyList_SetItem(metrics, 0, PyFloat_FromDouble(re->va_auc));
    PyList_SetItem(metrics, 1, PyFloat_FromDouble(re->te_auc));
    PyList_SetItem(metrics, 2, PyFloat_FromDouble(re->nonzero_wt));
    PyList_SetItem(metrics, 3, PyFloat_FromDouble(re->sparse_ratio));
    PyList_SetItem(metrics, 4, PyFloat_FromDouble(re->total_time));
    PyList_SetItem(metrics, 5, PyFloat_FromDouble(re->run_time));
    PyList_SetItem(metrics, 6, PyFloat_FromDouble(re->eval_time));
    PyTuple_SetItem(results, 0, wt);
    PyTuple_SetItem(results, 1, auc);
    PyTuple_SetItem(results, 2, rts);
    PyTuple_SetItem(results, 3, iters);
    PyTuple_SetItem(results, 4, online_aucs);
    PyTuple_SetItem(results, 5, metrics);
    return results;
}

void init_data(Data *data, PyArrayObject *x_tr_vals, PyArrayObject *x_tr_inds, PyArrayObject *x_tr_poss,
               PyArrayObject *x_tr_lens, PyArrayObject *data_y_tr, PyArrayObject *data_perm,
               PyArrayObject *tr_indices, PyArrayObject *va_indices, PyArrayObject *te_indices) {
    data->x_vals = (double *) PyArray_DATA(x_tr_vals);
    data->x_inds = (int *) PyArray_DATA(x_tr_inds);
    data->x_poss = (int *) PyArray_DATA(x_tr_poss);
    data->x_lens = (int *) PyArray_DATA(x_tr_lens);
    data->y = (double *) PyArray_DATA(data_y_tr);
    data->indices = (int *) PyArray_DATA(data_perm);
    data->tr_indices = (int *) PyArray_DATA(tr_indices);
    data->va_indices = (int *) PyArray_DATA(va_indices);
    data->te_indices = (int *) PyArray_DATA(te_indices);
    data->n_tr = (int) tr_indices->dimensions[0];
    data->n_va = (int) va_indices->dimensions[0];
    data->n_te = (int) te_indices->dimensions[0];
    data->n = data->n_tr + data->n_va + data->n_te;
}

void init_global_paras(GlobalParas *paras, PyArrayObject *global_paras) {
    double *arr_paras = (double *) PyArray_DATA(global_paras);
    paras->verbose = (int) arr_paras[0];
    paras->eval_step = (int) arr_paras[1];
    paras->record_aucs = (int) arr_paras[2];
}


static PyObject *wrap_algo_spauc(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_mu, para_l1; //SPAUC has two parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iO!dd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss,
                          &PyArray_Type, &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices,
                          &PyArray_Type, &tr_indices, &PyArray_Type, &va_indices, &PyArray_Type, &te_indices,
                          &data->p, &PyArray_Type, &global_paras, &para_mu, &para_l1)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p, data->n);
    _algo_spauc(data, paras, re, para_mu, para_l1);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}

static PyObject *wrap_algo_solam(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_r; //SOLAM has two parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iO!dd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss,
                          &PyArray_Type, &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices,
                          &PyArray_Type, &tr_indices, &PyArray_Type, &va_indices, &PyArray_Type, &te_indices,
                          &data->p, &PyArray_Type, &global_paras, &para_xi,
                          &para_r)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p, data->n);
    _algo_solam(data, paras, re, para_xi, para_r);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}


static PyObject *wrap_algo_spam(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_xi, para_l1, para_l2; //SPAM has three parameters.
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iiO!ddd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss,
                          &PyArray_Type, &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices,
                          &PyArray_Type, &tr_indices, &PyArray_Type, &va_indices, &PyArray_Type, &te_indices,
                          &data->p, &PyArray_Type, &global_paras, &para_xi, &para_l1, &para_l2)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p, data->n);
    _algo_spam(data, paras, re, para_xi, para_l1, para_l2);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}

static PyObject *wrap_algo_fsauc(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_r, para_g;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iiO!dd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss,
                          &PyArray_Type, &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices,
                          &PyArray_Type, &tr_indices, &PyArray_Type, &va_indices, &PyArray_Type, &te_indices,
                          &data->p, &PyArray_Type, &global_paras, &para_r, &para_g)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p, data->n);
    _algo_fsauc(data, paras, re, para_r, para_g);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}


static PyObject *wrap_algo_ftrl_auc(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_l1, para_l2, para_beta, para_gamma;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iO!dddd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss,
                          &PyArray_Type, &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices,
                          &PyArray_Type, &tr_indices, &PyArray_Type, &va_indices, &PyArray_Type, &te_indices,
                          &data->p, &PyArray_Type, &global_paras,
                          &para_l1, &para_l2, &para_beta, &para_gamma)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p, data->n);
    _algo_ftrl_auc(data, paras, re, para_l1, para_l2, para_beta, para_gamma);
    PyObject *results = get_results(data->p, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}

static PyObject *wrap_algo_ftrl_proximal(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_l1, para_l2, para_beta, para_gamma;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iiO!dddd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss, &PyArray_Type,
                          &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices, &PyArray_Type, &tr_indices,
                          &PyArray_Type, &va_indices, &PyArray_Type, &te_indices,
                          &data->p, &PyArray_Type, &global_paras,
                          &para_l1, &para_l2, &para_beta, &para_gamma)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p + 1, data->n);
    _algo_ftrl_proximal(data, paras, re, para_l1, para_l2, para_beta, para_gamma);
    PyObject *results = get_results(data->p + 1, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}


static PyObject *wrap_algo_rda_l1(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_lambda, para_gamma, para_rho;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iO!ddd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss, &PyArray_Type,
                          &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices, &PyArray_Type, &tr_indices,
                          &PyArray_Type, &va_indices, &PyArray_Type, &te_indices, &data->p, &PyArray_Type,
                          &global_paras, &para_lambda, &para_gamma, &para_rho)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p + 1, data->n);
    _algo_rda_l1(data, paras, re, para_lambda, para_gamma, para_rho);
    PyObject *results = get_results(data->p + 1, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}


static PyObject *wrap_algo_adagrad(PyObject *self, PyObject *args) {
    if (self != NULL) { printf("%zd", self->ob_refcnt); }
    PyArrayObject *x_vals, *x_inds, *x_poss, *x_lens, *y, *global_paras;
    PyArrayObject *indices, *tr_indices, *va_indices, *te_indices;
    Data *data = malloc(sizeof(Data));
    GlobalParas *paras = malloc(sizeof(GlobalParas));
    double para_lambda, para_eta, para_epsilon;
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!O!O!O!iO!ddd",
                          &PyArray_Type, &x_vals, &PyArray_Type, &x_inds, &PyArray_Type, &x_poss, &PyArray_Type,
                          &x_lens, &PyArray_Type, &y, &PyArray_Type, &indices, &PyArray_Type, &tr_indices,
                          &PyArray_Type, &va_indices, &PyArray_Type, &te_indices, &data->p, &PyArray_Type,
                          &global_paras, &para_lambda, &para_eta, &para_epsilon)) { return NULL; }
    init_global_paras(paras, global_paras);
    init_data(data, x_vals, x_inds, x_poss, x_lens, y, indices, tr_indices, va_indices, te_indices);
    AlgoResults *re = make_algo_results(data->p + 1, data->n);
    _algo_adagrad(data, paras, re, para_lambda, para_eta, para_epsilon);
    PyObject *results = get_results(data->p + 1, re);
    free_algo_results(re);
    free(paras);
    free(data);
    return results;
}


// wrap_algo_solam_sparse
static PyMethodDef sparse_methods[] = { // hello_name
        {"c_test",               test,                    METH_VARARGS, "docs"},
        {"c_algo_solam",         wrap_algo_solam,         METH_VARARGS, "docs"},
        {"c_algo_spauc",         wrap_algo_spauc,         METH_VARARGS, "docs"},
        {"c_algo_spam",          wrap_algo_spam,          METH_VARARGS, "docs"},
        {"c_algo_ftrl_auc",      wrap_algo_ftrl_auc,      METH_VARARGS, "docs"},
        {"c_algo_fsauc",         wrap_algo_fsauc,         METH_VARARGS, "docs"},
        {"c_algo_ftrl_proximal", wrap_algo_ftrl_proximal, METH_VARARGS, "docs"},
        {"c_algo_rda_l1",        wrap_algo_rda_l1,        METH_VARARGS, "docs"},
        {"c_algo_adagrad",       wrap_algo_adagrad,       METH_VARARGS, "docs"},
        {NULL, NULL, 0, NULL}};


#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "sparse_module",     /* m_name */
        "This is a module",  /* m_doc */
        -1,                  /* m_size */
        sparse_methods,      /* m_methods */
        NULL,                /* m_reload */
        NULL,                /* m_traverse */
        NULL,                /* m_clear */
        NULL,                /* m_free */
    };
#endif

PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3 // for Python3.x
PyInit_sparse_module(void){
     Py_Initialize();
     import_array(); // In order to use numpy, you must include this!
    return PyModule_Create(&moduledef);
}
#else // for Python2.x
initsparse_module(void) {
    Py_InitModule3("sparse_module", sparse_methods, "some docs for solam algorithm.");
    import_array(); // In order to use numpy, you must include this!
}

#endif

int main() {
    printf("test of main wrapper!\n");
}

