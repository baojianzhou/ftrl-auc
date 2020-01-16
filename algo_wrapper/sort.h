//
// Created by baojian on 8/29/19.
//

#ifndef SPARSE_AUC_SORT_H
#define SPARSE_AUC_SORT_H

#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

// descending sort an array
bool arg_sort_descend(const double *x, int *sorted_indices, int x_len);

// ascending sort an array
bool arg_sort_ascend(const double *x, int *sorted_indices, int x_len);

// descending sort array by its magnitude
bool arg_magnitude_sort_descend(const double *x, int *sorted_indices, int x_len);

// ascending sort array by its magnitude
bool arg_magnitude_sort_ascend(const double *x, int *sorted_indices, int x_len);

/**
 * top-k indices of largest magnitude
 * @param x
 * @param sorted_set
 * @param k
 * @param x_len
 * @return
 */
bool arg_magnitude_sort_top_k(const double *x, int *sorted_set, int k, int x_len);

// np.lexsort((y, x)) sort by x then by y
bool lex_sort_descend(const double *y, const double *x, int *sorted_set, int x_len);

// np.lexsort((y, x)) sort by x then by y
bool lex_sort_ascend(const double *y, const double *x, int *sorted_set, int x_len);

#endif //SPARSE_AUC_SORT_H
