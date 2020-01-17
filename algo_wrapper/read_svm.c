//
// Created by baojian on 1/17/20.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))
static char *line = NULL;
static int max_line_len;

void exit_input_error(int line_num) {
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(1);
}

struct feature_node {
    int index;
    double value;
};
struct problem {
    int l, n;
    double *y;
    struct feature_node **x;
    double bias;            /* < 0 if no bias term */
};
struct problem prob;
struct feature_node *x_space;
double bias;

static char *readline(FILE *input) {
    int len;
    if (fgets(line, max_line_len, input) == NULL) { return NULL; }
    while (strrchr(line, '\n') == NULL) {
        max_line_len *= 2;
        line = (char *) realloc(line, max_line_len);
        len = (int) strlen(line);
        if (fgets(line + len, max_line_len - len, input) == NULL) { break; }
    }
    return line;
}

void read_problem(const char *filename) {
    int max_index, inst_max_index, i;
    size_t elements, j;
    FILE *fp = fopen(filename, "r");
    char *endptr;
    char *idx, *val, *label;
    if (fp == NULL) {
        fprintf(stderr, "can't open input file %s\n", filename);
        exit(1);
    }
    prob.l = 0;
    elements = 0;
    max_line_len = 1024;
    line = Malloc(char, max_line_len);
    while (readline(fp) != NULL) {
        char *p = strtok(line, " \t"); // label
        // features
        while (1) {
            p = strtok(NULL, " \t");
            if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            elements++;
        }
        elements++; // for bias term
        prob.l++;
    }
    rewind(fp);
    prob.bias = bias;
    prob.y = Malloc(double, prob.l);
    prob.x = Malloc(struct feature_node *, prob.l);
    x_space = Malloc(struct feature_node, elements + prob.l);
    max_index = 0;
    j = 0;
    for (i = 0; i < prob.l; i++) {
        inst_max_index = 0; // strtol gives 0 if wrong format
        readline(fp);
        prob.x[i] = &x_space[j];
        label = strtok(line, " \t\n");
        if (label == NULL) // empty line
            exit_input_error(i + 1);
        prob.y[i] = strtod(label, &endptr);
        if (endptr == label || *endptr != '\0')
            exit_input_error(i + 1);
        while (1) {
            idx = strtok(NULL, ":");
            val = strtok(NULL, " \t");
            if (val == NULL)
                break;
            errno = 0;
            x_space[j].index = (int) strtol(idx, &endptr, 10);
            if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i + 1);
            else
                inst_max_index = x_space[j].index;
            errno = 0;
            x_space[j].value = strtod(val, &endptr);
            if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i + 1);
            ++j;
        }
        if (inst_max_index > max_index)
            max_index = inst_max_index;
        if (prob.bias >= 0)
            x_space[j++].value = prob.bias;
        x_space[j++].index = -1;
    }
    if (prob.bias >= 0) {
        prob.n = max_index + 1;
        for (i = 1; i < prob.l; i++)
            (prob.x[i] - 2)->index = prob.n;
        x_space[j - 2].index = prob.n;
    } else
        prob.n = max_index;
    fclose(fp);
}

int main() {
    read_problem("/network/rit/lab/ceashpc/bz383376/data/kdd20/"
                 "01_webspam/webspam_wc_normalized_trigram_small.svm");
}