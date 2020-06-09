#!/bin/bash
ROOT_PATH="--- config your path ---"
PYTHON_PATH=${ROOT_PATH}python-2.7.14/include/python2.7
NUMPY_PATH=${ROOT_PATH}env-python2.7.14/lib/python2.7/site-packages/numpy/core/include/
OPENBLAS_PATH=${ROOT_PATH}openblas-0.3.1/include/
OPENBLAS_LIB=${ROOT_PATH}openblas-0.3.1/lib/
PYTHON_LIB=${ROOT_PATH}python-2.7.14/lib/
FLAGS="-g -shared  -Wall -fPIC -std=c11 -O3"
INCLUDE="-I${PYTHON_PATH} -I${NUMPY_PATH} -I${OPENBLAS_PATH}"
LIB="-L${OPENBLAS_LIB} -L${PYTHON_LIB}"
SRC="algo_wrapper/main_wrapper.c algo_wrapper/sort.h algo_wrapper/sort.c 
algo_wrapper/online_methods.h algo_wrapper/online_methods.c algo_wrapper/loss.c algo_wrapper/loss.h"
gcc ${FLAGS} ${INCLUDE} ${LIB} ${SRC} -o sparse_module.so -lopenblas -lm
