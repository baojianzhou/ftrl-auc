cmake_minimum_required(VERSION 2.8)
include_directories(algo_wrapper/)
project(ftrl-auc C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-g")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -O3 -fPIC -lpython2.7 -pthread")

set(ROOT "/network/rit/lab/ceashpc/bz383376/opt/")
set(OPENBLAS_INCLUDE "${ROOT}openblas-0.3.1/include")
set(OPENBLAS_LIB "${ROOT}openblas-0.3.1/lib/")
set(NUMPY_INCLUDE "${ROOT}env-python2.7.14/lib/python2.7/site-packages/numpy/core/include/")
set(PYTHON_LIBRARY "${ROOT}python-2.7.14/lib/libpython2.7.so")
set(PYTHON_INCLUDE "${ROOT}python-2.7.14/include/python2.7/")

include_directories(${OPENBLAS_INCLUDE_DIR})
find_package(PythonLibs 2.7 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

add_library(sparse_module SHARED algo_wrapper/main_wrapper.c
        algo_wrapper/online_methods.h algo_wrapper/online_methods.c
        algo_wrapper/fast_pcst.c algo_wrapper/fast_pcst.h algo_wrapper/loss.c algo_wrapper/loss.h)
target_link_libraries(sparse_module python2.7 -std=c11 -Wall -Wextra -O3
        -I${OPENBLAS_INCLUDE} -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE} -L${OPENBLAS_LIB}
        -lm -lpython2.7 -lopenblas -lpthread)