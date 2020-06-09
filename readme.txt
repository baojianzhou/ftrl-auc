----------------------------------------------------------------------------------------
Thank you for taking the time to review our code and datasets. This readme.txt describes 
how to run FTRL-AUC and all other baseline methods. 

You will find the following 4 sub-folders and 1 supplement:
    1.  ~/code/: It contains the implementation of FTRL-AUC and all baseline methods.
    2.  ~/datasets/: As shown in our paper, we consider 7 datasets.
    3.  ~/results/: These generated results are presented in our paper.
    4.  ~/figures/: All figures and corresponding data used in our paper.
    5.  the supplement is here: https://github.com/taejj7t1wv/ftrl-auc

Our code is written by Python2.7 and C (with C11 standard). We assume your Operating 
System is GNU/Linux-based. However, if you have MacOS or MacBook, it will be okay. The 
dependencies of our programs are Python2.7, GCC and OpenBLAS-0.3.1.

Notice: Since our code depends on OpenBLAS-0.3.1 and Python2.7, for people who are not 
familiar with C and GNU/Linux environment, it may be difficult to run our code. However, 
we will provide our pip command if the paper got accepted.

----------------------------------------------------------------------------------------
This section is to tell you how to prepare the environment. It has three steps:
    1.  install Python2.7 and GCC (Linux/MacOS/MacBook already have them.)
    2.  install numpy, matplotlib (optional).
    3.  install OpenBLAS-0.3.1: download the source code from
                https://github.com/xianyi/OpenBLAS/releases/tag/v0.3.1

After set up above 3 steps, you are ready to generate a sparse_module.so file. To 
generate sparse_module.so file, you need to config Python2.7 Path and OpenBLAS-0.3.1 
Path correctly by replacing "--- config your path ---" with your own path. 
Please check build.sh (under code/) file to see how to generate it. 

After config it, please run the following command :
	./build.sh
If the sparse_module.so file is generated, then we are done for this section.

----------------------------------------------------------------------------------------
This section describes how to generate the figures reported.
---
To generate Figure-1, 
run >python draw_figure_1_2_9.py and call draw_figure_1() in main()

---
To generate Figure-2,
run >python draw_figure_1_2_9.py and call show_figure_2() in main()
(To generate the results, you need to cal get_single_test() first.

---
To generate Figure-3,
run >python test_on_high_dim all_convege_curves_iter

---
To generate Figure-4,
run >python test_on_high_dim all_para_select

---
To generate Figure-5,
run >python test_on_high_imbalance all_convege_curves_iter

---
To generate Figure-6,
run >python test_on_high_imbalance all_para_select

---
To generate Figure-7,
run >python test_on_high_dim show_curves_huge

---
To generate Figure-8,
run >python test_on_high_dim all_convege_curves

---
To generate Figure-9,
run >python draw_figure_1_2_9.py and call show_figure_9() in main()

---
To generate Figure-10, (Change the imbalance ratio to 0.05)
run >python test_on_high_imbalance all_convege_curves_iter

---
To generate Figure-11, (Change the imbalance ratio to 0.05)
run >python test_on_high_imbalance all_para_select

----------------------------------------------------------------------------------------
This section describes how to generate the tables reported.

---
To generate Table-2:
python test_on_high_dim show_auc 01_news20b
python test_on_high_dim show_auc 02_real_sim
python test_on_high_dim show_auc 03_rcv1_bin
python test_on_high_dim show_auc 04_farmads
python test_on_high_dim show_auc 05_imdb
python test_on_high_dim show_auc 06_reviews

---
To generate Table-3:
python test_on_high_imbalance show_auc 01_news20b
python test_on_high_imbalance show_auc 02_real_sim
python test_on_high_imbalance show_auc 03_rcv1_bin
python test_on_high_imbalance show_auc 04_farmads
python test_on_high_imbalance show_auc 05_imdb
python test_on_high_imbalance show_auc 06_reviews
