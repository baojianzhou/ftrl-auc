# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
from os.path import join
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def main():
    import matplotlib.pyplot as plt
    from pylab import rcParams
    plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble'] = '\usepackage{libertine}'
    plt.rcParams["font.size"] = 14
    rcParams['figure.figsize'] = 8, 3.5
    y_scores = []
    w = np.asarray([-1, 1], dtype=float)
    #     +     +     +   +    +       +    +      +     +    +     +     +    -
    x1 = [0.15, 0.1, 0.1, 0.30, 0.34, 0.34, 0.35, 0.36, 0.40, 0.50, 0.15, 0.6, 0.7, 0.3]
    x2 = [0.90, 0.8, 0.2, 0.60, 0.70, 0.80, 0.90, 0.70, 0.80, 0.90, 0.55, 0.7, 0.6, 0.2]
    y_true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    for x, y in zip(x1, x2):
        y_scores.append(np.dot(w, np.asarray([x, y])))
    fig, ax = plt.subplots(1, 2)
    ax[0].fill_between([0.0, 0.2, 0.5, 1.0], [0.0, 0.2, 0.5, 1.0], alpha=0.2, color='b')
    ax[0].fill_between([0.0, 0.0, 1.0, 0], [0.0, 1.0, 1.0, 0], alpha=0.2, color='r')
    ax[0].scatter(x1, x2, alpha=0.8, marker='P', c='r', edgecolors='none', s=80, label='Positive')
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_ylim([0.0, 1.0])
    ax[0].set_xlabel(r"$\displaystyle w_1$")
    ax[0].set_ylabel(r"$\displaystyle w_2$")
    x1 = [0.6, 0.90, 0.40, 0.30, 0.80, 0.90, 0.40, 0.6, 0.30, 0.80, 0.05]
    x2 = [0.4, 0.10, 0.30, 0.40, 0.42, 0.21, 0.17, 0.2, 0.53, 0.86, 0.15]
    for x, y in zip(x1, x2):
        y_scores.append(np.dot(w, np.asarray([x, y])))
    y_true.extend([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    ax[0].scatter(x1, x2, alpha=0.8, marker='X', c='b', edgecolors='none', s=80, label='Negative')

    ax[0].plot([0., 1.], [0., 1.], linewidth=1.5, color='g')
    ax[0].legend(fancybox=True, loc='upper center', framealpha=1.0, ncol=2,
                 bbox_to_anchor=(.5, 1.1), frameon=False, borderpad=0.01,
                 labelspacing=0.01, handletextpad=0.01, markerfirst=True)
    fpr, tpr, threholds = roc_curve(y_true=np.asarray(y_true), y_score=y_scores)
    auc = roc_auc_score(y_true=np.asarray(y_true), y_score=y_scores)
    ax[1].plot(fpr, tpr, alpha=0.8, c='k', linewidth=1.5)
    ax[0].set_xticks([0.2, 0.4, 0.6, 0.8])
    ax[0].set_xticklabels([0.2, 0.4, 0.6, 0.8])
    ax[0].set_yticks([0.2, 0.4, 0.6, 0.8])
    ax[0].set_yticklabels([0.2, 0.4, 0.6, 0.8])
    ax[1].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax[1].set_xlabel("FPR")
    ax[1].set_ylabel("TPR")
    ax[1].set_title('AUC=%.2f' % auc)
    print(fpr)
    ax[1].fill_between(fpr, tpr, alpha=0.2, color='gray')
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    f_name = '/home/baojian/Dropbox/Apps/ShareLaTeX/kdd20-oda-auc/figs/toy-example.pdf'
    fig.savefig(f_name, dpi=600, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()


if __name__ == '__main__':
    main()
