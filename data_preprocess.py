# -*- coding: utf-8 -*-
import os
import sys
import time
import pickle as pkl
import multiprocessing
from os.path import join
from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def data_process_00_sentiment(data_name='00_sentiment'):
    np.random.seed(int(time.time()))
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/00_sentiment/processed_acl/books/'
    data = dict()
    # sparse data to make it linear
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    data['words'] = dict()
    data['total_words'] = 0
    prev_posi, min_id, max_id, max_len = 0, np.inf, 0, 0
    with open(data_path + 'positive.review') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(1.0)
            cur_values, cur_indices = [], []
            for _ in items[:-1]:
                word, count = _.split(':')[0], float(_.split(':')[1])
                if word not in data['words']:
                    data['words'][word] = data['total_words']
                    cur_values.append(count)
                    cur_indices.append(data['total_words'])
                    data['total_words'] += 1
                else:
                    cur_values.append(count)
                    cur_indices.append(data['words'][word])
            xx = np.asarray(cur_values)

            data['x_tr_vals'].extend(list(xx / np.linalg.norm(xx)))
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)

    with open(data_path + 'negative.review') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(-1.0)
            cur_values, cur_indices = [], []
            for _ in items[:-1]:
                word, count = _.split(':')[0], float(_.split(':')[1])
                if word not in data['words']:
                    data['words'][word] = data['total_words']
                    cur_values.append(count)
                    cur_indices.append(data['total_words'])
                    data['total_words'] += 1
                else:
                    cur_values.append(count)
                    cur_indices.append(data['words'][word])
            xx = np.asarray(cur_values)
            data['x_tr_vals'].extend(list(xx / np.linalg.norm(xx)))
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning, all features are zeros! of %d' % index)
    print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = 195887
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['name'] = '00_sentiment'
    for run_id in range(10):
        rand_perm = np.random.permutation(data['n'])
        data['rand_perm_%d' % run_id] = rand_perm
    pkl.dump(data, open(os.path.join(data_path, 'data_00_sentiment.pkl'), 'wb'))


def data_process_01_webspam_small():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/01_webspam/'
    data = dict()
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, feature_index, features = 0, 0, dict()
    with open(data_path + 'webspam_wc_normalized_trigram_small.svm') as f:
        for ind, each_line in enumerate(f.readlines()):
            items = str(each_line).lstrip().rstrip().split(' ')
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) for _ in items[1:]]
            for item in cur_indices:
                if item not in features:
                    features[item] = feature_index
                    feature_index += 1
            data['y_tr'].append(float(items[0]))
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend([features[_] for _ in cur_indices])
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(items) == 1:
                print(each_line)
    print(min(features.keys()), max(features.keys()))
    print(min(features.values()), max(features.values()))
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(features.keys())
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['name'] = '01_webspam'
    for run_id in range(10):
        rand_perm = np.random.permutation(data['n'])
        data['rand_perm_%d' % run_id] = rand_perm
    pkl.dump(data, open(data_path + '01_webspam_10000.pkl', 'wb'))


def data_process_01_webspam_whole():
    data_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/01_webspam/'
    data = dict()
    data['x_tr_vals'] = []
    data['x_tr_inds'] = []
    data['x_tr_poss'] = []
    data['x_tr_lens'] = []
    data['y_tr'] = []
    prev_posi, feature_index, features = 0, 0, dict()
    with open(data_path + 'webspam_wc_normalized_trigram.svm') as f:
        for ind, each_line in enumerate(f.readlines()):
            items = str(each_line).lstrip().rstrip().split(' ')
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) for _ in items[1:]]
            for item in cur_indices:
                if item not in features:
                    features[item] = feature_index
                    feature_index += 1
            data['y_tr'].append(float(items[0]))
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend([features[_] for _ in cur_indices])
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(items) == 1:
                print(each_line)
    print(min(features.keys()), max(features.keys()))
    print(min(features.values()), max(features.values()))
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(features.keys())
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    data['name'] = '01_webspam'
    data['tr_indices'] = np.arange(280000)
    data['te_indices'] = np.arange(280000, 350000)
    pkl.dump(data, open(data_path + '01_webspam_350000.pkl', 'wb'))


def main():
    data_process_01_webspam_whole()


if __name__ == '__main__':
    main()
