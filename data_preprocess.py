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

root_path = '/network/rit/lab/ceashpc/bz383376/data/kdd20/'


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


def data_process_01_webspam(num_trials=10):
    np.random.seed(17)
    start_time = time.time()
    data = {'file_path': '/network/rit/lab/ceashpc/bz383376/data/kdd20/01_webspam/',
            'data_name': '01_webspam',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(data['file_path'] + 'raw_webspam') as f:
        for ind, each_line in enumerate(f.readlines()):
            items = str(each_line).lstrip().rstrip().split(' ')
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            for item in cur_indices:
                if item not in feature_indices:
                    feature_indices[item] = ''
            data['y_tr'].append(float(items[0]))
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % ind)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = max_id - min_id + 1
    if len(feature_indices.keys()) != data['p']:
        print('some features are all zeros.')
    data['k'] = np.ceil(len(data['x_tr_vals']) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    print(time.time() - start_time)
    return data


def data_process_02_news20b(num_trials=10):
    """
    number of classes: 2
    number of samples: 72,309
    number of features: 20,958
    URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    :return:
    """
    np.random.seed(int(time.time()))
    data = {'file_path': root_path + '02_news20b/raw_news20b',
            'data_name': '02_news20b',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(os.path.join(data['file_path']), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(feature_indices)
    data['k'] = np.ceil(len(data['x_tr_vals']) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    pkl.dump(data, open(os.path.join(root_path, '02_news20b/processed_02_news20b.pkl'), 'wb'))


def data_process_03_realsim(num_trials=10):
    """
    number of classes: 2
    number of samples: 72,309
    number of features: 20,958
    URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    :return:
    """
    np.random.seed(int(time.time()))
    data = {'file_path': root_path + '03_real_sim/raw_real_sim',
            'data_name': '03_realsim',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(os.path.join(data['file_path']), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(feature_indices)
    data['k'] = np.ceil(len(data['x_tr_vals']) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    pkl.dump(data, open(os.path.join(root_path, '03_real_sim/processed_03_real_sim.pkl'), 'wb'))


def data_process_04_webspam_u(num_trials=10):
    """
    number of classes: 2
    number of samples: 72,309
    number of features: 20,958
    URLs:   https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
            https://www.cc.gatech.edu/projects/doi/WebbSpamCorpus.html
    :return:
    """
    np.random.seed(int(time.time()))
    data = {'file_path': root_path + '04_webspam_u/raw_webspam_u',
            'data_name': '04_webspam_u',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(os.path.join(data['file_path']), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(feature_indices)
    data['k'] = np.ceil(float(len(data['x_tr_vals'])) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    print(len(data['x_tr_vals']) / float(data['n']))
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    pkl.dump(data, open(os.path.join(root_path, '04_webspam_u/processed_04_webspam_u.pkl'), 'wb'))


def data_process_05_rcv1_binary(num_trials=10):
    np.random.seed(int(time.time()))
    data = {'file_path': root_path + '05_rcv1_bin/raw_rcv1_bin',
            'data_name': '05_rcv1_bin',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(os.path.join(data['file_path']), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = max_id - min_id + 1
    if len(feature_indices) != (max_id - min_id):
        print('some features are all zeros.')
    assert data['p'] == (max_id - min_id + 1)  # TODO be careful
    data['k'] = np.ceil(len(data['x_tr_vals']) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    print('number of num_nonzeros: %d' % data['num_nonzeros'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    pkl.dump(data, open(os.path.join(root_path, '05_rcv1_bin/processed_05_rcv1_bin.pkl'), 'wb'))


def data_process_06_pcmac(num_trials=10):
    np.random.seed(int(time.time()))
    data = {'file_path': root_path + '06_pcmac/raw_pcmac',
            'data_name': '06_pcmac',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(os.path.join(data['file_path']), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = len(feature_indices)
    assert data['p'] == (max_id - min_id + 1)
    data['k'] = np.ceil(len(data['x_tr_vals']) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    print('number of num_nonzeros: %d' % data['num_nonzeros'])
    print('k: %d' % data['k'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    pkl.dump(data, open(os.path.join(root_path, '06_pcmac/processed_06_pcmac.pkl'), 'wb'))


def data_process_07_url(num_trials=10):
    np.random.seed(17)
    data = {'file_path': root_path + '07_url/raw_url',
            'data_name': '07_url',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(os.path.join(data['file_path']), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_values = np.asarray(cur_values) / np.linalg.norm(cur_values)
            cur_values = list(cur_values)
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = (max_id - min_id + 1)
    if data['p'] != (max_id - min_id + 1):
        print('number of nonzero features: %d' % len(feature_indices))
        assert data['p'] == (max_id - min_id + 1)
    data['k'] = np.ceil(len(data['x_tr_vals']) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    print('number of num_nonzeros: %d' % data['num_nonzeros'])
    print('k: %d' % data['k'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        print(all_indices[:5])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    sys.stdout.flush()
    return data


def data_process_08_farmads(num_trials=10):
    np.random.seed(17)
    data = {'file_path': root_path + '08_farmads/raw_farmads',
            'data_name': '08_farmads',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    with open(os.path.join(data['file_path']), 'rb') as f:
        for index, each_line in enumerate(f.readlines()):
            items = each_line.lstrip().rstrip().split(' ')
            data['y_tr'].append(int(items[0]))
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_values = np.asarray(cur_values) / np.linalg.norm(cur_values)
            cur_values = list(cur_values)
            cur_indices = [int(_.split(':')[0]) - 1 for _ in items[1:]]
            data['x_tr_vals'].extend(cur_values)
            data['x_tr_inds'].extend(cur_indices)
            data['x_tr_poss'].append(prev_posi)
            data['x_tr_lens'].append(len(cur_indices))
            prev_posi += len(cur_indices)
            if len(cur_indices) != 0:
                for feature in cur_indices:
                    feature_indices[feature] = ''
                min_id = min(min(cur_indices), min_id)
                max_id = max(max(cur_indices), max_id)
                max_len = max(len(cur_indices), max_len)
            else:
                print('warning for sample %d: all features are zeros!' % index)
        print(min_id, max_id, max_len)
    data['x_tr_vals'] = np.asarray(data['x_tr_vals'], dtype=float)
    data['x_tr_inds'] = np.asarray(data['x_tr_inds'], dtype=np.int32)
    data['x_tr_lens'] = np.asarray(data['x_tr_lens'], dtype=np.int32)
    data['x_tr_poss'] = np.asarray(data['x_tr_poss'], dtype=np.int32)
    data['y_tr'] = np.asarray(data['y_tr'], dtype=float)
    data['n'] = len(data['y_tr'])
    data['p'] = (max_id - min_id + 1)
    if data['p'] != (max_id - min_id + 1):
        print('number of nonzero features: %d' % len(feature_indices))
        assert data['p'] == (max_id - min_id + 1)
    data['k'] = np.ceil(len(data['x_tr_vals']) / float(data['n']))
    assert len(np.unique(data['y_tr'])) == 2  # we have total 2 classes.
    data['num_posi'] = len([_ for _ in data['y_tr'] if _ > 0])
    data['num_nega'] = len([_ for _ in data['y_tr'] if _ < 0])
    data['posi_ratio'] = float(data['num_posi']) / float(data['num_nega'])
    data['num_nonzeros'] = len(data['x_tr_vals'])
    print('number of positive: %d' % len([_ for _ in data['y_tr'] if _ > 0]))
    print('number of negative: %d' % len([_ for _ in data['y_tr'] if _ < 0]))
    print('number of num_nonzeros: %d' % data['num_nonzeros'])
    print('k: %d' % data['k'])
    for _ in range(num_trials):
        all_indices = np.random.permutation(data['n'])
        print(all_indices[:5])
        data['trial_%d_all_indices' % _] = np.asarray(all_indices, dtype=np.int32)
        assert data['n'] == len(data['trial_%d_all_indices' % _])
        tr_indices = all_indices[:int(len(all_indices) * 4. / 6.)]
        data['trial_%d_tr_indices' % _] = np.asarray(tr_indices, dtype=np.int32)
        va_indices = all_indices[int(len(all_indices) * 4. / 6.):int(len(all_indices) * 5. / 6.)]
        data['trial_%d_va_indices' % _] = np.asarray(va_indices, dtype=np.int32)
        te_indices = all_indices[int(len(all_indices) * 5. / 6.):]
        data['trial_%d_te_indices' % _] = np.asarray(te_indices, dtype=np.int32)
        n_tr = len(data['trial_%d_tr_indices' % _])
        n_va = len(data['trial_%d_va_indices' % _])
        n_te = len(data['trial_%d_te_indices' % _])
        assert data['n'] == (n_tr + n_va + n_te)
    sys.stdout.flush()
    return data


def main():
    data_process_01_webspam()


if __name__ == '__main__':
    main()
