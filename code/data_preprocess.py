# -*- coding: utf-8 -*-
import os
import sys
import numpy as np


root_path = '---configure your path ---'


def data_process_01_news20b(num_trials=10):
    """
    number of classes: 2
    number of samples: 72,309
    number of features: 20,958
    URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    :return:
    """
    np.random.seed(17)
    data = {'file_path': root_path + '01_news20b/raw_news20b',
            'data_name': '01_news20b',
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
    return data


def data_process_02_realsim(num_trials=10):
    """
    number of classes: 2
    number of samples: 72,309
    number of features: 20,958
    URL: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    :return:
    """
    np.random.seed(17)
    data = {'file_path': root_path + '02_real_sim/raw_real_sim',
            'data_name': '02_real_sim',
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
    sys.stdout.flush()
    return data


def data_process_03_rcv1_bin(num_trials=10):
    np.random.seed(17)
    data = {'file_path': root_path + '03_rcv1_bin/raw_rcv1_bin',
            'data_name': '03_rcv1_bin',
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
    sys.stdout.flush()
    return data


def data_process_04_farmads(num_trials=10):
    np.random.seed(17)
    data = {'file_path': root_path + '04_farmads/raw_farmads',
            'data_name': '04_farmads',
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
    print(len(feature_indices), data['p'])
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


def data_process_05_imdb(num_trials=10):
    np.random.seed(17)
    data = {'file_path': root_path + '05_imdb/raw_imdb',
            'data_name': '05_imdb',
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
            if int(items[0]) > 5:
                data['y_tr'].append(1)
            else:
                data['y_tr'].append(-1)
            cur_values = [float(_.split(':')[1]) for _ in items[1:]]
            cur_values = np.asarray(cur_values) / np.linalg.norm(cur_values)
            cur_values = list(cur_values)
            cur_indices = [int(_.split(':')[0]) for _ in items[1:]]
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
    print(len(feature_indices), data['p'])
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


def data_process_06_reviews(num_trials=10):
    np.random.seed(17)
    all_lines = []
    with open(os.path.join(root_path + '06_reviews/posi_reviews'), 'rb') as ff:
        for ind, each_line in enumerate(ff.readlines()):
            all_lines.append(each_line)
    with open(os.path.join(root_path + '06_reviews/nega_reviews'), 'rb') as ff:
        for each_line in ff:
            all_lines.append(each_line)
    data = {'file_path': root_path + '06_reviews/null',
            'data_name': '06_reviews',
            'x_tr_vals': [],
            'x_tr_inds': [],
            'x_tr_poss': [],
            'x_tr_lens': [],
            'y_tr': []}
    # sparse data to make it linear
    prev_posi, min_id, max_id, max_len, feature_indices = 0, np.inf, 0, 0, dict()
    index_feat = 0
    for index, each_line in enumerate(all_lines):
        items = each_line.lstrip().rstrip().split(' ')
        if items[-1].split(':')[1] == 'positive':
            data['y_tr'].append(1)
        else:
            data['y_tr'].append(-1)
        cur_values = [float(_.split(':')[1]) for _ in items[:-1]]
        cur_values = np.asarray(cur_values) / np.linalg.norm(cur_values)
        cur_values = list(cur_values)
        cur_indices = []
        for each_feat in [str(_.split(':')[0]) for _ in items[:-1]]:
            if each_feat not in feature_indices:
                feature_indices[each_feat] = index_feat
                cur_indices.append(index_feat)
                index_feat += 1
            else:
                cur_indices.append(feature_indices[each_feat])
        data['x_tr_vals'].extend(cur_values)
        data['x_tr_inds'].extend(cur_indices)
        data['x_tr_poss'].append(prev_posi)
        data['x_tr_lens'].append(len(cur_indices))
        prev_posi += len(cur_indices)
        if len(cur_indices) != 0:
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
    print(len(feature_indices), data['p'])
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


def data_process_07_avazu(num_trials=10):
    np.random.seed(17)
    # tr: 12,642,186
    # va: 1,953,951
    data = {'file_path': root_path + '07_avazu/raw_avazu',
            'data_name': '07_avazu',
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
            if int(items[0]) > 0:
                data['y_tr'].append(+1)
            else:
                data['y_tr'].append(-1)
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
    data['p'] = 1000000
    if data['p'] != (max_id - min_id + 1):
        print('number of nonzero features: %d' % len(feature_indices))
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
    sys.stdout.flush()
    return data


def main():
    pass


if __name__ == '__main__':
    main()
