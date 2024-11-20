import numpy as np
import torch
import mne

from utils.data_utils import time_cut
from utils.data_augment import DWTA


def data_preprocess_loader(dataset, args):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''


    X = np.load('/mnt/data2/sylyoung/EEG/DeepTransferEEG/data/' + dataset + '/X.npy')
    y = np.load('/mnt/data2/sylyoung/EEG/DeepTransferEEG/data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate, session_trials_arr = None, None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        presum_trials_arr = np.array([[288, 288],
                             [288, 288],
                             [288, 288],
                             [288, 288],
                             [288, 288],
                             [288, 288],
                             [288, 288],
                             [288, 288],
                             [288, 288]])

        if 'seta' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [left_hand, right_hand]
                    cnt = 0
                    for k in inds:
                        if y[k] in ['left_hand', 'right_hand']:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [right_hand, feet]
                    cnt = 0
                    for k in inds:
                        if y[k] in ['right_hand', 'feet']:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
            y = y.astype(int)
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        print('BNCI2014002 downsampled')
        X = mne.filter.resample(X, down=(512 / 250))
        sample_rate = 250

        presum_trials_arr = np.array([[100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60],
                             [100, 60]])

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [right_hand, feet]
                cnt = 0
                for k in inds:
                    if y[k] in ['right_hand', 'feet']:
                        indices.append(k)
                        cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'right_hand')] = 0
        y[np.where(y == 'feet')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014004':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

        presum_trials_arr = np.array([[120, 120, 160, 160, 160],
                               [120, 120, 160, 120, 160],
                               [120, 120, 160, 160, 160],
                               [120, 140, 160, 160, 160],
                               [120, 140, 160, 160, 160],
                               [120, 120, 160, 160, 160],
                               [120, 120, 160, 160, 160],
                               [160, 120, 160, 160, 160],
                               [120, 120, 160, 160, 160]])

        if 'seta' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [left_hand, right_hand]
                    cnt = 0
                    for k in inds:
                        if y[k] in ['left_hand', 'right_hand']:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        if 'setb' in args.method:
            print('BNCI2015001 downsampled')
            X = mne.filter.resample(X, down=(512 / 250))
            sample_rate = 250

        presum_trials_arr = [[200, 200],
                            [200, 200],
                            [200, 200],
                            [200, 200],
                            [200, 200],
                            [200, 200],
                            [200, 200],
                            [200, 200, 200],
                            [200, 200, 200],
                            [200, 200, 200],
                            [200, 200, 200],
                            [200, 200]]
        indices = []
        trials_arr = []
        for i in range(num_subjects):
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i][j]) + np.sum(np.sum([presum_trials_arr[k]]) for k in range(i))
                # only use two classes [right_hand, feet]
                cnt = 0
                for k in inds:
                    if y[k] in ['right_hand', 'feet']:
                        indices.append(k)
                        cnt += 1
                # fake cnt for split, due to unequal number of sessions across subjects
                trials_arr.append(cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'right_hand')] = 0
        y[np.where(y == 'feet')] = 1
        y = y.astype(int)
    elif dataset == 'Weibo2014':
        paradigm = 'MI'
        num_subjects = 10
        sample_rate = 200
        ch_num = 60

        print('Weibo2014 upsampled')
        X = mne.filter.resample(X, up=(250 / 200))
        sample_rate = 250

        #presum_trials_arr = np.array([[237], [237], [237], [237], [237], [237], [237], [237], [237], [237]])
        presum_trials_arr = np.array([[554], [554], [554], [554], [554], [554], [554], [554], [554], [554]])

        if 'seta' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [left_hand, right_hand]
                    cnt = 0
                    for k in inds:
                        if y[k] in ['left_hand', 'right_hand']:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [right_hand, feet]
                    cnt = 0
                    for k in inds:
                        if y[k] in ['right_hand', 'feet']:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
            y = y.astype(int)
    elif dataset == 'Zhou2016':
        paradigm = 'MI'
        num_subjects = 4
        sample_rate = 250
        ch_num = 14

        presum_trials_arr = np.array([[179, 150, 150],
                               [150, 135, 150],
                               [150, 151, 150],
                               [135, 150, 150]])

        if 'seta' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [left_hand, right_hand]
                    cnt = 0
                    for k in inds:
                        if y[k] in ['left_hand', 'right_hand']:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [right_hand, feet]
                    cnt = 0
                    for k in inds:
                        if y[k] in ['right_hand', 'feet']:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
            y = y.astype(int)

    elif 'MI1' in dataset:
        paradigm = 'MI'
        num_subjects = 5
        sample_rate = 250
        ch_num = 59
        class_num = 2

        # print('MI1 downsampled')
        # X = mne.filter.resample(X, down=4)
        # sample_rate = int(sample_rate // 4)

        # use only left and right hand 5 subjects, remove S0 and S5
        # X = np.concatenate([X[200:1000], X[1200:]])
        # y = np.concatenate([y[200:1000], y[1200:]])
        # num_subjects = 5

        presum_trials_arr = np.array([[200], [200], [200], [200], [200]])

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [left_hand, right_hand]
                cnt = 0
                for k in inds:
                    if y[k] in [0, 1]:
                        indices.append(k)
                        cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 0)] = 0
        y[np.where(y == 1)] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

        # time cut
        X = time_cut(X, cut_percentage=0.8)
        presum_trials_arr = np.array([[4200], [4200], [4200], [4200], [4200], [4200], [4200], [4200]])

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [left_hand, right_hand]
                cnt = 0
                for k in inds:
                    if y[k] in ['NonTarget', 'Target']:
                        indices.append(k)
                        cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
        class_num = 2

        presum_trials_arr = np.array([[576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576],
                                       [576, 576, 576]])

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [left_hand, right_hand]
                cnt = 0
                for k in inds:
                    if y[k] in ['NonTarget', 'Target']:
                        indices.append(k)
                        cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)

    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
        class_num = 2

        presum_trials_arr = np.array([[5400], [5400], [1800], [1800], [1800], [1800], [1800], [1800], [1800], [1800]])

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [left_hand, right_hand]
                cnt = 0
                for k in inds:
                    if y[k] in ['NonTarget', 'Target']:
                        indices.append(k)
                        cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'EPFLP300':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 32
        class_num = 2

        #print('EPFLP300 downsampled')
        #X = mne.filter.resample(X, down=8)
        #sample_rate = int(sample_rate // 8)

        X = X[:, :, :206]

        trials_arr = np.array([[850,816,825,813],
                                [793,823,818,850],
                                [820,848,831,805],
                                [777,843,831,823],
                                [802,826,810,818],
                                [850,816,817,848],
                                [823,856,817,799],
                                [793,795,862,818]])

        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'HighGamma':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 500
        ch_num = 128
        class_num = 4

        print('HighGamma downsampled')
        X = mne.filter.resample(X, down=2)
        sample_rate = int(sample_rate // 2)

        # all classes
        presum_trials_arr = np.array([[320], [813], [880], [897], [720], [880], [880], [654], [880], [880], [880], [880], [800], [880]])

        # [left_hand, right_hand, feet, rest]
        if 'seta' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [left_hand, right_hand]
                    cnt = 0
                    for k in inds:
                        if y[k] in [0, 1]:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 0)] = 0
            y[np.where(y == 1)] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                new_subject_cnt = []
                for j in range(len(presum_trials_arr[i])):
                    inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                    # only use two classes [right_hand, feet]
                    cnt = 0
                    for k in inds:
                        if y[k] in [1, 2]:
                            indices.append(k)
                            cnt += 1
                    new_subject_cnt.append(cnt)
                trials_arr.append(new_subject_cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 1)] = 0
            y[np.where(y == 2)] = 1
            y = y.astype(int)
    elif 'MI1' in dataset:
        paradigm = 'MI'
        num_subjects = 7
        sample_rate = 1000
        ch_num = 59
        class_num = 2

        print('MI1 downsampled')
        X = mne.filter.resample(X, down=4)
        sample_rate = int(sample_rate // 4)

        # use only left and right hand 5 subjects, remove S0 and S5
        X = np.concatenate([X[200:1000], X[1200:]])
        y = np.concatenate([y[200:1000], y[1200:]])
        num_subjects = 5

        presum_trials_arr = np.array([[200], [200], [200], [200], [200]])

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [left_hand, right_hand]
                cnt = 0
                for k in inds:
                    if y[k] in [0, 1]:
                        indices.append(k)
                        cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        y = y[indices]
        y[np.where(y == 0)] = 0
        y[np.where(y == 1)] = 1
        y = y.astype(int)
    elif 'Kaneshiro2015' in dataset:
        paradigm = 'visual'
        num_subjects = 10
        sample_rate = 62.5
        ch_num = 124
        class_num = 4

        print('Kaneshiro2015 upsampled')
        X = mne.filter.resample(X, up=250 / 62.5)
        sample_rate = 250

        presum_trials_arr = np.array([[2161, 2161],
                                      [2160, 2160],
                                      [2160, 2161],
                                      [2160, 2160],
                                      [2160, 2160],
                                      [2160, 2160],
                                      [2161, 2161],
                                      [2160, 2160],
                                      [2160, 2161],
                                      [2160, 2160]])

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                cnt = 0
                for k in inds:
                    indices.append(k)
                    cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        # cut to 0.5s? how did we have 128 timesample points?
        X = X[:, :, :126]
        y = y[indices]
        y = y.astype(int)
    elif 'ThingsEEG' in dataset:
        paradigm = 'visual'
        num_subjects = 10
        sample_rate = 100
        ch_num = 17
        class_num = 4

        print('ThingsEEG upsampled')
        X = mne.filter.resample(X, up=250 / 100)
        sample_rate = 250

        presum_trials_arr = np.array([[66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000],
                                      [66160, 16000]])
        indices = []
        trials_arr = []
        for i in range(num_subjects):
            new_subject_cnt = []
            for j in range(len(presum_trials_arr[i])):
                inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
                cnt = 0
                for k in inds:
                    indices.append(k)
                    cnt += 1
                new_subject_cnt.append(cnt)
            trials_arr.append(new_subject_cnt)
        X = X[indices]
        y = y[indices]
        y = y.astype(int)

    print('before strip:', X.shape)
    if 'set' in args.method:
        # strip to first 1000 points
        X = X[:, :, :1000]
        print('striping to first 1000 points')
    print('after strip:', X.shape)

    print('data shape:', X.shape, ' labels shape:', y.shape)
    print('trials arr:', trials_arr)
    return X, y, trials_arr


def data_preprocess_loader_hg(dataset, args, subj_id):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''


    X = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/X.npy')
    y = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate, session_trials_arr = None, None, None, None

    if dataset == 'HighGamma':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 250
        ch_num = 128
        class_num = 4

        # all classes
        presum_trials_arr = np.array([[320], [813], [880], [897], [720], [880], [880], [654], [880], [880], [880], [880], [800], [880]])

        # [left_hand, right_hand, feet, rest]
        if 'seta' in args.method:
            indices = []
            new_subject_cnt = []
            i = subj_id
            inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
            # only use two classes [left_hand, right_hand]
            cnt = 0
            for j in inds:
                if y[j] in [0, 1]:
                    indices.append(j)
                    cnt += 1
            new_subject_cnt.append(cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 0)] = 0
            y[np.where(y == 1)] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            indices = []
            new_subject_cnt = []
            i = subj_id
            inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
            # only use two classes [right_hand, feet]
            cnt = 0
            for j in inds:
                if y[j] in [1, 2]:
                    indices.append(j)
                    cnt += 1
            new_subject_cnt.append(cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 1)] = 0
            y[np.where(y == 2)] = 1
            y = y.astype(int)
        # # for all 4 classes, for SML, not for HTL
        # elif 'SML' in args.method:
        #     indices = []
        #     new_subject_cnt = []
        #     i = subj_id
        #     inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
        #     # only use two classes [right_hand, feet]
        #     cnt = 0
        #     for j in inds:
        #         if y[j] in [0, 1, 2, 3]:
        #             indices.append(j)
        #             cnt += 1
        #     new_subject_cnt.append(cnt)
        #     X = X[indices]
        #     y = y[indices]
        #     y[np.where(y == 0)] = 0
        #     y[np.where(y == 1)] = 1
        #     y[np.where(y == 2)] = 2
        #     y[np.where(y == 3)] = 3
        #     y = y.astype(int)
        # print('HighGamma downsampled')
        # X = mne.filter.resample(X, down=2)
        # sample_rate = int(sample_rate // 2)
        return X, y, new_subject_cnt


def data_preprocess_loader_thingseeg(dataset, args, subj_id):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''


    X = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/X.npy')
    y = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate, session_trials_arr = None, None, None, None

    if dataset == 'ThingsEEG':
        paradigm = 'visual'
        num_subjects = 10
        sample_rate = 100
        ch_num = 17
        class_num = 4

        # presum_trials_arr = np.array([[66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000],
        #                               [66160, 16000]])

        presum_trials_arr = np.array([[18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000],
                                      [18320, 16000]])

        indices_s1 = []
        indices_s2 = []
        i = subj_id
        j = 0
        inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
        for k in inds:
            indices_s1.append(k)
        j = 1
        inds = np.arange(presum_trials_arr[i, j]) + np.sum(presum_trials_arr[:i, :])
        for k in inds:
            indices_s2.append(k)

        X_s1 = X[indices_s1]
        y_s1 = y[indices_s1]
        y_s1 = y_s1.astype(int)
        X_s2 = X[indices_s2]
        y_s2 = y[indices_s2]
        y_s2 = y_s2.astype(int)

        print('ThingsEEG upsampled')
        X_s1 = mne.filter.resample(X_s1, up=250 / 100)
        X_s2 = mne.filter.resample(X_s2, up=250 / 100)
        sample_rate = 250
        print('after resample X_s1, X_s2', X_s1.shape, X_s2.shape)
        # trial was -0.2 to 0.8s w.r.t. stimulus onset
        # cut to 0.0 to 0.5s for 0.5s segment trial
        # [250 * 0.2: 250 * 0.7]
        X_s1 = X_s1[:, :, 50:176]
        X_s2 = X_s2[:, :, 50:176]
        print('after cut X_s1, X_s2', X_s1.shape, X_s2.shape)
        return X_s1, y_s1, X_s2, y_s2


def data_preprocess_loader_Grootswagers2019(dataset, args, subj_id):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    num_subjects, paradigm, sample_rate, session_trials_arr = None, None, None, None

    if dataset == 'Grootswagers2019':
        paradigm = 'visual'
        num_subjects = 16
        sample_rate = 100
        ch_num = 63
        class_num = 4

        presum_trials_arr = np.array([[8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000], [8000, 8000]])

        X1 = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/X' + str(subj_id) + '-0.npy', allow_pickle=True)
        y1 = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/y' + str(subj_id) + '-0.npy', allow_pickle=True)
        print(X1.shape, y1.shape)

        X_s1 = X1
        y_s1 = y1
        y_s1 = y_s1.astype(int)

        X2 = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/X' + str(subj_id) + '-1.npy', allow_pickle=True)
        y2 = np.load('/home/sylyoung/DeepTransferEEG/data/' + dataset + '/y' + str(subj_id) + '-1.npy', allow_pickle=True)
        print(X2.shape, y2.shape)

        X_s2 = X2
        y_s2 = y2
        y_s2 = y_s2.astype(int)

        print('Grootswagers2019 upsampled')
        X_s1 = mne.filter.resample(X_s1, up=250 / 100)
        X_s2 = mne.filter.resample(X_s2, up=250 / 100)
        print('after resample X_s1, X_s2', X_s1.shape, X_s2.shape)
        # trial was -0.2 to 0.8s w.r.t. stimulus onset
        # cut to 0.0 to 0.5s for 0.5s segment trial
        # [250 * 0.2: 250 * 0.7]
        X_s1 = X_s1[:, :, 50:176]
        X_s2 = X_s2[:, :, 50:176]
        print('after cut X_s1, X_s2', X_s1.shape, X_s2.shape)
        sample_rate = 250
        return X_s1, y_s1, X_s2, y_s2


def load_data(data_name, data_list, align, args=None, aug=None):
    try:
        if 'cmch' in args.method:
            append = '/cmch'
            if args.paradigm == 'ERP' and 'c4' in args.method:
                append = '/min4ch'
        else:
            append = ''
    except:
        append = ''

    if data_name == 'HighGamma':
        all_X = []
        all_y = []
        for i in range(14):
            if align:
                if data_list is not None:
                    X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + append + '/X-EA' + str(i) + '.npy')
                else:
                    X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + append  + '/X-EA' + str(i) + '.npy')
            else:
                if data_list is not None:
                    X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + append  + '/X' + str(i) + '.npy')
                else:
                    X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + append  + '/X' + str(i) + '.npy')
            y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + append  + '/y' + str(i) + '.npy')
            all_X.append(X)
            all_y.append(y)
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
    elif data_name == 'ThingsEEG' or data_name == 'Grootswagers2019':
        all_X = []
        all_y = []
        if data_name == 'ThingsEEG':
            N = 10
        elif data_name == 'Grootswagers2019':
            N = 16
        for i in range(N):
            if align:
                if data_list is not None:
                    X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/' + (data_list) + '/X-EA' + str(i) + '.npy')
                else:
                    X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/X-EA' + str(i) + '.npy')
            else:
                if data_list is not None:
                    X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/' + (data_list) + '/X' + str(i) + '.npy')
                else:
                    X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/X' + str(i) + '.npy')
            y = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/' + (data_list) + '/y' + str(i) + '.npy')
            all_X.append(X)
            all_y.append(y)
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
    elif data_name == 'Kaneshiro2015':
        if align:
            if data_list is not None:
                X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/' + (data_list) + '/X-EA.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/X-EA.npy')
        else:
            if data_list is not None:
                X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/' + (data_list) + '/X.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/X.npy')
        if data_list is not None:
            y = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/' + (data_list) + '/y.npy')
        else:
            y = np.load('/mnt/data2/sylyoung/EEG/' + data_name + '/y.npy')
    else:
        if align:
            if data_list is not None:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + append + '/X-EA.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name  + append + '/X-EA.npy')
        else:
            if data_list is not None:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + append + '/X.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name  + append + '/X.npy')
        if data_list is not None:
            y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + append + '/y.npy')
        else:
            y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + append + '/y.npy')
    if data_name == 'EPFLP300':
        X = np.real(X)
    print(X.shape, y.shape)

    if aug is not None:
        if aug == 'DWT-A':
            # randomly select two samples from the same class, regardless of whether these two samples
            # are from the same or different subjects
            class_num = len(np.unique(y))
            all_X = []
            all_y = []
            for k in range(class_num):
                inds_thisclass = np.where(y == k)[0]
                X_thisclass = X[inds_thisclass]
                y_thisclass = y[inds_thisclass]
                # append original
                all_X.append(X_thisclass)
                all_y.append(y_thisclass)

                inds_split = np.arange(len(X_thisclass))
                # shuffle the inds before division
                #np.random.shuffle(inds_split)
                try:
                    inds_split_two = np.split(inds_split, 2)
                except:
                    inds_split_two = []
                    inds_split_two.append(inds_split[:len(inds_split + 1) // 2])
                    inds_split_two.append(inds_split[len(inds_split + 1) // 2:])
                inds_left, inds_right = inds_split_two[0], inds_split_two[1]
                if len(inds_left) != len(inds_right):
                    if len(inds_left) > len(inds_right):
                        inds_right = np.concatenate((inds_right, np.expand_dims(inds_right[0], axis=0)))
                    else:
                        inds_left = np.concatenate((inds_left, np.expand_dims(inds_left[0], axis=0)))

                x_0_aug, x_1_aug = DWTA(X_thisclass[inds_left], X_thisclass[inds_right])
                # append augmented
                all_X.append(x_0_aug)
                all_X.append(x_1_aug)
                all_y.append(np.repeat(k, len(inds_left) * 2))
            X = np.concatenate(all_X)
            y = np.concatenate(all_y)
            print('augmented data:', X.shape, y.shape)

    return X, y


def load_data_target(args):
    try:
        if 'cmch' in args.method or 'tgtmin' in args.method:
            append = '/cmch'
            if args.paradigm == 'ERP' and 'c4' in args.method:
                append = '/min4ch'
        else:
            append = ''
    except:
        append = ''

    if args.tgtdata == 'HighGamma':
        all_X = []
        all_y = []
        for i in range(14):
            if args.align:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + '/' + (args.data_list) + append + '/X-EA' + str(i) + '.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + '/' + (args.data_list) + append + '/X' + str(i) + '.npy')
            y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + '/' + (args.data_list) + append + '/y' + str(i) + '.npy')
            all_X.append(X)
            all_y.append(y)
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
    elif args.tgtdata == 'ThingsEEG' or args.tgtdata == 'Grootswagers2019' or args.tgtdata == 'ThingsEEG-Avg':
        all_X = []
        all_y = []
        if 'b1conv' in args.method:
            num_subjects = args.tgtN
        else:
            num_subjects = args.N
        for i in range(num_subjects):
            if args.align:
                X = np.load('/mnt/data2/sylyoung/EEG/' + args.tgtdata + '/' + (args.data_list) + '/X-EA' + str(i) + '.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/' + args.tgtdata + '/' + (args.data_list) + '/X' + str(i) + '.npy')
            y = np.load('/mnt/data2/sylyoung/EEG/' + args.tgtdata + '/' + (args.data_list) + '/y' + str(i) + '.npy')
            all_X.append(X)
            all_y.append(y)
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
    elif hasattr(args, 'data_list') and args.data_list == 'visual':
        if args.align:
            X = np.load('/mnt/data2/sylyoung/EEG/' + args.tgtdata + '/' + (args.data_list) + '/X-EA.npy')
        else:
            X = np.load('/mnt/data2/sylyoung/EEG/' + args.tgtdata + '/' + (args.data_list) + '/X.npy')
        y = np.load('/mnt/data2/sylyoung/EEG/' + args.tgtdata + '/' + (args.data_list) + '/y.npy')
    else:
        if args.align:
            if hasattr(args, 'data_list'):
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + '/' + (args.data_list) + append + '/X-EA.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + append + '/X-EA.npy')
        else:
            if hasattr(args, 'data_list'):
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + '/' + (args.data_list) + append + '/X.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + append + '/X.npy')
        if hasattr(args, 'data_list'):
            y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + '/' + (args.data_list) + append + '/y.npy')
        else:
            y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + args.tgtdata + append + '/y.npy')
    #print(X.shape, y.shape)
    if args.tgtdata == 'EPFLP300' or args.tgtdata == 'Kaneshiro2015':
        X = np.real(X)

    return X, y


def load_data_test(data_name, data_list, align):

    if data_name == 'HighGamma-4':
        all_X = []
        all_y = []
        for i in range(14):
            if align:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + '/X-EA' + str(i) + '.npy')
            else:
                X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + '/X' + str(i) + '.npy')
            y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + '/y' + str(i) + '.npy')
            all_X.append(X)
            all_y.append(y)
        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
    else:
        if align:
            X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + '/X-EA.npy')
        else:
            X = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + '/X.npy')
        y = np.load('/mnt/data2/sylyoung/EEG/BNCI/' + data_name + '/' + (data_list) + '/y.npy')
    print(X.shape, y.shape)

    return X, y