# -*- coding: utf-8 -*-
# @Time    : 2023/7/11
# @Author  : Siyang Li
# @File    : dataloader.py
import numpy as np
import torch
import mne
import sys
from sklearn import preprocessing
from utils.data_utils import traintest_split_cross_subject, traintest_split_domain_classifier, domain_split_multisource, traintest_split_domain_classifier_pretest, time_cut, traintest_split_oodd

def data_process(dataset, args):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load(args.root_dir + 'data/' + 'BNCI2014001' + '/X.npy')
        y = np.load(args.root_dir + 'data/' + 'BNCI2014001' + '/labels.npy')
    elif dataset == 'MI1':
        X = np.load(args.root_dir + 'data/' + 'MI1' + '/X.npy')
        y = np.load(args.root_dir + 'data/' + 'MI1' + '/labels.npy')
    elif dataset == 'MI1-7':
        X = np.load(args.root_dir + 'data/' + 'MI1' + '/X-7.npy')
        y = np.load(args.root_dir + 'data/' + 'MI1' + '/labels-7.npy')
    elif 'DEAP' in dataset:
        X = np.load(args.root_dir + 'data/' + 'DEAP' + '/X.npy')
        y = np.load(args.root_dir + 'data/' + 'DEAP' + '/labels.npy')
    else:
        X = np.load(args.root_dir + 'data/' + dataset + '/X.npy')
        y = np.load(args.root_dir + 'data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate, trials_arr = None, None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        if 'setb' in args.method:
            # only use two classes [right_hand, feet]
            indices = []
            for i in range(len(y)):
                if y[i] in ['right_hand', 'feet']:
                    indices.append(i)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
            y = y.astype(int)
        else:  # seta, and commonly
            # only use two classes [left_hand, right_hand]
            indices = []
            for i in range(len(y)):
                if y[i] in ['left_hand', 'right_hand']:
                    indices.append(i)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        if 'setb' in args.method:
            print('BNCI2014002 downsampled')
            X = mne.filter.resample(X, down=(512 / 250))
            sample_rate = 250

        # only use session train, remove session test
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
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

        # only use session 1 first 120 trials, remove other sessions
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(120) + np.sum(presum_trials_arr[:i, :]))
        indices = np.concatenate(indices, axis=0)
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

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        y[np.where(y == 'right_hand')] = 0
        y[np.where(y == 'feet')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2015004':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 256
        ch_num = 30

        if 'setb' in args.method:
            print('BNCI2015004 downsampled')
            X = mne.filter.resample(X, down=(256 / 250))
            sample_rate = 250

        if 'setb' in args.method:
            # only use two classes [right_hand, feet]
            indices = []
            for i in range(len(y)):
                if y[i] in ['right_hand', 'feet']:
                    indices.append(i)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
            y = y.astype(int)

        presum_trials_arr = np.array([[80, 80],
                               [80, 80],
                               [80, 80],
                               [80, 70],
                               [80, 80],
                               [80, 80],
                               [80, 70],
                               [80, 80],
                               [80, 80]])

        # only use session 1 first 80 trials, remove other sessions
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(80) + np.sum(presum_trials_arr[:i, :]))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'Weibo2014':
        paradigm = 'MI'
        num_subjects = 10
        sample_rate = 200
        ch_num = 60

        if 'set' in args.method:
            print('Weibo2014 upsampled')
            X = mne.filter.resample(X, up=(250 / 200))
            sample_rate = 250

        if 'seta' in args.method:
            # only use two classes [right_hand, feet]
            indices = []
            for i in range(len(y)):
                if y[i] in ['left_hand', 'right_hand']:
                    indices.append(i)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            # only use two classes [right_hand, feet]
            indices = []
            for i in range(len(y)):
                if y[i] in ['right_hand', 'feet']:
                    indices.append(i)
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
            # only use session 1
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [left_hand, right_hand]
                cnt = 0
                for j in inds:
                    if y[j] in ['left_hand', 'right_hand']:
                        indices.append(j)
                        cnt += 1
                trials_arr.append(cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            # only use session 1
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [right_hand, feet]
                cnt = 0
                for j in inds:
                    if y[j] in ['right_hand', 'feet']:
                        indices.append(j)
                        cnt += 1
                trials_arr.append(cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
            y = y.astype(int)
        else:
            # only use session 1
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
                indices.extend(inds)
                trials_arr.append(len(inds))
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y[np.where(y == 'feet')] = 2
            y = y.astype(int)

    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y[np.where(y == 'feet')] = 2
        y[np.where(y == 'tongue')] = 3
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

        if 'seta' in args.method or dataset == 'MI1':
            X = np.concatenate([X[200:1000], X[1200:]])
            y = np.concatenate([y[200:1000], y[1200:]])
            num_subjects = 5
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

        # time cut
        X = time_cut(X, cut_percentage=0.8)
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
        class_num = 2
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
        class_num = 2

        # all classes
        trials_arr = np.array([5400, 5400, 1800, 1800, 1800, 1800, 1800, 1800, 1800, 1800])

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

        trials_arr = np.array([3304, 3284, 3304, 3274, 3256, 3331, 3295, 3268])

        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'HighGamma':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 500
        ch_num = 128
        class_num = 4

        # all classes
        trials_arr = np.array([320, 813, 880, 897, 720, 880, 880, 654, 880, 880, 880, 880, 800, 880])

        if 'seta' in args.method:
            # only use two classes [left_hand, right_hand]
            class_num = 2
            indices = []
            for i in range(len(y)):
                if y[i] == 0 or y[i] == 1:
                    indices.append(i)
            # two classes
            new_trials_arr = []
            for i in range(len(trials_arr)):
                subject_y = y[np.sum(trials_arr[:i]):np.sum(trials_arr[:(i + 1)])]
                trial_num = len(np.where(subject_y == 0)[0]) + len(np.where(subject_y == 1)[0])
                new_trials_arr.append(trial_num)
            trials_arr = new_trials_arr

            X = X[indices]
            y = y[indices]

            y[np.where(y == 0)] = 0
            y[np.where(y == 1)] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            # only use two classes [right_hand, feet]
            class_num = 2
            indices = []
            for i in range(len(y)):
                if y[i] == 1 or y[i] == 2:
                    indices.append(i)
            # two classes
            new_trials_arr = []
            for i in range(len(trials_arr)):
                subject_y = y[np.sum(trials_arr[:i]):np.sum(trials_arr[:(i + 1)])]
                trial_num = len(np.where(subject_y == 1)[0]) + len(np.where(subject_y == 2)[0])
                new_trials_arr.append(trial_num)
            trials_arr = new_trials_arr

            X = X[indices]
            y = y[indices]

            y[np.where(y == 1)] = 0
            y[np.where(y == 2)] = 1
            y = y.astype(int)
        else:
            # only use session 1
            class_num = 4

            y[np.where(y == 0)] = 0
            y[np.where(y == 1)] = 1
            y[np.where(y == 2)] = 2
            y[np.where(y == 3)] = 3
            y = y.astype(int)

        print('HighGamma downsampled')
        X = mne.filter.resample(X, down=2)
        sample_rate = int(sample_rate // 2)
    elif dataset == 'DEAP-v':
        paradigm = 'MI'
        num_subjects = 32
        sample_rate = 128
        ch_num = 32
        class_num = 2

        if args.backbone == 'TSeption':
            ch_num = 28
            args.chn = 28
            # drop middle z channels
            X = np.concatenate([X[:, :14, :], X[:, 16:18, :], X[:, 19:23, :], X[:, 24:, :]], axis=1)

        y = y[:, 0]
        y[np.where(y < 5)] = 0
        y[np.where(y > 5)] = 1

        presum_trials_arr = np.ones((32,)) * 600

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            inds = np.arange(presum_trials_arr[i]) + np.sum(presum_trials_arr[:i])
            inds = inds.astype(np.int)
            cnt = 0
            for j in inds:
                if y[j] != 5:
                    indices.append(j)
                    cnt += 1
            trials_arr.append(cnt)

        X = X[indices]
        y = y[indices]
        y = y.astype(int)
    elif dataset == 'DEAP-a':
        paradigm = 'MI'
        num_subjects = 32
        sample_rate = 128
        ch_num = 32
        class_num = 2

        if args.backbone == 'TSeption':
            ch_num = 28
            args.chn = 28
            # drop middle z channels
            X = np.concatenate([X[:, :14, :], X[:, 16:18, :], X[:, 19:23, :], X[:, 24:, :]], axis=1)

        y = y[:, 1]
        y[np.where(y < 5)] = 0
        y[np.where(y > 5)] = 1

        presum_trials_arr = np.ones((32,)) * 600

        indices = []
        trials_arr = []
        for i in range(num_subjects):
            inds = np.arange(presum_trials_arr[i]) + np.sum(presum_trials_arr[:i])
            inds = inds.astype(np.int)
            cnt = 0
            for j in inds:
                if y[j] != 5:
                    indices.append(j)
                    cnt += 1
            trials_arr.append(cnt)

        X = X[indices]
        y = y[indices]
        y = y.astype(int)
    elif dataset == 'CapgMyo':
        paradigm = 'MI'
        num_subjects = 18
        sample_rate = 1000
        ch_num = 128

        print('CapgMyo downsampled')
        X = mne.filter.resample(X, down=(1000 / 250))
        sample_rate = 250
        y = y - 1
        y = y.astype(int)
    elif dataset == 'Dreyer2023':
        paradigm = 'MI'
        num_subjects = 60
        sample_rate = 256
        ch_num = 27

        trials_arr = np.array([[240], [240], [240], [240], [240], [240], [240], [240], [240], [240], [240],
                                      [240], [240], [240], [240], [240], [240], [240], [240], [240], [240], [240],
                                      [240], [240], [240], [240], [240], [240], [240], [240], [240], [240], [240],
                                      [240], [240], [240], [240], [240], [240], [200], [240], [240], [240], [240],
                                      [240], [240], [240], [240], [240], [240], [240], [240], [240], [240], [240],
                                      [240], [240], [240], [160], [240]])

        y = y.astype(int)

    if 'DDPM' in args.method:
        print('downsampling for Diffusion')
        X = mne.filter.resample(X, down=(sample_rate / 250))
        sample_rate = 250
        # args.time_sample_num = X.shape[-1]

    if 'set' in args.method:
        print('before strip:', X.shape)
        # strip to first 1000 points
        X = X[:, :, :1000]
        print('striping to first 1000 points')
        print('after strip:', X.shape)

    if 'enc' in args.method or 'Enc' in args.method:
        # strip to even time samples
        X = X[:, :, :(X.shape[-1] // 2) * 2]
        print('striping to even time samples')
        print('after strip:', X.shape)

    print('data shape:', X.shape, ' labels shape:', y.shape)

    return X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr


def data_preprocess_loader(dataset, args):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''


    X = np.load('./data/' + dataset + '/X.npy')
    y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate, session_trials_arr = None, None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y[np.where(y == 'feet')] = 2
        y[np.where(y == 'tongue')] = 3
        y = y.astype(int)
        session_trials_arr = []
        for i in range(num_subjects):
            session_trials_arr.append([288, 288])
        session_trials_arr = np.stack(session_trials_arr)
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        print('BNCI2014002 downsampled')
        X = mne.filter.resample(X, down=(512 / 250))
        sample_rate = 250

        session_trials_arr = np.array([100, 60] * 14)

        y[np.where(y == 'right_hand')] = 0
        y[np.where(y == 'feet')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014004':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 3

        session_trials_arr = np.array([[120, 120, 160, 160, 160],
                               [120, 120, 160, 120, 160],
                               [120, 120, 160, 160, 160],
                               [120, 140, 160, 160, 160],
                               [120, 140, 160, 160, 160],
                               [120, 120, 160, 160, 160],
                               [120, 120, 160, 160, 160],
                               [160, 120, 160, 160, 160],
                               [120, 120, 160, 160, 160]])

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

        session_trials_arr = np.array([200] * 28)
        '''
        session_trials_arr = np.array([[200, 200],
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
                                       [200, 200]])
        '''
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

        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y[np.where(y == 'feet')] = 2
        y[np.where(y == 'hands')] = 3
        y[np.where(y == 'left_hand_right_foot')] = 4
        y[np.where(y == 'right_hand_left_foot')] = 5
        y = y.astype(int)
    elif dataset == 'Zhou2016':
        paradigm = 'MI'
        num_subjects = 4
        sample_rate = 250
        ch_num = 14

        session_trials_arr = np.array([[179, 150, 150],
                               [150, 135, 150],
                               [150, 151, 150],
                               [135, 150, 150]])

        y[np.where(y == 'left_hand')] = 0
        y[np.where(y == 'right_hand')] = 1
        y[np.where(y == 'feet')] = 2
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
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

        # time cut
        X = time_cut(X, cut_percentage=0.8)
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
        class_num = 2

        session_trials_arr = np.array([576, 576, 576] * 10)

        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)

    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
        class_num = 2

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
        session_trials_arr = np.array([320, 813, 880, 897, 720, 880, 880, 654, 880, 880, 880, 880, 800, 880])

        y[np.where(y == 0)] = 0  # left hand
        y[np.where(y == 1)] = 1  # right hand
        y[np.where(y == 2)] = 2  # feet
        y[np.where(y == 3)] = 3  # rest
        y = y.astype(int)

    print('before strip:', X.shape)
    if 'set' in args.method:
        # strip to first 1000 points
        X = X[:, :, :1000]
        print('striping to first 1000 points')
    print('after strip:', X.shape)

    args.session_trials_arr = session_trials_arr

    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y


def data_process_firstsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i)) # use first sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session test, remove session train
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(100) + (160 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session E, remove session T
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

        # time cut
        X = time_cut(X, cut_percentage=0.8)
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_process_secondsession(dataset):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session E, remove session T
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

        # only use two classes [left_hand, right_hand]
        indices = []
        for i in range(len(y)):
            if y[i] in ['left_hand', 'right_hand']:
                indices.append(i)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        # only use session test, remove session train
        indices = []
        for i in range(num_subjects):
            #indices.append(np.arange(100) + (160 * i))
            indices.append(np.arange(60) + (160 * i) + 100) # use second sessions
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]

    elif dataset == 'BNCI2015001':
        paradigm = 'MI'
        num_subjects = 12
        sample_rate = 512
        ch_num = 13

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            # use second sessions
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7) + 200)
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7) + 200)
            else:
                indices.append(np.arange(200) + (400 * i) + 200)

        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014001-4':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        # only use session E, remove session T
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i) + 288)
        indices = np.concatenate(indices, axis=0)
        X = X[indices]
        y = y[indices]
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

        # time cut
        X = time_cut(X, cut_percentage=0.8)
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num


def data_process_allsessions(dataset, args):
    '''

    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''

    if dataset == 'BNCI2014001-4':
        X = np.load('./data/' + 'BNCI2014001' + '/X.npy')
        y = np.load('./data/' + 'BNCI2014001' + '/labels.npy')
    elif dataset == 'MI1':
        X = np.load('./data/' + 'MI1' + '/X.npy')
        y = np.load('./data/' + 'MI1' + '/labels.npy')
    elif dataset == 'MI1-7':
        X = np.load('./data/' + 'MI1' + '/X-7.npy')
        y = np.load('./data/' + 'MI1' + '/labels-7.npy')
    elif 'DEAP' in dataset:
        X = np.load('./data/' + 'DEAP' + '/X.npy')
        y = np.load('./data/' + 'DEAP' + '/labels.npy')
    else:
        X = np.load('./data/' + dataset + '/X.npy')
        y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate, trials_arr = None, None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22

        if 'setb' in args.method:
            # only use two classes [right_hand, feet]
            indices = []
            for i in range(len(y)):
                if y[i] in ['right_hand', 'feet']:
                    indices.append(i)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
            y = y.astype(int)
        else:  # seta, and commonly
            # only use two classes [left_hand, right_hand]
            indices = []
            for i in range(len(y)):
                if y[i] in ['left_hand', 'right_hand']:
                    indices.append(i)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
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

        # only use session 1, remove session 2/3
        indices = []
        for i in range(num_subjects):
            if i in [7, 8, 9, 10]:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            elif i == 11:
                indices.append(np.arange(200) + (400 * 7) + 600 * (i - 7))
            else:
                indices.append(np.arange(200) + (400 * i))

        indices = np.concatenate(indices, axis=0)
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

        if 'set' in args.method:
            print('Weibo2014 upsampled')
            X = mne.filter.resample(X, up=(250 / 200))
            sample_rate = 250

        if 'seta' in args.method:
            # only use two classes [right_hand, feet]
            indices = []
            for i in range(len(y)):
                if y[i] in ['left_hand', 'right_hand']:
                    indices.append(i)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            # only use two classes [right_hand, feet]
            indices = []
            for i in range(len(y)):
                if y[i] in ['right_hand', 'feet']:
                    indices.append(i)
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
            # only use session 1
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [left_hand, right_hand]
                cnt = 0
                for j in inds:
                    if y[j] in ['left_hand', 'right_hand']:
                        indices.append(j)
                        cnt += 1
                trials_arr.append(cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'left_hand')] = 0
            y[np.where(y == 'right_hand')] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            # only use session 1
            indices = []
            trials_arr = []
            for i in range(num_subjects):
                inds = np.arange(presum_trials_arr[i, 0]) + np.sum(presum_trials_arr[:i, :])
                # only use two classes [right_hand, feet]
                cnt = 0
                for j in inds:
                    if y[j] in ['right_hand', 'feet']:
                        indices.append(j)
                        cnt += 1
                trials_arr.append(cnt)
            X = X[indices]
            y = y[indices]
            y[np.where(y == 'right_hand')] = 0
            y[np.where(y == 'feet')] = 1
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

        if 'seta' in args.method or dataset == 'MI1':
            X = np.concatenate([X[200:1000], X[1200:]])
            y = np.concatenate([y[200:1000], y[1200:]])
            num_subjects = 5
    elif dataset == 'BNCI2014008':
        paradigm = 'ERP'
        num_subjects = 8
        sample_rate = 256
        ch_num = 8
        class_num = 2

        # time cut
        X = time_cut(X, cut_percentage=0.8)
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'BNCI2014009':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 16
        class_num = 2
        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)

    elif dataset == 'BNCI2015003':
        paradigm = 'ERP'
        num_subjects = 10
        sample_rate = 256
        ch_num = 8
        class_num = 2

        y[np.where(y == 'NonTarget')] = 0
        y[np.where(y == 'Target')] = 1
        y = y.astype(int)
    elif dataset == 'HighGamma':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 500
        ch_num = 128
        class_num = 4

        # all classes
        trials_arr = np.array([320, 813, 880, 897, 720, 880, 880, 654, 880, 880, 880, 880, 800, 880])

        if 'seta' in args.method:
            # only use two classes [left_hand, right_hand]
            class_num = 2
            indices = []
            for i in range(len(y)):
                if y[i] == 0 or y[i] == 1:
                    indices.append(i)
            # two classes
            new_trials_arr = []
            for i in range(len(trials_arr)):
                subject_y = y[np.sum(trials_arr[:i]):np.sum(trials_arr[:(i + 1)])]
                trial_num = len(np.where(subject_y == 0)[0]) + len(np.where(subject_y == 1)[0])
                new_trials_arr.append(trial_num)
            trials_arr = new_trials_arr

            X = X[indices]
            y = y[indices]

            y[np.where(y == 0)] = 0
            y[np.where(y == 1)] = 1
            y = y.astype(int)
        elif 'setb' in args.method:
            # only use two classes [right_hand, feet]
            class_num = 2
            indices = []
            for i in range(len(y)):
                if y[i] == 1 or y[i] == 2:
                    indices.append(i)
            # two classes
            new_trials_arr = []
            for i in range(len(trials_arr)):
                subject_y = y[np.sum(trials_arr[:i]):np.sum(trials_arr[:(i + 1)])]
                trial_num = len(np.where(subject_y == 1)[0]) + len(np.where(subject_y == 2)[0])
                new_trials_arr.append(trial_num)
            trials_arr = new_trials_arr

            X = X[indices]
            y = y[indices]

            y[np.where(y == 1)] = 0
            y[np.where(y == 2)] = 1
            y = y.astype(int)

        print('HighGamma downsampled')
        X = mne.filter.resample(X, down=2)
        sample_rate = int(sample_rate // 2)

    print('before strip:', X.shape)
    if 'set' in args.method:
        # strip to first 1000 points
        X = X[:, :, :1000]
        print('striping to first 1000 points')
    print('after strip:', X.shape)

    if 'enc' in args.method or 'Enc' in args.method:
        # strip to even time samples
        X = X[:, :, :(X.shape[-1] // 2) * 2]
        print('striping to even time samples')
    print('after strip:', X.shape)

    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr


def read_mi_combine_tar(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args.data, args)

    args.trials_arr = trials_arr

    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt, trials_arr)

    return src_data, src_label, tar_data, tar_label


def read_mi_separate_sessions(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process_firstsession(args.data)

    if args.data == 'BNCI2014002' or args.data == 'BNCI2015001':
        print('downsampled')
        X = mne.filter.resample(X, down=(512 / 250))
        sample_rate = 250

    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)

    for i in range(len(data_subjects)):
        data_subjects[i] = data_subjects[i][:len(data_subjects[i]) // 2]
        labels_subjects[i] = labels_subjects[i][:len(labels_subjects[i]) // 2]

    X_test, y_test, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)

    if args.data == 'BNCI2014002' or args.data == 'BNCI2015001':
        print('downsampled')
        X_test = mne.filter.resample(X_test, down=(512 / 250))
        sample_rate = 250

    data_subjects_test = np.split(X_test, indices_or_sections=num_subjects, axis=0)
    labels_subjects_test = np.split(y_test, indices_or_sections=num_subjects, axis=0)

    for i in range(len(data_subjects_test)):
        data_subjects_test[i] = data_subjects_test[i][len(data_subjects_test[i]) // 2:]
        labels_subjects_test[i] = labels_subjects_test[i][len(labels_subjects_test[i]) // 2:]

    return data_subjects, labels_subjects, data_subjects_test, labels_subjects_test


def read_mi_combine_tar_secondsession(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)

    src_data, src_label, tar_data, tar_label = traintest_split_cross_subject(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_separate_domain_sessions(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args.data, args)

    if args.data == 'BNCI2014002' or args.data == 'BNCI2015001':
        print('downsampled')
        X = mne.filter.resample(X, down=(512 / 250))
        sample_rate = 250

    args.trials_arr = trials_arr

    train_x, train_y = traintest_split_oodd(args.data, X, y, num_subjects)
    for i in range(len(train_x)):
        print(train_x[i].shape, train_y[i].shape)
        train_x[i] = train_x[i][:len(train_x[i]) // 2]
        train_y[i] = train_y[i][:len(train_y[i]) // 2]
        print(train_x[i].shape, train_y[i].shape)

    # assuming using the basic datasets (14001, 14002, 15001), no difference of number of trials across subjects
    X_second, y_second, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)

    if args.data == 'BNCI2014002' or args.data == 'BNCI2015001':
        print('downsampled')
        X_second = mne.filter.resample(X_second, down=(512 / 250))
        sample_rate = 250

    test_x, test_y = traintest_split_oodd(args.data, X_second, y_second, num_subjects)

    return train_x, train_y, test_x, test_y


def read_mi_separate_domain_sessions_binary(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args.data, args)

    if args.data == 'BNCI2014002' or args.data == 'BNCI2015001':
        print('downsampled')
        X = mne.filter.resample(X, down=(512 / 250))
        sample_rate = 250

    args.trials_arr = trials_arr

    train_x, train_y = traintest_split_oodd(args.data, X, y, num_subjects)

    # assuming using the basic datasets (14001, 14002, 15001), no difference of number of trials across subjects
    X_second, y_second, num_subjects, paradigm, sample_rate, ch_num = data_process_secondsession(args.data)

    if args.data == 'BNCI2014002' or args.data == 'BNCI2015001':
        print('downsampled')
        X_second = mne.filter.resample(X_second, down=(512 / 250))
        sample_rate = 250

    test_x, test_y = traintest_split_oodd(args.data, X_second, y_second, num_subjects)

    return train_x, train_y, test_x, test_y


def read_mi_combine_domain(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args.data, args)

    args.trials_arr = trials_arr

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier(args.data, X, y, num_subjects, args.idt)

    return src_data, src_label, tar_data, tar_label


def read_mi_combine_domain_split(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args.data, args)

    args.trials_arr = trials_arr

    src_data, src_label, tar_data, tar_label = traintest_split_domain_classifier_pretest(args.data, X, y, num_subjects, args.ratio)

    return src_data, src_label, tar_data, tar_label


def read_mi_multi_source(args):
    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args.data, args)

    args.trials_arr = trials_arr

    data_subjects, labels_subjects = domain_split_multisource(args.data, X, y, num_subjects, args.idt)

    return data_subjects, labels_subjects


def read_mi_all(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(args.data, args)

    args.trials_arr = trials_arr

    return X, y


def read_mi_all_allsessions(args):

    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process_allsessions(args.data, args)

    args.trials_arr = trials_arr

    return X, y


def data_normalize(fea_de, norm_type):
    if norm_type == 'zscore':
        zscore = preprocessing.StandardScaler()
        fea_de = zscore.fit_transform(fea_de)

    return fea_de
