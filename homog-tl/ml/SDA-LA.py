# -*- coding: utf-8 -*-
# @Time    : 2023/10/27
# @Author  : Siyang Li
# @File    : SDA-LA.py
# Label Alignment for cross-subject analysis (using both source labeled data and target labeled calibration data)
import mne
import numpy as np
import torch
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from scipy.linalg import fractional_matrix_power

import random
import sys
import os

from utils.alg_utils import EA, EA_online, EA_ref
from utils.alg_utils import LA
from utils.data_utils import traintest_split_cross_subject


def apply_zscore(train_x, test_x, num_subjects):
    # train split into subjects
    train_z = []
    trial_num = int(train_x.shape[0] / (num_subjects - 1))
    for j in range(num_subjects - 1):
        scaler = preprocessing.StandardScaler()
        train_x_tmp = scaler.fit_transform(train_x[trial_num * j: trial_num * (j + 1), :])
        train_z.append(train_x_tmp)
    train_x = np.concatenate(train_z, axis=0)
    # test subject
    scaler = preprocessing.StandardScaler()
    test_x = scaler.fit_transform(test_x)
    return train_x, test_x


def data_process(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    X = np.load('./data/' + dataset + '/X.npy')
    y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014001':
        paradigm = 'MI'
        num_subjects = 9
        sample_rate = 250
        ch_num = 22
        class_num = 2

        # only use session T, remove session E
        indices = []
        for i in range(num_subjects):
            indices.append(np.arange(288) + (576 * i))
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
        class_num = 2

        # only use session train, remove session test
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
        class_num = 2

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

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num, class_num


def traintest_split_within_subject(dataset, X, y, num_subjects, test_subject_id, num, shuffle):
    # equal number of samples per class
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    subj_data = data_subjects.pop(test_subject_id)
    subj_label = labels_subjects.pop(test_subject_id)
    class_out = len(np.unique(subj_label))
    if shuffle:
        inds = np.arange(len(subj_data))
        np.random.shuffle(inds)
        subj_data = subj_data[inds]
        subj_label = subj_label[inds]
    if num < 1:  # percentage, all/each class
        num_int = int(len(subj_data) * num / class_out)
    else:  # numbers, for each class
        num_int = int(num)

    inds_all_train = []
    inds_all_test = []
    for class_num in range(class_out):
        inds_class = np.where(subj_label == class_num)[0]
        inds_all_train.append(inds_class[:num_int])
        inds_all_test.append(inds_class[num_int:])
    inds_all_train = np.concatenate(inds_all_train)
    inds_all_test = np.concatenate(inds_all_test)

    train_x = subj_data[inds_all_train]
    train_y = subj_label[inds_all_train]
    test_x = subj_data[inds_all_test]
    test_y = subj_label[inds_all_test]

    print('Within subject s' + str(test_subject_id))
    print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def ml_classifier(approach, output_probability, train_x, train_y, test_x, return_model=None, weight=None):
    if approach == 'LDA':
        clf = LinearDiscriminantAnalysis()
    elif approach == 'LR':
        clf = LogisticRegression(max_iter=1000)
    elif approach == 'AdaBoost':
        clf = AdaBoostClassifier()
    elif approach == 'GradientBoosting':
        clf = GradientBoostingClassifier()
    elif approach == 'xgb':
        clf = XGBClassifier()
        if weight:
            print('XGB weight:', weight)
            clf = XGBClassifier(scale_pos_weight=weight)

    clf.fit(train_x, train_y)

    if output_probability:
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    if return_model:
        return pred, clf
    else:
        print(pred)
        return pred


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def ml_cross(dataset, info, align, approach, calbr_n):
    X, y, num_subjects, paradigm, sample_rate, ch_num, class_num = data_process(dataset)
    print('X, y, num_subjects, paradigm, sample_rate:', X.shape, y.shape, num_subjects, paradigm, sample_rate)

    print('sample rate:', sample_rate)

    scores_arr = []

    for i in range(num_subjects):
        source_x, source_y, _, _ = traintest_split_cross_subject(dataset, X, y, num_subjects, i)
        calbr_ratio = calbr_n / ((len(y) // num_subjects))
        target_x, target_y, test_x, test_y = traintest_split_within_subject(dataset, X, y, num_subjects, i, calbr_ratio, False)

        print('source_x, source_y, target_x, target_y, test_x, test_y.shape', source_x.shape, source_y.shape, target_x.shape, target_y.shape, test_x.shape, test_y.shape)

        if align:
            source_x = data_alignment(source_x, num_subjects - 1)
            num_samples = calbr_n * class_num
            if calbr_n == 0:
                target_x = []
                R = 0
            else:
                target_x, R = EA_ref(target_x)

            test_x_aligned = []
            for ind in range(len(test_y)):
                curr = test_x[ind]
                R = EA_online(curr, R, num_samples)
                num_samples += 1
                sqrtRefEA = fractional_matrix_power(R, -0.5)
                curr_aligned = np.dot(sqrtRefEA, curr)
                test_x_aligned.append(curr_aligned)
            test_x = np.stack(test_x_aligned)

        if align:
            # source_x, source_y = LA(source_x, source_y, target_x, target_y, use_logeuclid=False)
            source_x, source_y = LA(source_x, source_y, target_x, target_y, use_logeuclid=True)

        if paradigm == 'MI':

            # CCSP
            combined_csp = mne.decoding.CSP(n_components=6)

            combined_x_csp = combined_csp.fit_transform(np.concatenate((source_x, target_x)),
                                                        np.concatenate((source_y, target_y)))

            test_x_csp_combined = combined_csp.transform(test_x)

            pred, model = ml_classifier(approach, False, combined_x_csp, np.concatenate((source_y, target_y)), test_x_csp_combined, return_model=True)

            score = np.round(accuracy_score(test_y, pred), 5)
            print('CCSP score', np.round(score, 5))

        scores_arr.append(score)

    print('#' * 30)
    for i in range(len(scores_arr)):
        scores_arr[i] = np.round(scores_arr[i] * 100)
    print('sbj scores', scores_arr)
    print('avg', np.round(np.average(scores_arr), 5))

    return scores_arr


if __name__ == '__main__':

    scores = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    clbr_num_arr = [10, 20, 50]

    dataset_arr = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    scores = np.zeros((len(dataset_arr), len(clbr_num_arr)))

    cnt0 = 0

    for dataset in dataset_arr:

        cnt1 = 0

        for approach in ['LDA']:

            for calbr_n in clbr_num_arr:
                # use LA
                align = True

                print(dataset, align, approach)

                scores_arr = ml_cross(dataset, None, align, approach, calbr_n)
                score_avg = np.round(np.average(scores_arr), 2)
                print(score_avg)
                scores[cnt0, cnt1] = score_avg

                cnt1 += 1
        cnt0 += 1

    print(scores)