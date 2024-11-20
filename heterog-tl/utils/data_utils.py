# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : data_utils.py
import random

import numpy as np


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    print(label_01)
    return label_01


def time_cut(data, cut_percentage):
    # Time Cutting: cut at a certain percentage of the time. data: (..., ..., time_samples)
    data = data[:, :, :int(data.shape[2] * cut_percentage + 1)]  # add one for ERP BNCI2014008
    return data


def traintest_split_cross_subject(dataset, X, y, num_subjects, test_subject_id, trials_arr=None):
    if trials_arr is not None:
        accum_arr = []
        for t in range(len(trials_arr)):
            accum_arr.append(np.sum([trials_arr[:(t + 1)]]))
        #print(accum_arr)
        data_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        labels_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
    else:
        data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
        labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    test_x = data_subjects.pop(test_subject_id)
    test_y = labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    train_y = np.concatenate(labels_subjects, axis=0)
    #print('Test subject s' + str(test_subject_id))
    #print('Training/Test split:', train_x.shape, test_x.shape)
    return train_x, train_y, test_x, test_y


def traintest_split_domain_classifier(dataset, X, y, num_subjects, test_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)
    data_subjects.pop(test_subject_id)
    labels_subjects.pop(test_subject_id)
    train_x = np.concatenate(data_subjects, axis=0)
    for i in range(num_subjects - 1):
        labels_subjects[i] = np.ones((int(len(labels_subjects[i]))),) * i
    train_y = np.concatenate(labels_subjects, axis=0)
    print('Test subject s' + str(test_subject_id))
    print('Training:', train_x.shape, train_y.shape)
    return train_x, train_y, None, None


def traintest_split_domain_classifier_pretest(dataset, X, y, num_subjects, ratio):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    train_x_all = []
    train_y_all = []
    test_x_all = []
    test_y_all = []
    for i in range(num_subjects):
        data = data_subjects[i]
        random.shuffle(data)
        train_x_all.append(data[:int(len(data) * ratio)])
        train_y_all.append(np.ones((int(len(data) * ratio)),) * i)
        test_x_all.append(data[int(len(data) * ratio):])
        test_y_all.append(np.ones((int(len(data) * (1 - ratio))),) * i)
    train_x = np.concatenate(train_x_all, axis=0)
    train_y = np.concatenate(train_y_all, axis=0)
    test_x = np.concatenate(test_x_all, axis=0)
    test_y = np.concatenate(test_y_all, axis=0)
    print('Training/Test split:', train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    return train_x, train_y, test_x, test_y


def domain_split_multisource(dataset, X, y, num_subjects, target_subject_id):
    data_subjects = np.split(X, indices_or_sections=num_subjects, axis=0)
    labels_subjects = np.split(y, indices_or_sections=num_subjects, axis=0)

    data_subjects.pop(target_subject_id)
    labels_subjects.pop(target_subject_id)

    print(num_subjects, 'Subjects of', data_subjects[0].shape, labels_subjects[0].shape)
    return data_subjects, labels_subjects
