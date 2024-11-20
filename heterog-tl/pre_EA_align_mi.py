# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : pre_EA_align_mi.py
# Since datasets are too big, we prealign data using Euclidean alignment (EA)
# Note that we follow online incremental EA for each session of each subject of each dataset, which means that the target subject was also done so, satisfying online test-time adaptation manner as in T-TIME
# However, you might find it the performance was no different of online EA and offline EA, as EA can take advantage of mean covariance matrix from only a few samples. So if you find it too slow to preprocess data, feel free to change the code to offline EA.
import argparse
import numpy as np
from utils.dataloader import data_preprocess_loader, data_preprocess_loader_hg
from utils.algorithms import EA_online

import os
import gc
from pathlib import Path

# TODO: please replace path with your specified file path, this path will be used for any algorithms to load data from, which means the ./data/ path will only be a temporary one
DATA_PATH = 'name_your_data_path/'


def data_process(data_list, data_name_list):
    for data_name in data_name_list:

        print(data_name)

        if data_name == 'BNCI2014001': paradigm, N, chn = 'MI', 9, 22
        if data_name == 'BNCI2014002': paradigm, N, chn = 'MI', 14, 15
        if data_name == 'BNCI2014004': paradigm, N, chn = 'MI', 9, 3
        if data_name == 'BNCI2015001': paradigm, N, chn = 'MI', 12, 13
        if data_name == 'MI1': paradigm, N, chn = 'MI', 5, 59
        if data_name == 'HighGamma': paradigm, N, chn = 'MI', 14, 128
        if data_name == 'Weibo2014': paradigm, N, chn = 'MI', 10, 60
        if data_name == 'Zhou2016': paradigm, N, chn = 'MI', 4, 14

        args = argparse.Namespace(N=N, chn=chn, paradigm=paradigm, data_name=data_name)

        args.method = data_list

        X, y, trials_arr = data_preprocess_loader(data_name, args)

        trials_arr_flatten = np.array(trials_arr).reshape(-1, )
        print(trials_arr_flatten.shape)
        accum_arr = []
        for i in range(len(trials_arr_flatten) - 1):
            ind = i + 1
            accum_arr.append(np.sum([trials_arr_flatten[:ind]]))
        X_subjects = np.split(X, indices_or_sections=accum_arr, axis=0)
        Y_subjects = np.split(y, indices_or_sections=accum_arr, axis=0)
        all_aligned_X = []
        for i in range(len(X_subjects)):
            aligned_x = EA_online(X_subjects[i])
            all_aligned_X.append(aligned_x)
        all_aligned_X = np.concatenate(all_aligned_X)
        all_y = np.concatenate(Y_subjects)

        print('X.shape, all_aligned_X.shape, all_y.shape:', X.shape, all_aligned_X.shape, all_y.shape)

        if not os.path.isdir(DATA_PATH + data_name + '/' + (data_list) + '/'):
            path = Path(DATA_PATH + data_name + '/' + (data_list) + '/')
            path.mkdir(parents=True)

        np.save(DATA_PATH + data_name + '/' + (data_list) + '/X', X)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/X-EA', all_aligned_X)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/y', all_y)


def data_process_hg(i):
    data_lists = ['seta', 'setb']

    for data_list in data_lists:

        data_name = 'HighGamma'
        print(data_name)

        if data_name == 'HighGamma': paradigm, N, chn = 'MI', 14, 128

        args = argparse.Namespace(N=N, chn=chn, paradigm=paradigm, data_name=data_name)

        args.method = data_list

        X_subject, y_subject, new_subject_cnt = data_preprocess_loader_hg(data_name, args, i)

        aligned_x = EA_online(X_subject)

        print('X.shape, subject_aligned_X.shape, subject_y.shape:', X_subject.shape, aligned_x.shape, y_subject.shape)

        if not os.path.isdir(DATA_PATH + data_name + '/' + (data_list) + '/'):
            path = Path(DATA_PATH + data_name + '/' + (data_list) + '/')
            path.mkdir(parents=True)

        np.save(DATA_PATH + data_name + '/' + (data_list) + '/X' + str(i), X_subject)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/X-EA' + str(i), aligned_x)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/y' + str(i), y_subject)

        gc.collect()


if __name__ == '__main__':

    # has to be done separately for each subject in HighGamma dataset, or the entire dataset uses too much memory due to 128 electrodes of high dimensionality
    for i in range(14):
        data_process_hg(i)

    data_list = 'seta'
    data_name_list = ['BNCI2014001', 'BNCI2014004', 'MI1', 'Weibo2014']
    data_process(data_list, data_name_list)

    data_list = 'setb'
    data_name_list = ['BNCI2014001', 'BNCI2015001', 'Weibo2014', 'Zhou2016']
    data_process(data_list, data_name_list)



