# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : pre_EA_align_p300_cmch.py
# this file prealigns EEG data, while only using the common shared subset of electrodes (8 for P300), for a baseline comparison of reducing the heterogeneous TL problem to a homogeneous one
import argparse
import numpy as np
from utils.dataloader import data_preprocess_loader
from utils.algorithms import EA_online

import os
from pathlib import Path


# TODO: please replace path with your specified file path, this path will be used for any algorithms to load data from, which means the ./data/ path will only be a temporary one
DATA_PATH = 'name_your_data_path/'


def data_process(data_name_list):
    for data_name in data_name_list:

        print(data_name)

        if data_name == 'BNCI2014008': paradigm, N, chn = 'ERP', 8, 8
        if data_name == 'BNCI2014009': paradigm, N, chn = 'ERP', 10, 16
        if data_name == 'BNCI2015003': paradigm, N, chn = 'ERP', 10, 8
        if data_name == 'EPFLP300': paradigm, N, chn = 'ERP', 8, 32

        args = argparse.Namespace(N=N, chn=chn, paradigm=paradigm, data_name=data_name)

        args.method = ''

        X, y, trials_arr = data_preprocess_loader(data_name, args)

        if data_name == 'BNCI2014008':
            common_mat = [0, 1, 2, 3, 4, 5, 6, 7]
        elif data_name == 'BNCI2014009':
            common_mat = [0, 1, 2, 3, 4, 5, 6, 7]
        elif data_name == 'BNCI2015003':
            common_mat = [0, 1, 3, 6, 2, 4, 5, 7]
        elif data_name == 'EPFLP300':
            common_mat = [30, 31, 12, 15, 11, 18, 10, 19]
        X = X[:, common_mat, :]

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

        if not os.path.isdir(DATA_PATH + data_name + '/cmch'):
            path = Path(DATA_PATH + data_name + '/cmch')
            path.mkdir(parents=True)
        np.save(DATA_PATH + data_name + '/' + 'cmch/X', X)
        np.save(DATA_PATH + data_name + '/' + 'cmch/X-EA', all_aligned_X)
        np.save(DATA_PATH + data_name + '/' + 'cmch/y', all_y)


if __name__ == '__main__':

    data_name_list = ['BNCI2014008', 'BNCI2014009', 'BNCI2015003', 'EPFLP300']

    data_process(data_name_list)




