# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : pre_EA_align_p300.py
# Since datasets are too big, we prealign data using Euclidean alignment (EA)
# Note that we follow online incremental EA for each session of each subject of each dataset, which means that the target subject was also done so, satisfying online test-time adaptation manner as in T-TIME
# However, you might find it the performance was no different of online EA and offline EA, as EA can take advantage of mean covariance matrix from only a few samples. So if you find it too slow to preprocess P300 data which are a lot, feel free to change the code to offline EA.
import argparse
import numpy as np
from utils.dataloader import data_preprocess_loader, data_preprocess_loader_hg
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

        if not os.path.isdir(DATA_PATH + data_name + '/'):
            path = Path(DATA_PATH + data_name + '/')
            path.mkdir(parents=True)

        np.save(DATA_PATH + data_name + '/' + '/X', X)
        np.save(DATA_PATH + data_name + '/' + '/X-EA', all_aligned_X)
        np.save(DATA_PATH + data_name + '/' + '/y', all_y)


if __name__ == '__main__':

    data_name_list = ['BNCI2014008', 'BNCI2014009', 'BNCI2015003', 'EPFLP300']

    data_process(data_name_list)




