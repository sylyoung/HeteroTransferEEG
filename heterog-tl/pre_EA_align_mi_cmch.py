# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : pre_EA_align_mi_cmch.py
# this file prealigns EEG data, while only using the common shared subset of electrodes (3 for MI), for a baseline comparison of reducing the heterogeneous TL problem to a homogeneous one
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

        if data_list == 'seta':
            if data_name == 'BNCI2014001':
                common_mat = [7, 9, 11]
            elif data_name == 'BNCI2014004':
                common_mat = [0, 1, 2]
            elif data_name == 'MI1':
                common_mat = [26, 28, 30]
            elif data_name == 'HighGamma':
                common_mat = [14, 15, 16]
            elif data_name == 'Weibo2014':
                common_mat = [25, 27, 29]
        elif data_list == 'setb':
            if data_name == 'BNCI2014001':
                common_mat = [7, 9, 11]
            elif data_name == 'BNCI2015001':
                common_mat = [4, 6, 8]
            elif data_name == 'HighGamma':
                common_mat = [14, 15, 16]
            elif data_name == 'Weibo2014':
                common_mat = [25, 27, 29]
            elif data_name == 'Zhou2016':
                common_mat = [5, 6, 7]
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

        if not os.path.isdir(DATA_PATH + data_name + '/' + (data_list) + '/cmch/'):
            path = Path(DATA_PATH + data_name + '/' + (data_list) + '/cmch/')
            path.mkdir(parents=True)

        np.save(DATA_PATH + data_name + '/' + (data_list) + '/cmch/X', X)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/cmch/X-EA', all_aligned_X)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/cmch/y', all_y)


def data_process_hg(i):
    data_lists = ['seta', 'setb']

    for data_list in data_lists:

        data_name = 'HighGamma'
        print(data_name)

        if data_name == 'HighGamma': paradigm, N, chn = 'MI', 14, 128

        args = argparse.Namespace(N=N, chn=chn, paradigm=paradigm, data_name=data_name)

        args.method = data_list

        X_subject, y_subject, new_subject_cnt = data_preprocess_loader_hg(data_name, args, i)

        common_mat = [14, 15, 16]

        X_subject = X_subject[:, common_mat, :]

        aligned_x = EA_online(X_subject)

        print('X.shape, subject_aligned_X.shape, subject_y.shape:', X_subject.shape, aligned_x.shape, y_subject.shape)

        if not os.path.isdir(DATA_PATH + data_name + '/' + (data_list) + '/cmch/'):
            path = Path(DATA_PATH + data_name + '/' + (data_list) + '/cmch/')
            path.mkdir(parents=True)

        np.save(DATA_PATH + data_name + '/' + (data_list) + '/cmch/X' + str(i), X_subject)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/cmch/X-EA' + str(i), aligned_x)
        np.save(DATA_PATH + data_name + '/' + (data_list) + '/cmch/y' + str(i), y_subject)

        gc.collect()


if __name__ == '__main__':

    for i in range(14):
        data_process_hg(i)

    data_list = 'seta'
    data_name_list = ['BNCI2014001', 'BNCI2014004', 'MI1', 'Weibo2014']

    data_process(data_list, data_name_list)

    data_list = 'setb'
    data_name_list = ['BNCI2014001', 'BNCI2015001', 'Weibo2014', 'Zhou2016']
    data_process(data_list, data_name_list)



