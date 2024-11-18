# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : download_data.py
# download from MOABB and process EEG data
import os
import sys
import math

import numpy as np
import moabb

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, \
    BNCI2014004, BNCI2015001, PhysionetMI, Cho2017, Wang2016, Schirrmeister2017, Weibo2014, Zhou2016, Ofner2017, \
    Nakanishi2015  # , Kalunga2016, Lee2019_SSVEP
from moabb.paradigms import MotorImagery, P300, SSVEP


def dataset_to_file(dataset_name, data_save):
    moabb.set_log_level("ERROR")
    if dataset_name == 'BNCI2014001':
        dataset = BNCI2014001()
        paradigm = MotorImagery(n_classes=4)
        # (5184, 22, 1001) (5184,) 250Hz 9subjects * 4classes * (72+72)trials for 2sessions
    elif dataset_name == 'BNCI2014002':
        dataset = BNCI2014002()
        paradigm = MotorImagery(n_classes=2)
        # (2240, 15, 2561) (2240,) 512Hz 14subjects * 2classes * (50+30)trials * 2sessions(not namely separately)
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions

    if data_save:
        print('preparing data...')

        X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:])

        ar_unique, cnts = np.unique(labels, return_counts=True)
        print("labels:", ar_unique)
        print("Counts:", cnts)
        print(X.shape, labels.shape)
        np.save('./data/' + dataset_name + '/X', X)
        np.save('./data/' + dataset_name + '/labels', labels)
        meta.to_csv('./data/' + dataset_name + '/meta.csv')
    else:
        if isinstance(paradigm, MotorImagery):
            for i in range(len(dataset.subject_list)):
                X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[i]])
                print(X.shape)
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            print(X.ch_names)
            return X.info


if __name__ == '__main__':
    dataset_names = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    for dataset_name in dataset_names:
        print(dataset_name)

        # this line for saving data
        info = dataset_to_file(dataset_name, data_save=True)

        # this line for getting info only
        # info = dataset_to_file(dataset_name, data_save=False)

        np.save('./data/' + dataset_name + '/info', info)
        print(info)
