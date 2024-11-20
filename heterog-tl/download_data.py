# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : download_data.py
# download from MOABB and process EEG data
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
    elif dataset_name == 'BNCI2014004':
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
    elif dataset_name == 'Zhou2016':
        dataset = Zhou2016()
        paradigm = MotorImagery(n_classes=3)
    elif dataset_name == 'Weibo2014':
        dataset = Weibo2014()
        paradigm = MotorImagery(n_classes=2)
    elif dataset_name == 'BNCI2014008':
        dataset = BNCI2014008()
        paradigm = P300()
    elif dataset_name == 'BNCI2014009':
        dataset = BNCI2014009()
        paradigm = P300()
    elif dataset_name == 'BNCI2015003':
        dataset = BNCI2015003()
        paradigm = P300()
    elif dataset_name == 'EPFLP300':
        dataset = EPFLP300()
        paradigm = P300()
    elif dataset_name == 'Schirrmeister2017':
        dataset = Schirrmeister2017()
        paradigm = MotorImagery(n_classes=4)

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
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=[dataset.subject_list[0]], return_epochs=True)
            print(X.ch_names)
            return X.info

if __name__ == '__main__':
    dataset_names = ['BNCI2014001', 'BNCI2014004', 'BNCI2015001', 'Schirrmeister2017', 'Weibo2014', 'Zhou2016', 'BNCI2014008', 'BNCI2014009', 'BNCI2015003', 'EPFLP300']
    for dataset_name in dataset_names:
        print(dataset_name)

        # this line for saving data
        info = dataset_to_file(dataset_name, data_save=True)

        # this line for getting info only
        # info = dataset_to_file(dataset_name, data_save=False)

        np.save('./data/' + dataset_name + '/info', info)
        print(info)
