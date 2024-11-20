# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : process_HighGamma.py
# this script preprocess Schirrmeister2017 (HighGamma) dataset, in case cannot download from MOABB due to some unknown network issue like I encountered
# credit: code snippet originally written by Herui Zhang
# download data from link https://gin.g-node.org/robintibor/high-gamma-dataset/
import numpy as np
import mne
import os


def preprocessing(data, fs):
    data = mne.filter.filter_data(data, l_freq=8, h_freq=32, sfreq=fs)
    return data


def get_source_eeg_Schirrmeister2017(person_id, current_working_dir, train, useall=True, resample_fs=250):
    if useall:
        path = current_working_dir + '/train/' + f'{person_id}.edf'
    else:
        if train:
            path = current_working_dir + 'train/' + f'{person_id}.edf'
        else:
            path = current_working_dir + 'test/' + f'{person_id}.edf'
    rawDataEDF = mne.io.read_raw_edf(path, preload=True, exclude=['EOG EOGh', 'EOG EOGv', 'EMG EMG_RH', 'EMG EMG_LH', 'EMG EMG_RF'])
    fs = int(rawDataEDF.info['sfreq'])
    event_position = rawDataEDF.annotations.onset
    event_type = rawDataEDF.annotations.description
    temp = rawDataEDF.to_data_frame().drop(['time'], axis=1)
    chan_time_all = temp.T.to_numpy()
    pre_data = preprocessing(chan_time_all, fs)
    if resample_fs != fs:
        pre_data = mne.filter.resample(pre_data, down=(fs / resample_fs))
        fs = resample_fs
    data = []
    label = []
    if useall:
        for xuhao, type_mi in enumerate(event_type):
            event_start_position = int(event_position[xuhao] * fs)
            if type_mi == 'left_hand':
                data.append(pre_data[:, event_start_position:event_start_position + fs * 4])
                label.append(0)
            elif type_mi == 'right_hand':
                data.append(pre_data[:, event_start_position:event_start_position + fs * 4])
                label.append(1)
            elif type_mi == 'feet':
                data.append(pre_data[:, event_start_position:event_start_position + fs * 4])
                label.append(2)
            elif type_mi == 'rest':
                data.append(pre_data[:, event_start_position:event_start_position + fs * 4])
                label.append(3)
        data = np.array(data)
        label = np.array(label).reshape(-1, )
    if useall:
        return data, label


def EA(X):
    num_trial, num_channel, num_sampls = np.shape(X)
    R = np.zeros((num_channel, num_channel))
    for i in range(num_trial):
        XTemp = np.squeeze(X[i, :, :])
        R = R + np.dot(XTemp, XTemp.T)
    R = R / num_trial
    R = Zsolve(R)
    for i in range(num_trial):
        XTemp = np.squeeze(X[i, :, :])
        XTemp = np.dot(R, XTemp)
        X[i, :, :] = XTemp
    return X


def Zsolve(R):
    v, Q = np.linalg.eig(R)
    ss1 = np.diag(v ** (-0.5))
    ss1[np.isnan(ss1)] = 0
    re = np.dot(Q, np.dot(ss1, np.linalg.inv(Q)))
    return np.real(re)


if __name__ == '__main__':

    # TODO: please replace path with your downloaded data file path https://gin.g-node.org/robintibor/high-gamma-dataset/
    working_dir = 'path_to_replace/Schirrmeister2017/'

    resample_fs = 250
    all_data = []
    all_label = []
    for person_id in range(1, 15):
        data, label = get_source_eeg_Schirrmeister2017(person_id=person_id, current_working_dir=working_dir, train=True, useall=True, resample_fs=resample_fs)
        print('subject id, data.shape, label.shape:', person_id, data.shape, label.shape)
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data)
    all_label = np.concatenate(all_label)
    print('all data.shape, label.shape:', all_data.shape, all_label.shape)
    np.save('./data/HighGamma/X.npy', all_data)
    np.save('./data/HighGamma/labels.npy', all_label)