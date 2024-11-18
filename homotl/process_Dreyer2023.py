# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : download_data.py
# download from link and process EEG data
# credit: code snippet originally written by Yunlu Tu
import mne
import os
import shutil
import numpy as np
import scipy.io as sio

# channel list : Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6,
#                EOG1, EOG2, EOG3, EMGg, EMGd, F4, FC2, FC4, FC6, CP2,
#                CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3
# highpass: 0.0 Hz
# lowpass: 256.0 Hz
# sfreq: 512.0 Hz


shamfed = "_acquisition.gdf"
realfed = "_onlineT.gdf"
CEtail = "_CE_baseline.gdf"
OEtail = "_OE_baseline.gdf"
RUN = 6  # A59 missing R5 and R6
REST_RUN = 2
TRIAL = 40
NA = 60
NB = 21
NC = 6
REST_TIME = 180
sfreq = 512
CHAN = 32
MITIME = 8
TRIALTIME = 5


def preprocess_Dreyer2023(data_folder):

    all_data = []
    all_label = []
    lens = []
    for i in range(1, 60 + 1):
        mat = sio.loadmat(data_folder + "/" + 'A' + str(i) + ".mat")

        data = mat['data']
        label = mat['task_label']

        if i == 40:
            data = np.concatenate((data[:2, :, :, :], data[3:, :, :, :]))
            label = np.concatenate((label[:2], label[3:]))
        if i == 59:
            data = data[:4, :, :, :]
            label = label[:4]

        data = np.concatenate(data)
        label = np.concatenate(label)
        label = label - 1
        data = data[:, :, 512 * 3 - 1:]
        data = np.real(data)
        data = data.astype(np.float64)
        data = mne.filter.filter_data(data, l_freq=8, h_freq=32, sfreq=512)

        data = mne.filter.resample(data, down=(512 / 256))

        lens.append(len(data))

        all_data.append(data)
        all_label.append(label)

        print(i, data.shape, label.shape)

    data = np.concatenate(all_data)
    labels = np.concatenate(all_label)
    labels = labels.reshape(-1,)
    print(lens)

    print(data.shape, labels.shape)

    if not os.path.exists('./data/Dreyer2023'):
        os.makedirs('./data/Dreyer2023')
    np.save('./data/Dreyer2023/X.npy', data)
    np.save('./data/Dreyer2023/labels.npy', labels)
    print('done')


if __name__ == '__main__':

    # download data from link from original paper of Scientific Data by P. Dreyer
    # https://zenodo.org/records/8089820
    data_folder = './data/BCI Database'

    preprocess_Dreyer2023(data_folder)