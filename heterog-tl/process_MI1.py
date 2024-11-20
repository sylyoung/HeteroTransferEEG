# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : process_MI1.py
# process MI1 (Blankertz2007) EEG data
# download data from link following https://www.bbci.de/competition/iv/ data sets1 ‹motor imagery, uncued classifier application› provided by the Berlin BCI group: Technische Universität Berlin
import numpy as np
import scipy.io as sio
import os
import mne


def process_MI1(path):

    print('preparing ' + 'MI1' + ' 1000Hz data...')
    paradigm = 'MI'
    num_subjects = 7
    sample_rate = 1000
    ch_num = 59

    data = []
    labels = []

    names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    for i in range(1, num_subjects+1):
        # skip subject 1 and 6, only use 5 out of 7 subjects
        '''
        1 left foot
        2 left right
        3 left right
        4 left right
        5 left right
        6 left foot
        7 left right
        '''
        if i == 1 or i == 6:
           continue
        mat = sio.loadmat(path + "/BCICIV_calib_ds1" + names[i - 1] +  "_1000Hz.mat")
        X = mat['cnt']
        X = np.transpose(X, (1, 0))
        print(X.shape)
        #X = np.transpose(X, (2, 0, 1))
        mrk = mat['mrk']
        pos = mrk['pos'][0, 0][0]

        trials = []
        for start in pos:
            start = int(start)
            trial = X[:, start - 1:start - 1 + 8000]

            ch_names = ['AF3','AF4','F5','F3','F1','Fz','F2','F4','F6','FC5','FC3','FC1','FCz','FC2','FC4','FC6','CFC7','CFC5','CFC3','CFC1','CFC2','CFC4','CFC6','CFC8','T7','C5','C3','C1','Cz','C2','C4','C6','T8','CCP7','CCP5','CCP3','CCP1','CCP2','CCP4','CCP6','CCP8','CP5','CP3','CP1','CPz','CP2','CP4','CP6','P5','P3','P1','Pz','P2','P4','P6','PO1','PO2','O1','O2']
            info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types=['eeg'] * 59)

            raw = mne.io.RawArray(trial, info)
            data_freqs = []

            trial = raw.get_data()

            print(trial.shape)

            trial = mne.filter.resample(trial, down=(1000 / 250))
            trial = mne.filter.filter_data(trial, l_freq=8, h_freq=32, sfreq=250)

            print('4 seconds')
            trial = trial[:, :1000]
            print(trial.shape)
            trials.append(trial)
        trials = np.stack(trials)

        y = mrk['y'][0, 0]
        y = (y + 1) / 2
        y = np.array(y).astype(int)
        y = y.reshape(200, )
        print(trials.shape, y.shape)  # (59, 300, 200) (200, 1)
        data.append(trials)
        labels.append(y)
    data = np.concatenate(data)
    labels = np.concatenate(labels)
    print(data.shape, labels.shape)
    if not os.path.exists('./data/MI1'):
        os.makedirs('./data/MI1')
    np.save('./data/MI1/X.npy', data)
    np.save('./MI1/labels.npy', labels)
    print('done')


if __name__ == '__main__':

    # TODO: please replace path with your downloaded data file path https://www.bbci.de/competition/iv/desc_1.html
    path = 'path_to_replace/BCICompetitionIV/BCICIV_1calib_1000Hz_mat'
    process_MI1(path)
