# -*- coding: utf-8 -*-
# @Time    : 2023/10/16
# @Author  : Siyang Li
# @File    : tsne_raw.py
# plot BNCI2014002 and Dreyer2023 t-SNE distributions of raw extracted CSP features
import mne
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from xgboost import XGBClassifier
from scipy.linalg import fractional_matrix_power

from utils.alg_utils import EA

import os
import random


def data_process(dataset):
    '''
    :param dataset: str, dataset name
    :return: X, y, num_subjects, paradigm, sample_rate
    '''
    mne.set_log_level('warning')

    X = np.load('./data/' + dataset + '/X.npy')
    y = np.load('./data/' + dataset + '/labels.npy')
    print(X.shape, y.shape)

    num_subjects, paradigm, sample_rate = None, None, None

    if dataset == 'BNCI2014002':
        paradigm = 'MI'
        num_subjects = 14
        sample_rate = 512
        ch_num = 15

        trials_arr = np.array([100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60, 100, 60])
    elif dataset == 'Dreyer2023':
        paradigm = 'MI'
        num_subjects = 60
        sample_rate = 256
        ch_num = 27

        trials_arr = np.array([240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240,
                                      240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240,
                                      240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240,
                                      240, 240, 240, 240, 240, 240, 200, 240, 240, 240, 240,
                                      240, 240, 240, 240, 240, 240, 240, 240, 240, 240, 240,
                                      240, 240, 240, 160, 240])

        y = y.astype(int)

    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    print('data shape:', X.shape, ' labels shape:', y.shape)
    return X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr


def data_alignment(X, num_subjects):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    #print('before EA:', X.shape)
    out = []
    for i in range(num_subjects):
        tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    #print('after EA:', X.shape)
    return X


def tsne_subjects(dataset, align):
    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(dataset)
    print('X, y, num_subjects, paradigm, sample_rate, trials_arr:', X.shape, y.shape, num_subjects, paradigm, sample_rate, trials_arr)

    print('sample rate:', sample_rate)

    # align
    if align:
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
            aligned_x = EA(X_subjects[i])
            all_aligned_X.append(aligned_x)
        X = np.concatenate(all_aligned_X)
        y = np.concatenate(Y_subjects)

    print('X, y', X.shape, y.shape)

    if paradigm == 'MI':
        # CSP
        csp = mne.decoding.CSP(n_components=6)

        if dataset == 'Dreyer2023':
            accum_arr = []
            for t in range(len(trials_arr)):
                accum_arr.append(np.sum([trials_arr[:t]]))
            Xs = np.split(X, accum_arr)
            ys = np.split(y, accum_arr)
            print('len(Xs)', len(Xs))
            all_X = []
            all_y = []
            for X_subject, y_subject in zip(Xs, ys):
                X_subject = X_subject[:int(160 * 0.2)]
                y_subject = y_subject[:int(160 * 0.2)]
                all_X.append(X_subject)
                all_y.append(y_subject)
            X = np.concatenate(all_X)
            y = np.concatenate(all_y)

        X_csp = csp.fit_transform(X, y)
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(X_csp)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)

        csfont = {'fontname': "Times New Roman"}

        labels = []
        for i in range(num_subjects):
            labels.append(np.ones(len(y) // num_subjects) * i)  # assume equal num of trials across subjects
        labels = np.concatenate(labels)

        labels_colors = []
        markers = []

        labels_colors.append('red')
        labels_colors.append('blue')
        labels_colors.append('green')
        labels_colors.append('orange')
        labels_colors.append('purple')
        labels_colors.append('brown')
        labels_colors.append('pink')
        labels_colors.append('coral')
        labels_colors.append('grey')
        labels_colors.append('moccasin')
        labels_colors.append('sienna')
        labels_colors.append('lightgreen')
        labels_colors.append('darkviolet')
        labels_colors.append('azure')

        # Adding more colors
        labels_colors.append('gold')
        labels_colors.append('crimson')
        labels_colors.append('slateblue')
        labels_colors.append('darkgreen')
        labels_colors.append('lightblue')
        labels_colors.append('tan')
        labels_colors.append('darkred')
        labels_colors.append('teal')
        labels_colors.append('magenta')
        labels_colors.append('khaki')
        labels_colors.append('salmon')
        labels_colors.append('indigo')
        labels_colors.append('turquoise')
        labels_colors.append('chocolate')
        labels_colors.append('lime')
        labels_colors.append('lavender')
        labels_colors.append('plum')
        labels_colors.append('navy')
        labels_colors.append('chartreuse')
        labels_colors.append('peachpuff')
        labels_colors.append('mintcream')
        labels_colors.append('orangered')
        labels_colors.append('seagreen')
        labels_colors.append('deepskyblue')
        labels_colors.append('slategray')
        labels_colors.append('maroon')
        labels_colors.append('wheat')
        labels_colors.append('dodgerblue')
        labels_colors.append('orchid')
        labels_colors.append('cadetblue')
        labels_colors.append('palegreen')
        labels_colors.append('firebrick')
        labels_colors.append('mistyrose')
        labels_colors.append('midnightblue')
        labels_colors.append('greenyellow')
        labels_colors.append('tomato')
        labels_colors.append('peru')
        labels_colors.append('hotpink')
        labels_colors.append('mediumorchid')
        labels_colors.append('darkorange')
        labels_colors.append('palevioletred')
        labels_colors.append('forestgreen')
        labels_colors.append('royalblue')
        labels_colors.append('cornflowerblue')
        labels_colors.append('lightseagreen')
        labels_colors.append('darkkhaki')
        labels_colors.append('skyblue')
        labels_colors.append('springgreen')
        labels_colors.append('ivory')

        '''
        markers.append('^')
        markers.append('o')
        markers.append('x')
        markers.append('<')
        markers.append('>')
        markers.append('.')
        markers.append(',')
        markers.append('1')
        markers.append('2')
        '''

        X_2d = data

        for i in range(num_subjects):
            if len(markers) == 0:
                plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=3, c=labels_colors[i])
            else:
                plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=3, c=labels_colors[i], marker=markers[i])


        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 12,
                 }

        plt.tight_layout()

        if not os.path.exists("./files/"):
            os.makedirs("./files/")

        save_path = './files/' + dataset + '_subjects'
        if align:
            plt.savefig(save_path + '_EA.png', dpi=100, format='png')
            plt.savefig(save_path + '_EA.pdf', dpi=100, format='pdf')
            plt.savefig(save_path + '_EA.eps', dpi=100, format='eps')
        else:
            plt.savefig(save_path + '.png', dpi=100, format='png')
            plt.savefig(save_path + '.pdf', dpi=100, format='pdf')
            plt.savefig(save_path + '.eps', dpi=100, format='eps')
        plt.clf()


def tsne_labels(dataset, align):
    X, y, num_subjects, paradigm, sample_rate, ch_num, trials_arr = data_process(dataset)
    print('X, y, num_subjects, paradigm, sample_rate, trials_arr:', X.shape, y.shape, num_subjects, paradigm, sample_rate, trials_arr)

    print('sample rate:', sample_rate)

    # align
    if align:
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
            aligned_x = EA(X_subjects[i])
            all_aligned_X.append(aligned_x)
        X = np.concatenate(all_aligned_X)
        y = np.concatenate(Y_subjects)

    print('X, y', X.shape, y.shape)

    if paradigm == 'MI':
        # CSP
        csp = mne.decoding.CSP(n_components=6)

        if dataset == 'Dreyer2023':
            accum_arr = []
            for t in range(len(trials_arr)):
                accum_arr.append(np.sum([trials_arr[:t]]))
            Xs = np.split(X, accum_arr)
            ys = np.split(y, accum_arr)
            print('len(Xs)', len(Xs))
            all_X = []
            all_y = []
            for X_subject, y_subject in zip(Xs, ys):
                X_subject = X_subject[:int(160 * 0.2)]
                y_subject = y_subject[:int(160 * 0.2)]
                all_X.append(X_subject)
                all_y.append(y_subject)
            X = np.concatenate(all_X)
            y = np.concatenate(all_y)

        X_csp = csp.fit_transform(X, y)

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(X_csp)
        x_min, x_max = np.min(features, 0), np.max(features, 0)
        data = (features - x_min) / (x_max - x_min)

        csfont = {'fontname': "Times New Roman"}

        labels = y

        labels_colors = []
        markers = []

        labels_colors.append('red')
        labels_colors.append('blue')

        '''
        markers.append('^')
        markers.append('o')
        markers.append('x')
        markers.append('<')
        markers.append('>')
        markers.append('.')
        markers.append(',')
        markers.append('1')
        markers.append('2')
        '''

        X_2d = data

        for i in range(2):
            if len(markers) == 0:
                plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=3, c=labels_colors[i])
            else:
                plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=3, c=labels_colors[i], marker=markers[i])

        font1 = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': 12,
                 }

        plt.tight_layout()

        if not os.path.exists("./files/"):
            os.makedirs("./files/")

        save_path = './files/' + dataset + '_labels'
        if align:
            plt.savefig(save_path + '_EA.png', dpi=100, format='png')
            plt.savefig(save_path + '_EA.pdf', dpi=100, format='pdf')
            plt.savefig(save_path + '_EA.eps', dpi=100, format='eps')
        else:
            plt.savefig(save_path + '.png', dpi=100, format='png')
            plt.savefig(save_path + '.pdf', dpi=100, format='pdf')
            plt.savefig(save_path + '.eps', dpi=100, format='eps')
        plt.clf()


if __name__ == '__main__':

    scores = []

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_arr = ['Dreyer2023', 'BNCI2014002']
    for dataset in dataset_arr:
        tsne_subjects(dataset, align=False)
        tsne_subjects(dataset, align=True)
        tsne_labels(dataset, align=False)
        tsne_labels(dataset, align=True)


