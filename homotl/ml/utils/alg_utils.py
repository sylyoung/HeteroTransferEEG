# -*- coding: utf-8 -*-
# @Time    : 2023/10/24
# @Author  : Siyang Li
# @File    : alg_utils.py
import numpy as np
import torch
import sys
import pyriemann

import torch.nn.functional as F
from scipy.linalg import fractional_matrix_power


def EA(x):
    """
    Euclidean Alignment by He, He
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA


def EA_online(x, R, sample_num):
    """
    Parameters
    ----------
    x : numpy array
        sample of shape (num_channels, num_time_samples)
    R : numpy array
        current reference matrix (num_channels, num_channels)
    sample_num: int
        previous number of samples used to calculate R

    Returns
    ----------
    refEA : numpy array
        data of shape (num_channels, num_channels)
    """

    cov = np.cov(x)
    refEA = (R * sample_num + cov) / (sample_num + 1)
    return refEA


def EA_ref(x):
    """
    Parameters
    ----------
    x : numpy array
        data of shape (num_samples, num_channels, num_time_samples)

    Returns
    ----------
    XEA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    """
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA, refEA


def LA(Xs, Ys, Xt, Yt, use_logeuclid=True):
    """
    Label Alignment by He, He
    Parameters
    ----------
    Xs : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    Xt : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    Ys : numpy array, label of 0, 1, ... (int)
        data of shape (num_samples, )
    Yt : numpy array, label of 0, 1, ... (int)
        data of shape (num_samples, )
    use_logeuclid: boolean, whther to use log euclidean mean

    Returns
    ----------
    XLA : numpy array
        data of shape (num_samples, num_channels, num_time_samples)
    YLA : numpy array
        data of shape (num_samples, )
    """

    assert Xs.shape[1] == Xt.shape[1], print('LA Error, channel mismatch!')
    assert Xs.shape[2] == Xt.shape[2], print('LA Error, time sample mismatch!')
    label_space_s, cnts_class_s = np.unique(Ys, return_counts=True)
    label_space_t, cnts_class_t = np.unique(Yt, return_counts=True)
    assert len(label_space_s) == len(label_space_t), print('LA Error, label space mismatch!')
    num_classes = len(label_space_s)

    Xs_by_labels = []
    Xt_by_labels = []
    for c in range(num_classes):
        inds_class = np.where(Ys == c)[0]
        Xs_by_labels.append(Xs[inds_class])
        inds_class = np.where(Yt == c)[0]
        Xt_by_labels.append(Xt[inds_class])

    covxs = []
    covxt = []

    for c in range(num_classes):
        covxs.append(np.zeros((cnts_class_s[c], Xs.shape[1], Xs.shape[1])))
        covxt.append(np.zeros((cnts_class_t[c], Xt.shape[1], Xt.shape[1])))

    XLA = []
    YLA = []

    for c in range(num_classes):
        for i in range(len(covxs[c])):
            covxs[c][i] = np.cov(Xs_by_labels[c][i])
        for i in range(len(covxt[c])):
            covxt[c][i] = np.cov(Xt_by_labels[c][i])

        if use_logeuclid:
            covxs_class = pyriemann.utils.mean.mean_logeuclid(covxs[c])
            covxt_class = pyriemann.utils.mean.mean_logeuclid(covxt[c])
        else:
            covxs_class = np.mean(covxs[c], axis=0)
            covxt_class = np.mean(covxt[c], axis=0)
        sqrtCs = fractional_matrix_power(covxs_class, -0.5)
        sqrtCt = fractional_matrix_power(covxt_class, 0.5)
        A = np.dot(sqrtCt, sqrtCs)
        for i in range(len(Xs_by_labels[c])):
            XLA.append(np.dot(A, Xs_by_labels[c][i]))
            YLA.append(c)

    XLA = np.array(XLA)
    YLA = np.array(YLA).reshape(-1,)
    assert XLA.shape == Xs.shape, print('LA Error, X shape problem!')
    assert YLA.shape == Ys.shape, print('LA Error, Y shape problem!')
    assert np.unique(YLA, return_counts=True)[1][0] == cnts_class_s[0], print('LA Error, labels problem!')

    return XLA, YLA
