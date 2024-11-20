# -*- coding: utf-8 -*-
# @Time    : 2023/12/22
# @Author  : Siyang Li
# @File    : algorithms.py
import numpy as np
import torch
import torch.nn.functional as F

from scipy.linalg import fractional_matrix_power


def EA_offline(x):
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
    return XEA


def EA_online(x):
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
    x_aligned = []
    R = 0
    num_samples = 0
    for ind in range(len(x)):
        curr = x[ind]
        cov = np.cov(curr)
        R = (R * num_samples + cov) / (num_samples + 1)
        num_samples += 1
        sqrtRefEA = fractional_matrix_power(R, -0.5)
        curr_aligned = np.dot(sqrtRefEA, curr)
        x_aligned.append(curr_aligned)
    XEA = np.array(x_aligned)
    return XEA