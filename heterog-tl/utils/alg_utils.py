# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : alg_utils.py
import numpy as np
import torch
import pyriemann
import torch.nn.functional as F

from scipy.linalg import fractional_matrix_power
from sklearn.metrics import accuracy_score


def EA(x):
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


def EA_online_continual(x, R, sample_num):
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


def predict_pseudo_center(model, loader, args):
    """
    predict target pseudo labels with model and find class centers using pseudo prediction labels

    Parameters
    ----------
    model: torch.nn.module, EEGNet
    data_iter: torch.utils.data.DataLoader
    args: argparse.Namespace, for transfer learning

    Returns
    -------
    target_centers: torch tensors, cpu
    """
    start_test = True
    model.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_inputs = inputs.cpu()
                all_output = outputs.cpu()
                #all_label = labels.float().cpu()
                start_test = False
            else:
                all_inputs = torch.cat((all_inputs, inputs.cpu()), 0)
                all_output = torch.cat((all_output, outputs.cpu()), 0)
                #all_label = torch.cat((all_label, labels.float().cpu()), 0)
    all_output = torch.nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    pred = torch.squeeze(predict).float()
    #true = all_label.cpu()
    #acc = accuracy_score(true, pred)
    for cls_id in range(args.class_num):
        indices = torch.where(pred == cls_id)[0]
        #print('cls_id:', cls_id)
        #print('indices:', indices)
        cls_data = torch.index_select(all_inputs, dim=0, index=indices)
        #print('cls_data.shape:', cls_data.shape)
        cls_center = torch.mean(cls_data, dim=0)
        #print('cls_center.shape:', cls_center.shape)
        if cls_id != 0:
            target_centers = torch.cat((cls_center, target_centers.float().cpu()), 0)
        else:
            target_centers = cls_center
    #print('target_centers.shape:', target_centers.shape)
    return target_centers


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


class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T
    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T ** 2) / y_s.shape[0]
        return loss