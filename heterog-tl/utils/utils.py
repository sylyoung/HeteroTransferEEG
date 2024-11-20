# -*- coding: utf-8 -*-
# @Time    : 2023/07/13
# @Author  : Siyang Li
# @File    : utils.py
import os.path as osp
import os
import numpy as np
import random
import sys

import torch as tr
import torch.nn as nn
import torch.utils.data
import torch.utils.data as Data
from torch.utils.data import BatchSampler
from torch.utils.data.sampler import WeightedRandomSampler
import moabb
import mne
import learn2learn as l2l
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score, f1_score
from scipy.linalg import fractional_matrix_power
#from learn2learn.data.transforms import NWays, KShots, LoadData
import torch.fft as fft

from utils.alg_utils import EA, EA_online
from utils.aug_utils import leftrightflipping_transform
from utils.data_augment import data_aug

from moabb.datasets import BNCI2014001, BNCI2014002, BNCI2014008, BNCI2014009, BNCI2015003, BNCI2015004, EPFLP300, \
    BNCI2014004, BNCI2015001
from moabb.paradigms import MotorImagery, P300


def split_data(data, axis, times):
    # Splitting data into multiple sections. data: (trials, channels, time_samples)
    data_split = np.split(data, indices_or_sections=times, axis=axis)
    return data_split


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
    elif dataset_name == 'BNCI2014004':
        dataset = BNCI2014004()
        paradigm = MotorImagery(n_classes=2)
        # (6520, 3, 1126) (6520,) 250Hz 9subjects * 2classes * (?)trials * 5sessions
    elif dataset_name == 'BNCI2015001':
        dataset = BNCI2015001()
        paradigm = MotorImagery(n_classes=2)
        # (5600, 13, 2561) (5600,) 512Hz 12subjects * 2 classes * (200 + 200 + (200 for Subj 8/9/10/11)) trials * (2/3)sessions
    elif dataset_name == 'MI1':
        info = None
        return info
        # (1400, 59, 300) (1400,) 100Hz 7subjects * 2classes * 200trials * 1session
    elif dataset_name == 'BNCI2015004':
        dataset = BNCI2015004()
        paradigm = MotorImagery(n_classes=2)
        # [160, 160, 160, 150 (80+70), 160, 160, 150 (80+70), 160, 160]
        # (1420, 30, 1793) (1420,) 256Hz 9subjects * 2classes * (80+80/70)trials * 2sessions
    elif dataset_name == 'BNCI2014008':
        dataset = BNCI2014008()
        paradigm = P300()
        # (33600, 8, 257) (33600,) 256Hz 8subjects 4200 trials * 1session
    elif dataset_name == 'BNCI2014009':
        dataset = BNCI2014009()
        paradigm = P300()
        # (17280, 16, 206) (17280,) 256Hz 10subjects 1728 trials * 3sessions
    elif dataset_name == 'BNCI2015003':
        dataset = BNCI2015003()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 2520 trials * 1session
    elif dataset_name == 'EPFLP300':
        dataset = EPFLP300()
        paradigm = P300()
        # (25200, 8, 206) (25200,) 256Hz 10subjects 1session
    elif dataset_name == 'ERN':
        ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
                    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2',
                    'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3',
                    'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'POz', 'P08', 'O1', 'O2']
        info = mne.create_info(ch_names=ch_names, sfreq=200, ch_types=['eeg'] * 56)
        return info
        # (5440, 56, 260) (5440,) 200Hz 16subjects 1session
    # SEED (152730, 62, 5*DE*)  (152730,) 200Hz 15subjects 3sessions

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
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info
        elif isinstance(paradigm, P300):
            X, labels, meta = paradigm.get_data(dataset=dataset, subjects=dataset.subject_list[:], return_epochs=True)
            return X.info


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def fix_random_seed(SEED):
    tr.manual_seed(SEED)
    tr.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


def create_folder(dir_name, data_env, win_root):
    if not osp.exists(dir_name):
        os.system('mkdir -p ' + dir_name)
    if not osp.exists(dir_name):
        if data_env == 'gpu':
            os.mkdir(dir_name)
        elif data_env == 'local':
            os.makedirs(win_root + dir_name)


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def lr_scheduler_full(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def cal_loss(loader, model, flag=True, fc=None, args=None):
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if hasattr(args, 'target_common_mat'):
                inputs = inputs[:, :, args.target_common_mat, :]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            inputs = inputs
            # for cross-device common channel XDVC-CMCH
            #if args.target_common_mat:
            #    inputs = inputs[:, :, args.target_common_mat, :]
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs), 0)
                all_label = tr.cat((all_label, labels), 0)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(all_output, all_label)
    return loss.item()


def cal_acc(loader, netF, netC, args=None):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    accuracy = accuracy_score(true, pred)

    return accuracy * 100, all_output


def cal_bca(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    bca = balanced_accuracy_score(true, pred)
    return bca * 100, all_output


def cal_auc(loader, netF, netC):
    start_test = True
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cuda()
            labels = data[1].float()
            outputs = netC(netF(inputs))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    true = all_label.cpu()
    pred = all_output[:, 1].detach().numpy()
    auc = roc_auc_score(true, pred)
    return auc * 100, all_output


def cal_acc_comb(loader, model, flag=True, fc=None, args=None):
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            # if hasattr(args, 'target_common_mat'):
            #     inputs = inputs[:, :, args.target_common_mat, :]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            inputs = inputs
            # for cross-device common channel XDVC-CMCH
            #if args.target_common_mat:
            #    inputs = inputs[:, :, args.target_common_mat, :]
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    if args.paradigm == 'MI':
        acc = accuracy_score(true, pred)
        return acc * 100, all_output
    elif args.paradigm == 'ERP':
        #bca = balanced_accuracy_score(true, pred)
        try:
            auc = roc_auc_score(true, all_output[:, 1])
        except:
            np.set_printoptions(threshold=sys.maxsize)
            print(all_output)
            sys.exit(0)
        return auc * 100, all_output
    elif args.paradigm == 'aBCI':
        acc = accuracy_score(true, pred)
        bca = balanced_accuracy_score(true, pred)
        f1 = f1_score(true, pred, average='micro')
        return acc * 100, bca * 100, f1 * 100, all_output
    elif args.paradigm == 'visual':
        acc = balanced_accuracy_score(true, pred)
        return acc * 100, all_output


def cal_acc_minet(loader, model, flag=True, fc=None, args=None, mats=None):
    left_mat, middle_mat, right_mat = mats
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            inputs_left, inputs_middle, inputs_right = [inputs[:, :, left_mat, :],
                                                        inputs[:, :, middle_mat, :],
                                                        inputs[:, :, right_mat, :]]
            inputs = [inputs_left, inputs_middle, inputs_right]

            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred)

    return acc * 100, all_output


def cal_metrics_comb(loader, model, flag=True, fc=None, args=None):
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            inputs = inputs
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = tr.max(all_output, 1)
    pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    acc = accuracy_score(true, pred)
    bca = balanced_accuracy_score(true, pred)
    auc = roc_auc_score(true, all_output[:, 1])
    f1 = f1_score(true, pred, average='micro')

    return acc * 100, bca * 100, auc * 100, f1 * 100


def convert_label(labels, axis, threshold):
    # Converting labels to 0 or 1, based on a certain threshold
    label_01 = np.where(labels > threshold, 1, 0)
    # print(label_01)
    return label_01

"""
# for TTA paper
def cal_score_online(loader, model, args):
    y_true = []
    y_pred = []
    model.eval()
    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cpu()
            labels = data[1]
            if i == 0:
                data_cum = inputs.float().cpu()
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)

            if args.align:
                # update reference matrix
                R = EA_online(inputs.reshape(args.chn, args.time_sample_num), R, i)
                sqrtRefEA = fractional_matrix_power(R, -0.5)
                # transform current test sample
                inputs = np.dot(sqrtRefEA, inputs)
                inputs = inputs.reshape(1, 1, args.chn, args.time_sample_num)

            inputs = torch.from_numpy(inputs).to(torch.float32)
            if args.data_env != 'local':
                inputs = inputs.cuda()
            _, outputs = model(inputs)
            outputs = outputs.float().cpu()
            labels = labels.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

            if i == 0:
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    if args.balanced:
        score = accuracy_score(y_true, y_pred)
    else:
        all_output = nn.Softmax(dim=1)(all_output)
        true = all_label.cpu()
        pred = all_output[:, 1].detach().numpy()
        score = roc_auc_score(true, pred)

    return score * 100
"""


def cal_score_online(loader, model, args):
    y_true = []
    y_pred = []
    model.eval()
    # initialize test reference matrix for Incremental EA
    if args.align:
        R = 0
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0].cpu()
            labels = data[1]
            if i == 0:
                data_cum = inputs.float().cpu()
            else:
                data_cum = tr.cat((data_cum, inputs.float().cpu()), 0)

            if args.align:
                # update reference matrix
                R = EA_online(inputs.reshape(args.chn, args.time_sample_num), R, i)
                sqrtRefEA = fractional_matrix_power(R, -0.5)
                # transform current test sample
                inputs = np.dot(sqrtRefEA, inputs)
                inputs = inputs.reshape(1, 1, args.chn, args.time_sample_num)
                inputs = torch.from_numpy(inputs).to(torch.float32)

            if args.data_env != 'local':
                inputs = inputs.cuda()
            _, outputs = model(inputs)
            outputs = outputs.float().cpu()
            labels = labels.float().cpu()
            _, predict = tr.max(outputs, 1)
            pred = tr.squeeze(predict).float()
            y_pred.append(pred.item())
            y_true.append(labels.item())

            if i == 0:
                all_output = outputs.float().cpu()
                all_label = labels.float()
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)

    if args.paradigm == 'MI':
        score = accuracy_score(y_true, y_pred)
    elif args.paradigm == 'ERP':
        score = balanced_accuracy_score(y_true, y_pred)
    else:
        all_output = nn.Softmax(dim=1)(all_output)
        true = all_label.cpu()
        pred = all_output[:, 1].detach().numpy()
        score = roc_auc_score(true, pred)

    return score * 100


def cal_auc_comb(loader, model, flag=True, fc=None, args=None):
    start_test = True
    model.eval()
    with tr.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            if args.data_env != 'local':
                inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    outputs, _ = model(inputs)  # modified
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = tr.cat((all_output, outputs.float().cpu()), 0)
                all_label = tr.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    # _, predict = tr.max(all_output, 1)
    # pred = tr.squeeze(predict).float()
    true = all_label.cpu()
    pred = all_output[:, 1].detach().numpy()
    auc = roc_auc_score(true, pred)

    return auc * 100, all_output


def cal_metrics_multisource(loader, nets, args, metrics, idt):
    # mode 'avg', 'vote'
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            all_probs = None
            for i in range(args.N):
                if args.data_env != 'local':
                    x = x.cuda()
                    y = y.cuda()
                if i == idt:  # skip test domain model
                    continue
                outputs = nets[i][0](x)
                _, outputs = nets[i][1](outputs)
                predicted_probs = torch.nn.functional.softmax(outputs, dim=1)
                if all_probs is None:
                    all_probs = torch.zeros((x.shape[0], args.class_num))
                    if args.data_env != 'local':
                        all_probs = all_probs.cuda()
                else:
                    all_probs += predicted_probs.reshape(x.shape[0], args.class_num)

                _, predicted = torch.max(predicted_probs, 1)

                if args.mode == 'vote':
                    votes = torch.zeros((x.shape[0], args.class_num))
                    if args.data_env != 'local':
                        votes = votes.cuda()
                    for i in range(x.shape[0]):
                        votes[i, predicted[i]] += 1
            if args.mode == 'vote':
                _, predicted = torch.max(votes, 1)  # VOTING
            if args.mode == 'avg':
                _, predicted = torch.max(all_probs, 1)  # AVERAGING
            y_true.append(y.cpu())
            y_pred.append(predicted.cpu())
    score = metrics(np.concatenate(y_true).reshape(-1, ).tolist(), np.concatenate(y_pred)).reshape(-1, )[0]
    return score * 100


def data_alignment(X, num_subjects, args):
    '''
    :param X: np array, EEG data
    :param num_subjects: int, number of total subjects in X
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    if args.data == 'BNCI2015003' and len(
            X) < 141:  # check is dataset BNCI2015003 and is downsampled and is not testset
        # upsampling for unequal distributions across subjects, i.e., each subject is upsampled to different num of trials
        print('before EA:', X.shape)
        out = []
        inds = [140, 140, 140, 140, 640, 840, 840, 840, 840, 840]
        inds = np.delete(inds, args.idt)
        for i in range(num_subjects):
            tmp_x = EA(X[np.sum(inds[:i]):np.sum(inds[:i + 1]), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    elif args.data == 'BNCI2015003' and len(X) > 25200:  # check is dataset BNCI2015003 and is upsampled
        # upsampling for unequal distributions across subjects, i.e., each subject is upsampled to different num of trials
        print('before EA:', X.shape)
        out = []
        inds = [4900, 4900, 4900, 4900, 4400, 4200, 4200, 4200, 4200, 4200]
        inds = np.delete(inds, args.idt)
        for i in range(num_subjects):
            tmp_x = EA(X[np.sum(inds[:i]):np.sum(inds[:i + 1]), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    else:
        print('before EA:', X.shape)
        out = []
        for i in range(num_subjects):
            tmp_x = EA(X[X.shape[0] // num_subjects * i:X.shape[0] // num_subjects * (i + 1), :, :])
            out.append(tmp_x)
        X = np.concatenate(out, axis=0)
        print('after EA:', X.shape)
    return X


def data_alignment_returnref(x):
    '''
    :param X: np array, EEG data
    :return: np array, aligned EEG data; np array, reference matrix
    '''
    cov = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
    for i in range(x.shape[0]):
        cov[i] = np.cov(x[i])
    refEA = np.mean(cov, 0)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5)
    XEA = np.zeros(x.shape)
    for i in range(x.shape[0]):
        XEA[i] = np.dot(sqrtRefEA, x[i])
    return XEA, refEA


def data_alignment_uneven(X, trials_arr):
    '''
    :param X: np array, EEG data
    :param trials_arr: np array of ints, number of trials per each subject
    :return: np array, aligned EEG data
    '''
    # subject-wise EA
    print('before EA:', X.shape)
    out = []
    for i in range(len(trials_arr)):
        print(int(np.sum(trials_arr[:(i)])), int(np.sum(trials_arr[:(i+1)])))
        tmp_x = EA(X[int(np.sum(trials_arr[:i])):int(np.sum(trials_arr[:(i+1)])), :, :])
        out.append(tmp_x)
    X = np.concatenate(out, axis=0)
    print('after EA:', X.shape)
    return X


def data_loader_target(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # cross-subject target loader
    dset_loaders = {}

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    Xt = Xt.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys, Xt, Yt = Xs.cuda(), Ys.cuda(), Xt.cuda(), Yt.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar = Data.TensorDataset(Xt, Yt)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=args.target_batch_size, shuffle=True, drop_last=True)
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=args.target_batch_size, shuffle=True, drop_last=True)


    # for TL test
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=args.target_batch_size * 3, shuffle=False, drop_last=False)

    # source specific loaders
    if hasattr(args, 'tgttrials_arr'):
        t_arr = args.tgttrials_arr.copy()
        minus_one = False
        accum_arr = []
        for t in range(args.tgtN - 1):
            if t == args.idt:  # remove target
                t_arr = np.delete(t_arr, t)
                minus_one = True
            else:
                ind = t + 1
                if minus_one:
                    ind -= 1
                accum_arr.append(np.sum([t_arr[:ind]]))
        Xs_subjects = torch.tensor_split(Xs, tensor_indices_or_sections=torch.tensor(accum_arr), dim=0)
        Ys_subjects = torch.tensor_split(Ys, tensor_indices_or_sections=torch.tensor(accum_arr), dim=0)
    else:
        Xs_subjects = torch.split(Xs, len(Xs) // (args.tgtN - 1))
        Ys_subjects = torch.split(Ys, len(Xs) // (args.tgtN - 1))
    data_valid_train = []
    label_valid_train = []
    data_valid_test = []
    label_valid_test = []

    dset_loaders["sources"] = []
    for i in range(args.tgtN - 1):
        data_subject = Data.TensorDataset(Xs_subjects[i], Ys_subjects[i])
        dset_loaders["sources"].append(Data.DataLoader(data_subject, batch_size=args.target_batch_size, shuffle=True, drop_last=True))
        data_valid_train.append(Xs_subjects[i][:int(len(Xs_subjects[i]) * 0.8)])
        label_valid_train.append(Ys_subjects[i][:int(len(Ys_subjects[i]) * 0.8)])
        data_valid_test.append(Xs_subjects[i][int(len(Xs_subjects[i]) * 0.8):])
        label_valid_test.append(Ys_subjects[i][int(len(Ys_subjects[i]) * 0.8):])

    data_valid_train = torch.concat(data_valid_train)
    label_valid_train = torch.concat(label_valid_train)
    data_valid_test = torch.concat(data_valid_test)
    label_valid_test = torch.concat(label_valid_test)
    dataset_valid_train = Data.TensorDataset(data_valid_train, label_valid_train)
    dataset_valid_test = Data.TensorDataset(data_valid_test, label_valid_test)

    dset_loaders["source-val-train"] = Data.DataLoader(dataset_valid_train, batch_size=args.target_batch_size, shuffle=True, drop_last=True)
    dset_loaders["source-val-test"] = Data.DataLoader(dataset_valid_test, batch_size=args.target_batch_size, shuffle=False, drop_last=False)

    return dset_loaders


def data_loader_all(X, y, N, trials_arr, args):
    # all data loader
    dset_loaders = {}
    train_bs = args.batch_size

    X, y = tr.from_numpy(X).to(
        tr.float32), tr.from_numpy(y.reshape(-1, )).to(tr.long)
    X = X.unsqueeze_(3)
    # if 'EEGNet' or 'MINet' in args.backbone:
    X = X.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        X, y = X.cuda(), y.cuda()

    data = Data.TensorDataset(X, y)

    # for train
    dset_loaders["train"] = Data.DataLoader(data, batch_size=train_bs, shuffle=True, drop_last=True)

    # source specific loaders
    t_arr = trials_arr.copy()
    accum_arr = []
    for t in range(1, N+1):
        accum_arr.append(np.sum([t_arr[:t]]))
    print(accum_arr)
    Xs_subjects = torch.tensor_split(X, tensor_indices_or_sections=torch.tensor(accum_arr), dim=0)
    Ys_subjects = torch.tensor_split(y, tensor_indices_or_sections=torch.tensor(accum_arr), dim=0)
    dset_loaders["sources"] = []
    # dset_loaders["sources-dg"] = []
    #subject_batch = train_bs // N
    #if train_bs % N != 0:
    #    subject_batch += 1
    for i in range(N):
        # print(Xs_subjects[i].shape, Ys_subjects[i].shape)
        data_subject = Data.TensorDataset(Xs_subjects[i], Ys_subjects[i])
        dset_loaders["sources"].append(Data.DataLoader(data_subject, batch_size=train_bs, shuffle=True, drop_last=True))
        # dset_loaders["sources-dg"].append(Data.DataLoader(data_subject, batch_size=train_bs, shuffle=True, drop_last=True))

    return dset_loaders

def data_loader_1dataset(X, y, N, trials_arr, args, train_bs=None):
    # all data loader
    dset_loaders = {}
    if train_bs is None:
        train_bs = args.batch_size

    X, y = tr.from_numpy(X).to(
        tr.float32), tr.from_numpy(y.reshape(-1, )).to(tr.long)
    X = X.unsqueeze_(3)
    X = X.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        X, y = X.cuda(), y.cuda()

    data = Data.TensorDataset(X, y)

    # for train
    dset_loaders["train"] = Data.DataLoader(data, batch_size=train_bs, shuffle=True, drop_last=True)

    return dset_loaders


def data_convert(X, y, N, trials_arr, args, train_bs=None):
    # all data loader
    dset_loaders = {}
    if train_bs is None:
        train_bs = args.batch_size

    X, y = tr.from_numpy(X).to(
        tr.float32), tr.from_numpy(y.reshape(-1, )).to(tr.long)
    X = X.unsqueeze_(3)
    X = X.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        X, y = X.cuda(), y.cuda()

    data = Data.TensorDataset(X, y)

    return data


def data_loader_augbaseline(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # cross-subject loader
    # augmentation approach baselines
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_copy = Xt.copy()

    Xt_copy_online = Xt.copy()
    Yt_copy_online = Yt.copy()

    if args.align:
        Xs = data_alignment(Xs, args.N - 1, args)
        Xt = data_alignment(Xt, 1, args)

    # mult_flag, noise_flag, neg_flag, freq_mod_flag
    if 'mult' in args.method:
        flag_aug = [True, False, False, False]
    elif 'noise' in args.method:
        flag_aug = [False, True, False, False]
    elif 'neg' in args.method:
        flag_aug = [False, False, True, False]
    elif 'freq' in args.method:
        flag_aug = [False, False, False, True]
    elif 'all' in args.method:
        flag_aug = [True, True, True, True]
    print('flag_aug:', flag_aug)

    print('original data:', Xs.shape, Ys.shape)
    aug_out = data_aug(np.transpose(Xs, (0, 2, 1)), Ys, Xs.shape[-1], flag_aug)
    Xs, Ys = aug_out
    Xs = np.transpose(Xs, (0, 2, 1))
    print('augmented data:', Xs.shape, Ys.shape)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    Xt_copy = tr.from_numpy(Xt_copy).to(
        tr.float32)
    Xt_copy = Xt_copy.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_copy = Xt_copy.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys, Xt, Yt, Xt_copy = Xs.cuda(), Ys.cuda(), Xt.cuda(), Yt.cuda(), Xt_copy.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar = Data.TensorDataset(Xt, Yt)

    data_tar_online = Data.TensorDataset(Xt_copy, Yt)

    test_x_aligned = []
    R = 0
    num_samples = 0
    for ind in range(len(Yt_copy_online)):
        curr = Xt_copy_online[ind]
        R = EA_online(curr, R, num_samples)
        num_samples += 1
        sqrtRefEA = fractional_matrix_power(R, -0.5)
        curr_aligned = np.dot(sqrtRefEA, curr)
        test_x_aligned.append(curr_aligned)
    test_x = np.stack(test_x_aligned)
    test_x = tr.from_numpy(test_x).to(
        tr.float32)
    test_x = test_x.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        test_x = test_x.permute(0, 3, 1, 2)
    Yt_copy_online = tr.from_numpy(Yt_copy_online.reshape(-1, )).to(tr.long)
    if args.data_env != 'local':
        test_x, Yt_copy_online = test_x.cuda(), Yt_copy_online.cuda()
    data_tar_online_test = Data.TensorDataset(test_x, Yt_copy_online)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online TL
    dset_loaders["Target-Online"] = Data.DataLoader(data_tar_online, batch_size=1, shuffle=False, drop_last=False)

    # for online TL test, aligned
    dset_loaders["target-online"] = Data.DataLoader(data_tar_online_test, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def data_loader_sdaTL(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # supervised domain adaptation loader
    dset_loaders = {}
    train_bs = args.batch_size

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    Xs = Xs.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys = Xs.cuda(), Ys.cuda()

    Source_data = Data.TensorDataset(Xs, Ys)

    dset_loaders["Source"] = Data.DataLoader(Source_data, batch_size=train_bs, shuffle=True, drop_last=True)

    Xt_train, Xt_test = Xt[:args.num_target_train], Xt[args.num_target_train:]
    Yt_train, Yt_test = Yt[:args.num_target_train], Yt[args.num_target_train:]

    Xt_train, Yt_train = tr.from_numpy(Xt_train).to(
        tr.float32), tr.from_numpy(Yt_train.reshape(-1, )).to(tr.long)
    Xt_train = Xt_train.unsqueeze_(3)
    Xt_train = Xt_train.permute(0, 3, 1, 2)

    Xt_test, Yt_test = tr.from_numpy(Xt_test).to(
        tr.float32), tr.from_numpy(Yt_test.reshape(-1, )).to(tr.long)
    Xt_test = Xt_test.unsqueeze_(3)
    Xt_test = Xt_test.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xt_train, Xt_test, Yt_train, Yt_test = Xt_train.cuda(), Xt_test.cuda(), Yt_train.cuda(), Yt_test.cuda()
    data_tar_train = Data.TensorDataset(Xt_train, Yt_train)
    data_tar_test = Data.TensorDataset(Xt_test, Yt_test)

    #print('Xt_test, Yt_test', Xt_test.shape, Yt_test.shape)

    dset_loaders["target-train"] = Data.DataLoader(data_tar_train, batch_size=args.target_batch_size, shuffle=True, drop_last=False)
    #dset_loaders["target-unsup"] = Data.DataLoader(data_tar_test, batch_size=train_bs, shuffle=True, drop_last=False)
    dset_loaders["Target-test"] = Data.DataLoader(data_tar_test, batch_size=train_bs * 6, shuffle=False, drop_last=False)

    return dset_loaders


def data_sdaTL_convert(Xs=None, Ys=None, Xt=None, Yt=None, args=None, return_other=False):
    train_bs = args.batch_size

    Xt_train = Xt[:args.num_target_train]
    Yt_train = Yt[:args.num_target_train]

    Xt_train, Yt_train = tr.from_numpy(Xt_train).to(
        tr.float32), tr.from_numpy(Yt_train.reshape(-1, )).to(tr.long)
    Xt_train = Xt_train.unsqueeze_(3)
    Xt_train = Xt_train.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xt_train, Yt_train = Xt_train.cuda(), Yt_train.cuda()

    data_tar_train = Data.TensorDataset(Xt_train, Yt_train)

    if return_other:
        Xs, Ys = tr.from_numpy(Xs).to(
            tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
        Xs = Xs.unsqueeze_(3)
        if 'EEGNet' or 'ShallowCNN' or 'DeepCNN' in args.backbone:
            Xs = Xs.permute(0, 3, 1, 2)
        if args.data_env != 'local':
            Xs, Ys = Xs.cuda(), Ys.cuda()
        return data_tar_train, Xs, Ys
    return data_tar_train


def data_loader_crossTL(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # cross loader
    dset_loaders = {}

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    Xt = Xt.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Xt, Ys, Yt = Xs.cuda(), Xt.cuda(), Ys.cuda(), Yt.cuda()

    Source_data = Data.TensorDataset(Xs, Ys)
    Target_data = Data.TensorDataset(Xt, Yt)

    dset_loaders["Source"] = Data.DataLoader(Source_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dset_loaders["Target"] = Data.DataLoader(Target_data, batch_size=args.batch_size * 6, shuffle=False, drop_last=False)

    # split into calibration and test, where Source-Sup means target-train, Target-Sup means target-test
    if hasattr(args, 'num_target_train'):
        Xt_train, Xt_test = Xt[:args.num_target_train], Xt[args.num_target_train:]
        Yt_train, Yt_test = Yt[:args.num_target_train], Yt[args.num_target_train:]

        Source_data_sup = Data.TensorDataset(Xt_train, Yt_train)
        Target_data_sup = Data.TensorDataset(Xt_test, Yt_test)

        dset_loaders["target-train"] = Data.DataLoader(Source_data_sup, batch_size=args.target_batch_size, shuffle=False, drop_last=False)
        dset_loaders["target-test"] = Data.DataLoader(Target_data_sup, batch_size=args.batch_size * 6, shuffle=False, drop_last=False)

    return dset_loaders


def data_loader_sda(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # for CR aug
    # supervised transfer loader
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_train, Xt_test = Xt[:args.num_target_train], Xt[args.num_target_train:]
    Yt_train, Yt_test = Yt[:args.num_target_train], Yt[args.num_target_train:]

    Xt_train_copy_online = Xt_train.copy()
    Xt_test_copy_online = Xt_test.copy()
    Yt_test_copy_online = Yt_test.copy()

    if args.align:
        Xs = data_alignment(Xs, args.N - 1, args)
        Xt_train, R = data_alignment_returnref(Xt_train)  # save reference matrix
        Xt_test = data_alignment(Xt_test, 1, args)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt_train, Yt_train = tr.from_numpy(Xt_train).to(
        tr.float32), tr.from_numpy(Yt_train.reshape(-1, )).to(tr.long)
    Xt_train = Xt_train.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_train = Xt_train.permute(0, 3, 1, 2)

    Xt_test, Yt_test = tr.from_numpy(Xt_test).to(
        tr.float32), tr.from_numpy(Yt_test.reshape(-1, )).to(tr.long)
    Xt_test = Xt_test.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_test = Xt_test.permute(0, 3, 1, 2)

    if args.align:
        test_x_aligned = []
        num_samples = len(Yt_train)
        for ind in range(len(Yt_test_copy_online)):
            curr = Xt_test_copy_online[ind]
            R = EA_online(curr, R, num_samples)
            num_samples += 1
            sqrtRefEA = fractional_matrix_power(R, -0.5)
            curr_aligned = np.dot(sqrtRefEA, curr)
            test_x_aligned.append(curr_aligned)
        test_x = np.stack(test_x_aligned)
        test_x = tr.from_numpy(test_x).to(tr.float32)
        test_x = test_x.unsqueeze_(3)
        if 'EEGNet' in args.backbone:
            test_x = test_x.permute(0, 3, 1, 2)
        Yt_test_copy_online = tr.from_numpy(Yt_test_copy_online.reshape(-1, )).to(tr.long)
        if args.data_env != 'local':
            test_x, Yt_test_copy_online = test_x.cuda(), Yt_test_copy_online.cuda()
        data_tar_online_test = Data.TensorDataset(test_x, Yt_test_copy_online)

    # TODO
    if 'CR' in args.method:
        # mirroring
        if args.data_name == 'BNCI2014001':
            left_mat = [1, 2,
                        6, 7, 8, 13, 14,
                        18]
            right_mat = [5, 4,
                         12, 11, 10, 17, 16,
                         20]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = 1 - Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                #aug_labels_source = 1 - Ys
                aug_labels_source = Ys  # CS
        elif args.data_name == 'BNCI2014002':
            left_mat = [0, 3, 4, 5, 6, 12]
            right_mat = [2, 11, 10, 9, 8, 14]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = Ys
        elif args.data_name == 'BNCI2014004':
            left_mat = [0]
            right_mat = [2]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = 1 - Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                #aug_labels_source = 1 - Ys
                aug_labels_source = Ys  # CS
        elif args.data_name == 'BNCI2015001':
            left_mat = [0,
                        3, 4, 5, 10]
            right_mat = [2,
                         9, 8, 7, 12]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = Ys
        elif args.data_name == 'PhysionetMI':
            left_mat = [0, 1, 2, 21, 24, 25, 29, 30, 31, 32,
                        14, 15, 16, 38, 40, 42, 44, 46,
                        7, 8, 47, 48, 49,
                        55, 56, 60]
            right_mat = [6, 5, 4, 23, 28, 27, 37, 36, 35, 34,
                         20, 19, 18, 39, 41, 43, 45, 54,
                         13, 12, 53, 52, 51,
                         59, 58, 62]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = Ys
        elif 'MI1' in args.data_name:
            left_mat = [0, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 33, 34, 35, 36, 41, 42, 43, 48, 49,
                        50, 55, 57]
            right_mat = [1, 8, 7, 6, 15, 14, 13, 23, 22, 21, 20, 32, 31, 30, 29, 40, 39, 38, 37, 47, 46, 45, 54, 53,
                         52, 56, 58]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = 1 - Yt_train

            if args.idt == 0 or args.idt == 5:
                aug_inputs_tgttrain = Xt_train
                aug_labels_tgttrain = Yt_train

            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                #aug_labels_source = 1 - Ys
                aug_labels_source = Ys  # CS

        Xt_train = torch.concat((Xt_train, aug_inputs_tgttrain))
        Yt_train = torch.concat((Yt_train, aug_labels_tgttrain))

        print('CR augmented:', Xt_train.shape, Yt_train.shape)

        if 'CRb' in args.method:
            Xs = torch.concat((Xs, aug_inputs_source))
            Ys = torch.concat((Ys, aug_labels_source))
            print('CR augmented source:', Xs.shape, Ys.shape)

    if args.data_env != 'local':
        Xs, Ys, Xt_train, Yt_train, Xt_test, Yt_test = Xs.cuda(), Ys.cuda(), Xt_train.cuda(), Yt_train.cuda(), Xt_test.cuda(), Yt_test.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar_train = Data.TensorDataset(Xt_train, Yt_train)
    data_tar_test = Data.TensorDataset(Xt_test, Yt_test)

    # for TL train
    # assumes class-balanced batches in order
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=False, drop_last=False)
    dset_loaders["target-train"] = Data.DataLoader(data_tar_train, batch_size=train_bs // 8, shuffle=False, drop_last=False)
    dset_loaders["target-test"] = Data.DataLoader(data_tar_test, batch_size=train_bs // 8, shuffle=False, drop_last=False)

    # for TL combined train
    # no drop last just in case to keep all target train data
    data_train = Data.TensorDataset(torch.concat((Xs, Xt_train)), torch.concat((Ys, Yt_train)))
    dset_loaders["train"] = Data.DataLoader(data_train, batch_size=train_bs, shuffle=True, drop_last=False)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target-train"] = Data.DataLoader(data_tar_train, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target-test"] = Data.DataLoader(data_tar_test, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online TL test, aligned
    dset_loaders["target-online"] = Data.DataLoader(data_tar_online_test, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def data_loader_sda_augbaseline(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # supervised domain adaptation loader
    # augmentation approach baselines
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_train, Xt_test = Xt[:args.num_target_train], Xt[args.num_target_train:]
    Yt_train, Yt_test = Yt[:args.num_target_train], Yt[args.num_target_train:]

    Xt_train_copy_online = Xt_train.copy()
    Xt_test_copy_online = Xt_test.copy()
    Yt_test_copy_online = Yt_test.copy()

    if args.align:
        Xs = data_alignment(Xs, args.N - 1, args)
        Xt_train, R = data_alignment_returnref(Xt_train)  # save reference matrix
        Xt_test = data_alignment(Xt_test, 1, args)

    # mult_flag, noise_flag, neg_flag, freq_mod_flag
    if 'mult' in args.method:
        flag_aug = [True, False, False, False]
        train_bs = int(2 / 3 * train_bs)
    elif 'noise' in args.method:
        flag_aug = [False, True, False, False]
    elif 'neg' in args.method:
        flag_aug = [False, False, True, False]
    elif 'freq' in args.method:
        flag_aug = [False, False, False, True]
        train_bs = int(2 / 3 * train_bs)
    print('flag_aug:', flag_aug)

    print('original data source:', Xs.shape, Ys.shape)
    aug_out = data_aug(np.transpose(Xs, (0, 2, 1)), Ys, Xs.shape[-1], flag_aug)
    Xs, Ys = aug_out
    Xs = np.transpose(Xs, (0, 2, 1))
    print('augmented data source:', Xs.shape, Ys.shape)

    print('original data target train:', Xt_train.shape, Yt_train.shape)
    aug_out = data_aug(np.transpose(Xt_train, (0, 2, 1)), Yt_train, Xt_train.shape[-1], flag_aug)
    Xt_train, Yt_train = aug_out
    Xt_train = np.transpose(Xt_train, (0, 2, 1))
    print('augmented data target train:', Xt_train.shape, Yt_train.shape)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt_train, Yt_train = tr.from_numpy(Xt_train).to(
        tr.float32), tr.from_numpy(Yt_train.reshape(-1, )).to(tr.long)
    Xt_train = Xt_train.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_train = Xt_train.permute(0, 3, 1, 2)

    Xt_test, Yt_test = tr.from_numpy(Xt_test).to(
        tr.float32), tr.from_numpy(Yt_test.reshape(-1, )).to(tr.long)
    Xt_test = Xt_test.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_test = Xt_test.permute(0, 3, 1, 2)

    if args.align:
        test_x_aligned = []
        num_samples = len(Yt_train)
        for ind in range(len(Yt_test_copy_online)):
            curr = Xt_test_copy_online[ind]
            R = EA_online(curr, R, num_samples)
            num_samples += 1
            sqrtRefEA = fractional_matrix_power(R, -0.5)
            curr_aligned = np.dot(sqrtRefEA, curr)
            test_x_aligned.append(curr_aligned)
        test_x = np.stack(test_x_aligned)
        test_x = tr.from_numpy(test_x).to(tr.float32)
        test_x = test_x.unsqueeze_(3)
        if 'EEGNet' in args.backbone:
            test_x = test_x.permute(0, 3, 1, 2)
        Yt_test_copy_online = tr.from_numpy(Yt_test_copy_online.reshape(-1, )).to(tr.long)
        if args.data_env != 'local':
            test_x, Yt_test_copy_online = test_x.cuda(), Yt_test_copy_online.cuda()
        data_tar_online_test = Data.TensorDataset(test_x, Yt_test_copy_online)

    if args.data_env != 'local':
        Xs, Ys, Xt_train, Yt_train, Xt_test, Yt_test = Xs.cuda(), Ys.cuda(), Xt_train.cuda(), Yt_train.cuda(), Xt_test.cuda(), Yt_test.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar_train = Data.TensorDataset(Xt_train, Yt_train)
    data_tar_test = Data.TensorDataset(Xt_test, Yt_test)

    # for TL train
    # assumes class-balanced batches in order
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=False, drop_last=False)
    dset_loaders["target-train"] = Data.DataLoader(data_tar_train, batch_size=train_bs // 8, shuffle=False, drop_last=False)
    dset_loaders["target-test"] = Data.DataLoader(data_tar_test, batch_size=train_bs // 8, shuffle=False, drop_last=False)

    # for TL combined train
    # no drop last just in case to keep all target train data
    data_train = Data.TensorDataset(torch.concat((Xs, Xt_train)), torch.concat((Ys, Yt_train)))
    dset_loaders["train"] = Data.DataLoader(data_train, batch_size=train_bs, shuffle=True, drop_last=False)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target-train"] = Data.DataLoader(data_tar_train, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target-test"] = Data.DataLoader(data_tar_test, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online TL test, aligned
    dset_loaders["target-online"] = Data.DataLoader(data_tar_online_test, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def data_loader_sda_reverse(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # aug first, then EA
    # supervised domain adaptation loader
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_train, Xt_test = Xt[:args.num_target_train], Xt[args.num_target_train:]
    Yt_train, Yt_test = Yt[:args.num_target_train], Yt[args.num_target_train:]

    Xt_train_copy_online = Xt_train.copy()
    Xt_test_copy_online = Xt_test.copy()
    Yt_test_copy_online = Yt_test.copy()

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt_train, Yt_train = tr.from_numpy(Xt_train).to(
        tr.float32), tr.from_numpy(Yt_train.reshape(-1, )).to(tr.long)
    Xt_train = Xt_train.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_train = Xt_train.permute(0, 3, 1, 2)

    Xt_test, Yt_test = tr.from_numpy(Xt_test).to(
        tr.float32), tr.from_numpy(Yt_test.reshape(-1, )).to(tr.long)
    Xt_test = Xt_test.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_test = Xt_test.permute(0, 3, 1, 2)

    if 'CR' in args.method:
        # mirroring
        if args.data_name == 'BNCI2014001':
            left_mat = [1, 2,
                        6, 7, 8, 13, 14,
                        18]
            right_mat = [5, 4,
                         12, 11, 10, 17, 16,
                         20]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = 1 - Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = 1 - Ys
        elif args.data_name == 'BNCI2014002':
            left_mat = [0, 3, 4, 5, 6, 12]
            right_mat = [2, 11, 10, 9, 8, 14]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = Ys
        elif args.data_name == 'BNCI2014004':
            left_mat = [0]
            right_mat = [2]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = 1 - Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = 1 - Ys
        elif args.data_name == 'BNCI2015001':
            left_mat = [0,
                        3, 4, 5, 10]
            right_mat = [2,
                         9, 8, 7, 12]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = Ys
        elif args.data_name == 'PhysionetMI':
            left_mat = [0, 1, 2, 21, 24, 25, 29, 30, 31, 32,
                        14, 15, 16, 38, 40, 42, 44, 46,
                        7, 8, 47, 48, 49,
                        55, 56, 60]
            right_mat = [6, 5, 4, 23, 28, 27, 37, 36, 35, 34,
                         20, 19, 18, 39, 41, 43, 45, 54,
                         13, 12, 53, 52, 51,
                         59, 58, 62]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = Ys
        elif 'MI1' in args.data_name:
            left_mat = [0, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 33, 34, 35, 36, 41, 42, 43, 48, 49,
                        50, 55, 57]
            right_mat = [1, 8, 7, 6, 15, 14, 13, 23, 22, 21, 20, 32, 31, 30, 29, 40, 39, 38, 37, 47, 46, 45, 54, 53,
                         52, 56, 58]
            aug_inputs_tgttrain = leftrightflipping_transform(Xt_train, left_mat, right_mat)
            aug_labels_tgttrain = 1 - Yt_train
            if 'CRb' in args.method:
                aug_inputs_source = leftrightflipping_transform(Xs, left_mat, right_mat)
                aug_labels_source = 1 - Ys

        Xt_train = torch.concat((Xt_train, aug_inputs_tgttrain))
        Yt_train = torch.concat((Yt_train, aug_labels_tgttrain))

        print('CR augmented:', Xt_train.shape, Yt_train.shape)

        if 'CRb' in args.method:
            Xs = torch.concat((Xs, aug_inputs_source))
            Ys = torch.concat((Ys, aug_labels_source))
            print('CR augmented source:', Xs.shape, Ys.shape)

    if args.align:

        if 'EEGNet' in args.backbone:
            Xs = Xs.squeeze().numpy()
            Xt_train = Xt_train.squeeze().numpy()
            Xt_test = Xt_test.squeeze().numpy()

        Xs = data_alignment(Xs, args.N - 1, args)
        Xt_train, R = data_alignment_returnref(Xt_train)  # save reference matrix
        Xt_test = data_alignment(Xt_test, 1, args)

        test_x_aligned = []
        num_samples = len(Yt_train)
        for ind in range(len(Yt_test_copy_online)):
            curr = Xt_test_copy_online[ind]
            R = EA_online(curr, R, num_samples)
            num_samples += 1
            sqrtRefEA = fractional_matrix_power(R, -0.5)
            curr_aligned = np.dot(sqrtRefEA, curr)
            test_x_aligned.append(curr_aligned)
        test_x = np.stack(test_x_aligned)
        test_x = tr.from_numpy(test_x).to(tr.float32)
        test_x = test_x.unsqueeze_(3)
        if 'EEGNet' in args.backbone:
            test_x = test_x.permute(0, 3, 1, 2)
        Yt_test_copy_online = tr.from_numpy(Yt_test_copy_online.reshape(-1, )).to(tr.long)
        if args.data_env != 'local':
            test_x, Yt_test_copy_online = test_x.cuda(), Yt_test_copy_online.cuda()
        data_tar_online_test = Data.TensorDataset(test_x, Yt_test_copy_online)

        if 'EEGNet' in args.backbone:
            Xs = torch.from_numpy(Xs).to(tr.float32).unsqueeze_(1)
            Xt_train = torch.from_numpy(Xt_train).to(tr.float32).unsqueeze_(1)
            Xt_test = torch.from_numpy(Xt_test).to(tr.float32).unsqueeze_(1)

    if args.data_env != 'local':
        Xs, Ys, Xt_train, Yt_train, Xt_test, Yt_test = Xs.cuda(), Ys.cuda(), Xt_train.cuda(), Yt_train.cuda(), Xt_test.cuda(), Yt_test.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar_train = Data.TensorDataset(Xt_train, Yt_train)
    data_tar_test = Data.TensorDataset(Xt_test, Yt_test)

    # for TL train
    # assumes class-balanced batches in order
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=False, drop_last=False)
    dset_loaders["target-train"] = Data.DataLoader(data_tar_train, batch_size=train_bs // 8, shuffle=False, drop_last=False)
    dset_loaders["target-test"] = Data.DataLoader(data_tar_test, batch_size=train_bs // 8, shuffle=False, drop_last=False)

    # for TL combined train
    # no drop last just in case to keep all target train data
    data_train = Data.TensorDataset(torch.concat((Xs, Xt_train)), torch.concat((Ys, Yt_train)))
    dset_loaders["train"] = Data.DataLoader(data_train, batch_size=train_bs, shuffle=True, drop_last=False)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target-train"] = Data.DataLoader(data_tar_train, batch_size=train_bs * 3, shuffle=False, drop_last=False)
    dset_loaders["Target-test"] = Data.DataLoader(data_tar_test, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online TL test, aligned
    dset_loaders["target-online"] = Data.DataLoader(data_tar_online_test, batch_size=train_bs * 3, shuffle=False, drop_last=False)


    return dset_loaders


"""
def data_loader_meta(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # multi-source meta cross-subject loader
    dset_loaders = {}
    train_bs = args.batch_size

    Xt_copy = Xt
    if args.align:
        for i in range(len(Xs)):
            Xs[i] = data_alignment(Xs[i], 1, args)
        Xt = data_alignment(Xt, 1, args)

    for i in range(len(Xs)):
        Xs[i], Ys[i] = tr.from_numpy(Xs[i]).to(
            tr.float32), tr.from_numpy(Ys[i].reshape(-1, )).to(tr.long)
        Xs[i] = Xs[i].unsqueeze_(3)
        if 'EEGNet' in args.backbone:
            Xs[i] = Xs[i].permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    Xt_copy = tr.from_numpy(Xt_copy).to(
        tr.float32)
    Xt_copy = Xt_copy.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt_copy = Xt_copy.permute(0, 3, 1, 2)

    sources_ms = []
    for i in range(args.N - 1):
        if args.data_env != 'local':
            Xs[i], Ys[i] = Xs[i].cuda(), Ys[i].cuda()
        source = Data.TensorDataset(Xs[i], Ys[i])
        sources_ms.append(source)

    if args.data_env != 'local':
        Xt, Yt, Xt_copy = Xt.cuda(), Yt.cuda(), Xt_copy.cuda()

    data_tar = Data.TensorDataset(Xt, Yt)

    data_tar_online = Data.TensorDataset(Xt_copy, Yt)

    # for TL test
    dset_loaders["target"] = Data.DataLoader(data_tar, batch_size=train_bs, shuffle=True, drop_last=True)
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    # for online TL test
    dset_loaders["Target-Online"] = Data.DataLoader(data_tar_online, batch_size=1, shuffle=False, drop_last=False)

    # for multi-sources
    loader_arr = []
    for i in range(args.N - 1):
        loader = Data.DataLoader(sources_ms[i], batch_size=train_bs, shuffle=True, drop_last=True)
        loader_arr.append(loader)
    dset_loaders["sources"] = loader_arr

    loader_arr_S = []
    for i in range(args.N - 1):
        loader = Data.DataLoader(sources_ms[i], batch_size=train_bs, shuffle=False, drop_last=False)
        loader_arr_S.append(loader)
    dset_loaders["Sources"] = loader_arr_S

    return dset_loaders
"""

def data_loader_multisource(X=None, Y=None, num_sources=None, args=None):
    # multi-subjects loader
    dset_loaders = {}
    train_bs = args.batch_size

    if args.align:
        for i in range(len(X)):
            X[i] = data_alignment(X[i], 1, args)

    for i in range(len(X)):
        X[i], Y[i] = tr.from_numpy(X[i]).to(
            tr.float32), tr.from_numpy(Y[i].reshape(-1, )).to(tr.long)
        X[i] = X[i].unsqueeze_(3)
        if 'EEGNet' in args.backbone:
            X[i] = X[i].permute(0, 3, 1, 2)

    domains = []
    for i in range(num_sources):
        if args.data_env != 'local':
            X[i], Y[i] = X[i].cuda(), Y[i].cuda()
        domain = Data.TensorDataset(X[i], Y[i])
        domains.append(domain)

    # for training
    loader_arr = []
    for i in range(num_sources):
        loader = Data.DataLoader(domains[i], batch_size=train_bs, shuffle=True, drop_last=True)
        loader_arr.append(loader)
    dset_loaders["domains"] = loader_arr

    # for test
    loader_arr_S = []
    for i in range(num_sources):
        loader = Data.DataLoader(domains[i], batch_size=train_bs * 3, shuffle=False, drop_last=False)
        loader_arr_S.append(loader)
    dset_loaders["Domains"] = loader_arr_S

    return dset_loaders


def data_loader_without_tar(Xs=None, Ys=None, args=None):
    # no target process
    dset_loaders = {}
    train_bs = args.batch_size

    if args.align:
        Xs = data_alignment(Xs, args.N - 1, args)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys = Xs.cuda(), Ys.cuda()

    data_src = Data.TensorDataset(Xs, Ys)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)

    # for TL test
    dset_loaders["Source"] = Data.DataLoader(data_src, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders


def data_loader_split(Xs=None, Ys=None, Xt=None, Yt=None, args=None):
    # within-subject loader
    dset_loaders = {}
    train_bs = args.batch_size

    if args.align:
        Xs = data_alignment(Xs, args.N, args)
        Xt = data_alignment(Xt, args.N, args)

    Xs, Ys = tr.from_numpy(Xs).to(
        tr.float32), tr.from_numpy(Ys.reshape(-1, )).to(tr.long)
    Xs = Xs.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xs = Xs.permute(0, 3, 1, 2)

    Xt, Yt = tr.from_numpy(Xt).to(
        tr.float32), tr.from_numpy(Yt.reshape(-1, )).to(tr.long)
    Xt = Xt.unsqueeze_(3)
    if 'EEGNet' in args.backbone:
        Xt = Xt.permute(0, 3, 1, 2)

    if args.data_env != 'local':
        Xs, Ys, Xt, Yt = Xs.cuda(), Ys.cuda(), Xt.cuda(), Yt.cuda()

    data_src = Data.TensorDataset(Xs, Ys)
    data_tar = Data.TensorDataset(Xt, Yt)

    # for TL train
    dset_loaders["source"] = Data.DataLoader(data_src, batch_size=train_bs, shuffle=True, drop_last=True)

    # for TL test
    dset_loaders["Target"] = Data.DataLoader(data_tar, batch_size=train_bs * 3, shuffle=False, drop_last=False)

    return dset_loaders