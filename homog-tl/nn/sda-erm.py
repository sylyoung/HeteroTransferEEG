# -*- coding: utf-8 -*-
# @Time    : 2023/10/18
# @Author  : Siyang Li
# @File    : sda-erm.py
# this is the baseline for Supervised Domain Adaptation, separate ERM (S+T) for both labeled data from both domains
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader_sdaTL

import gc
import sys


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    # for Dreyer2023 dataset
    # add this part to other files as well if want to try for this dataset
    if args.data == 'Dreyer2023':
        if args.gender:
            gender_list = args.gender_list
            gender = gender_list[args.idt]
            if args.same:
                inds = np.where(np.array(gender_list) == gender)[0]
                print('remove before', len(inds), inds)
                inds = inds[inds != args.idt]
                print('remove after', len(inds), inds)
            else:
                inds = np.where(np.array(gender_list) != gender)[0]
            print(gender, inds)

            data = []
            label = []
            for i in range(len(inds)):
                assert inds[i] != args.idt, print('inds error')
                if inds[i] > args.idt:
                    data.append(X_src[inds[i] - 1])
                    label.append(y_src[inds[i] - 1])
                else:
                    data.append(X_src[inds[i]])
                    label.append(y_src[inds[i]])

            trials_arr = []
            for i in range(args.N):
                if i == args.idt:
                    continue
                if (gender_list[i] == gender and args.same) or (gender_list[i] != gender and not args.same):
                    trials_arr.append(args.trials_arr[i])

            print(trials_arr)
            args.trials_arr = trials_arr

            args.srcN = len(inds)

            X_src = np.concatenate(data)
            y_src = np.concatenate(label)
        if args.age:
            age_list = args.age_list
            age = age_list[args.idt]
            if args.same:
                if age >= 1997:
                    inds = np.where(np.array(age_list) >= 1997)[0]
                elif age <= 1982:
                    inds = np.where(np.array(age_list) <= 1982)[0]
                else:
                    print('age error')
                    sys.exit(0)
                print('remove before', len(inds), inds)
                inds = inds[inds != args.idt]
                print('remove after', len(inds), inds)
            else:
                if age >= 1997:
                    inds = np.where(np.array(age_list) <= 1982)[0]
                elif age <= 1982:
                    inds = np.where(np.array(age_list) >= 1997)[0]
                else:
                    print('age error')
                    sys.exit(0)

            print(age, inds)

            data = []
            label = []
            for i in range(len(inds)):
                assert inds[i] != args.idt, print('inds error')
                if inds[i] > args.idt:
                    data.append(X_src[inds[i] - 1])
                    label.append(y_src[inds[i] - 1])
                else:
                    data.append(X_src[inds[i]])
                    label.append(y_src[inds[i]])

            trials_arr = []
            for i in range(args.N):
                if i == args.idt:
                    continue
                if args.same:
                    if age >= 1997 and args.age_list[i] >= 1997 and i != args.idt:
                        trials_arr.append(args.trials_arr[i])
                    elif age <= 1982 and args.age_list[i] <= 1982 and i != args.idt:
                        trials_arr.append(args.trials_arr[i])
                else:
                    if age >= 1997 and args.age_list[i] <= 1982:
                        trials_arr.append(args.trials_arr[i])
                    elif age <= 1982 and args.age_list[i] >= 1997:
                        trials_arr.append(args.trials_arr[i])

            print(trials_arr)
            args.trials_arr = trials_arr

            args.srcN = len(inds)

            X_src = np.concatenate(data)
            y_src = np.concatenate(label)
            print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    else:
        print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)

    dset_loaders = data_loader_sdaTL(X_src, y_src, X_tar, y_tar, args)

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    criterion = nn.CrossEntropyLoss()

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

    max_iter = args.max_epoch * len(dset_loaders["source"])
    interval_iter = max_iter // args.max_epoch
    args.max_iter = max_iter
    iter_num = 0
    base_network.train()

    while iter_num < max_iter:
        try:
            inputs_source, labels_source = next(iter_source)
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source = next(iter_source)

        if inputs_source.size(0) == 1:
            continue

        try:
            inputs_tgttrain, labels_tgttrain = next(iter_tgttrain)
        except:
            iter_tgttrain = iter(dset_loaders["target-train"])
            inputs_tgttrain, labels_tgttrain = next(iter_tgttrain)

        if inputs_tgttrain.size(0) == 1:
            continue

        iter_num += 1

        features_source, outputs_source = base_network(inputs_source)
        features_tgttrain, outputs_tgttrain = base_network(inputs_tgttrain)

        source_classifier_loss = criterion(outputs_source, labels_source)
        tgttrain_classifier_loss = criterion(outputs_tgttrain, labels_tgttrain)
        loss = (source_classifier_loss + tgttrain_classifier_loss * args.weight) / 2

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer_f.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()
            if args.align:
                acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)
            else:
                acc_t_te, _ = cal_acc_comb(dset_loaders["target-test"], base_network, args=args)

            log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
            args.log.record(log_str)
            print(log_str)

            base_network.train()

    print('Test Acc = {:.2f}%'.format(acc_t_te))

    print('saving model...')

    if args.align:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt')
    else:
        torch.save(base_network.state_dict(),
                   './runs/' + str(args.data_name) + '/' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_noEA' + '.ckpt')

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return acc_t_te


if __name__ == '__main__':

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']
    # data_name_list = ['Dreyer2023']

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640

        if data_name == 'Dreyer2023': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 60, 27, 2, 1281, 256, -1, 320

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name)

        # learning rate
        args.lr = 0.001

        args.root_dir = '/mnt/data2/sylyoung/EEG/DeepTransferEEG/'

        args.num_target_train = 50
        args.method = 'SDA-ERM-' + str(args.num_target_train)

        args.backbone = 'EEGNet'

        # whether to use EA
        args.align = True

        # For Dreyer2023 dataset
        # Default to cross group transfer, e.g., male to female
        # only set one property to True for gender/age
        args.gender_list = [1,1,2,1,1,2,2,1,2,2,2,2,2,1,2,2,1,1,1,2,1,2,1,1,1,1,2,1,2,2,1,1,1,2,2,1,1,1,2,2,1,1,1,2,2,2,2,1,1,1,1,1,2,2,2,2,1,2,2,2]
        args.age_list = [1993,1993,1969,1982,1985,1970,1997,1992,1996,1997,1997,1993,1997,1994,1988,1996,1997,1995,1985,1996,1988,1989,1994,1985,1999,1998,1981,1995,1997,1996,1978,1969,1992,1993,1993,1990,1959,1973,1996,1999,1989,1994,1980,1988,1977,1993,1990,1997,1981,1997,1975,1997,1991,1989,1996,1998,1996,1996,1991,1968]
        args.gender = False
        args.age = False
        assert (data_name == 'Dreyner2023' and (args.gender or args.age)) or not data_name == 'Dreyner2023', print('Please use cross gender/age transfer for Dreyer2023 dataset')
        assert (data_name == 'Dreyner2023' and not (args.gender and args.age)) or not data_name == 'Dreyner2023', print('Please set consider gender or age to True')
        # True for transfer from same groups, e.g., male to male
        args.same = False
        if args.gender:
            args.method += '-gender'
        if args.age:
            args.method += '-age'
        if args.same:
            args.method += '-same'

        # train batch size, also target train batch size
        args.batch_size = 10

        # training epochs
        args.max_epoch = 50
        if data_name == 'Dreyer2023':
            args.max_epoch = 20

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'

        total_acc = []

        # train multiple randomly initialized models
        for s in [1, 2, 3, 4, 5]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)

            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

            sub_acc_all = np.zeros(N)
            for idt in range(N):
                # note here for Dreyer2023 dataset
                if args.age is True:
                    if args.age_list[idt] > 1982 and args.age_list[idt] < 1997:
                        continue
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log

                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)

        print(str(total_acc))

        args.log.record(str(total_acc))

        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        print(subject_mean)
        print(total_mean)
        print(total_std)

        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))

        # result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        # for i in range(len(subject_mean)):
        #     result_dct['s' + str(i)] = subject_mean[i]

        # dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    # dct.to_csv('./logs/' + str(args.method) + ".csv")