# -*- coding: utf-8 -*-
# @Time    : 2023/10/25
# @Author  : Siyang Li
# @File    : sda-dann.py
# Ganin et al. - 2017 - Domain-Adversarial Training of Neural Networks
# adapted from https://github.com/thuml/Transfer-Learning-Library by Wen Zhang
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from utils.network import backbone_net, feat_classifier
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader_sdaTL
from utils.loss import CELabelSmooth_raw, Entropy, ReverseLayerF

import gc
import sys


def train_target(args):
    X_src, y_src, X_tar, y_tar = read_mi_combine_tar(args)
    print('X_src, y_src, X_tar, y_tar:', X_src.shape, y_src.shape, X_tar.shape, y_tar.shape)
    dset_loaders = data_loader_sdaTL(X_src, y_src, X_tar, y_tar, args)

    netF, netC = backbone_net(args, return_type='xy')
    if args.data_env != 'local':
        netF, netC = netF.cuda(), netC.cuda()
    base_network = nn.Sequential(netF, netC)

    ad_net = feat_classifier(type=args.layer, class_num=args.class_num, hidden_dim=args.feature_deep_dim).cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
    optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(ad_net.parameters(), lr=args.lr)

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

        p = float(iter_num) / max_iter
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        reverse_source, reverse_target = ReverseLayerF.apply(features_source, alpha), ReverseLayerF.apply(
            features_tgttrain,
            alpha)
        domain_output_s = ad_net(reverse_source)
        domain_output_t = ad_net(reverse_target)
        domain_label_s = torch.ones(inputs_source.size()[0]).long().cuda()
        domain_label_t = torch.zeros(inputs_tgttrain.size()[0]).long().cuda()

        adv_loss = nn.CrossEntropyLoss()(domain_output_s, domain_label_s) + nn.CrossEntropyLoss()(domain_output_t,
                                                                                                  domain_label_t)

        loss = (source_classifier_loss + tgttrain_classifier_loss) / 2 + adv_loss * args.alignment_weight

        optimizer_f.zero_grad()
        optimizer_c.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_f.step()
        optimizer_c.step()
        optimizer_d.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            base_network.eval()

            acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)
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

    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num, layer='wn',
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name)

        args.root_dir = '/mnt/data2/sylyoung/EEG/DeepTransferEEG/'

        args.num_target_train = 50
        args.method = 'SDA-DANN-' + str(args.num_target_train)

        args.backbone = 'EEGNet'

        # whether to use EA
        args.align = True

        # alignment loss weight
        args.alignment_weight = 1.

        # learning rate
        args.lr = 0.001

        # train batch size, also target train batch size
        args.batch_size = 10

        # training epochs
        args.max_epoch = 50

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

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]

        dct = dct.append(result_dct, ignore_index=True)

    # save results to csv
    dct.to_csv('./logs/' + str(args.method) + ".csv")