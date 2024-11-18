# -*- coding: utf-8 -*-
# @Time    : 2024/2/1
# @Author  : Siyang Li
# @File    : tSNE-deep.py
# t-SNE visualization of deep representations
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.network import backbone_net
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_combine_tar
from utils.utils import fix_random_seed, cal_acc_comb, data_loader_sdaTL

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

    if args.data_env != 'local':
        base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt'))
    else:
        base_network.load_state_dict(torch.load('./runs/' + str(args.data_name) + '/' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt', map_location=torch.device('cpu')))

    iter_source = iter(dset_loaders["source"])
    source_fea, source_labels = [], []

    iter_tgttrain = iter(dset_loaders["target-train"])
    tgttrain_fea, tgttrain_labels = [], []

    iter_test = iter(dset_loaders["target-online"])
    test_fea, test_labels = [], []

    for i in range(len(iter_source)):
        inputs_source, labels_source = next(iter_source)
        features, _ = base_network(inputs_source)
        source_fea.append(features)
        source_labels.append(labels_source)
    source_fea = torch.concat(source_fea).detach().cpu().numpy()
    source_labels = torch.concat(source_labels).detach().cpu().numpy()

    for i in range(len(iter_tgttrain)):
        inputs_tgttrain, labels_tgttrain = next(iter_tgttrain)
        features, _ = base_network(inputs_tgttrain)
        tgttrain_fea.append(features)
        tgttrain_labels.append(labels_tgttrain)
    tgttrain_fea = torch.concat(tgttrain_fea).detach().cpu().numpy()
    tgttrain_labels = torch.concat(tgttrain_labels).detach().cpu().numpy() + 2  # assumes binary

    for i in range(len(iter_test)):
        inputs_test, labels_test = next(iter_test)
        features, _ = base_network(inputs_test)
        test_fea.append(features)
        test_labels.append(labels_test)
    test_fea = torch.concat(test_fea).detach().cpu().numpy()
    test_labels = torch.concat(test_labels).detach().cpu().numpy() + 4  # assumes binary

    print('shapes')
    print(source_fea.shape, source_labels.shape, tgttrain_fea.shape, tgttrain_labels.shape, test_fea.shape, test_labels.shape)

    # tsne source+targettrain+targettest
    data = np.concatenate([source_fea, tgttrain_fea, test_fea])
    labels = np.concatenate([source_labels, tgttrain_labels, test_labels])

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features = tsne.fit_transform(data)
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    data = (features - x_min) / (x_max - x_min)

    X_2d = data
    print("t-SNE'd shape", X_2d.shape)

    labels_colors = []
    markers = []

    labels_colors.append('red')
    labels_colors.append('blue')
    labels_colors.append('red')
    labels_colors.append('blue')
    labels_colors.append('red')
    labels_colors.append('blue')

    markers.append('^')
    markers.append('^')
    markers.append('X')
    markers.append('X')
    markers.append('o')
    markers.append('o')

    X_2d = data

    ps = []

    for i in range(6):
        if len(markers) == 0:
            p = plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=10, edgecolors=labels_colors[i])
        else:
            p = plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=10, edgecolors=labels_colors[i],
                        marker=markers[i], facecolors='none')
        ps.append(p)

    star_0 = np.mean(X_2d[np.where(labels == 0)[0], 0])
    star_1 = np.mean(X_2d[np.where(labels == 0)[0], 1])
    plt.scatter([star_0], [star_1], s=450, edgecolors='white', marker='^', facecolors='salmon')

    star_0 = np.mean(X_2d[np.where(labels == 1)[0], 0])
    star_1 = np.mean(X_2d[np.where(labels == 1)[0], 1])
    plt.scatter([star_0], [star_1], s=450, edgecolors='white', marker='^', facecolors='cornflowerblue')

    star_0 = np.mean(X_2d[np.where(labels == 2)[0], 0])
    star_1 = np.mean(X_2d[np.where(labels == 2)[0], 1])
    plt.scatter([star_0], [star_1], s=600, edgecolors='white', marker='X', facecolors='salmon')

    star_0 = np.mean(X_2d[np.where(labels == 3)[0], 0])
    star_1 = np.mean(X_2d[np.where(labels == 3)[0], 1])
    plt.scatter([star_0], [star_1], s=600, edgecolors='white', marker='X', facecolors='cornflowerblue')

    if args.legend == True:
        plt.legend((ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]), ('Source 0', 'Source 1', 'Target Calibration 0', 'Target Calibration 1', 'Target Test 0', 'Target Test 1'), markerscale=2, fontsize=12, loc='upper left')#,
        #           scatterpoints=1, loc='lower left', ncol=3, fontsize=8)
    plt.grid(False)

    if 'TL' in args.method:
        plt.title('w/o DPL', fontsize=24)
    elif 'DPL' in args.method:
        plt.title('w/ DPL', fontsize=24)

    plt.tight_layout()

    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_legend.png', dpi=600, format='png')
    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_legend.pdf', dpi=600, format='pdf')
    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_legend.eps', dpi=600, format='eps')
    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_legend.jpg', dpi=600, format='jpg')

    plt.clf()

    # tsne targettrain+targettest
    data = np.concatenate([tgttrain_fea, test_fea])
    labels = np.concatenate([tgttrain_labels, test_labels]) - 2

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features = tsne.fit_transform(data)
    x_min, x_max = np.min(features, 0), np.max(features, 0)
    data = (features - x_min) / (x_max - x_min)

    X_2d = data
    print("t-SNE'd shape", X_2d.shape)

    labels_colors = []
    markers = []

    labels_colors.append('red')
    labels_colors.append('blue')
    labels_colors.append('red')
    labels_colors.append('blue')

    markers.append('X')
    markers.append('X')
    markers.append('o')
    markers.append('o')

    ps = []
    for i in range(4):
        if len(markers) == 0:
            p = plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=24, edgecolors=labels_colors[i])
        else:
            p = plt.scatter(X_2d[np.where(labels == i), 0], X_2d[np.where(labels == i), 1], s=24, edgecolors=labels_colors[i],
                        marker=markers[i], facecolors='none')
        ps.append(p)

    star_0 = np.mean(X_2d[np.where(labels == 0)[0], 0])
    star_1 = np.mean(X_2d[np.where(labels == 0)[0], 1])
    plt.scatter([star_0], [star_1], s=400, edgecolors='white', marker='X', facecolors='salmon')

    star_0 = np.mean(X_2d[np.where(labels == 1)[0], 0])
    star_1 = np.mean(X_2d[np.where(labels == 1)[0], 1])
    plt.scatter([star_0], [star_1], s=400, edgecolors='white', marker='X', facecolors='cornflowerblue')

    if args.legend == True:
        plt.legend((ps[0], ps[1], ps[2], ps[3]), ('Target Calibration 0', 'Target Calibration 1', 'Target Test 0', 'Target Test 1'), markerscale=2, fontsize=12, loc='upper left')#,
        #           scatterpoints=1, loc='lower left', ncol=3, fontsize=8)

    plt.grid(False)

    if 'TL' in args.method:
        plt.title('w/o DPL', fontsize=24)
    elif 'DPL' in args.method:
        plt.title('w/ DPL', fontsize=24)

    plt.tight_layout()

    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_nosource' + '.png', dpi=600, format='png')
    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_nosource' + '.pdf', dpi=600, format='pdf')
    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_nosource' + '.eps', dpi=600, format='eps')
    plt.savefig('./figures/' + str(args.data_name) + '_' + str(args.method) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_nosource' + '.jpg', dpi=600, format='jpg')

    plt.clf()

    gc.collect()
    if args.data_env != 'local':
        torch.cuda.empty_cache()

    return 0


if __name__ == '__main__':

    data_name_list = ['BNCI2014001', 'BNCI2014002', 'BNCI2015001']

    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])


    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 9, 22, 2, 1001, 250, 144, 248
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 14, 15, 2, 2561, 512, 100, 640
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim = 'MI', 12, 13, 2, 2561, 512, 200, 640

        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name)

        args.method = 'SDA-TL-50'
        #args.method = 'SDA-DPL-50-0.1-0.1-0.1-5-5'
        args.backbone = 'EEGNet'

        args.num_target_train = 50

        if args.method == 'SDA-TL-50':
            args.legend = True
        else:
            args.legend = False

        # whether to use EA
        args.align = True

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
                break  # only plot first subject # TODO
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