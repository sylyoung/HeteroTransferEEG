# -*- coding: utf-8 -*-
# @Time    : 2023/07/07
# @Author  : Siyang Li
# @File    : network.py
import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm

from models.EEGNet import EEGNet_feature, EEGNet, EEGNet_feature_ld, EEGNet_feature_map, EEGNet_BN_feature, EEGNet_block1, EEGNet_block2
from models.FC import FC, FC_xy, FC_xy_flat
from models.MINet import MINet_feature
from models.ShallowCNN import ShallowConvNet, ShallowConvNet_noBN
from models.DeepCNN import DeepConvNet, DeepConvNet_block1, DeepConvNet_block1_noBN, DeepConvNet_block23
from models.TSception import TSception
from models.EEGNet import EEGNet_block2_noBN, EEGNet_block2_BN, EEGNet_block1_noBN, EEGNet_block1_BN, EEGNet_block1_encoder, EEGNet_feature_preTransformer
from models.Conformer import Conformer_patchembedding, Conformer_patchembedding_noBN, Conformer_encoder, Conformer
from models.EEGTransformer import EEGTransformer_Encoder
from models.Deformer import Deformer, Deformer_Conv, Deformer_Encoder, Deformer_Conv_noBN
from models.SSVEPNet import SSVEPNet
from models.DeepCNN import DeepConvNet_block1_b4, DeepConvNet_block234_b4, DeepConvNet_4b, DeepConvNet_block23_b4, DeepConvNet_b3_4b


def backbone_net(chn, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    #print('F1, D, F2:', F1, D, F2)
    if backbone == 'EEGNet':
        netF = EEGNet_feature(n_classes=class_num,
                            Chans=chn,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,#0.5,
                            norm_rate=0.5)
    elif backbone == 'ShallowCNN':
        netF = ShallowConvNet(Chans=chn,
                              Samples=time_sample_num)
    elif backbone == 'DeepCNN':
        # netF = DeepConvNet(Chans=chn,
        #                    Samples=time_sample_num)
        if paradigm == 'MI':
            netF = DeepConvNet_4b(Chans=chn,
                               Samples=time_sample_num)
        elif paradigm == 'ERP':
            netF = DeepConvNet_b3_4b(Chans=chn,
                               Samples=time_sample_num)
    elif backbone == 'TSception':
        netF = TSception(input_size=(1, chn, time_sample_num),
                         sampling_rate=sample_rate,
                         num_T=15,
                         num_S=15)
    elif backbone == 'Conformer':
        netF = Conformer(chn=chn)
    elif backbone == 'Deformer':
        netF = Deformer(num_chan=chn,
                        num_time=time_sample_num,
                        temporal_kernel=11,
                        num_kernel=64,
                        depth=4,
                        heads=16,
                        mlp_dim=16,
                        dim_head=16,
                        dropout=0.5)
    elif backbone == 'SSVEPNet':
        netF = SSVEPNet(num_channels=chn,
                        T=time_sample_num)
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netF, netC


def backbone_net_noBN(chn, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    #print('F1, D, F2:', F1, D, F2)
    if backbone == 'EEGNet':
        netF = EEGNet_feature(n_classes=class_num,
                            Chans=chn,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,#0.5,
                            norm_rate=0.5)
    elif backbone == 'ShallowCNN':
        netF = ShallowConvNet(Chans=chn,
                              Samples=time_sample_num)
    elif backbone == 'DeepCNN':
        netF = DeepConvNet(Chans=chn,
                           Samples=time_sample_num)
    elif backbone == 'Conformer':
        netF = Conformer(chn=chn)
    elif backbone == 'Deformer':
        netF = Deformer(num_chan=chn,
                        num_time=time_sample_num,
                        temporal_kernel=11,
                        num_kernel=64,
                        depth=4,
                        heads=16,
                        mlp_dim=16,
                        dim_head=16,
                        dropout=0.5)
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netF, netC


def backbone_net_cross_dataset(chns, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    if backbone == 'EEGNet':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_block1(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.5,
                                norm_rate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = EEGNet_block2(n_classes=class_num,
                            Chans=-1,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,#0.5,
                            norm_rate=0.5)
    elif backbone == 'ShallowCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = ShallowConvNet(Chans=chns[i],
                              Samples=time_sample_num)
            netFspecifics.append(netFspecific)
        netFshared = None
    elif backbone == 'DeepCNN':
        netFspecifics = []
        for i in range(len(chns)):
            # netFspecific = DeepConvNet_block1(Chans=chns[i],
            netFspecific = DeepConvNet_block1_b4(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
            netFspecifics.append(netFspecific)
        # netFshared = DeepConvNet_block23(Chans=chns[i],
        if paradigm == 'MI':
            netFshared = DeepConvNet_block234_b4(Chans=chns[i],
                                    Samples=time_sample_num,
                                    dropoutRate=0.5)
        elif paradigm == 'ERP':
            netFshared = DeepConvNet_block23_b4(Chans=chns[i],
                                    Samples=time_sample_num,
                                    dropoutRate=0.5)
    elif backbone == 'Conformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Conformer_patchembedding(chn=chns[i])
            netFspecifics.append(netFspecific)
        netFshared = Conformer_encoder(chn=-1)
    elif backbone == 'EEGTransformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_feature_preTransformer(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.25,
                                norm_rate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = EEGTransformer_Encoder(Layers=3,
                                            embed_dim=64,
                                            num_heads=4)
    elif backbone == 'Deformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Deformer_Conv(num_chan=chns[i],
                                         num_time=time_sample_num,
                                         temporal_kernel=11,
                                         num_kernel=64)
            netFspecifics.append(netFspecific)
        netFshared = Deformer_Encoder(num_time=time_sample_num,
                                      temporal_kernel=11,
                                      num_kernel=64,
                                      depth=4,
                                      heads=16,
                                      mlp_dim=16,
                                      dim_head=16,
                                      dropout=0.5)
    elif backbone == 'SSVEPNet':
        return
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netFspecifics, netFshared, netC


def backbone_net_cross_dataset_dsclf(chns, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    if backbone == 'EEGNet':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_block1(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.5,
                                norm_rate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = EEGNet_block2(n_classes=class_num,
                            Chans=-1,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,#0.5,
                            norm_rate=0.5)
    elif backbone == 'ShallowCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = ShallowConvNet(Chans=chns[i],
                              Samples=time_sample_num)
            netFspecifics.append(netFspecific)
        netFshared = None
    elif backbone == 'DeepCNN':
        netFspecifics = []
        for i in range(len(chns)):
            # netFspecific = DeepConvNet_block1(Chans=chns[i],
            netFspecific = DeepConvNet_block1_b4(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
            netFspecifics.append(netFspecific)
        # netFshared = DeepConvNet_block23(Chans=chns[i],
        if paradigm == 'MI':
            netFshared = DeepConvNet_block234_b4(Chans=chns[i],
                                    Samples=time_sample_num,
                                    dropoutRate=0.5)
        elif paradigm == 'ERP':
            netFshared = DeepConvNet_block23_b4(Chans=chns[i],
                                    Samples=time_sample_num,
                                    dropoutRate=0.5)
    elif backbone == 'Conformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Conformer_patchembedding(chn=chns[i])
            netFspecifics.append(netFspecific)
        netFshared = Conformer_encoder(chn=-1)
    elif backbone == 'EEGTransformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_feature_preTransformer(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.25,
                                norm_rate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = EEGTransformer_Encoder(Layers=3,
                                            embed_dim=64,
                                            num_heads=4)
    elif backbone == 'Deformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Deformer_Conv(num_chan=chns[i],
                                         num_time=time_sample_num,
                                         temporal_kernel=11,
                                         num_kernel=64)
            netFspecifics.append(netFspecific)
        netFshared = Deformer_Encoder(num_time=time_sample_num,
                                      temporal_kernel=11,
                                      num_kernel=64,
                                      depth=4,
                                      heads=16,
                                      mlp_dim=16,
                                      dim_head=16,
                                      dropout=0.5)
    elif backbone == 'SSVEPNet':
        return
    netCs = []
    for i in range(len(chns)):
        if return_type == 'y':
            if 'reduce' in method:
                netC = FC(2, class_num)
            else:
                netC = FC(feature_deep_dim, class_num)
        elif return_type == 'xy':
            if 'reduce' in method:
                netC = FC_xy(2, class_num)
            else:
                netC = FC_xy(feature_deep_dim, class_num)
        netCs.append(netC)
    return netFspecifics, netFshared, netCs


def backbone_net_cross_dataset_exfc(chns, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    if backbone == 'EEGNet':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_feature(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.5,
                                norm_rate=0.5)
            netFspecifics.append(netFspecific)
    elif backbone == 'ShallowCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = ShallowConvNet(Chans=chns[i],
                              Samples=time_sample_num)
            netFspecifics.append(netFspecific)
    elif backbone == 'DeepCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = DeepConvNet(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
            netFspecifics.append(netFspecific)
    elif backbone == 'Conformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Conformer(chn=chns[i])
            netFspecifics.append(netFspecific)
    elif backbone == 'Deformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Deformer(num_chan=chns[i],
                                    num_time=time_sample_num,
                                    temporal_kernel=11,
                                    num_kernel=64,
                                    depth=4,
                                    heads=16,
                                    mlp_dim=16,
                                    dim_head=16,
                                    dropout=0.5)
            netFspecifics.append(netFspecific)
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netFspecifics, netC


def backbone_net_cross_dataset_noBN(chns, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    if backbone == 'EEGNet':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_block1_noBN(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.5,
                                norm_rate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = EEGNet_block2(n_classes=class_num,
                            Chans=-1,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,#0.5,
                            norm_rate=0.5)
    elif backbone == 'ShallowCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = ShallowConvNet_noBN(Chans=chns[i],
                              Samples=time_sample_num)
            netFspecifics.append(netFspecific)
        netFshared = None
    elif backbone == 'DeepCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = DeepConvNet_block1_noBN(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = DeepConvNet_block23(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
    elif backbone == 'Conformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Conformer_patchembedding_noBN(chn=chns[i])
            netFspecifics.append(netFspecific)
        netFshared = Conformer_encoder(chn=-1)
    elif backbone == 'Deformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Deformer_Conv_noBN(num_chan=chns[i],
                                         num_time=time_sample_num,
                                         temporal_kernel=25,
                                         num_kernel=64)
            netFspecifics.append(netFspecific)
        netFshared = Deformer_Encoder(num_time=time_sample_num,
                                      temporal_kernel=11,
                                      num_kernel=64,
                                      depth=4,
                                      heads=16,
                                      mlp_dim=16,
                                      dim_head=16,
                                      dropout=0.5)
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netFspecifics, netFshared, netC


def backbone_net_encoder_cross_dataset(chns, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, depth, emb_size, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    if backbone == 'EEGNet':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_block1_encoder(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.5,
                                norm_rate=0.5,
                                depth=depth,
                                emb_size=emb_size)
            netFspecifics.append(netFspecific)
        netFshared = EEGNet_block2(n_classes=class_num,
                            Chans=-1,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,#0.5,
                            norm_rate=0.5)
    elif backbone == 'ShallowCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = ShallowConvNet(Chans=chns[i],
                              Samples=time_sample_num)
            netFspecifics.append(netFspecific)
        netFshared = None
    elif backbone == 'DeepCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = DeepConvNet_block1(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = DeepConvNet_block23(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
    elif backbone == 'Conformer':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = Conformer_patchembedding(chn=chns[i])
            netFspecifics.append(netFspecific)
        netFshared = Conformer_encoder(chn=-1)
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netFspecifics, netFshared, netC


def backbone_net_cross_dataset_bn(chns, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    if backbone == 'EEGNet':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = EEGNet_block1_noBN(n_classes=class_num,
                                Chans=chns[i],
                                Samples=time_sample_num,
                                kernLenght=int(sample_rate // 2),
                                F1=F1,
                                D=D,
                                F2=F2,
                                dropoutRate=0.25,#0.5,
                                norm_rate=0.5)
            netFspecifics.append(netFspecific)
        netFBN = EEGNet_block1_BN(n_classes=class_num,
                            Chans=-1,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.25,#0.5,
                            norm_rate=0.5)
        netFshared = EEGNet_block2(n_classes=class_num,
                            Chans=-1,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.25,#0.5,
                            norm_rate=0.5)
    elif backbone == 'ShallowCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = ShallowConvNet(Chans=chns[i],
                              Samples=time_sample_num)
            netFspecifics.append(netFspecific)
        netFshared = None
    elif backbone == 'DeepCNN':
        netFspecifics = []
        for i in range(len(chns)):
            netFspecific = DeepConvNet_block1(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
            netFspecifics.append(netFspecific)
        netFshared = DeepConvNet_block23(Chans=chns[i],
                                Samples=time_sample_num,
                                dropoutRate=0.5)
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netFspecifics, netFBN, netFshared, netC


def backbone_net_1_dataset(chn, time_sample_num, sample_rate, class_num, paradigm, method, backbone, feature_deep_dim, F1=None, D=None, F2=None, return_type='y'):
    if F1 is None:
        if paradigm == 'MI':
            F1, D, F2 = 4, 2, 8
        elif paradigm == 'ERP' or paradigm == 'aBCI':
            F1, D, F2 = 8, 2, 16
    if backbone == 'EEGNet':
        netFspecific = EEGNet_block1(n_classes=class_num,
                            Chans=chn,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,
                            norm_rate=0.5)
        netFshared = EEGNet_block2(n_classes=class_num,
                            Chans=-1,
                            Samples=time_sample_num,
                            kernLenght=int(sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.5,#0.5,
                            norm_rate=0.5)
    if return_type == 'y':
        if 'reduce' in method:
            netC = FC(2, class_num)
        else:
            netC = FC(feature_deep_dim, class_num)
    elif return_type == 'xy':
        if 'reduce' in method:
            netC = FC_xy(2, class_num)
        else:
            netC = FC_xy(feature_deep_dim, class_num)
    return netFspecific, netFshared, netC


def backbone_net_3d(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    if args.backbone == 'EEGNet':
        netF = EEGNet_feature_map(n_classes=args.class_num,
                            Chans=args.chn,
                            Samples=args.time_sample_num,
                            kernLenght=int(args.sample_rate // 2),
                            F1=F1,
                            D=D,
                            F2=F2,
                            dropoutRate=0.25,
                            norm_rate=0.5)
    if return_type == 'xy':
        netC = FC_xy_flat(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_minet(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = MINet_feature(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLenght=int(args.sample_rate // 2),
                        F1=F1,
                        D=D,
                        F2=F2,
                        dropoutRate=0.25,
                        norm_rate=0.5)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_relationnet(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = EEGNet_feature_map(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLenght=int(args.sample_rate // 2),
                        F1=F1,
                        D=D,
                        F2=F2,
                        dropoutRate=0.25,
                        norm_rate=0.5)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC


def backbone_net_ld(args, return_type='y'):
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    print('F1, D, F2:', F1, D, F2)
    netF = EEGNet_feature_ld(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLenght=int(args.sample_rate // 2),
                        F1=F1,
                        D=D,
                        F2=F2,
                        dropoutRate=0.25,
                        norm_rate=0.5,
                        feature_deep_dim=args.feature_deep_dim,
                        reduced_dimension=args.reduced_dimension)
    if return_type == 'y':
        netC = FC(args.reduced_dimension, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.reduced_dimension, args.class_num)
    return netF, netC


def GAN_disc(args):
    # discriminator
    try:
        F1, D, F2 = args.F1, args.D, args.F2
    except:
        F1, D, F2 = 4, 2, 8
    netF = EEGNet_feature(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLenght=int(args.sample_rate // 2),
                        F1=F1,
                        D=D,
                        F2=F2,
                        dropoutRate=0.25,
                        norm_rate=0.5)
    netC = FC(args.feature_deep_dim, 2)  # 2 for {0: real data, 1: generated data}
    return netF, netC


# dynamic change the weight of the domain-discriminator
def calc_coeff(iter_num, alpha=10.0, max_iter=10000.0):
    return float(2.0 / (1.0 + np.exp(-alpha * iter_num / max_iter)) - 1)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class Net_ln2(nn.Module):
    def __init__(self, n_feature, n_hidden, bottleneck_dim):
        super(Net_ln2, self).__init__()
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(n_feature, n_hidden)
        self.ln1 = nn.LayerNorm(n_hidden)
        self.fc2 = nn.Linear(n_hidden, bottleneck_dim)
        self.fc2.apply(init_weights)
        self.ln2 = nn.LayerNorm(bottleneck_dim)

    def forward(self, x):
        x = self.act(self.ln1(self.fc1(x)))
        x = self.act(self.ln2(self.fc2(x)))
        x = x.view(x.size(0), -1)
        return x


class Net_CFE(nn.Module):
    def __init__(self, input_dim=310, bottleneck_dim=64):
        if input_dim < 256:
            print('\nwarning', 'input_dim < 256')
        super(Net_CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(input_dim, 256),
            # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, bottleneck_dim),  # default 64
            # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, hidden_dim, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(hidden_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(hidden_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class feat_classifier_xy(nn.Module):
    def __init__(self, class_num, bottleneck_dim, type="linear"):
        super(feat_classifier_xy, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        y = self.fc(x)
        return x, y


class scalar(nn.Module):
    def __init__(self, init_weights):
        super(scalar, self).__init__()
        self.w = nn.Parameter(tr.tensor(1.) * init_weights)

    def forward(self, x):
        x = self.w * tr.ones((x.shape[0]), 1).cuda()
        x = tr.sigmoid(x)
        return x


def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


class Discriminator(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ln1 = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.ln2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.ln1(x))
        x = self.ln2(self.bn(x))
        y = tr.sigmoid(x)
        return y


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size1, hidden_size2):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size1)
        self.ad_layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.ad_layer3 = nn.Linear(hidden_size2, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]



