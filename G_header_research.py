#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pythonProject -> E_header_research.py
@IDE    ：PyCharm
@Author ：Leanne
@Date   ：2022/3/24 18:30
================================================='''
# Step7：对最佳sens和corr值下的电极通道筛选结果进行分析，输出N-2次训练中每个电极被筛选出的次数，用于Excel进行百分比计算，Visio绘图
import pandas as pd
import numpy as np
import sys
import channel_choose

sys.path.append(r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\pythonProject')

names = locals()


def data_read(sub, fea_aim, psd_path):
    fea_l = pd.read_csv(psd_path + 'sub{}_'.format(sub) + fea_aim + '_low.csv', encoding="gbk")
    fea_h = pd.read_csv(psd_path + 'sub{}_'.format(sub) + fea_aim + '_high.csv', encoding="gbk")  # axis=0行拼接，axis=1列拼接
    return fea_l, fea_h


def fea_extr(fea_load, fea_aim, header_in, header_de):
    if header_in[fea_aim + '_in']:
        if header_de[fea_aim + '_de']:
            fea1 = np.average(fea_load.loc[:, header_in[fea_aim + '_in']], axis=1) - np.average(
                fea_load.loc[:, header_de[fea_aim + '_de']], axis=1)
        else:
            fea1 = np.average(fea_load.loc[:, header_in[fea_aim + '_in']], axis=1) - 0
    elif header_de[fea_aim + '_de']:
        fea1 = 0 - np.average(fea_load.loc[:, header_de[fea_aim + '_de']], axis=1)
    else:
        fea1 = []
    return fea1


def header_choose(header_in, header_de):
    if header_in and header_de:
        header_in_new1 = set([x for item in list(header_in.values()) for x in item]) - set(
            [x for item in list(header_de.values()) for x in item])
        header_de_new1 = set([x for item in list(header_de.values()) for x in item]) - set(
            [x for item in list(header_in.values()) for x in item])
        header_in_new = {list(header_in.keys())[0]: list(header_in_new1)}
        header_de_new = {list(header_de.keys())[0]: list(header_de_new1)}
    else:
        header_in_new = header_in
        header_de_new = header_de
    return header_in_new, header_de_new


def del_fea_none(fea_new):
    for k in list(fea_new.keys()):
        if fea_new[k].size == 0:
            del fea_new[k]
    return fea_new


def header_count(header_sum):
    dic = {}
    for i in header_sum:
        dic[i] = dic.get(i, 0) + 1
    return dic


if __name__ == '__main__':
    sub_all = [3]  # 1, 2, 3, 4, 5, 6, 7, 9, 11, 13
    corr_thresh = [0.95]
    sens = [0.75]  # 注意每个被试best sens有所不同，依次为：0.6, 0.7, 0.75, 0.8, 0.85, 0.65, 0.55, 0.65, 0.8, 0.85
    fea = ['delta', 'theta', 'alpha', 'beta']
    train_stop1 = [96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768, 816]
    test_num = 48
    header_aim = ['FP1', 'FP2', 'F7', 'F3', 'FZ',
                  'F4', 'F8', 'FC3', 'FCZ',
                  'FC4', 'T7', 'C3', 'CZ', 'C4',
                  'T8', 'CP3', 'CPZ', 'CP4', 'P7', 'P3',
                  'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']
    save_path = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【终极版】\【6】channel_show\\'
    for c in range(len(corr_thresh)):  # 遍历设置的阈值
        for s in range(len(sens)):  # 遍历设置的敏感系数
            header_delta_in = []
            header_theta_in = []
            header_alpha_in = []
            header_beta_in = []
            header_delta_de = []
            header_theta_de = []
            header_alpha_de = []
            header_beta_de = []
            for train_stop in train_stop1:
                for sub in sub_all:  # 遍历所有被试
                    psd_path = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\【3】roll_psd\sub{}\\'.format(
                        sub)
                    fea_new_train = {}
                    fea_new_test = {}
                    for f in range(len(fea)):  # 遍历所有频段
                        fea_l, fea_h = data_read(sub, fea[f], psd_path)
                        fea1_l = fea_l[:train_stop]
                        fea1_h = fea_h[:train_stop]
                        fea2_l = fea_l[train_stop:train_stop + test_num]
                        fea2_h = fea_h[train_stop:train_stop + test_num]
                        header_in, header_de = channel_choose.ch_all(fea[f], fea1_l, fea1_h, corr_thresh[c],
                                                                     train_stop, sens[s],
                                                                     header_aim)  # 使用训练数据集找的增大减小敏感电极
                        header_in, header_de = header_choose(header_in, header_de)
                        in_key = list(header_in.keys())
                        de_key = list(header_de.keys())
                        names['header_' + in_key[0]] = names['header_' + in_key[0]] + (header_in[in_key[0]])
                        names['header_' + de_key[0]] = names['header_' + de_key[0]] + (header_de[de_key[0]])
                        # print('{}-increase-channel(cor:{},sens:{}:'.format(fea[f], corr_thresh[c], sens[s]), header_in)
                        # print('{}-decrease-channel(cor:{},sens:{}:'.format(fea[f], corr_thresh[c], sens[s]), header_de)
                        fea_new_l1 = fea_extr(fea_l, fea[f], header_in, header_de)
                        fea_new_h1 = fea_extr(fea_h, fea[f], header_in, header_de)  # 使用筛选出来的电极计算特征均值，降维
                        fea_new_train.update(
                            {'cor{}_'.format(corr_thresh[c]) + 'sens{}_'.format(sens[s]) + fea[f] + '_train': np.hstack(
                                (fea_new_l1[:train_stop], fea_new_h1[:train_stop]))})
                        fea_new_test.update(
                            {'cor{}_'.format(corr_thresh[c]) + 'sens{}_'.format(sens[s]) + fea[f] + '_train': np.hstack(
                                (fea_new_l1[train_stop:train_stop + test_num],
                                 fea_new_h1[train_stop:train_stop + test_num]))})
                    fea_new_train = del_fea_none(fea_new_train)
                    fea_new_test = del_fea_none(fea_new_test)
                    if fea_new_train:
                        fea_train_df = pd.DataFrame(fea_new_train)  # 得到训练特征数据
                        fea_test_df = pd.DataFrame(fea_new_test)  # 得到测试特征数据
                    else:
                        best_score = 0
                        acc = 0
            # print('header_in_delta:', header_delta_in)
            # print('header_de_delta:', header_delta_de)
            delta_in_count = header_count(header_delta_in)
            delta_de_count = header_count(header_delta_de)
            theta_in_count = header_count(header_theta_in)
            theta_de_count = header_count(header_theta_de)
            alpha_in_count = header_count(header_alpha_in)
            alpha_de_count = header_count(header_alpha_de)
            beta_in_count = header_count(header_beta_in)
            beta_de_count = header_count(header_beta_de)
            print('delta in sum:', delta_in_count)
            print('delta de sum:', delta_de_count)
            print('theta in sum:', theta_in_count)
            print('theta de sum:', theta_de_count)
            print('alpha in sum:', alpha_in_count)
            print('alpha de sum:', alpha_de_count)
            print('beta in sum:', beta_in_count)
            print('beta de sum:', beta_de_count)
