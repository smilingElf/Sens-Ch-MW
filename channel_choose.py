#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pythonProject -> channel_choose.py
@IDE    ：PyCharm
@Author ：Leanne
@Date   ：2022/3/24 18:13
================================================='''
# 敏感电极筛选程序
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set()
names = locals()


def ch_meth(fea_l, fea_h, train_stop, sens, header_aim):
    """
    :param fea_aim: delta,theta,alpha,beta
    :param sub: 被试编号，用于调取特征值文件
    :param psd_path: 特征值文件所在路径
    :param train_stop: 划分训练集和测试集数据节点
    :param sens: 将某通道出现正差值次数/数据总数，作为高到低psd增大的敏感系数；将某通道出现负差值次数/数据总数作为高到低负荷任务psd降低的敏感系数。设置敏感阈值为0.9
    如果用该敏感系数得到的减小敏感通道或增大敏感通道为空，则舍去该频段的特征，视为该频段特征不够稳定，高低负荷下无稳定的线性关系。
    :return: fea_l,fea_h,in_ch和de_ch，即增大敏感通道和减小敏感通道名
    """
    psd_diff = (fea_h.loc[:, header_aim] - fea_l.loc[:, header_aim]).dropna(axis=0, how='all')
    denum_dic = {}
    innum_dic = {}
    # 统计训练数据集中每个通道特征值，由低负荷到高负荷psd相对降低和相对增大的数据次数
    for j in header_aim:
        names[j + '-de_num'] = 0
        names[j + '-in_num'] = 0
        for i in range(train_stop):
            if psd_diff[j][i] < 0:
                names[j + '-de_num'] += 1
            elif psd_diff[j][i] > 0:
                names[j + '-in_num'] += 1
        denum_dic.update({j + '-de_num': names[j + '-de_num']})
        innum_dic.update({j + '-in_num': names[j + '-in_num']})
    # 设置敏感电极百分比
    in_ch = [k for k, v in innum_dic.items() if v > (train_stop * sens)]
    de_ch = [k for k, v in denum_dic.items() if v > (train_stop * sens)]
    ch_type = {'in': in_ch, 'de': de_ch}
    return ch_type


def cor_ch(fea_l, fea_h, fea_aim, header_aim):
    """
    :param fea_l: 提取到的低负荷特征值
    :param fea_h: 提取到的高负荷特征值
    :param fea_aim: 特征频段名
    :return: corr，即高低负荷数据混在一起后计算的各通道皮尔森相关系数
    """
    fea = pd.concat([fea_l, fea_h], axis=0)
    corr = fea[header_aim].corr(method='pearson')  # 计算高低负荷混合psd各个通道的皮尔森相关性
    # 绘制相关系数热力图
    f, ax = plt.subplots(figsize=(20, 20))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    sns.heatmap(corr, annot=True, annot_kws={'size': 7})
    ax.tick_params(labelsize=8)
    plt.title('{}'.format(fea_aim))
    plt.close('all')
    return corr


def ch_all(fea_aim, fea_l, fea_h, corr_thresh, train_stop, sens, header_aim):
    """
    :param sub: 被试编号，用于调取特征值文件
    :param psd_path: 特征值文件所在路径
    :param corr_thresh: 相关系数阈值
    :param train_stop: 划分训练集和测试集数据节点
    :param sens: 将某通道出现正差值次数/数据总数，作为高到低psd增大的敏感系数；将某通道出现负差值次数/数据总数作为高到低负荷任务psd降低的敏感系数。设置敏感阈值为0.9
    如果用该敏感系数得到的减小敏感通道或增大敏感通道为空，则舍去该频段的特征，视为该频段特征不够稳定，高低负荷下无稳定的线性关系。
    :return: ch_in_final, ch_de_final, del_in, del_de。结合相关系数筛选后的增大通道和减小通道以及变化不稳定频段
    计算特征值时应使用psd(ch_in_final)-psd(ch_de_final)，包含在del_in和del_de中的部分psd算作0
    """
    ch_in_final = {}
    ch_de_final = {}
    ch_type = ch_meth(fea_l, fea_h, train_stop, sens, header_aim)
    corr = cor_ch(fea_l, fea_h, fea_aim, header_aim)
    for i in ch_type.items():
        ch_final2 = []
        for m in range(len(i[1])):
            corr_aim = corr.drop(corr[corr[i[1][m][:-7]] < corr_thresh].index)  # 删除第一列中小于0.95的行的索引
            r_index = corr_aim.index  # 获取行索引
            for n in r_index:
                r_ser = corr_aim.loc[n, :] < corr_thresh
                del_r_corr = r_ser[r_ser.values == 1].index
                corr_aim = corr_aim.drop(del_r_corr, axis='columns')  # 删除第一行中小于0.95的列
            ch1 = corr_aim.columns.values.tolist()
            ch2 = (corr_aim.index).values.tolist()
            ch_final1 = list(set(ch1).union(set(ch2)))
            ch_final2 = list(set(ch_final1).union(set(ch_final2)))
        if i[0] == 'in':
            ch_in_final.update({fea_aim + '_in': ch_final2})
        elif i[0] == 'de':
            ch_de_final.update({fea_aim + '_de': ch_final2})
    return ch_in_final, ch_de_final
