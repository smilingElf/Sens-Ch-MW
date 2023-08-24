#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：ROC_analysis.py -> rest_hotmap.py
@IDE    ：PyCharm
@Author ：Leanne
@Date   ：2022/4/5 19:13
================================================='''
# Step2：绘制流程图所需heatmap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def loaddata(data_path, day, sub, time, freq):
    """
    读取传入得数据表
    """
    high_df = pd.read_csv(data_path + '{}sub{}{}_{}_high.csv'.format(day, sub, time, freq), encoding="gbk",
                          header=None)
    high_psd = high_df.values[1:, 1:].astype('float64')
    # 26个电极通道ch
    x_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz',
            'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']
    high_num = high_psd.shape[0]
    return high_psd, x_ch, high_num


def hotmap(psd_all, x_ch, high_num, day, time, sub, fre_seg):
    plt.figure(figsize=(13, 7))
    ax = sns.heatmap(psd_all, xticklabels=x_ch, cmap=plt.get_cmap('seismic'), cbar=False)
    cb = ax.figure.colorbar(ax.collections[0])  # 显示colorbar
    cb.ax.tick_params(labelsize=25)  # 设置colorbar刻度字体大小。
    # 设置色标说明
    cb.set_label(r'$E_{{\{},ch}}$'.format(fre_seg), fontsize=32)
    # y轴坐标刻度以及轴坐标字体大小显示
    ax.set_yticks([0, high_num])
    ax.set_yticklabels((1, high_num + 1))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=25)
    ax.set_xlabel('channel', fontsize=32)  # x轴标题
    ax.set_ylabel('epoch', fontsize=32)
    # plt.show()
    figure = ax.get_figure()
    figure.savefig(save_path + 'sub{}day{}{}_{}.jpeg'.format(sub, day, time, fre_seg), dpi=200)  # 保存图片


# 主函数
if __name__ == "__main__":
    # 导入数据表文件名，注意sub6day2AM和sub12、sub13的day6PM无
    sub_all = [1]  # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    day_all = [1]  # 1, 2, 3, 4, 5, 6, 7, 8, 9
    time_all = ['AM']  # 'AM', 'PM'
    fre_seg_all = ['delta', 'theta', 'alpha', 'beta']  # 'delta', 'theta', 'alpha', 'beta'
    for s in sub_all:
        for d in day_all:
            for t in time_all:
                for f in fre_seg_all:
                    psd_path = r"E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\【1】epochs_psd\sub{}\\".format(
                        s)
                    save_path = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【终极版】\【2】epochs_show\4\\'
                    psd, x_ch, high_n = loaddata(psd_path, d, s, t, f)
                    hotmap(psd, x_ch, high_n, d, t, s, f)
