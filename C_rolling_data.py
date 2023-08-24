#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pythonProject -> C_rolling_data.py
@IDE    ：PyCharm
@Author ：Leanne
@Date   ：2022/3/24 14:48
================================================='''
# Step3：使用A生成的数据进行滚动均值计算，为敏感电极筛选做数据准备
import pandas as pd
import os
import re

names = locals()


def get_file(data_path, freq_all):
    # 高低两个负荷四种节律的能量文件名分类
    files_all = os.listdir(data_path)
    for freq in freq_all:
        names['files_' + freq + '_l'] = []
        names['files_' + freq + '_h'] = []
    for file in files_all:  # [6:8]
        if 'low' in file:
            temp = re.compile('_(.+)_')
            freq_l = temp.findall(file)
            if freq_l:
                names['files_' + freq_l[0] + '_l'].append(file)
        if 'high' in file:
            temp = re.compile('_(.+)_')
            freq_h = temp.findall(file)
            if freq_h:
                names['files_' + freq_h[0] + '_h'].append(file)
    return files_delta_l, files_delta_h, files_theta_l, files_theta_h, files_alpha_l, files_alpha_h, files_beta_l, files_beta_h


def roll_mean(df, windowsize, step, sub, day, time):
    # 创建一个空的 DataFrame
    psd_mean = pd.DataFrame(
        columns=['fea_info', 'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC3', 'FCZ', 'FC4', 'T7', 'C3', 'CZ', 'C4',
                 'T8', 'CP3', 'CPZ', 'CP4', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2'])
    step_times = int((len(df) - windowsize) / step)
    for n in range(step_times + 1):
        narray_data = (df.values[n * step:n * step + windowsize, 1:]).mean(axis=0)
        roll_df = pd.DataFrame(
            {'fea_info': 'sub{}_DAY{}{}{}'.format(sub, day, time, n), 'FP1': narray_data[0], 'FP2': narray_data[1],
             'F7': narray_data[2], 'F3': narray_data[3], 'FZ': narray_data[4], 'F4': narray_data[5],
             'F8': narray_data[6], 'FC3': narray_data[7], 'FCZ': narray_data[8], 'FC4': narray_data[9],
             'T7': narray_data[10], 'C3': narray_data[11], 'CZ': narray_data[12], 'C4': narray_data[13],
             'T8': narray_data[14], 'CP3': narray_data[15], 'CPZ': narray_data[16], 'CP4': narray_data[17],
             'P7': narray_data[18], 'P3': narray_data[19], 'PZ': narray_data[20], 'P4': narray_data[21],
             'P8': narray_data[22], 'O1': narray_data[23], 'OZ': narray_data[24], 'O2': narray_data[25]}, index=[n])
        psd_mean = pd.concat([psd_mean, roll_df])
    return psd_mean


if __name__ == "__main__":
    # 设置移动窗口长90epochs，步进值为5个epochs,读取A程序生成的单个epoch能量值
    sub_all = [8, 9,10]
    fre_all = ['delta', 'theta', 'alpha', 'beta']
    windowsize = 90
    step = 5
    for s in sub_all:
        epoch_path = r'E:\研\postgraduate\生物电\论文写作材料\-定稿实验数据&分析结果整理\data_process\【1】epochs_psd\sub{}\\'.format(
            s)
        save_path = r'E:\研\postgraduate\生物电\论文写作材料\-定稿实验数据&分析结果整理\data_process\【3】roll_psd\sub{}\\'.format(
            s)
        files_delta_l, files_delta_h, files_theta_l, files_theta_h, files_alpha_l, files_alpha_h, files_beta_l, files_beta_h = get_file(
            epoch_path, fre_all)
        for f in fre_all:
            header = pd.DataFrame(
                columns=['fea_info', 'FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FC3', 'FCZ', 'FC4', 'T7',
                         'C3', 'CZ', 'C4',
                         'T8', 'CP3', 'CPZ', 'CP4', 'P7', 'P3', 'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2'])
            header.to_csv(save_path + 'sub{}_{}_low.csv'.format(s, f), mode='a', index=False, header=True)
            header.to_csv(save_path + 'sub{}_{}_high.csv'.format(s, f), mode='a', index=False, header=True)
            for file_l in names['files_' + f + '_l']:
                d = file_l[0]
                t = file_l[-(9 + len(f) + 2):-(9 + len(f))]
                fea_l = pd.read_csv(epoch_path + file_l, encoding="gbk")
                psd_l = roll_mean(fea_l[:], windowsize=windowsize, step=step, sub=s, day=d, time=t)
                psd_l.to_csv(save_path + 'sub{}_{}_low.csv'.format(s, f), mode='a', index=False, header=False)
            for file_h in names['files_' + f + '_h']:
                d = file_h[0]
                t = file_h[-(10 + len(f) + 2):-(10 + len(f))]
                fea_h = pd.read_csv(epoch_path + file_h, encoding="gbk")
                psd_h = roll_mean(fea_h[:], windowsize=90, step=5, sub=s, day=d, time=t)
                psd_h.to_csv(save_path + 'sub{}_{}_high.csv'.format(s, f), mode='a', index=False, header=False)
