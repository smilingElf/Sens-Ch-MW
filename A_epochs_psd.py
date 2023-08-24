#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pythonProject -> A_epochs_psd.py
@IDE    ：PyCharm
@Author ：Leanne
@Date   ：2022/3/23 15:51
================================================='''
# Step1：计算四种节律下的单个epochs相对psd值，并输出保存为csv文件
import mne
import scipy.io as io
import numpy as np
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
'''
若有警告提示：
进入validation.py函数，第985行：
y = column_or_1d(y, warn=True)改为：
y = column_or_1d(y.ravel(), warn=True)即可解决警告问题
'''


class EEGTrain(object):
    """
    EEG train
    """

    def __init__(self, data_path, sub, day, ti, task):
        self.data_path = data_path
        self.sub = sub
        self.day = day
        self.ti = ti
        self.task = task

    def fil_data(self, load_rest):
        rest_len = load_rest.shape[1]
        info1 = mne.create_info(ch_names=['Fp1', 'Fp2', 'F11', 'F7', 'F3', 'Fz',
                                          'F4', 'F8', 'F12', 'FT11', 'FC3', 'FCz',
                                          'FC4', 'FT12', 'T7', 'C3', 'Cz', 'C4',
                                          'T8', 'CP3', 'CPz', 'CP4', 'M1', 'M2',
                                          'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'],
                                ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', ],
                                sfreq=1024)
        raws = mne.io.RawArray(load_rest, info1)
        raws.set_eeg_reference(ref_channels=['M1', 'M2'])
        raws.filter(l_freq=1, h_freq=30, n_jobs=1, fir_design='firwin2')
        filtered_data = raws.get_data(picks=['Fp1', 'Fp2', 'F7', 'F3', 'Fz',
                                             'F4', 'F8', 'FC3', 'FCz',
                                             'FC4', 'T7', 'C3', 'Cz', 'C4',
                                             'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3',
                                             'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'])
        fil_data_rest = filtered_data[:, 0: rest_len]
        return fil_data_rest

    def get_psd(self, filtered_data):
        data = filtered_data[:,
               30720:706560]  # 截取中间11分钟的数据
        info2 = mne.create_info(ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz',
                                          'F4', 'F8', 'FC3', 'FCz',
                                          'FC4', 'T7', 'C3', 'Cz', 'C4',
                                          'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3',
                                          'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2'],

                                ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg',
                                          'eeg', 'eeg'],
                                sfreq=1024)
        raw = mne.io.RawArray(data, info2)
        new_events = mne.make_fixed_length_events(raw, duration=2.)
        epochs = mne.Epochs(raw, new_events)
        biosemi_montage = mne.channels.make_standard_montage('standard_1020')
        epochs.set_montage(biosemi_montage, on_missing='raise')
        delta_df = epochs.plot_psd_topomap(ch_type='eeg', bands=[(1, 4, 'delta')],
                                           normalize=True, vlim=(0, 2))
        theta_df = epochs.plot_psd_topomap(ch_type='eeg', bands=[(4, 8, 'theta')],
                                           normalize=True, vlim=(0, 2))
        alpha_df = epochs.plot_psd_topomap(ch_type='eeg', bands=[(8, 12, 'Alpha')],
                                           normalize=True, vlim=(0, 2))
        beta_df = epochs.plot_psd_topomap(ch_type='eeg', bands=[(12, 30, 'Beta')],
                                          normalize=True, vlim=(0, 2))
        delta_dic = {'数据': np.array(list(range(1, delta_df.shape[0] + 1))), 'FP1': delta_df[:, 0],
                     'FP2': delta_df[:, 1], 'F7': delta_df[:, 2],
                     'F3': delta_df[:, 3],
                     'FZ': delta_df[:, 4], 'F4': delta_df[:, 5], 'F8': delta_df[:, 6], 'FC3': delta_df[:, 7],
                     'FCZ': delta_df[:, 8],
                     'FC4': delta_df[:, 9], 'T7': delta_df[:, 10], 'C3': delta_df[:, 11], 'CZ': delta_df[:, 12],
                     'C4': delta_df[:, 13],
                     'T8': delta_df[:, 14], 'CP3': delta_df[:, 15], 'CPZ': delta_df[:, 16], 'CP4': delta_df[:, 17],
                     'P7': delta_df[:, 18], 'P3': delta_df[:, 19], 'PZ': delta_df[:, 20], 'P4': delta_df[:, 21],
                     'P8': delta_df[:, 22],
                     'O1': delta_df[:, 23], 'OZ': delta_df[:, 24], 'O2': delta_df[:, 25]}
        theta_dic = {'数据': np.array(list(range(1, delta_df.shape[0] + 1))), 'FP1': theta_df[:, 0],
                     'FP2': theta_df[:, 1], 'F7': theta_df[:, 2],
                     'F3': theta_df[:, 3],
                     'FZ': theta_df[:, 4], 'F4': theta_df[:, 5], 'F8': theta_df[:, 6], 'FC3': theta_df[:, 7],
                     'FCZ': theta_df[:, 8],
                     'FC4': theta_df[:, 9], 'T7': theta_df[:, 10], 'C3': theta_df[:, 11], 'CZ': theta_df[:, 12],
                     'C4': theta_df[:, 13],
                     'T8': theta_df[:, 14], 'CP3': theta_df[:, 15], 'CPZ': theta_df[:, 16], 'CP4': theta_df[:, 17],
                     'P7': theta_df[:, 18], 'P3': theta_df[:, 19], 'PZ': theta_df[:, 20], 'P4': theta_df[:, 21],
                     'P8': theta_df[:, 22],
                     'O1': theta_df[:, 23], 'OZ': theta_df[:, 24], 'O2': theta_df[:, 25]}
        alpha_dic = {'数据': np.array(list(range(1, delta_df.shape[0] + 1))), 'FP1': alpha_df[:, 0],
                     'FP2': alpha_df[:, 1], 'F7': alpha_df[:, 2],
                     'F3': alpha_df[:, 3],
                     'FZ': alpha_df[:, 4], 'F4': alpha_df[:, 5], 'F8': alpha_df[:, 6], 'FC3': alpha_df[:, 7],
                     'FCZ': alpha_df[:, 8],
                     'FC4': alpha_df[:, 9], 'T7': alpha_df[:, 10], 'C3': alpha_df[:, 11], 'CZ': alpha_df[:, 12],
                     'C4': alpha_df[:, 13],
                     'T8': alpha_df[:, 14], 'CP3': alpha_df[:, 15], 'CPZ': alpha_df[:, 16], 'CP4': alpha_df[:, 17],
                     'P7': alpha_df[:, 18], 'P3': alpha_df[:, 19], 'PZ': alpha_df[:, 20], 'P4': alpha_df[:, 21],
                     'P8': alpha_df[:, 22],
                     'O1': alpha_df[:, 23], 'OZ': alpha_df[:, 24], 'O2': alpha_df[:, 25]}
        beta_dic = {'数据': np.array(list(range(1, delta_df.shape[0] + 1))), 'FP1': beta_df[:, 0], 'FP2': beta_df[:, 1],
                    'F7': beta_df[:, 2],
                    'F3': beta_df[:, 3],
                    'FZ': beta_df[:, 4], 'F4': beta_df[:, 5], 'F8': beta_df[:, 6], 'FC3': beta_df[:, 7],
                    'FCZ': beta_df[:, 8],
                    'FC4': beta_df[:, 9], 'T7': beta_df[:, 10], 'C3': beta_df[:, 11], 'CZ': beta_df[:, 12],
                    'C4': beta_df[:, 13],
                    'T8': beta_df[:, 14], 'CP3': beta_df[:, 15], 'CPZ': beta_df[:, 16], 'CP4': beta_df[:, 17],
                    'P7': beta_df[:, 18], 'P3': beta_df[:, 19], 'PZ': beta_df[:, 20], 'P4': beta_df[:, 21],
                    'P8': beta_df[:, 22],
                    'O1': beta_df[:, 23], 'OZ': beta_df[:, 24], 'O2': beta_df[:, 25]}
        delta_df1 = pd.DataFrame(delta_dic)
        theta_df1 = pd.DataFrame(theta_dic)
        alpha_df1 = pd.DataFrame(alpha_dic)
        beta_df1 = pd.DataFrame(beta_dic)

        return delta_df1, theta_df1, alpha_df1, beta_df1

    '''
    topomap函数修改：
    1908行：注释掉，不进行epochs时间维度降维处理
    2018行：各个节律下的能量均值计算代码更改：data=agg_fun(psds[:,:, freq_mask], axis=-1)
    2030行函数返回值修改：
        注释掉：
                        _plot_topomap_multi_cbar(data, pos, ax, title=title, vmin=vmin,
                                 vmax=vmax, cmap=cmap, outlines=outlines,
                                 colorbar=True, unit=unit, cbar_fmt=cbar_fmt,
                                 sphere=sphere, ch_type=ch_type)
        tight_layout(fig=fig)
        fig.canvas.draw()
        plt_show(show)
        return fig  
        改为：
        return data  
    '''

    def process(self):
        rest_raw = io.loadmat(self.data_path + 'sub{}_DAY{}{}.mat'.format(self.sub, self.day, self.ti))['Y']  # rest20起
        fil_data_rest = self.fil_data(rest_raw)
        delta_r1, theta_r1, alpha_r1, beta_r1 = self.get_psd(fil_data_rest)
        # # roll_psd保存
        delta_r1.to_csv(
            save_path2 + '{}sub{}{}_delta_{}.csv'.format(self.day, self.sub, time_all[self.ti - 1], self.task),
            mode='a',
            index=False,
            header=True)
        theta_r1.to_csv(
            save_path2 + '{}sub{}{}_theta_{}.csv'.format(self.day, self.sub, time_all[self.ti - 1], self.task),
            mode='a',
            index=False,
            header=True)
        alpha_r1.to_csv(
            save_path2 + '{}sub{}{}_alpha_{}.csv'.format(self.day, self.sub, time_all[self.ti - 1], self.task),
            mode='a',
            index=False,
            header=True)
        beta_r1.to_csv(
            save_path2 + '{}sub{}{}_beta_{}.csv'.format(self.day, self.sub, time_all[self.ti - 1], self.task),
            mode='a',
            index=False,
            header=True)


if __name__ == '__main__':
    # 注意sub6day2AM和sub12、sub13的day6PM数据读入有问题
    sub_all = [1, 2, 3, 4, 5, 6, 7, 8]  # 9, 10, 11, 12, 13, 14, 15
    task_all = ['low']  # 'low','high'
    day_all = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 1, 2, 3, 4, 5, 6,7,8,9
    time1_all = [1, 2]  # 依次对应AM和PM
    time_all = ['AM', 'PM']
    for s in sub_all:
        for t in task_all:
            for d in day_all:
                for time in time1_all:
                    raw_data_path = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\data\MATB\青岛\{}\sub{}\\'.format(t, s)
                    save_path1 = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\【1】epochs_psd\用于【2】绘制epochs云图\sub{}\\'.format(
                        s)
                    save_path2 = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\【1】epochs_psd\用于【3】epochs滚动均值计算\sub{}\\'.format(
                        s)
                    feature_box = ['delta', 'theta', 'alpha', 'beta']  # 选择特征不低于两类，绘制的散点图只会画前两个特征的二维图，但训练分类器是用所有特征进行训练的
                    EEG_Train = EEGTrain(raw_data_path, s, d, time, t)
                    EEG_Train.process()
