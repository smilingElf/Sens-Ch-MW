#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：channel_choose.py -> ROC_analysis.py
@IDE    ：PyCharm
@Author ：Leanne
@Date   ：2022/4/3 12:57
================================================='''
# Step6：计算绘制并保存每个模型测试的ROC曲线图，计算并保存AUC值，用于origin绘制雷达图，进行分类器稳定性分析
import pandas as pd
import numpy as np
import sys
import channel_choose
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

sys.path.append(r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\pythonProject')
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

names = locals()


def data_read(sub, fea_aim, psd_path):
    fea_l = pd.read_csv(psd_path + 'sub{}_'.format(sub) + fea_aim + '_low.csv', encoding="gbk")
    fea_h = pd.read_csv(psd_path + 'sub{}_'.format(sub) + fea_aim + '_high.csv', encoding="gbk")  # axis=0行拼接，axis=1列拼接
    return fea_l, fea_h


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


def del_fea_none(fea_new):
    for k in list(fea_new.keys()):
        if fea_new[k].size == 0:
            del fea_new[k]
    return fea_new


def svm_train(fea_train_df, train_stop):
    X = fea_train_df.values
    y = [0] * train_stop + [1] * train_stop
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    c_grid = [0.1, 1, 10, 100]
    gamma_grid = [0.01, 0.1, 1, 10]
    kernel_grid = ['linear', 'rbf']
    param_grid = {'C': c_grid, 'gamma': gamma_grid, 'kernel': kernel_grid}
    grid_search = GridSearchCV(SVC(decision_function_shape='ovr', probability=True), param_grid,
                               scoring='accuracy', cv=5)
    grid_search.fit(x_train, y_train)
    print("Train score:{:.2f}".format(grid_search.score(x_train, y_train)))
    print("Test score:{:.2f}".format(grid_search.score(x_test, y_test)))
    best_score = grid_search.best_score_
    best_parameters = grid_search.best_params_
    print("Best parameters:{}".format(best_parameters))
    print("Best cross-validation  score:{:.2f}".format(best_score))
    clf = grid_search.best_estimator_
    return clf, best_score


def cal_probablity(clf, test_fea, test_lab1):
    x_train, x_test, y_train, y_test = train_test_split(test_fea, test_lab1, test_size=0.7, random_state=42)
    prob = clf.predict_proba(x_test)
    prob = (prob[:, 1]).tolist()
    return y_test, prob


def multi_models_roc(y_test_all, prob_all, col_num, sub, s, c):
    """一个被试所有模型的ROC一起绘图"""
    # 定义色板
    color_all = ['#515151', '#F1402E', '#1A6FDF', '#37AD6B', '#B177DE', '#CC9900', '#00CBCC', '#7D4E4E', '#8E8E00',
                 '#CD6501', '#6699CC', '#6FB802']
    color_map = np.array(color_all[:col_num])
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 设置figure窗体的颜色(解决生成的图片默认灰色网格背景的问题)
    plt.rcParams['figure.facecolor'] = 'white'
    # 设置axes绘图区的颜色
    plt.rcParams['axes.facecolor'] = 'white'
    plt.grid(None)
    ax = plt.gca()
    # 绘制图片方框线
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    num = 1
    auc_dic = {}
    for (x, y, z) in zip(y_test_all, prob_all, color_map):
        fpr, tpr, thresholds = roc_curve(y_test_all[x], prob_all[y], pos_label=1)
        plt.plot(fpr, tpr, lw=2, label='Classification{} (AUC={:.3f})'.format(num, auc(fpr, tpr)), color=z)
        plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')
        plt.tick_params(labelsize=15)
        plt.xlabel('FPR', fontsize=20)
        plt.ylabel('TPR', fontsize=20)
        # plt.title('ROC', fontsize=20)
        plt.legend(loc='lower right', fontsize=12)
        # 存储AUC
        temp_dic = {'model{}'.format(num): '%.4f' % auc(fpr, tpr)}
        auc_dic.update(temp_dic)
        num += 1
    auc_df = pd.DataFrame([auc_dic])
    # 保存AUC计算结果
    auc_df.to_csv(
        save_path + 'sub{}.csv'.format(sub), mode='a', index=False, header=True)
    # plt.show()
    # 保存ROC曲线图
    plt.savefig(save_path + 'sub{}ROC_sen{}_corr{}.png'.format(sub, s, c))
    return


if __name__ == '__main__':
    sub_all = [1]  # 1, 2, 3, 4, 5, 6, 7, 9, 11, 13
    corr_thresh = [0.95]
    sens = [0.6]  # 注意每个被试best sens有所不同，依次为：0.6, 0.7, 0.75, 0.8, 0.85, 0.65, 0.55, 0.65, 0.8, 0.85
    fea = ['delta', 'theta', 'alpha', 'beta']
    train_stop1 = [96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576,
                   624]  # train: 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576; test: 624~816 or 864
    color_num = len(train_stop1)
    header_aim = ['FP1', 'FP2', 'F7', 'F3', 'FZ',
                  'F4', 'F8', 'FC3', 'FCZ',
                  'FC4', 'T7', 'C3', 'CZ', 'C4',
                  'T8', 'CP3', 'CPZ', 'CP4', 'P7', 'P3',
                  'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']
    save_path = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【终极版】\【10】模型与数据量研究\python绘ROC\model11_AUC\\'
    for c in range(len(corr_thresh)):  # 遍历设置的阈值
        for s in range(len(sens)):  # 遍历设置的敏感系数
            test_all = {}
            prob_all = {}
            num = 1
            for train_stop in train_stop1:
                for sub in sub_all:  # 遍历所有被试
                    psd_path = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\【3】roll_psd\sub{}\\'.format(sub)
                    fea_new_train = {}
                    fea_new_test = {}
                    for f in range(len(fea)):  # 遍历所有频段
                        fea_l, fea_h = data_read(sub, fea[f], psd_path)
                        fea1_l = fea_l[:train_stop]
                        fea1_h = fea_h[:train_stop]
                        fea2_l = fea_l[480:]
                        fea2_h = fea_h[480:]
                        test_lab = [0] * fea2_l.shape[0] + [1] * fea2_h.shape[0]
                        header_in, header_de = channel_choose.ch_all(fea[f], fea1_l, fea1_h, corr_thresh[c],
                                                                     train_stop, sens[s],
                                                                     header_aim)  # 使用训练数据集找的增大减小敏感电极
                        header_in, header_de = header_choose(header_in, header_de)
                        print('{}-increase-channel(cor:{},sens:{}:'.format(fea[f], corr_thresh[c], sens[s]), header_in)
                        print('{}-decrease-channel(cor:{},sens:{}:'.format(fea[f], corr_thresh[c], sens[s]), header_de)
                        fea_new_l1 = fea_extr(fea_l, fea[f], header_in, header_de)
                        fea_new_h1 = fea_extr(fea_h, fea[f], header_in, header_de)  # 使用筛选出来的电极计算特征均值，降维
                        fea_new_train.update(
                            {'cor{}_'.format(corr_thresh[c]) + 'sens{}_'.format(sens[s]) + fea[f] + '_train': np.hstack(
                                (fea_new_l1[:train_stop], fea_new_h1[:train_stop]))})
                        fea_new_test.update(
                            {'cor{}_'.format(corr_thresh[c]) + 'sens{}_'.format(sens[s]) + fea[f] + '_test': np.hstack(
                                (fea_new_l1[480:], fea_new_h1[480:]))})
                    fea_new_train = del_fea_none(fea_new_train)
                    fea_new_test = del_fea_none(fea_new_test)
                    if fea_new_train:
                        fea_train_df = pd.DataFrame(fea_new_train)  # 得到训练特征数据
                        fea_test_df = pd.DataFrame(fea_new_test)  # 得到测试特征数据
                        clf, best_score = svm_train(fea_train_df, train_stop)
                        test_label, test_prob = cal_probablity(clf, fea_test_df, test_lab)
                        fpr, tpr, thresholds = roc_curve(test_label, test_prob, pos_label=1)
                        label_dic = {'Classification{}'.format(num): test_label}
                        prob_dic = {'Classification{}'.format(num): test_prob}
                        test_all.update(label_dic)
                        prob_all.update(prob_dic)
                        num += 1
            multi_models_roc(test_all, prob_all, color_num, sub_all[0], sens[s], corr_thresh[c])
