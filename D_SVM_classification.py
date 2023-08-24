#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：pythonProject -> D_SVM_acc.py
@IDE    ：PyCharm
@Author ：Leanne
@Date   ：2022/3/24 18:12
================================================='''
# Step4：SVM分类器训练测试。最佳sens的研究，保存每次模型精度和测试精度，用于origin绘制箱型图；corr值研究，绘制测试精度曲线图
import pandas as pd
import numpy as np
import sys
import channel_choose
import csv

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
    y = [1] * train_stop + [3] * train_stop
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    c_grid = [0.1, 1, 10, 100]
    gamma_grid = [0.01, 0.1, 1, 10]
    kernel_grid = ['linear', 'rbf']
    param_grid = {'C': c_grid, 'gamma': gamma_grid, 'kernel': kernel_grid}
    grid_search = GridSearchCV(SVC(decision_function_shape='ovr'), param_grid,
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


def svm_test(clf, fea_test_df):
    fea_test = fea_test_df.values
    label = [1] * test_num + [3] * test_num
    level = clf.predict(fea_test)
    true = 0
    fall = 0
    for i in range(len(label)):
        if level[i] == label[i]:
            true += 1
        else:
            fall += 1
    acc = true / len(label)
    print("Test accuracy:", acc)
    return acc


if __name__ == '__main__':
    sub_all = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 14, 15]  # 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    corr_thresh = [0.95]  # 0.95, 0.8, 0.85, 0.75, 0.7, 0.65, 0.6
    sens = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]  # 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9
    fea = ['delta', 'theta', 'alpha', 'beta']
    train_stop1 = [96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768 ,816]  # 96, 144, 192, 240, 288, 336, 384, 432, 480, 528, 576, 624, 672, 720, 768 ,816
    test_num = 48
    header_aim = ['FP1', 'FP2', 'F7', 'F3', 'FZ',
                  'F4', 'F8', 'FC3', 'FCZ',
                  'FC4', 'T7', 'C3', 'CZ', 'C4',
                  'T8', 'CP3', 'CPZ', 'CP4', 'P7', 'P3',
                  'PZ', 'P4', 'P8', 'O1', 'OZ', 'O2']
    save_path = r'E:\研\postgraduate\助教\\'
    for c in range(len(corr_thresh)):  # 遍历设置的阈值
        for s in range(len(sens)):  # 遍历设置的敏感系数
            for train_stop in train_stop1:
                f1 = open(save_path + 'cor{}_sens{}_score.csv'.format(corr_thresh[c], sens[s]), 'a+',
                          encoding='utf-8-sig',
                          newline='')
                csv_writer1 = csv.writer(f1)
                csv_writer1.writerow(["被试编号", "{}train-{}test".format(train_stop, test_num)])
                f2 = open(save_path + 'cor{}_sens{}_acc.csv'.format(corr_thresh[c], sens[s]), 'a+',
                          encoding='utf-8-sig',
                          newline='')
                csv_writer2 = csv.writer(f2)
                csv_writer2.writerow(["被试编号", "{}train-{}test".format(train_stop, test_num)])
                score_dic = {}
                acc_dic = {}
                for sub in sub_all:  # 遍历所有被试
                    psd_path = r'E:\研\postgraduate\生物电\论文写作材料\实验总结\实验结果MATB【用于论文】\【3】roll_psd\sub{}\\'.format(sub)
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
                        print('{}-increase-channel(cor:{},sens:{}:'.format(fea[f], corr_thresh[c], sens[s]), header_in)
                        print('{}-decrease-channel(cor:{},sens:{}:'.format(fea[f], corr_thresh[c], sens[s]), header_de)
                        fea_new_l1 = fea_extr(fea_l, fea[f], header_in, header_de)
                        fea_new_h1 = fea_extr(fea_h, fea[f], header_in, header_de)  # 使用筛选出来的电极计算特征均值，降维
                        fea_new_train.update(
                            {'cor{}_'.format(corr_thresh[c]) + 'sens{}_'.format(sens[s]) + fea[f] + '_train': np.hstack(
                                (fea_new_l1[:train_stop], fea_new_h1[:train_stop]))})
                        fea_new_test.update(
                            {'cor{}_'.format(corr_thresh[c]) + 'sens{}_'.format(sens[s]) + fea[f] + '_test': np.hstack(
                                (fea_new_l1[train_stop:train_stop + test_num],
                                 fea_new_h1[train_stop:train_stop + test_num]))})
                    fea_new_train = del_fea_none(fea_new_train)
                    fea_new_test = del_fea_none(fea_new_test)
                    if fea_new_train:
                        fea_train_df = pd.DataFrame(fea_new_train)  # 得到训练特征数据
                        fea_test_df = pd.DataFrame(fea_new_test)  # 得到测试特征数据
                        clf, best_score = svm_train(fea_train_df, train_stop)
                        acc = svm_test(clf, fea_test_df)
                    else:
                        best_score = 0
                        acc = 0
                    score_dic.update()
                    csv_writer1.writerow(["sub{}".format(sub), best_score])
                    csv_writer2.writerow(["sub{}".format(sub), acc])
