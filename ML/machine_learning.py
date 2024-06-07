"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/9/12
Last Updated: 2023/11/20
Version: 2.0.0
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import KFold
import sys
import math
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

sys.path.append(os.path.abspath('../'))

from preprocessing.dependency_matrix import get_java_files
from preprocessing.dependency_matrix import parse_relation_file
from preprocessing.dependency_matrix import build_matrix
from preprocessing.dependency_matrix import load_pretrained_word2vec_and_calculate_similarity
from preprocessing.dependency_similarity import calculate_contribution_matrix
from preprocessing.dependency_similarity import graph_embedding_similarity
import warnings

# 忽略特定的警告
warnings.filterwarnings("ignore", category=UserWarning, message="`use_label_encoder` is deprecated")

#RQ2_XGBoost
def xgboost_classification_kfold(dataset_name, feature_file_name):
    df = pd.read_excel('../../datasets/'+dataset_name+'/'+feature_file_name+'.xlsx')
    labels = pd.read_excel('../../datasets/' + dataset_name + '/labels.xlsx')

    labels = labels.iloc[:, 0].values  # 将 labels 转换为一个 1D numpy 数组

    # 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)
    #十折交叉验证
    kf = KFold(n_splits=10, shuffle=True)

    f1_scores = []
    precision_scores = []
    recall_scores = []

    #进行50次十折交叉验证
    for t in range(50):
        print(t)
        for train_index, test_index in kf.split(X_normalized, labels):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            # 创建XGBoost模型并训练
            model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X_train, y_train)

            # 预测测试集
            y_pred = model.predict(X_test)

            # 计算并记录F1分数
            f1_scores.append(f1_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))

    # 返回F1分数的均值和标准差
    return np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)

#RQ2_XWCoDe
def XWCoDe(dataset_name, feature_file_name, ci, mi, wi, dependency_matrix):
    df = pd.read_excel('../../datasets/' + dataset_name + '/' + feature_file_name + '.xlsx')
    labels = pd.read_excel('../../datasets/' + dataset_name + '/labels.xlsx')
    # 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)

    # 假设calculate_dependency_importance等其他必要的函数已经在上下文中定义
    attention_weight = [ci, mi, wi]
    sigma1 = 0.25
    sigma2 = 0.25
    sigma3 = 0.50

    # 十折交叉验证
    kf = KFold(n_splits=10, shuffle=True)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    # 进行50次十折交叉验证
    for t in range(50):
        print(t)
        for train_index, test_index in kf.split(X_normalized, labels):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]

            train_label = labels.iloc[train_index].values.ravel().tolist()
            df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                         dependency_matrix, train_label)
            threshold_count = int(sigma2 * len(df))
            # 获取代码依赖关系较前的下标
            top_idx = df['Test_Index'][:threshold_count].tolist()
            # 获取代码依赖关系较后的下标
            if threshold_count > 0:
                last_idx = df['Test_Index'][-threshold_count:].tolist()
            else:
                last_idx = []
            for i in range(len(y_pred)):
                if (y_pred[i] >= sigma3 - sigma1 and y_pred[i] <= sigma3 and test_index[i] in top_idx):
                    y_pred[i] = 1  # 改为正
                elif (y_pred[i] >= sigma3 and y_pred[i] <= sigma3 + sigma2 and test_index[i] in last_idx):
                    y_pred[i] = 0  # 改为负
            for i in range(len(y_pred)):
                if (y_pred[i] < sigma3):
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
            precision_scores.append(precision_score(y_test, y_pred))
            recall_scores.append(recall_score(y_test, y_pred))
    return np.mean(precision_scores), np.mean(recall_scores), np.mean(f1_scores)

def XGB_CD(dataset_name, feature_file_name, dependency_matrix):
    df = pd.read_excel('../../datasets/' + dataset_name + '/' + feature_file_name + '.xlsx')
    labels = pd.read_excel('../../datasets/' + dataset_name + '/labels.xlsx')
    # 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)

    # 假设calculate_dependency_importance等其他必要的函数已经在上下文中定义
    sigma1 = 0.04
    sigma2 = 0.15
    sigma3 = 0.17

    # 十折交叉验证
    kf = KFold(n_splits=10, shuffle=True)
    f1_scores = []
    # 进行50次十折交叉验证
    for t in range(50):
        print(t)
        for train_index, test_index in kf.split(X_normalized, labels):
            X_train, X_test = X_normalized[train_index], X_normalized[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]

            train_label = labels.iloc[train_index].values.ravel().tolist()
            df = dependency_based_resort_without_GE(dataset_name, test_index, train_index, y_pred,
                                         dependency_matrix, train_label)
            threshold_count = int(sigma2 * len(df))
            # 获取代码依赖关系较前的下标
            top_idx = df['Test_Index'][:threshold_count].tolist()
            # 获取代码依赖关系较后的下标
            if threshold_count > 0:
                last_idx = df['Test_Index'][-threshold_count:].tolist()
            else:
                last_idx = []
            for i in range(len(y_pred)):
                if (y_pred[i] >= sigma3 - sigma1 and y_pred[i] <= sigma3 and test_index[i] in top_idx):
                    y_pred[i] = 1  # 改为正
                elif (y_pred[i] >= sigma3 and y_pred[i] <= sigma3 + sigma2 and test_index[i] in last_idx):
                    y_pred[i] = 0  # 改为负
            for i in range(len(y_pred)):
                if (y_pred[i] < 0.17):
                    y_pred[i] = 0
                else:
                    y_pred[i] = 1
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)
    return np.mean(f1_scores)



#RQ3-Other
def other_classification_model(dataset_name, feature_file_name, model, test_size):
    df = pd.read_excel('../../datasets/' + dataset_name + '/' + feature_file_name + '.xlsx')
    labels = pd.read_excel('../../datasets/' + dataset_name + '/labels.xlsx')

    labels = labels.iloc[:, 0].values  # 将 labels 转换为一个 1D numpy 数组

    # 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)

    f1_scores = []

    # 进行50次十折交叉验证
    for t in range(50):
        print(t)
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X_normalized, labels, test_size=test_size, shuffle=True)

        # 根据传入的模型名选择模型
        if model == 'RF':
            model = RandomForestClassifier(n_jobs=-1)
        elif model == 'LR':
            model = LogisticRegression(n_jobs=-1)
        elif model == 'KNN':
            model = KNeighborsClassifier(n_jobs=-1, n_neighbors=5)
        elif model == 'NB':
            model = GaussianNB()
        elif model == 'SVM':
            model = SVC()

        # 训练模型
        model.fit(X_train, y_train)

        # 预测测试集
        y_pred = model.predict(X_test)

        # 计算并记录F1分数
        f1_scores.append(f1_score(y_test, y_pred))

    # 返回F1分数的均值和标准差
    return np.mean(f1_scores)

#RQ3_ML_CD
def XGB_CD_GE_by_testsize(dataset_name, feature_file_name, ci, mi, wi, dependency_matrix, test_size):
    df = pd.read_excel('../../datasets/' + dataset_name + '/' + feature_file_name + '.xlsx')
    labels = pd.read_excel('../../datasets/' + dataset_name + '/labels.xlsx')
    # 数据归一化
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(df)

    # 假设calculate_dependency_importance等其他必要的函数已经在上下文中定义
    attention_weight = [ci, mi, wi]
    sigma1 = 0.04
    sigma2 = 0.15
    sigma3 = 0.17

    f1_scores = []
    # 进行50次十折交叉验证
    for t in range(50):
        print(t)
        indices = np.arange(len(X_normalized))
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test, train_index, test_index = train_test_split(X_normalized, labels, indices, test_size=test_size, shuffle=True)
        clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]

        train_label = labels.iloc[train_index].values.ravel().tolist()
        df = dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight,
                                         dependency_matrix, train_label)
        threshold_count = int(sigma2 * len(df))
        # 获取代码依赖关系较前的下标
        top_idx = df['Test_Index'][:threshold_count].tolist()
        # 获取代码依赖关系较后的下标
        if threshold_count > 0:
            last_idx = df['Test_Index'][-threshold_count:].tolist()
        else:
            last_idx = []
        for i in range(len(y_pred)):
            if (y_pred[i] >= sigma3 - sigma1 and y_pred[i] <= sigma3 and test_index[i] in top_idx):
                y_pred[i] = 1  # 改为正
            elif (y_pred[i] >= sigma3 and y_pred[i] <= sigma3 + sigma2 and test_index[i] in last_idx):
                y_pred[i] = 0  # 改为负
        for i in range(len(y_pred)):
            if (y_pred[i] < 0.17):
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        f1 = f1_score(y_test, y_pred)
        f1_scores.append(f1)
    return np.mean(f1_scores)

def calculate_dependency_importance(dataset_name, datasets):
    sims = []
    # 创建调用矩阵列表
    dependency_matrix = []
    for dataset in datasets:
        relation_file = '../../datasets/'+dataset+'/'+dataset.lower()+'_raw_callgraph.txt'  # 设置关系文件路径
        directory = '../../datasets/'+dataset+'/code/'  # 设置 Java 文件的目录

        # 获取 Java 文件
        java_files = get_java_files(directory)

        # 解析关系文件
        class_relations, method_relations = parse_relation_file(relation_file, java_files)

        # 构建类调用矩阵和方法调用矩阵
        class_matrix = build_matrix(class_relations, java_files, "class")
        method_matrix = build_matrix(method_relations, java_files, "method")

        file_path = '../../datasets/'+dataset+'/cc_doc.txt'
        model_path = '../../GoogleNews-vectors-negative300.bin'
        w2v_matrix = load_pretrained_word2vec_and_calculate_similarity(file_path, model_path)

        if(dataset_name == dataset):
            dependency_matrix.append(class_matrix)
            dependency_matrix.append(method_matrix)
            dependency_matrix.append(w2v_matrix)

        cm = calculate_contribution_matrix('../../datasets/'+dataset+'/code', '../../datasets/'+dataset+'/'+dataset.lower()+'_solution_links.txt')
        class_matrix_sim = graph_embedding_similarity(class_matrix, cm)
        method_matrix_sim = graph_embedding_similarity(method_matrix, cm)
        w2v_matrix_sim = graph_embedding_similarity(w2v_matrix, cm)
        # print("dataset:"+dataset_name+"c:"+str(class_matrix_sim)+"m:"+str(method_matrix_sim)+"w:"+str(w2v_matrix_sim))
        #使用注意力机制计算每个依赖的重要程度
        if(dataset_name != dataset):
            c_tot = math.pow(math.e, class_matrix_sim) + math.pow(math.e, method_matrix_sim) + math.pow(math.e, w2v_matrix_sim)
            ci = math.pow(math.e, class_matrix_sim)/c_tot
            mi = math.pow(math.e, method_matrix_sim)/c_tot
            wi = math.pow(math.e, w2v_matrix_sim)/c_tot
            sims.append((ci, mi, wi))
    c_res, m_res, w_res = 0, 0, 0
    for item in sims:
        c_res += item[0]
        m_res += item[1]
        w_res += item[2]
    c_res /= (len(datasets)-1)
    m_res /= (len(datasets)-1)
    w_res /= (len(datasets)-1)
    return c_res, m_res, w_res, dependency_matrix

def get_dependency_matrix(dataset_name, datasets):
    # 创建调用矩阵列表
    dependency_matrix = []
    for dataset in datasets:
        relation_file = '../../datasets/' + dataset + '/' + dataset.lower() + '_raw_callgraph.txt'  # 设置关系文件路径
        directory = '../../datasets/' + dataset + '/code/'  # 设置 Java 文件的目录

        # 获取 Java 文件
        java_files = get_java_files(directory)

        # 解析关系文件
        class_relations, method_relations = parse_relation_file(relation_file, java_files)

        # 构建类调用矩阵和方法调用矩阵
        class_matrix = build_matrix(class_relations, java_files, "class")
        method_matrix = build_matrix(method_relations, java_files, "method")

        file_path = '../../datasets/' + dataset + '/cc_doc.txt'
        model_path = '../../GoogleNews-vectors-negative300.bin'
        w2v_matrix = load_pretrained_word2vec_and_calculate_similarity(file_path, model_path)

        if (dataset_name == dataset):
            dependency_matrix.append(class_matrix)
            dependency_matrix.append(method_matrix)
            dependency_matrix.append(w2v_matrix)

    return dependency_matrix


def dependency_based_resort(dataset_name, test_index, train_index, y_pred, attention_weight, dependency_matrix, train_label):
    df = pd.DataFrame({
        'Test_Index': test_index,
        'Prediction': y_pred
    })
    cc_num = len(os.listdir('../../datasets/'+dataset_name+'/code'))
    uc_num = len(os.listdir('../../datasets/'+dataset_name+'/req'))
    sample_num = cc_num*uc_num
    each_train_set_num = np.zeros(uc_num)
    in_train_set = np.zeros(sample_num).reshape(uc_num, cc_num)
    train_label_set = np.zeros(sample_num).reshape(uc_num, cc_num)
    for i, idx in enumerate(train_index):
        each_train_set_num[int(idx / cc_num)] += 1
        in_train_set[int(idx / cc_num)][int(idx % cc_num)] = 1
        train_label_set[int(idx / cc_num)][int(idx % cc_num)] = train_label[i]
    ci = attention_weight[0]
    mi = attention_weight[1]
    wi = attention_weight[2]

    class_matrix = dependency_matrix[0]
    method_matrix = dependency_matrix[1]
    w2v_matrix = dependency_matrix[2]
    sims = []

    for item in df['Test_Index']:
        uc_item = int(item / cc_num)
        cc_item = int(item % cc_num)
        sim = 0

        for i in range(cc_num):
            if train_label_set[uc_item][i] == 1:
                if class_matrix[cc_item][i] >= 1:
                    sim += ci * class_matrix[cc_item][i]
                if method_matrix[cc_item][i] >= 1:
                    sim += mi * method_matrix[cc_item][i]
                sim += wi * w2v_matrix[cc_item][i]
        if each_train_set_num[uc_item] == 0:
            sims.append(0)
        else:
            sims.append(sim/each_train_set_num[uc_item])
    df['similarity'] = sims
    df_sorted = df.sort_values(by='similarity', ascending=False)
    return df_sorted

def dependency_based_resort_without_GE(dataset_name, test_index, train_index, y_pred, dependency_matrix, train_label):
    df = pd.DataFrame({
        'Test_Index': test_index,
        'Prediction': y_pred
    })
    cc_num = len(os.listdir('../../datasets/'+dataset_name+'/code'))
    uc_num = len(os.listdir('../../datasets/'+dataset_name+'/req'))
    sample_num = cc_num*uc_num
    each_train_set_num = np.zeros(uc_num)
    in_train_set = np.zeros(sample_num).reshape(uc_num, cc_num)
    train_label_set = np.zeros(sample_num).reshape(uc_num, cc_num)
    for i, idx in enumerate(train_index):
        each_train_set_num[int(idx / cc_num)] += 1
        in_train_set[int(idx / cc_num)][int(idx % cc_num)] = 1
        train_label_set[int(idx / cc_num)][int(idx % cc_num)] = train_label[i]

    class_matrix = dependency_matrix[0]
    method_matrix = dependency_matrix[1]
    w2v_matrix = dependency_matrix[2]
    sims = []

    for item in df['Test_Index']:
        uc_item = int(item / cc_num)
        cc_item = int(item % cc_num)
        sim = 0

        for i in range(cc_num):
            if train_label_set[uc_item][i] == 1:
                if class_matrix[cc_item][i] >= 1:
                    sim += 1/3 * class_matrix[cc_item][i]
                if method_matrix[cc_item][i] >= 1:
                    sim += 1/3 * method_matrix[cc_item][i]
                sim += 1/3 * w2v_matrix[cc_item][i]
        if each_train_set_num[uc_item] == 0:
            sims.append(0)
        else:
            sims.append(sim/each_train_set_num[uc_item])
    df['similarity'] = sims
    df_sorted = df.sort_values(by='similarity', ascending=False)
    return df_sorted

def calculate_averages(results_by_test_size):
    # 初始化字典来累积不同配置的评估指标和计数
    config_sums = {}
    for repeat_results in results_by_test_size:
        for up, down, threshold, t3, precision, recall, f1 in repeat_results:
            key = (up, down, threshold, t3)
            if key not in config_sums:
                config_sums[key] = {'precision': 0, 'recall': 0, 'f1': 0,  'count': 0}
            config_sums[key]['precision'] += precision
            config_sums[key]['recall'] += recall
            config_sums[key]['f1'] += f1
            config_sums[key]['count'] += 1

    # 计算平均值
    averaged_results = []
    for key, values in config_sums.items():
        averaged_precision = values['precision'] / values['count']
        averaged_recall = values['recall'] / values['count']
        averaged_f1 = values['f1'] / values['count']
        averaged_results.append(key + (averaged_precision, averaged_recall, averaged_f1))

    return averaged_results
