# -*-coding:utf-8-*-
# @Time : 2022/10/10 20:44
# @Author : 邓洋、李幸阜

import pandas as pd
import warnings

import SCQ
import Specificity_Entropy
import Specificity_ICTF
import Specificity_IDF
import Specificity_QS
import Specificity_SCS
import Document_Statistics as dt
import VAR
import PMI

warnings.filterwarnings(action='ignore')


def QQ_feature_generation(fname, tname, output_fname=None):
    """
    生成两个制品之间链接向量(笛卡尔积)
    :param fname: 制品1
    :param tname: 制品2
    :return: 链接向量，不带label
    """
    #  PMI.avgPMI,
    #  PMI.maxPMI,

    QQ_feature = pd.DataFrame()
    options = [
        VAR.SumVAR,
        VAR.AvgAVR,
        VAR.MaxVAR,
        Specificity_Entropy.Dev_Entropy,
        Specificity_Entropy.Med_Entropy,
        Specificity_Entropy.Avg_Entropy,
        Specificity_Entropy.Max_Entropy,
        Specificity_ICTF.AvgICDF,
        Specificity_ICTF.MaxICDF,
        Specificity_ICTF.DevICDF,
        Specificity_IDF.AvgIDF,
        Specificity_IDF.MaxIDF,
        Specificity_IDF.DevIDF,
        Specificity_SCS.KL_similarity,
        Specificity_QS.QS,
        SCQ.avgSCQ,
        SCQ.sumSCQ,
        SCQ.maxSCQ,
        dt.UniqueWordCount,
        dt.TotalWordCount,
        dt.PairOverlap,
       ]
    name = ["SumVAR",#
            "AvgAVR",#
            "MaxVAR",#
            "Dev_Entropy",#
            "Med_Entropy",#
            "Avg_Entropy",#
            "Max_Entropy",#
            "AvgICDF",#
            "MaxICDF",#
            "DevICDF",#
            'AvgIDF',#
            'MaxIDF',#
            "DevIDF",#
            'KL_similarity',#SCS
            'QS',#
            "avgSCQ",#
            "sumSCQ",#
            "maxSCQ",#
            'UniqueWordCount',
            'TotalWordCount',
            "PairOverlap",
            ]#

    code_feature = ['code']
    name_columns = []
    index = 0
    flag = 0
    for option in options:
        print(option)
        sim = option(fname, tname)  # tname为查询集，fname为被查询集
        QQ_feature[flag] = pd.concat([sim.iloc[i] for i in range(sim.shape[0])], axis=0,
                                     ignore_index=True)  # 将sim所有行转化为IR_based_feature的一列
        name_columns.append(name[index] + '_1')
        flag = flag + 1

        if dt.PairOverlap != option:
            sim = option(tname, fname)  # 查询集和被查询集交换
            QQ_feature[flag] = pd.concat([sim.iloc[:, i] for i in range(sim.shape[1])], axis=0,
                                         ignore_index=True)  # 将sim所有列转化为IR_based_feature的一列
            flag = flag + 1
            name_columns.append(name[index] + '_2')
        index = index + 1

    if output_fname is not None:
        QQ_feature.columns = name_columns
        QQ_feature.to_excel(output_fname + ".xlsx")
    return QQ_feature


if __name__ == '__main__':
    datasets = ['eANCI', 'eTour', 'iTrust', 'smos']
    for dataset in datasets:
        fname = "../datasets/" + dataset + "/cc_doc.txt"
        tname = "../datasets/" + dataset + "/uc_doc.txt"
        output_fname = "../datasets/" + dataset + "/QQ"
        print(QQ_feature_generation(fname, tname, output_fname))

