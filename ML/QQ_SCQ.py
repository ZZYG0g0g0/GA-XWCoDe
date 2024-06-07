# -*-coding:utf-8-*-
# @Time : 2022/10/10 20:44
# @Author : 邓洋、李幸阜
import pandas as pd

import warnings
import SCQ

warnings.filterwarnings(action='ignore')


def QQ_feature_generation(fname, tname, output_fname=None):
    QQ_feature = pd.DataFrame()
    code_feature = ['CN_MN_VN_CMT']
    flag = 0
    options = [SCQ.calcSCQList_final]
    # print(options)
    for option in options:
        df_avg, df_sum, df_max = option(fname + code_feature[0] + '_clear.txt', tname)  # tname为查询集，fname为被查询集
        QQ_feature[flag] = pd.concat([df_avg.iloc[i] for i in range(df_avg.shape[0])], axis=0,
                                     ignore_index=True)  # 将sim所有行转化为IR_based_feature的一列
        # print(option)
        flag = flag + 1
        QQ_feature[flag] = pd.concat([df_sum.iloc[i] for i in range(df_sum.shape[0])], axis=0,
                                     ignore_index=True)  # 将sim所有行转化为IR_based_feature的一列
        flag = flag + 1
        QQ_feature[flag] = pd.concat([df_max.iloc[i] for i in range(df_max.shape[0])], axis=0,
                                     ignore_index=True)  # 将sim所有行转化为IR_based_feature的一列
        flag = flag + 1

        df_avg, df_sum, df_max = option(tname, fname + code_feature[0] + '_clear.txt')  # 查询集和被查询集交换
        QQ_feature[flag] = pd.concat([df_avg.iloc[:, i] for i in range(df_avg.shape[1])], axis=0,
                                     ignore_index=True)  # 将sim所有列转化为IR_based_feature的一列
        flag = flag + 1

        QQ_feature[flag] = pd.concat([df_sum.iloc[:, i] for i in range(df_sum.shape[1])], axis=0,
                                     ignore_index=True)  # 将sim所有列转化为IR_based_feature的一列
        flag = flag + 1

        QQ_feature[flag] = pd.concat([df_max.iloc[:, i] for i in range(df_max.shape[1])], axis=0,
                                     ignore_index=True)  # 将sim所有列转化为IR_based_feature的一列
        flag = flag + 1

    if output_fname is not None:
        QQ_feature.to_excel(output_fname + ".xlsx")
    return QQ_feature


if __name__ == '__main__':
    fname = "../easyclinic_ID_UC/"
    tname = "../easyclinic_ID_UC/uc_clear.txt"
    output_fname = "../easyclinic_ID_UC/SCQ"
    QQ_feature_generation(fname, tname, output_fname)
