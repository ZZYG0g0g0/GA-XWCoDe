#!/usr/bin/python
# -*-coding:GBK-*-
# @Time : 2022/10/29 20:33
# @Author : 邓洋
from gensim import corpora
import numpy as np
import sys
import pandas as pd

sys.path.append('Set_generation.py')
import Set_generation


def ICDF(queried_file, query_file):
    # 计算IDF
    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)
    # 生成词典和计算词频corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in query_line]
    # print(corpus)
    result = []
    D = len(query_line)
    for i in corpus:
        sentence = []
        for j in i:
            tf = Set_generation.tf_D(j[0], corpus)
            icdf = np.log10(D / tf)
            sentence.append((j[0], icdf))
        result.append(sentence)
    row = len(query_line)
    colum = len(queried_line)
    return result, row, colum


def AvgICDF(queried_file, query_file):
    # 计算每个文本的ICDF平均值
    result, row, colum = ICDF(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # 每一个查询集
        if result[i] == null:
            avg_ICDF = (0, 0)
        else:
            icdf = np.asarray(result[i])
            avg_ICDF = np.mean(icdf, 0)
        for j in range(colum):
            df.iloc[i][j] = avg_ICDF[1]
    print("AvgICDF", df)
    return df


def MaxICDF(queried_file, query_file):
    # 计算每个文本的ICDF最大值
    result, row, colum = ICDF(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # 每一个查询集
        if result[i] == null:
            max_icdf = (0, 0)
        else:
            icdf = np.asarray(result[i])
            max_icdf = np.amax(icdf, 0)
        for j in range(colum):
            df.iloc[i][j] = max_icdf[1]
    print("MaxICDF", df)
    return df


def DevICDF(queried_file, query_file):
    # 计算每个文本的ICDF标准差
    result, row, colum = ICDF(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # 每一个查询集
        if result[i] == null:
            dev_icdf = (0, 0)
        else:
            icdf = np.asarray(result[i])
            dev_icdf = np.std(icdf, 0)
        for j in range(colum):
            df.iloc[i][j] = dev_icdf[1]
    print("MaxICDF", df)
    return df


if __name__ == "__main__":
    query_file = "./UC_clear.txt"
    queried_file = "./CN_MN_VN_CMT_clear.txt"

    AvgICDF(queried_file, query_file)
    MaxICDF(queried_file, query_file)
    DevICDF(queried_file, query_file)
