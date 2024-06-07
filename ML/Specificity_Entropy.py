#!/usr/bin/python
# -*-coding:utf-8-*-
# @Time : 2022/11/1 14:00
# @Author : ????

from gensim import corpora
import sys
import pandas as pd

sys.path.append('Set_generation.py')
import Set_generation
import numpy as np
import warnings

warnings.filterwarnings(action='ignore')


def Entropy(queried_file, query_file):
    # ????IDF
    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)
    # ????????????corpus
    dictionary = corpora.Dictionary(queried_line)
    corpus = [dictionary.doc2bow(text) for text in query_line]
    corpus2 = [dictionary.doc2bow(text) for text in queried_line]
    corpus2_matrix = np.zeros((len(queried_line), len(dictionary)))
    row = 0
    for document in corpus2:
        for word_id, freq in document:
            corpus2_matrix[row][word_id] = freq
        row = row + 1
    entrop_dic = Set_generation.pd_entropy(corpus2_matrix)
    # print(entrop_dic)
    # ???洢???????entropy
    entrop_result = []
    for i in corpus:
        sentence_result = []
        for j in i:
            sentence_result.append((j[0], entrop_dic[j[0]]))
        entrop_result.append(sentence_result)

    row = len(query_line)
    colum = len(queried_line)
    return entrop_result, row, colum


def Avg_Entropy(queried_file, query_file):
    entrop_result, row, colum = Entropy(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # ?????????
        if entrop_result[i] != null:
            entrop = np.asarray(entrop_result[i])
            avg_entrop = np.mean(entrop, 0)
        else:
            avg_entrop = (0, 0)
        for j in range(colum):
            df.iloc[i][j] = avg_entrop[1]
    # print("avg_result", df)
    return df


def Med_Entropy(queried_file, query_file):
    entrop_result, row, colum = Entropy(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):  # ?????????
        null = []
        for i in range(row):  # ?????????
            if entrop_result[i] != null:
                entrop = np.asarray(entrop_result[i])
                avg_entrop = np.median(entrop, 0)
            else:
                avg_entrop = (0, 0)
            for j in range(colum):
                df.iloc[i][j] = avg_entrop[1]
    # print("med_result", df)
    return df


def Max_Entropy(queried_file, query_file):
    entrop_result, row, colum = Entropy(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # ?????????
        if entrop_result[i] != null:
            entrop = np.asarray(entrop_result[i])
            avg_entrop = np.amax(entrop, 0)
        else:
            avg_entrop = (0, 0)
        for j in range(colum):
            df.iloc[i][j] = avg_entrop[1]
    # print("max_result", df)
    return df


def Dev_Entropy(queried_file, query_file):
    entrop_result, row, colum = Entropy(queried_file, query_file)
    df = pd.DataFrame(index=range(row), columns=range(colum))
    null = []
    for i in range(row):  # ?????????
        if entrop_result[i] != null:
            entrop = np.asarray(entrop_result[i])
            avg_entrop = np.std(entrop, 0)
        else:
            avg_entrop = (0, 0)
        for j in range(colum):
            df.iloc[i][j] = avg_entrop[1]
    # print("dev_result", df)
    return df


if __name__ == "__main__":
    query_file = "./UC_clear.txt"
    queried_file = "./CN_MN_VN_CMT_clear.txt"
    Max_Entropy(queried_file, query_file)
    # Dev_Entropy(queried_file, query_file)
    # Med_Entropy(queried_file, query_file)
    # Avg_Entropy(queried_file, query_file)
