#!/usr/bin/python
# -*-coding:GBK-*-
# Instantiate the count vectorizer and tfidf for the corpus

import numpy as np
import pandas as pd
from math import log
import itertools
from sklearn.feature_extraction.text import CountVectorizer

from SCQ import createCorpusFromDocumentList
from Set_generation import set_generation


import warnings

warnings.filterwarnings(action='ignore')


def createTermPairs(cv):
    terms = list(cv.vocabulary_.keys())

    # Create all possible pair combinations from the terms in the query
    pairCombinationList = list(itertools.combinations(terms, 2))

    return (pairCombinationList)  # ('chosen', 'glucos')


# 计算词频
# Method to find out how often a term occurs in a document
def findTermFrequencies(cv, docCollection):
    terms = list(cv.vocabulary_.keys())
    termFrequencies = {}
    for term in terms:
        termCounter = 0
        for document in docCollection:
            if isinstance(document, list):
                if term in document:
                    termCounter = termCounter + 1
        termFrequencies[term] = termCounter
    return (termFrequencies)


# Method to find out how often both terms occur in a document.
def findTermPairFrequencies(termPairs, docCollection):
    termPairFrequencies = {}
    for termPair in termPairs:
        termPairCount = 0
        for document in docCollection:
            if (isinstance(document, list)):
                if all(i in document for i in termPair):
                    termPairCount = termPairCount + 1
        termPairFrequencies[termPair] = termPairCount
    # print(termPairFrequencies)
    return (termPairFrequencies)


def calcPMIList(query, termFrequencies, termPairFrequencies, docCollection):
    if isinstance(query, list):
        # Find the frequencies of the individual terms and the pairs
        pairCombinationList = list(itertools.combinations(query, 2))
        termOccurances = []
        for pair in pairCombinationList:
            try:
                q1Freq = termFrequencies[pair[0]]
            except:
                q1Freq = 0
            try:
                q2Freq = termFrequencies[pair[1]]
            except:
                q2Freq = 0
            try:
                q1q2Freq = termPairFrequencies[pair]
            except:
                q1q2Freq = 0

            termOccurances.append({'q1Freq': q1Freq,
                                   'q2Freq': q2Freq,
                                   'q1q2Freq': q1q2Freq})

        docCount = len(docCollection)
        pmiList = []
        for term in termOccurances:
            pq1 = term['q1Freq'] / docCount
            pq2 = term['q2Freq'] / docCount
            pq1q2 = term['q1q2Freq'] / docCount

            try:
                pmi = log(pq1q2 / (pq1 * pq2))
            except:
                pmi = np.nan
            pmiList.append(pmi)
        return (pmiList)
    else:
        return 0


def calcAvgPMI(pmiList):
    if (isinstance(pmiList, list)):
        pairCount = len(pmiList)
        if (pairCount != 0):
            # Calculate the average of all the entropies
            avgPMI = np.nansum(pmiList) / pairCount
        else:
            avgPMI = 0
        return (avgPMI)
    return 0


def calcMaxPMI(pmiList):
    if (isinstance(pmiList, list)):
        pairCount = len(pmiList)
        if (pairCount != 0):
            maxPMI = np.nanmax(pmiList)
        else:
            maxPMI = np.nan
        return (maxPMI)
    return 0


def calcPMIList_final(query_file, queried_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    corpus = createCorpusFromDocumentList(queried_line)
    cv = CountVectorizer()
    cv.fit_transform(corpus)
    termPairs = createTermPairs(cv)

    termPairFrequencies = findTermPairFrequencies(termPairs, query_line)
    termFrequencies = findTermFrequencies(cv, query_line)

    row = len(query_line)
    colum = len(queried_line)
    df_avg = pd.DataFrame(index=range(row), columns=range(colum))
    df_max = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        pmiList = calcPMIList(i, termFrequencies, termPairFrequencies, query_line)
        for j in range(colum):
            df_avg.iloc[i][j] = calcAvgPMI(pmiList)
            df_max.iloc[i][j] = calcMaxPMI(pmiList)
    # print(df_max)

    return df_avg, df_max


def avgPMI(query_file, queried_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    corpus = createCorpusFromDocumentList(queried_line)
    cv = CountVectorizer()
    cv.fit_transform(corpus)
    termPairs = createTermPairs(cv)

    termPairFrequencies = findTermPairFrequencies(termPairs, query_line)
    termFrequencies = findTermFrequencies(cv, query_line)

    row = len(query_line)
    colum = len(queried_line)
    df_avg = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        pmiList = calcPMIList(i, termFrequencies, termPairFrequencies, query_line)
        for j in range(colum):
            df_avg.iloc[i][j] = calcAvgPMI(pmiList)
    return df_avg


def maxPMI(query_file, queried_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)

    corpus = createCorpusFromDocumentList(queried_line)
    cv = CountVectorizer()
    cv.fit_transform(corpus)
    termPairs = createTermPairs(cv)

    termPairFrequencies = findTermPairFrequencies(termPairs, query_line)
    termFrequencies = findTermFrequencies(cv, query_line)

    row = len(query_line)
    colum = len(queried_line)
    df_max = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        pmiList = calcPMIList(i, termFrequencies, termPairFrequencies, query_line)
        for j in range(colum):
            df_max.iloc[i][j] = calcAvgPMI(pmiList)
    return df_max


if __name__ == "__main__":
    query_file = "./CN_MN_VN_CMT_clear.txt"
    queried_file = "./UC_clear.txt"
    output_fname = "./QQ"
    print(avgPMI(query_file, queried_file))

