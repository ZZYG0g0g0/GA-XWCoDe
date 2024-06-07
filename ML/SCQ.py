# Instantiate the count vectorizer and tfidf for the corpus
import numpy as np
import pandas as pd
from math import log, sqrt
from sklearn.feature_extraction.text import CountVectorizer
from Set_generation import set_generation
from createFittedTF_IDF import *
import warnings

warnings.filterwarnings(action='ignore')


# The average of the collection-query similarity (SCQ) over all query terms
def calcSCQList(query, cv, fittedTF_IDF):
    SCQList = []
    if isinstance(query, list):
        documentString = ' '.join(query)  # 生成被查询集

        # Calculate the Term Frequency of the document
        inputDocs = [documentString]

        # count matrix
        count_vector = cv.transform(inputDocs)

        # tf-idf scores
        tf_idf_vector = fittedTF_IDF.transform(count_vector)
        feature_names = cv.get_feature_names_out()
        # feature_names = cv.get_feature_names()   #pip install scikit-learn==0.20.0
        # place tf-idf values in a pandas data frame
        df = pd.DataFrame(tf_idf_vector.T.todense(),
                          index=feature_names, columns=["tfidf"])

        # print(df)
        # Find the tfidf of the term
        for queryTerm in query:
            try:
                tfidf = df["tfidf"][queryTerm]
                SCQ = (1 + log(tfidf))
                SCQList.append(SCQ)
            except:
                continue

    # avgSCQ = sum(SCQList) / documentCount
    # print(avgSCQ)
    return (SCQList)


# The average of the collection-query similarity (SCQ) over all query terms
def calcAvgSCQ(SCQList, documentCount):
    avgSCQ = sum(SCQList) / documentCount
    return (avgSCQ)


# The average of the collection-query similarity (SCQ) over all query terms
def calcMaxSCQ(SCQList):
    termCount = len(SCQList)
    if (termCount != 0):
        maxSCQ = np.amax(SCQList)
    else:
        maxSCQ = np.NaN
    return (maxSCQ)


# The average of the collection-query similarity (SCQ) over all query terms
def calcSumSCQ(SCQList):
    sumSCQ = sum(SCQList)
    return (sumSCQ)


def createCorpusFromDocumentList(token_column):
    # token_list = token_column.tolist()
    token_list = token_column
    corpus_list = []
    for document in token_list:
        # Only join to the string when a list. When it is not a list, then it is np.NaN, thus no changes
        if (isinstance(document, list)):
            # Transform list to a string for SKLEARN to accept the input.
            token_string = ' '.join(document)
            # Add string to the corpus list
            corpus_list.append(token_string)

    return (corpus_list)


def calcSCQList_final(queried_file, query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)
    corpus = createCorpusFromDocumentList(queried_line)
    cv = CountVectorizer()
    # cv.fit_transform(corpus)
    fittedTF_IDF = createFittedTF_IDF(cv, corpus)
    row = len(query_line)
    colum = len(queried_line)
    df_avg = pd.DataFrame(index=range(row), columns=range(colum))
    df_sum = pd.DataFrame(index=range(row), columns=range(colum))
    df_max = pd.DataFrame(index=range(row), columns=range(colum))
    len_d = len(query_line)
    for i in range(row):
        SCQList = calcSCQList(query_line[i], cv, fittedTF_IDF)
        for j in range(colum):
            df_avg.iloc[i][j] = calcAvgSCQ(SCQList, len_d)
            df_sum.iloc[i][j] = calcSumSCQ(SCQList)
            df_max.iloc[i][j] = calcMaxSCQ(SCQList)
    # print(df_avg)
    return df_avg, df_sum, df_max


def avgSCQ(queried_file, query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)
    corpus = createCorpusFromDocumentList(queried_line)
    cv = CountVectorizer()
    fittedTF_IDF = createFittedTF_IDF(cv, corpus)
    row = len(query_line)
    colum = len(queried_line)
    df_avg = pd.DataFrame(index=range(row), columns=range(colum))
    len_d = len(query_line)
    for i in range(row):
        SCQList = calcSCQList(query_line[i], cv, fittedTF_IDF)
        for j in range(colum):
            df_avg.iloc[i][j] = calcAvgSCQ(SCQList, len_d)
    return df_avg


def sumSCQ(queried_file, query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)
    corpus = createCorpusFromDocumentList(queried_line)
    cv = CountVectorizer()
    fittedTF_IDF = createFittedTF_IDF(cv, corpus)
    row = len(query_line)
    colum = len(queried_line)
    df_sum = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        SCQList = calcSCQList(query_line[i], cv, fittedTF_IDF)
        for j in range(colum):
            df_sum.iloc[i][j] = calcSumSCQ(SCQList)
    return df_sum


def maxSCQ(queried_file, query_file):
    queried_line = set_generation(queried_file)
    query_line = set_generation(query_file)
    corpus = createCorpusFromDocumentList(queried_line)
    cv = CountVectorizer()
    fittedTF_IDF = createFittedTF_IDF(cv, corpus)
    row = len(query_line) #代码
    colum = len(queried_line) #uc
    df_max = pd.DataFrame(index=range(row), columns=range(colum))
    for i in range(row):
        SCQList = calcSCQList(query_line[i], cv, fittedTF_IDF)
        for j in range(colum):
            df_max.iloc[i][j] = calcMaxSCQ(SCQList)
    return df_max


if __name__ == '__main__':
    query_file = "./CN_MN_VN_CMT_clear.txt"
    queried_file = "./UC_clear.txt"
    output_fname = "./SCQ"
    maxSCQ(query_file, queried_file)

    # for i in query_line:
    # calcIDFList(i, cv, tfidf_transformer)
    # calcICTFList(i,cv,len(query_line))
    # calcEntropyList(query, cv, documentCount, docCollection)
