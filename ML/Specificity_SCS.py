from gensim import corpora
import sys

sys.path.append('Set_generation.py')
import Set_generation
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings(action='ignore')


# SCS KL散度
def KL_similarity(queried_file, query_file, output_fname=None):
    queried_line = Set_generation.set_generation(queried_file)
    query_line = Set_generation.set_generation(query_file)

    # 被查询集生成词典和corpus
    dictionary = corpora.Dictionary(queried_line + query_line)
    corpus = [dictionary.doc2bow(text) for text in queried_line]
    corpus2 = [dictionary.doc2bow(text) for text in query_line]

    A_matrix = np.zeros((len(queried_line), len(dictionary)))
    B_matrix = np.zeros((len(query_line), len(dictionary)))

    row = 0
    for document in corpus:
        for word_id, freq in document:
            A_matrix[row][word_id] = freq
        row = row + 1

    row = 0
    for document in corpus2:
        for word_id, freq in document:
            B_matrix[row][word_id] = freq
        row = row + 1
    # print(corpus)
    sum_matrix = np.sum(np.vstack((A_matrix, B_matrix)), axis=0)
    probability_A = A_matrix / sum_matrix
    probability_B = B_matrix / sum_matrix

    sim = KL_Sim(probability_A, probability_B)
    # print("KL_sim:",sim)
    if output_fname is not None:
        sim.to_excel(output_fname)
    return sim


def KL_Sim(A_set, B_set) -> pd.DataFrame:
    df = pd.DataFrame(index=range(len(B_set)), columns=range(len(A_set)))
    # 开始计算KL相似度
    for row in range(len(B_set)):
        for column in range(len(A_set)):
            df.iloc[[row], [column]] = KL_divergence(B_set[row], A_set[column])  # B_set为查询集，所以放前面
    return df


def KL_divergence(p, q):
    pk = np.asarray(p)
    pk2 = np.asarray(q)
    kl = 0
    for i in range(len(pk)):
        if ((pk[i] != 0) & (pk2[i] != 0)):
            kl = kl + pk[i] * np.log(pk[i] / pk2[i])
            # all=all+1
    return kl
