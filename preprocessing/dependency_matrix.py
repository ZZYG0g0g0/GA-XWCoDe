"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/1/24
Last Updated: 2024/1/24
Version: 1.0.0
"""
import numpy as np
import os
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')

def parse_relation_file(relation_file, java_files):
    """ 解析关系文件 """
    class_relations = set()
    method_relations = set()

    with open(relation_file, 'r') as file:
        for line in file:
            if line.startswith('C:'):
                # 类调用关系
                caller = line.split(' ')[0].split('.')[-1]
                callee = line.split(' ')[1].split('.')[-1]

                if caller.strip()+'.java' in java_files and callee.strip()+'.java' in java_files:
                    class_relations.add((caller.strip(), callee.strip()))
            elif line.startswith('M:'):
                # 方法调用关系
                caller = line.split(' ')[0].split(':')[-2].split('.')[-1]
                callee = line.split(' ')[1].split(":")[-2].split('.')[-1]
                if caller.strip()+'.java' in java_files and callee.strip()+'.java' in java_files:
                    method_relations.add((caller.strip(), callee.strip()))

    return class_relations, method_relations

def build_matrix(relations, elements, type):
    """ 构建关系矩阵 """
    N = len(elements)
    matrix = np.zeros((N, N), dtype=int)
    for i in range(N):
        elements[i] = elements[i].split('.')[0]
    element_index = {element: i for i, element in enumerate(elements)}

    if type == "class":
        for caller, callee in relations:
            if caller in element_index and callee in element_index:
                i = element_index[caller]
                j = element_index[callee]
                matrix[i][j] = 1

    elif type == "method":
        for caller, callee in relations:
            if caller in element_index and callee in element_index:
                i = element_index[caller]
                j = element_index[callee]
                matrix[i][j] += 1
    elif type == "extends":
        for caller, callee in relations:
            if caller in element_index and callee in element_index:
                i = element_index[caller]
                j = element_index[callee]
                matrix[i][j] = 1

    elif type == "implements":
        for caller, callee in relations:
            if caller in element_index and callee in element_index:
                i = element_index[caller]
                j = element_index[callee]
                matrix[i][j] = 1
    return matrix

def load_pretrained_word2vec_and_calculate_similarity(file_path, model_path):
    # 加载模型
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

    # 读取和预处理文本
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        documents = [word_tokenize(line.lower()) for line in file]

    # 文档转向量
    def document_vector(doc):
        valid_words = [word for word in doc if word in model.key_to_index]
        if valid_words:
            return np.mean(model[valid_words], axis=0)
        else:
            return np.zeros(model.vector_size)

    # 计算相似度矩阵
    document_vectors = np.array([document_vector(doc) for doc in documents])
    return cosine_similarity(document_vectors)


def get_java_files(directory):
    """ 获取目录中的所有 Java 文件 """
    return [file for file in os.listdir(directory) if file.endswith('.java')]
