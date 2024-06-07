"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/1/24
Last Updated: 2024/1/24
Version: 1.0.0
"""
import os

import numpy as np
import math
import preprocessing.dependency_matrix as dm
from node2vec import Node2Vec
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool


def calculate_contribution_matrix(dataset_path, true_set_path):
    c_names = [cname.split('.')[0] for cname in os.listdir(dataset_path)]
    c_num = len(c_names)
    contribution_matrix = np.zeros((c_num, c_num))

    file_to_requirements = {}
    with open(true_set_path, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            r_name, c_name = line.split()
            c_name = c_name.split('.')[0]
            if c_name in c_names:
                file_to_requirements.setdefault(c_name, set()).add(r_name)

    for i in range(c_num):
        for j in range(i + 1, c_num):
            requirements_i = file_to_requirements.get(c_names[i], set())
            requirements_j = file_to_requirements.get(c_names[j], set())

            m = len(requirements_i | requirements_j)
            n = len(requirements_i & requirements_j)

            if m == 0:
                contribution = 0
            else:
                contribution = math.log(m + 1) * (math.exp(n / m) - 1)

            contribution_matrix[i][j] = contribution
            contribution_matrix[j][i] = contribution
    return contribution_matrix

def graph_embedding_similarity(graph_matrix1, graph_matrix2, dimensions=64, walk_length=30, num_walks=200, workers=28):
    # 将邻接矩阵转换为 NetworkX 图对象
    G1 = nx.from_numpy_matrix(graph_matrix1)
    G2 = nx.from_numpy_matrix(graph_matrix2)

    # 初始化 Node2Vec 模型
    node2vec1 = Node2Vec(G1, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    node2vec2 = Node2Vec(G2, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)

    # 训练 Node2Vec 模型
    model1 = node2vec1.fit(window=10, min_count=1, batch_words=4)
    model2 = node2vec2.fit(window=10, min_count=1, batch_words=4)

    # 获取所有节点的嵌入向量
    embeddings1 = np.array([model1.wv[str(i)] for i in range(len(graph_matrix1))])
    embeddings2 = np.array([model2.wv[str(i)] for i in range(len(graph_matrix2))])

    # 计算两个图的平均嵌入向量
    avg_embedding1 = np.mean(embeddings1, axis=0)
    avg_embedding2 = np.mean(embeddings2, axis=0)

    # 计算余弦相似度
    similarity = cosine_similarity([avg_embedding1], [avg_embedding2])

    return similarity[0][0]
