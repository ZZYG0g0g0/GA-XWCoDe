"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2023/9/10
Last Updated: 2023/9/10
Version: 1.0.0
"""
import os

import pandas as pd

# files = ['./dataset/iTrust/IR.xlsx', './dataset/iTrust/QQ.xlsx', './dataset/iTrust/labels.xlsx']


#把特征拼起来
def feature_generate(files):
    dfs = []
    for file in files:
        if file != "../datasets/" + dataset + "/labels.xlsx":
            df = pd.read_excel(file, index_col=0)
        else:
            df = pd.read_excel(file)
        # df.reset_index(drop=True, inplace=True)  # 重置索引并去除原来的索引列
        dfs.append(df)
    final_df = pd.concat(dfs,axis=1)
    return final_df

#生成标签
def label_generate(dataset_name):
    with open('../datasets/'+dataset_name+"/"+dataset_name.lower()+'_solution_links.txt','r',encoding='ISO-8859-1')as f:
        true_links = [line.strip() for line in f.readlines()]
    #获取uc和cc名字
    uc_files = [f.replace('.txt','') for f in os.listdir('../datasets/'+dataset_name+'/req')]
    cc_files = [f for f in os.listdir('../datasets/'+dataset_name+'/code')]
    #初始化一个列表来保存标签
    labels = []
    #遍历每一个uc和cc
    for uc in uc_files:
        for cc in cc_files:
            #检查是否在真集中
            if f"{uc}.txt: {cc}" in true_links:
                labels.append(1)
            else:
                labels.append(0)
    #将标签转为一个DataFrame
    labels_df = pd.DataFrame(labels,columns=["label"])
    labels_df.to_excel("../datasets/"+dataset_name+"/labels.xlsx", index=False)

#用皮尔逊相关系数选择特征
def feature_select(df, store_name, dataset_name):
    # 计算所有特征与目标变量之间的皮尔逊相关系数
    correlations = df.iloc[:, :-1].apply(lambda x: x.corr(df.iloc[:, -1]))
    # 根据相关系数的绝对值排序特征
    sorted_features = correlations.abs().sort_values(ascending=False)
    # 计算要选择的特征数量
    num_selected = int(0.75 * len(sorted_features))
    # 选择前50%的特征
    selected_features = sorted_features.head(num_selected).index.tolist()
    # 使用所选特征创建新的DataFrame
    selected_df = df[selected_features]
    # 将新DataFrame写入Excel
    selected_df.to_excel('../datasets/'+dataset_name+'/'+store_name+'.xlsx', index=False)
    # print(selected_features)
    return selected_features


if __name__ == '__main__':
    datasets = ['eANCI', 'eTour', 'iTrust', 'smos']
    for dataset in datasets:
        label_generate(dataset)
        file = ["../datasets/" + dataset + "/IR.xlsx", "../datasets/" + dataset + "/QQ.xlsx",
                "../datasets/"+dataset+"/fine_grain.xlsx" , "../datasets/" + dataset + "/labels.xlsx"]
        test = feature_generate(file)
        feature_select(test, "selected_features", dataset)