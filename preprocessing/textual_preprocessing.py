"""
Author: 邹致远
Email: www.pisyongheng@foxmail.com
Date Created: 2024/1/24
Last Updated: 2024/1/24
Version: 1.0.0
"""
# @Time : 2023/09/06 15:21
# @Author : zzy
import os
import re

import stanfordcorenlp
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import json
from nltk.stem import PorterStemmer

nlp = stanfordcorenlp.StanfordCoreNLP(r'../CoreNLP/stanford-corenlp-4.5.4')

#1.去除驼峰命名
def split_camel_case(s):
    # 将字符串s切分为单词列表
    words = s.split()
    # 对于列表中的每一个单词，如果它是驼峰命名的，则拆分它
    for index, word in enumerate(words):
        splitted = ' '.join(re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', word))
        if splitted:  # 如果成功拆分，则替换原单词
            words[index] = splitted

    # 将处理后的单词列表连接起来，并返回
    return ' '.join(words)

#2.分词
def tokenize_text(text):
    return word_tokenize(text)

#3.将词转为小写
def words_to_lowercase(words):
    return [word.lower() for word in words]

#4.去除停用词，标点和数字
def filter_words(word_list):
    # nltk.download('stopword')

    stop_words = set(stopwords.words('english'))  # 停用词列表
    punctuation_symbols = set(string.punctuation)  # 标点符号列表
    words = []
    for word in word_list:
        if '.' in word:
            for each in word.split('.'):
                words.append(each)
        words.append(word)
    # 过滤停用词、标点符号和数字
    filtered_words = [word for word in words if word.lower() not in stop_words and
                      not any(char in punctuation_symbols for char in word) and
                      word.isalpha()]
    return filtered_words

#5.词形还原
def extract_restore(text):
    # Initialize NLTK's PorterStemmer
    stemmer = PorterStemmer()

    def split_text(text, max_length):
        return [text[i:i + max_length] for i in range(0, len(text), max_length)]

    chunks = split_text(text, 50000)
    all_stems = []
    for chunk in chunks:
        doc = nlp.annotate(chunk, properties={
            'annotators': 'lemma',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        })
        doc = json.loads(doc)
        lemmas = [word['lemma'] for sentence in doc['sentences'] for word in sentence['tokens']]
        stems = [stemmer.stem(token) for token in lemmas]
        all_stems.extend(stems)

    return ' '.join(all_stems)

#预处理
def preprocessing(dataset_name):
    file_names_cc = os.listdir('../datasets/'+ dataset_name +'/code')#读文件夹cc的文件
    file_names_uc = os.listdir('../datasets/' + dataset_name + '/req')  # 读文件夹uc的文件
    cc_dir = '../datasets/' + dataset_name
    uc_dir = '../datasets/' + dataset_name
    if not os.path.exists(cc_dir):
        os.makedirs(cc_dir)
    if not os.path.exists(uc_dir):
        os.makedirs(uc_dir)
    open(cc_dir + '/cc_doc.txt', 'w').close()
    open(uc_dir + '/uc_doc.txt', 'w').close()
    for file_name in file_names_cc:
        with open('../datasets/'+dataset_name+'/code/'+file_name,'r',encoding='ISO8859-1')as cf:
            text = ""
            lines = cf.readlines()
            for line in lines:
                text += line.strip()
            res = split_camel_case(text)
            res = tokenize_text(res)
            res = words_to_lowercase(res)
            res = filter_words(res)
            res = extract_restore(' '.join(res))
        with open('../datasets/'+dataset_name+'/cc_doc.txt','a',encoding='ISO8859-1')as cwf:
            cwf.write(res)
            cwf.write('\n')
    for file_name in file_names_uc:
        with open('../datasets/'+dataset_name+'/req/'+file_name,'r',encoding='ISO8859-1')as uf:
            text = ""
            lines = uf.readlines()
            for line in lines:
                text += line.strip()
            res = split_camel_case(text)
            res = tokenize_text(res)
            res = words_to_lowercase(res)
            res = filter_words(res)
            res = extract_restore(' '.join(res))
        with open('../datasets/'+dataset_name+'/uc_doc.txt','a',encoding='ISO8859-1')as uwf:
            uwf.write(res)
            uwf.write('\n')

if __name__ == '__main__':
    datasets = ['eAnci','eTour','iTrust','smos']
    for dataset in datasets:
        preprocessing(dataset)
