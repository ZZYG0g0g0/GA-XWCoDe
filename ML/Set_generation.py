#!/usr/bin/python
# -*-coding:ISO8859-1-*-
# @Time : 2022/10/30 14:00
# @Author : µËÑó
import re
import numpy as np


# Éú³É²éÑ¯¼¯»ò±»²éÑ¯¼¯(Éú³ÉÊý¾Ý¼¯)
def set_generation(query_file):
    # Éú³É²éÑ¯¼¯»ò±»²éÑ¯¼¯(Éú³ÉÊý¾Ý¼¯)
    #:param query_file: ·Ö´ÊºÍÈ¥³ýÍ£ÓÃ´ÊºóµÄÊý¾Ý¼¯
    #:return: ·µ»ØÒ»¸öÁÐ±í£¬ÁÐ±íµÄÃ¿¸öÔªËØÒ²ÊÇÒ»¸öÁÐ±í£¬ºóÕßÖÐµÄÁÐ±íµÄÃ¿¸öÔªËØ¶¼ÊÇÃ¿Ò»ÌõÊý¾ÝÖÐµÄµ¥´Ê¡£
    with open(query_file, "r", encoding="ISO8859-1") as ft:
        lines_T = ft.readlines()
    setline = []
    for line in lines_T:
        word = line.split(' ')
        word = [re.sub('\s', '', i) for i in word]
        word = [i for i in word if len(i) > 0]
        setline.append(word)
    return setline


def tf_D(index, corpus):
    # ¼ÆËãtf£¨t,D£©=ÊõÓïtÔÚÎÄÕÂDÖÐ³öÏÖµÄ´ÎÊý/ÎÄÕÂDµÄ×Ü´ÊÊý
    #:param : index£º£¨0£¬1£©µÄ0£¬´ÊÓïµÄ±êÇ©
    #:param corpus: doc2bowÖ®ºóµÄ½á¹û
    sum = 0  # sum:¸Ã´ÊÔÚËùÓÐÎÄµµÖÐ³öÏÖµÄ´ÎÊý
    all = 0  # all£ºÎÄµµµÄ×Ü´ÊÊý
    null = []
    for i in corpus:
        if i == null:
            all = all
        else:
            all = np.sum(np.asarray(i), 0)[1] + all
        for j in i:
            if index == j[0]:
                sum = sum + j[1]

    tf = sum / all
    # print("tf_D",tf)
    return tf


def tf_d(index, corpus):
    # ¼ÆËãtf£¨t,d£©=ÊõÓïtÔÚÎÄµµdÖÐ³öÏÖµÄ´ÎÊý/ÎÄµµdµÄ×Ü´ÊÊý
    #:param : index£º£¨0£¬1£©µÄ0£¬´ÊÓïµÄ±êÇ©
    #:param corpus: Ã¿Ò»¸öÎÄµµµÄdoc2bowÖ®ºóµÄ½á¹û[(0,1),(1,2)]
    all = np.sum(np.asarray(corpus), 0)[1]  # all£ºÎÄµµµÄ×Ü´ÊÊý
    tf = index[1] / all
    return tf


def pd(index, corpus):
    # ¼ÆËãp(d)=ÊõÓïtÔÚÎÄµµtÖÐ³öÏÖµÄ´ÎÊý/ÊõÓïtÔÚÎÄµµDÖÐ³öÏÖµÄ´ÎÊý
    #:param : index£º£¨0£¬1£©µÄ0£¬´ÊÓïµÄ±êÇ©
    #:param corpus: Ã¿Ò»¸öÎÄµµµÄdoc2bowÖ®ºóµÄ½á¹û[(0,1),(1,2)]
    sum = 0  # sum:¸Ã´ÊÔÚËùÓÐÎÄµµÖÐ³öÏÖµÄ´ÎÊý
    for i in corpus:
        for j in i:
            if index[0] == j[0]:
                sum = sum + j[1]
    pd_num = index[1] / sum
    # print(index[0],sum)
    return pd_num


def pd_entropy(corpus):
    # ¼ÆËãÃ¿¸ö´ÊµÄentropy
    # ¼ÆËãp(d)=ÊõÓïtÔÚÎÄµµtÖÐ³öÏÖµÄ´ÎÊý/ÊõÓïtÔÚÎÄµµDÖÐ³öÏÖµÄ´ÎÊý
    #:param : index£º£¨0£¬1£©µÄ0£¬´ÊÓïµÄ±êÇ©
    # print(corpus)
    sum = corpus / np.sum(corpus, axis=0)  # ¼ÆËãp(d)

    result_doc = np.fabs(sum * (np.log(sum) / np.log(len(corpus))))  # ¸ù¾ÝÃ¿Ò»ÁÐ£¬¼ÆËãÃ¿¸ö´ÊµÄentropy
    result_doc[np.isnan(result_doc)] = 0
    result = np.sum(result_doc, axis=0)
    return result
