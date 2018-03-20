# -*- coding: utf-8 -*-
# @Time    : 2018/3/20 13:39
# @Author  : chen
# @File    : 词袋模型ceshi.py
# @Software: PyCharm

# -*- coding: utf-8 -*-
import numpy as np


def load_data():
    """ 1. 导入train_x, train_y """
    train_x = [["my", "dog", "has", "flea", "problems", "help", "please"],
               ["maybe", "not", "take", "him", "to", "dog", "park", "stupid"],
               ["my", "dalmation", "is", "so", "cute", "I", "love", "him"],
               ["stop", "posting", "stupid", "worthless", "garbage"],
               ["him", "licks", "ate", "my", "steak", "how", "to", "stop", "him"],
               ["quit", "buying", "worthless", "dog", "food", "stupid"],
               ["my", "my", "dog", "dog"]]
    label = [0, 1, 0, 1, 0, 1, 0]
    return train_x, label


def setOfWord(train_x):
    """ 2. 所有单词不重复的汇总到一个列表
    train_x: 文档合集, 一个样本构成一个文档
    wordSet: 所有单词生成的集合的列表
    """
    wordList = []

    length = len(train_x)
    for sample in range(length):
        wordList.extend(train_x[sample])
    wordSet = list(set(wordList))
    return wordSet


def create_wordVec(sample, wordSet, mode="wordSet"):
    """ 3. 将一个样本生成一个词向量 """
    length = len(wordSet)
    wordVec = [0] * length

    if mode == "wordSet":
        for i in range(length):
            if wordSet[i] in sample:
                wordVec[i] = 1
    elif mode == "wordBag":
        for i in range(length):
            for j in range(len(sample)):
                if sample[j] == wordSet[i]:
                    wordVec[i] += 1
    else:
        raise (Exception("The mode must be wordSet or wordBag."))
    return wordVec


def main(mode="wordSet"):
    train_x, label = load_data()
    wordSet = setOfWord(train_x)

    sampleCnt = len(train_x)
    train_matrix = []
    for i in range(sampleCnt):
        train_matrix.append(create_wordVec(train_x[i], wordSet, "wordBag"))
    return train_matrix


if __name__ == "__main__":
    train_x, label = load_data()
    wordSet = setOfWord(train_x)
    train_matrix = main("wordSet")