# -*- coding: utf-8 -*-
# @Time    : 2018/3/17 16:43
# @Author  : chen
# @File    : tf-idf2.py
# @Software: PyCharm

# coding:utf-8
from sklearn.feature_extraction.text import CountVectorizer

#语料
corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
#将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
'''
这个将投影出向量来（x, y)
x：句子的记号
y: 该单词在字典中的位置
比如(1, 2)表示为在第二个句子中的该单词在字典的位置为2
'''
X = vectorizer.fit_transform(corpus)
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()
print(word)
#查看词频结果
print(X.toarray())

from sklearn.feature_extraction.text import TfidfTransformer

#类调用
transformer = TfidfTransformer()
print(transformer)
#将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X) # 这个已经可以生成词向量了
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重
print(tfidf.toarray())