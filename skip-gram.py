# -*- coding: utf-8 -*-
# @Time    : 2018/3/19 9:33
# @Author  : chen
# @File    : skip-gram.py
# @Software: PyCharm

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import string
import requests
import collections
import io
import gzip
import tarfile
import urllib.request
from nltk.corpus import stopwords
from tensorflow.python.framework import ops

ops.reset_default_graph() # 清除默认图的堆栈，并设置全局图为默认图

os.chdir(os.path.dirname(os.path.realpath(__file__))) # 不知道是干什么用的

sess = tf.Session()

# 定义一些参数
batch_size = 100
embedding_size = 200 # 代码的关键参数，表示词向量的维数
vocabulary_size = 10000
generations = 100000
print_loss_every = 2000

num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
window_size = 2  # 设置上下文的词的大小

# 声明英语的停用词
stops = stopwords.words('english')

# 测试单词
print_valid_every = 5000
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']


# Later we will have to transform these into indices

# 下载数据集
def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

    # Check if files are already downloaded
    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

        # Save tar.gz file
        req = requests.get(movie_data_url, stream=True)
        with open('temp_movie_review_temp.tar.gz', 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        # Extract tar.gz file into temp folder
        tar = tarfile.open('temp_movie_review_temp.tar.gz', "r:gz")
        tar.extractall(path='temp')
        tar.close()

    pos_data = []
    with open(pos_file, 'r', encoding='latin-1') as f:
        for line in f:
            pos_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    pos_data = [x.rstrip() for x in pos_data]

    neg_data = []
    with open(neg_file, 'r', encoding='latin-1') as f:
        for line in f:
            neg_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    neg_data = [x.rstrip() for x in neg_data]

    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] * len(neg_data) # 这个目标生成的方法需要注意

    return (texts, target)


texts, target = load_movie_data()

# 对数据集进行预处理
def normalize_text(texts, stops):
    texts = [line.lower() for line in texts]
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    texts = [' '.join([word for word in x.split(' ') if word not in stopwords]) for x in texts] # 去停用词，这个可以使用
    texts = [' '.join(x.split()) for x in texts]

    return texts

texts = normalize_text(texts, stops)

# 预处理数据集，删除单词较少的数据集
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]

# 生成字典，计数越大的单词标记越小
def build_dictionary(sentences, vocabulary_size):
    split_sentences = [s.split() for s in sentences]
    words = [x for sublist in split_sentences for x in sublist] # 这个方法需要注意
    count = [['RARE', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    word_dict = {}
    for word, word_count in count:
        word_dict[word] = len(word_dict)
    return word_dict

# 本段函数利用字典将一个句子标记为一个vocabulary_size大小的向量
def text_to_numbers(sentences, word_dict):
    data = []
    for sentence in sentences:
        sentence_data = []
        for word in sentence:
            if word in word_dict:
                word_ix = word_dict[word]
            else:
                word_ix = 0
            sentence_data.append(word_ix)
        data.append(sentence_data)
    return data

word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

# 用测试单词进行测试下
valid_examples = [word_dictionary[x] for x in valid_words]

# 核心代码，生成skip-gram模型
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size