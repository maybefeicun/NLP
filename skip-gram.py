# -*- coding: utf-8 -*-
# @Time    : 2018/3/19 9:33
# @Author  : chen
# @File    : skip-gram.py
# @Software: PyCharm

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import urllib.request
import io

sess = tf.Session()

# 声明一些参数
embedding_size = 200 # 这个很重要表明每个单词的嵌套大小是一个长度为200的向量

# 声明数据加载函数
def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polarity.neg')
    # 检查文件是否已经下载了
    if os.path.exists(save_folder_name):
        pos_data = []
        with open(pos_file, 'r') as temp_pos_file:
            for row in temp_pos_file:
                pos_data.append(row)
        neg_data = []
        with open(neg_file, 'r') as temp_neg_file:
            for row in temp_neg_file:
                neg_data.append(row)
    else:
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
        stream_data = urllib.request.urlopen(movie_data_url)
        temp = io.