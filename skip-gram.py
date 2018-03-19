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
generations = 5000
print_loss_every = 1000

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
    texts = [' '.join([word for word in x.split(' ') if word not in stops]) for x in texts] # 去停用词，这个可以使用
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
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys())) # 这个取反向字典的方法需要注意
text_data = text_to_numbers(texts, word_dictionary)

# 用测试单词进行测试下
valid_examples = [word_dictionary[x] for x in valid_words]

# 核心代码，生成skip-gram模型
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        rand_sentence = np.random.choice(sentences) # 随机选取一个向量化的句子
        '''
        下面两段代码十分重要
        window_sequences取的是中心词的上下文的list(包括中心词)
        label_indeces取的是中心词在window_sequences列表中的位置
        '''
        window_sequences = [rand_sentence[max((ix - window_size), 0) : (ix + window_size + 1)]
                            for ix, x in enumerate(rand_sentence)]
        label_indices = [ix if ix < window_size else window_size
                         for ix, x in enumerate(window_sequences)]
        if method == 'skip_gram':
            # batch_and_labels得到一个训练集
            batch_and_labels = [(x[y], x[:y] + x[(y+1):])
                                for x, y in zip(window_sequences, label_indices)]
            # tuple_data = [(x, y_) for y_ in y for x, y in batch_and_labels ]
            '''
            上面的写法是一个错误的写法，for循环是一个向后传递的
            '''
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
        else:
            raise ValueError('Method {} not implmented yet'.format(method))

        batch, labels = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(labels[:batch_size])

    batch_data = batch_data[:batch_size]
    label_data = label_data[:batch_size]
    batch_data = np.array(batch_data)
    label_data = np.transpose(np.array([label_data]))

    return batch_data, label_data

# 初始化词嵌入矩阵，声明占位符与嵌套查找函数
# random_uniform是生成一个-1.0至1.0的数字，注意变量的形状
embeddings = tf.Variable(tf.random_uniform(shape=[vocabulary_size, embedding_size], minval=-1.0, maxval=1.0))
# init = tf.initialize_all_variables()
# sess.run(init)
# embeddings_temp = sess.run(embeddings)
x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# lookup the word embedding
embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# softmax是用来实现多分类问题的，但本例目标是10000个不同的分类，会导致稀疏性非常高
# 最后我们采用nce这种损失函数，将问题转变为一个二值问题，预测单词的分类以及随机噪音
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, # 以后千万别不写weights=这些东西，这不是第一次出错了
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))

# 创建函数寻找验证单词周围的单词。我们将计算验证单词集和所有词向量的余弦相似度，打印出每个验证单词最接近的单词
'''
keep_dims是用来保证结果的二位特性，即计算的结果的矩阵形式不变,相当于省略了expand_dims这个功能
'''
'''
下面是计算相似度的结果
总共有10000个单词每个单词200维
测试为5个单词每个单词200维
similarity为5*10000的张量
'''
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm # 对词嵌入矩阵进行了归一化处理
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True) # transpose_b厉害了

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
train_step = optimizer.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

loss_vec = []
loss_x_vec = []
for i in range(generations):
    # generate_batch_data实现一个生成一个batch_size的训练集，写的很好
    batch_input, batch_labels = generate_batch_data(text_data, batch_size, window_size)
    feed_dict = {x_inputs : batch_input, y_target : batch_labels}
    sess.run(train_step, feed_dict=feed_dict)
    # 打印结果
    if (i + 1) % print_loss_every == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i + 1)
        print("Loss at step {} : {}".format(i + 1, loss_val))

        # Validation: Print some random words and top 5 related words
    if (i + 1) % print_valid_every == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            top_k = 5  # number of nearest neighbors
            nearest = (-sim[j, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(top_k):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)