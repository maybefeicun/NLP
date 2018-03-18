# -*- coding: utf-8 -*-
# @Time    : 2018/3/17 15:01
# @Author  : chen
# @File    : TF-IDF.py
# @Software: PyCharm

'''
本代码是用来进行tf-idf的实验，利用了垃圾邮件数据集进行训练与判断
与前文的词袋bag of word的方法相比基本一致，只是在BOW的基础上加上了词的权重，以增加关键词对于判断的影响
'''

import csv
import os
import string
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer # 进行tf-idf的计算
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 200
max_features = 1000 # 总算懂了，这个是设置单词的总数，只对1000个单词进行标记，文本单词远不止1000（我猜测是选取频率较大的单词）

# nltk.download('punkt')
sess = tf.Session()

# 加载数据集
dataset = []
save_filename = os.path.join('temp', 'temp_spam_data.csv')
if os.path.isfile(save_filename):
    with open(save_filename, 'r') as file_read:
        lines = csv.reader(file_read)
        for line in lines:
            if line != []:
                dataset.append(line)

texts = [data[1] for data in dataset]
labels = [data[0] for data in dataset]
target = [1. if x == 'spam' else 0. for x in labels]

# 预处理数据集
texts = [x.lower() for x in texts]
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]

# tokens = set()
# for text in texts:
#     for word in text.split(' '):
#         tokens.add(word)
# 通过该计算，我可以确定单词总数不止1000

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

# 核心代码，主要要注意max_features这个参数
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features) # 初始化tfidf
sparse_tfidf_texts = tfidf.fit_transform(texts) # 有关这段代码的解说可以看tf-idf2.py文件中的解释

# 选取训练集与测试集，这个一定要会写
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], round(0.8*sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))
texts_train = sparse_tfidf_texts[train_indices] # 其实就是将一个句子变成了一个二维数对
texts_test = sparse_tfidf_texts[test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# 声明变量与占位符
A = tf.Variable(tf.random_normal(shape=[max_features, 1], dtype=tf.float32))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# 声明损失函数
model_output = tf.add(tf.matmul(x_data, A), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target)) # 这里又犯错了

# 这些都是为了计算测试集所使用的
prediction = tf.round(tf.sigmoid(model_output)) # 这一点要注意，之所以使用round函数就是因为要进行准确度中的equal函数
acc = tf.cast(tf.equal(prediction, y_target), dtype=tf.float32)
accuracy = tf.reduce_mean(acc)

my_opt = tf.train.GradientDescentOptimizer(0.0025)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)

# 进行训练过程
train_loss = []
test_loss = []
train_acc = []
test_acc = []
i_data = []
for i in range(10000):
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size, replace=False) # 注意这里不要写成shape()
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data : rand_x, y_target : rand_y})

    # 记录输出数据
    if (i + 1) % 100 == 0:
        i_data.append(i + 1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy,
                                 feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)
    if (i + 1) % 500 == 0:
        acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
        print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(
            *acc_and_loss))

# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
