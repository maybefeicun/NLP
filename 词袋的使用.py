# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 14:15
# @Author  : chen
# @File    : 词袋的使用.py
# @Software: PyCharm

"""
本代码使用“词袋”的方法来分析UCI中的垃圾短信文本数据集
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import string
import requests
import io
from zipfile import ZipFile # 解压库
from tensorflow.contrib import learn # 获取数据集

sess = tf.Session()

# 1. 进行数据处理
dir_name = 'temp'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

save_file_name = os.path.join('temp', 'temp_spam_data.csv')
if os.path.isfile(save_file_name):
    # 如果save_file_name存在我们就直接读取文件内容，赋值给text_data
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            if row != []:
                text_data.append(row)
else: # 反之从url上提取shujuji
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')
    # 格式化数据
    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x) >= 1]

    # 将数据写入csv文件中
    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data) # 这个厉害了直接一下写完所有的数据

# 抽取文本与真是结果
texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]
# 重新定义结果 'spam' as 1， 'ham' as 0
target = [1 if x=='spam' else 0 for x in target]

# 同时对原始数据进行预处理消除大小写与数字的影响
"""
str = "-";
seq = ("a", "b", "c"); # 字符串序列
print str.join( seq );
"""
texts = [x.lower() for x in texts] # 变化大小写
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts] # 删除标点符号
texts = [''.join(c for c in x if c not in '0123456789') for x in texts] # 删除数字
texts = [' '.join(x.split()) for x in texts] # 删除长空格的影响

# Plot histogram of text lengths 画出最长句子的直方图
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x < 50]
plt.hist(text_lengths, bins=25)
plt.title('Histogram of # of Words in Texts')

# 从直方图中我们看可以看出一些趋势，故做出下面的参数定义
sentence_size = 25 # 句子的大小
min_word_fred = 3 # 最小的单词频率

# 2. 处理预处理后的文本，生成词表，并生成相应的词标号,主要生成句子的向量
# 在这里利用了tensorflow中自带的分词工具
"""
sentence_size 表示 处理的句子的最大长度为25，如果超过这个长度则进行截取，而没有达到的句子则补为0
min_frequency 表示 单词的频率如果小于3，则该单词就不会记录在词表中
这个生成的并不是向量类型的，而是统计整个词表的长度，然后每次单词分配一个数字,不在此表的词则设置为零
"""
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_fred)
# Have to fit transform to get length of unique words.
'''
transform将文本转换为id向量
'''
# temp_text = vocab_processor.transform(texts) # 这个生成了一个可迭代的元素
# for x in vocab_processor.transform(texts):
#     continue
'''
注意这段带只是生成一个迭代器，但是经过这段代码后我们可以得到单词的标记号
'''
temp_result = vocab_processor.fit_transform(texts) #
sentence_line = [x for x in vocab_processor.transform(texts)] # 这段代码是用来生成句子的直方图向量
embedding_size = len([x for x in vocab_processor.transform(texts)])
vocabulary = vocab_processor.vocabulary_._mapping # 通过这段代码我们可以查看到生成的单词表

# 3. 生成训练集与测试集
# round() 方法返回浮点数x的四舍五入值，怪不得每次要使用round函数
train_indices = np.random.choice(len(texts), round(len(texts) * 0.8), replace=False)
# texts_train1 = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_train = np.array(texts)[train_indices]
target_train = np.array(target)[train_indices]
test_indices = list(set(range(len(texts))) - set(train_indices))
texts_test = np.array(texts)[test_indices]
target_test = np.array(target)[test_indices]

# 声明词嵌入矩阵，将句子单词转成索引，再将索引转成one-hot向量，该向量为单位矩阵
# 我们使用该矩阵为每个单词查找稀疏向量
"""
  词嵌入矩阵的作用就是用来将一维词投影到一个二维的平面上面
  # 'diagonal' is [1, 2, 3, 4]
  tf.diag(diagonal) ==> [[1, 0, 0, 0]
                         [0, 2, 0, 0]
                         [0, 0, 3, 0]
                         [0, 0, 0, 4]]
"""
# 4. 生成一个单位矩阵
"""
代码的核心去理解这个词嵌入矩阵，也就是将一个句子中的25个单词投到这embedding_size维的向量空间中
"""
identity_mat = tf.diag(tf.ones(shape=[embedding_size]))

# 首先我们要声明逻辑回归变量然后声明占位符，所有的公式基本离不开Ax + b = y这个方式
# 这样就可以理解为线性规划的原则
A = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

x_data = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y_target = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# 使用tensorflow的嵌入查找函数来映射句子中的单词为单位矩阵的one-hot向量，然后把前面的词向量求和
"""
embedding_lookup(embeding, x_data)
embeding为一个单位向量，x_data为输入值
简单的讲就是根据x_data中的id，寻找embedding中的对应元素。
比如，input_ids=[1,3,5]，则找出embedding中下标为1,3,5的向量组成一个矩阵返回。
"""
'''
有关embedding_lookup的方法在url:http://blog.csdn.net/u013041398/article/details/60955847
'''
x_embed = tf.nn.embedding_lookup(identity_mat, x_data)
x_col_sums = tf.reduce_sum(x_embed, 0) # 等于句子的向量

# 根据x_col_sums操作的结果，定义输出公式
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
# 注意matmul与multiply函数的区别
model_output = tf.add(tf.matmul(x_col_sums_2D, A), b)


# 声明训练模型的损失函数，预测函数，优化器
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
prediction = tf.sigmoid(model_output)
my_opt = tf.train.GradientDescentOptimizer(0.001)
train_step = my_opt.minimize(loss)

init = tf.initialize_all_variables()
sess.run(init)

# 开始训练过程
loss_vec = []
train_acc_all = []
train_acc_avg = []
temp_fit = vocab_processor.fit_transform(texts_train)
for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]
    sess.run(train_step, feed_dict={x_data : t, y_target : y_data})
    temp_loss = sess.run(loss, feed_dict={x_data : t, y_target : y_data})
    loss_vec.append(temp_loss)

    embed = sess.run(x_embed, feed_dict={x_data : t, y_target : y_data})

    if (ix + 1) % 10 == 0:
        # diag = sess.run(identity_mat, feed_dict={x_data : t, y_target : y_data})
        print('Training Observation #' + str(ix + 1) + ': Loss = ' + str(temp_loss))

    [[temp_pred]] = sess.run(prediction, feed_dict={x_data: t, y_target: y_data})
    # Get True/False if prediction is accurate
    train_acc_temp = target_train[ix] == np.round(temp_pred) # 这个写法挺好的
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
        train_acc_avg.append(np.mean(train_acc_all[-50:]))

# 测试代码
test_acc_all = []
for xi, t in enumerate(vocab_processor.fit_transform(texts_test)):
    y_data = [[target_test[xi]]]
    [[temp_pred]] = sess.run(prediction, feed_dict={x_data : t, y_target : y_data})
    test_acc_temp = np.round(temp_pred)==target_test[xi]
    test_acc_all.append(test_acc_temp)

print('Test Accuracy : {}'.format(np.mean(test_acc_all)))

# word_vec = vocab_processor.fit_transform('like')
# result = tf.nn.embedding_lookup(identity_mat, word_vec)
# result1 = sess.run(result)
# pass