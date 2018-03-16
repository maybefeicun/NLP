# -*- coding: utf-8 -*-
# @Time    : 2018/3/16 10:44
# @Author  : chen
# @File    : MinHash.py
# @Software: PyCharm

from datasketch import MinHash

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

str1 = '延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 思维 方式 电信 企业 甲方 思维 优越感 强 过得 设备 制造商 合作伙伴 互联网 公司 危机意识 注重 创新'
str1 = str1.split(' ')
str2 = '延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 之三 产品 研发 电信 企业 功能 核心 追求 大而全 研发 周期长 上线 长 时间 改变 互联网 公司 客户 体验 中心 迭代 式 开发 小步 跑 不断改进'
str2 = str2.split(' ')
str3 = '延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 之五 客户服务 电信 企业 人工 号 装机 放号 话务员 客户 解决 互联网 公司 自助 互助 模式 快速 响应 需求 苹果 三星 粉丝团 网上 求助 客服 成本 变得 非常低'
str3 = str3.split(' ')
str4 = '延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 思维'
str4 = str4.split(' ')

m1, m2, m3 = MinHash(num_perm=256), MinHash(num_perm=256), MinHash(num_perm=256)
m4 = MinHash(num_perm=256)
for d in str1:
    m1.update(d.encode('utf-8'))
for d in str2:
    m2.update(d.encode('utf-8'))
for d in str3:
    m3.update(d.encode('utf-8'))
for d in str4:
    m4.update(d.encode('utf-8'))

print('estimated jaccard for data1 and data2 is', m1.jaccard(m2), '-', m1.jaccard(m3), '-', m1.jaccard(m4))


from datasketch import MinHash, MinHashLSH

'''
延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 思维 方式 电信 企业 甲方 思维 优越感 强 过得 设备 制造商 合作伙伴 互联网 公司 危机意识 注重 创新
延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 之三 产品 研发 电信 企业 功能 核心 追求 大而全 研发 周期长 上线 长 时间 改变 互联网 公司 客户 体验 中心 迭代 式 开发 小步 跑 不断改进
延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 之五 客户服务 电信 企业 人工 号 装机 放号 话务员 客户 解决 互联网 公司 自助 互助 模式 快速 响应 需求 苹果 三星 粉丝团 网上 求助 客服 成本 变得 非常低
'''
str1 = '延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 思维 方式 电信 企业 甲方 思维 优越感 强 过得 设备 制造商 合作伙伴 互联网 公司 危机意识 注重 创新'
str1 = str1.split(' ')
str2 = '延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 之三 产品 研发 电信 企业 功能 核心 追求 大而全 研发 周期长 上线 长 时间 改变 互联网 公司 客户 体验 中心 迭代 式 开发 小步 跑 不断改进'
str2 = str2.split(' ')
str3 = '延边 分公司 企业 文化论坛 电信 企业 互联网 公司 差异 之五 客户服务 电信 企业 人工 号 装机 放号 话务员 客户 解决 互联网 公司 自助 互助 模式 快速 响应 需求 苹果 三星 粉丝团 网上 求助 客服 成本 变得 非常低'
str3 = str3.split(' ')

set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'datasets'])
set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])
set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for',
            'estimating', 'the', 'similarity', 'between', 'documents'])

m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in str1:
    m1.update(d.encode('utf8'))
for d in str2:
    m2.update(d.encode('utf8'))
for d in str3:
    m3.update(d.encode('utf8'))

# Create LSH index
lsh = MinHashLSH(threshold=0.0, num_perm=128)
lsh.insert("m2", m2)
lsh.insert("m3", m3)
result = lsh.query(m1)
print("Approximate neighbours with Jaccard similarity > 0.5", result)

s1 = set(data1)
s2 = set(data2)
actual_jaccard = float(len(s1.intersection(s2))) / float(len(s1.union(s2)))
print('actual jaccard is :', actual_jaccard)