
import tensorflow as tf
import numpy as np
import zipfile
import collections
import random
import math

'''
word2vec Explain:（近似解释）
    相当于构造一个 Auto-Decoder-Encoder，类似与两层神经网结构：
    
    Layer 1（词汇表，输出层）：  词1   词2   词3  。。。。。。
    
    Layer 2（隐含层，词向量层）：     向量1   向量2   。。。。。。
    
    Layer 3（词汇表，输入层）：  词1   词2   词3  。。。。。。
    
    训练目标：
        1，CBOW 模型 则输入 n 个词向量拼接，输出层采用 sotfmax 
        2，skip-gram 则输入 1 个词向量，输出层采用 softmax 
'''

'''
# 默认模式r,读
azip = zipfile.ZipFile('bb.zip')  # ['bb/', 'bb/aa.txt']
# 返回所有文件夹和文件
print(azip.namelist())
# # 返回该zip的文件名
print(azip.filename)
'''

filePath = 'C:/Users/bladesaber/Desktop/tensorflow_cpu/DataSet/work2vec.zip'

def read_data(filename=filePath):
    with zipfile.ZipFile(filename) as f:
        # 这里获得单词的列表
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data()
print('Word List Size: ',len(words))

vocabulary_size = 50000

def build_dataset(words):

    # 这里 UNK 作为未知词
    count = [['UNK',-1]]

    # 上文的 words 是词列表，相当于文章的词列表
    # collections 是用于统计此列表的 词频数
    # 这里选取最频繁出现的前 vocabulary_size 个词
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    dictionary = dict()
    for word,_ in count:
        # 这里相当于 分配每个词一个编码
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count +=1
        # 这里将 以词本身呈现的 words 列表转换为 以 index 呈现的 data 列表
        data.append(index)
    count[0][1] = unk_count
    # 这是 反向词典 用于将 index 转为 词本身
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))

    # data 是 index列表，count 是 (name，frequency) 列表，dictionary 是 词本身 与 index 对应词典
    return data,count,dictionary,reverse_dictionary

data,count,dictionary,reverse_dictionary = build_dataset(words)

del words

data_index = 0
def generate_batch(batch_size,num_skips,skip_window):
    global data_index

    # assert 断言： 判断是否满足条件，否则弹出异常
    assert batch_size % num_skips == 0
    assert num_skips <= skip_window*2
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)
    span = 2 * skip_window + 1

    # collections.deque 相当于构造一个 queue(队列)，最大长度为span
    # 当超出长度后，后续数据替换前面数据
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index+1) % len(data)

    # // 符号运算是整除的意思,即是取商的部分
    for i in range(batch_size // num_skips):

        # 这里的 target 指目标词所在的位置
        target = skip_window
        # target_to_avoid 指当后面的抽样避免抽取的值
        target_to_avoid = [skip_window]
        # num_skips 相当于抽样次数
        for j in range(num_skips):
            while target in target_to_avoid:
                # 已抽取的内容不做考虑，所以进行 重新抽样
                target = random.randint(0,span-1)
            target_to_avoid.append(target)
            # 这里 batch 装载 目标单词，labels 装载附近的相关词
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j,0] = buffer[target]

        # 取样窗口移动
        buffer.append(data[data_index])
        data_index = (data_index+1) % len(data)
    return batch,labels

batch_size = 128
# 词向量大小
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window,valid_size,replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(dtype=tf.int32,shape=[batch_size])
    train_labels = tf.placeholder(dtype=tf.int32,shape=[batch_size,1])
    valid_dataset = tf.constant(value=valid_examples,dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(
            initial_value=tf.random_uniform(shape=[vocabulary_size,embedding_size],
                                            minval=-1.0,
                                            maxval=1.0,dtype=tf.float32))
        # tf.nn.embedding_lookup 指 从 向量表 中选取 ids 对应的向量
        embed = tf.nn.embedding_lookup(params=embeddings,ids=train_inputs)

        nce_weight = tf.Variable(
            initial_value= tf.truncated_normal(shape=[vocabulary_size,embedding_size],
                                               stddev=1.0/math.sqrt(embedding_size)))
        nce_biases = tf.Variable(initial_value=tf.zeros(shape=[vocabulary_size]))

    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weight,
                       biases=nce_biases,
                       labels=train_labels,
                       inputs=embed,
                       num_sampled=num_sampled,
                       num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # 以横向轴 的平方求和 再开方
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),axis=1,keep_dims=True))
    # 使分布标准化
    normalized_embeddings = embeddings / norm

    valid_embeddings = tf.nn.embedding_lookup(params=normalized_embeddings,ids=valid_dataset)

    # 将 b 矩阵 倒置 再相乘
    similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)

    init = tf.global_variables_initializer

num_steps = 100001
with tf.Session(graph=graph) as sess:
    sess.run(init)

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size=batch_size, num_skips=num_skips, skip_window=skip_window)
        feed_dict ={
            train_inputs:batch_inputs,
            train_labels:batch_labels
        }

        _,loss_val = sess.run(fetches=[optimizer,loss],
                              feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            print('Average loss at step: ',step,' : ',average_loss)
            average_loss = 0

        if step % 10000 == 0:
            sim = similarity.eval()
            top_k = 8
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                nearest = (-sim[i,:]).argsort()[1:top_k+1]
                log_str = 'Nearest to %s: ' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str,close_word)
            print(log_str)
final_embeddings = normalized_embeddings.eval()
















