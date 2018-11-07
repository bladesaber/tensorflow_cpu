import tensorflow as tf
import numpy as np
import tensorflow.contrib.rnn as rnnTool

'''
这是双向 RNN 网络，相当于将训练序列数据（T1-->Tn）分为两种（1，T1-->Tn{不变} 2，Tn-->T1{反向}）
这两种数据分别训练两个 RNN 网络，然后将两个网络的状态输出拼接，再通过全连接层进行分类
目的是 使训练数据 包含前后文信息
'''

learning_rate = 0.1
max_example = 40000
batch_size = 128
display_step = 10

n_input = 28
n_step = 28
n_hidden = 256
n_classes = 10

x = tf.placeholder(dtype=tf.float32,shape=[None,n_step,n_input])
y = tf.placeholder(dtype=tf.float32,shape=[None,n_classes])

weights = tf.Variable(initial_value=tf.random_normal(shape=[2*n_hidden,n_classes]))
biases = tf.Variable(initial_value=tf.random_normal(shape=[n_classes]))

def BiRNN(x,weights,biases):

    '''
    # ------  这部分工作用于快速运算
    1,输入矩阵转换为  [ n_step , batch_size , input_line]
    2,转换为 [n_step * batch_size , n_input]
    3,转换为 [n_step , batch_size * input_line]
    '''
    # tf.transpose 用于矩阵的维度重排，例如 tf.transpose(x,[2,1,0]) 相当于将维度1和3互换位置
    x = tf.transpose(x,[1,0,2])
    x = tf.reshape(x,[-1,n_input])

    # tf.split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，如果是0就表示对第0维度进行切割。
    # num_split就是切割的数量，如果是2就表示输入张量被切成2份，每一份是一个列表。
    # 默认 0 轴切分
    x = tf.split(x,n_step)

    lstm_fw_cell = rnnTool.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)
    lstm_bw_cell = rnnTool.BasicLSTMCell(num_units=n_hidden,forget_bias=1.0)

    outputs,_,_ = rnnTool.static_bidirectional_rnn(
        cell_fw=lstm_fw_cell,
        cell_bw=lstm_bw_cell,
        inputs=x,
        dtype=tf.float32)

    # 这里 static_bidirectional_rnn 运算 n_step 次
    return tf.matmul(outputs[-1],weights) + biases

pred = BiRNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correctt_pred = tf.equal(tf.arg_max(pred,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correctt_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        sess.run(init)
        step = 1
        while step * batch_size < max_example:
            batch_x,batch_y = ''''''
            batch_x = batch_x.reshape((batch_size,n_step,n_input))
            sess.run(fetches=optimizer,
                     feed_dict={
                         x:batch_x,
                         y:batch_y})
            if step % display_step == 0:
                acc,loss = sess.run(fetches=[accuracy,cost],
                               feed_dict={
                                   x: batch_x,
                                   y: batch_y})
                print('Iter ',str(step*batch_size),' loss: ',loss,' accuracy is: ',acc)
