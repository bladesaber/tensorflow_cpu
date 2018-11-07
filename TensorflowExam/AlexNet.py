import tensorflow as tf
import numpy as np

'''
using ImageNet DataSet
image size: 224*224*3 Channel Last
AlexNet:
Conv Layer:          size:11*11  filter:64  strides:4
MaxPooling Layer:    size:3*3               strides:2
Conv Layer:          size:5*5    filter:192 strides:1
MaxPooling Layer:    size:3*3               strides:2
Conv Layer:          size:3*3    filter:384 strides:1
Conv Layer:          size:3*3    filter:256 strides:1
Conv Layer:          size:3*3    filter:256 strides:1
MaxPooling Layer:    size:3*3               strides:2
Dense Layer:         size:4096
Dense Layer:         size:4096
Dense Layer:         size:1000

1，第一次使用深度网络进行识别任务
2，第一次使用 Relu 解决深层网络传导问题
3，第一次使用 CNN
4，提出 LRN 层，虽然后期证实提升效果不明显
5，提出使用 数据增强 data augmentation

'''

def print_activation(t):
    print(t.op.name,' ',t.get_shape().as_list())

batch_size = 32
num_batch = 100

Image_holder = tf.placeholder(dtype=tf.float32,shape=[batch_size,224,224,3])
Label_holder = tf.placeholder(dtype=tf.float32,shape=[batch_size,1000])

parameters = []

# name_scope 产生命名空间，它会为区间内每一个变量创建一个 conv1/... 的名字
with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[11,11,3,64],stddev=0.1,mean=0.0,dtype=tf.float32),name='weights')
    biases = tf.Variable(initial_value=tf.constant(0.0,dtype=tf.float32,shape=[64]),name='biases')
    conv1 = tf.nn.relu(tf.nn.conv2d(input=Image_holder,filter=kernel,strides=[1,4,4,1],padding='SAME') + biases,name=scope)
    print_activation(conv1)
    parameters += [kernel,biases]

with tf.name_scope('maxPool1') as scope:
    maxPool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name=scope)
    parameters += [maxPool1]
    print_activation(maxPool1)

with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[5,5,64,192],mean=0.0,stddev=0.1,dtype=tf.float32),name='weights')
    biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[192]),name='biases')
    conv2 = tf.nn.relu(tf.nn.conv2d(input=maxPool1,filter=kernel,strides=[1,1,1,1],padding='SAME') + biases,name=scope)
    print_activation(conv2)
    parameters += [kernel,biases]

with tf.name_scope('maxPool2') as scope:
    maxPool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name=scope)
    print_activation(maxPool2)
    parameters += [maxPool2]

with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,192,384],mean=0.0,stddev=0.1,dtype=tf.float32),name='weights')
    biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[384]),name='biases')
    conv3 = tf.nn.relu(tf.nn.conv2d(input=maxPool2,filter=kernel,strides=[1,1,1,1],padding='SAME') + biases,name=scope)
    print_activation(conv3)
    parameters += [kernel,biases]

with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,384,256],mean=0.0,stddev=0.1,dtype=tf.float32),name='weights')
    biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name='biases')
    conv4 = tf.nn.relu(tf.nn.conv2d(input=conv3,filter=kernel,strides=[1,1,1,1],padding='SAME') + biases,name=scope)
    print_activation(conv4)
    parameters += [kernel,biases]

with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[3,3,256,256],mean=0.0,stddev=0.1,dtype=tf.float32),name='weights')
    biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),name='biases')
    conv5 = tf.nn.relu(tf.nn.conv2d(input=conv4,filter=kernel,strides=[1,1,1,1],padding='SAME') + biases,name=scope)
    print_activation(conv5)
    parameters += [kernel,biases]

with tf.name_scope('maxPool3') as scope:
    maxPool3 = tf.nn.max_pool(conv5,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name=scope)
    print_activation(maxPool3)
    parameters += [maxPool3]

with tf.name_scope('dense1') as scope:
    reshape = tf.reshape(tensor=maxPool3,shape=[batch_size,-1])
    dim = reshape.get_shape()[1].value
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=[dim,4096],mean=0.0,stddev=np.sqrt(1.0/4096),dtype=tf.float32),name='weights')
    biases = tf.Variable(initial_value=tf.constant(0.1,shape=[4096],name='biases'))
    dense1 = tf.nn.relu(tf.matmul(reshape,weight) + biases,name=scope)
    print_activation(dense1)
    parameters += [weight,biases]

with tf.name_scope('dense2') as scope:
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=[4096, 4096], mean=0.0, stddev=np.sqrt(1.0 / 4096), dtype=tf.float32),name='weights')
    biases = tf.Variable(initial_value=tf.constant(0.1, shape=[4096], name='biases'))
    dense2 = tf.nn.relu(tf.matmul(dense1, weight) + biases, name=scope)
    print_activation(dense2)
    parameters += [weight, biases]

with tf.name_scope('dense3') as scope:
    weight = tf.Variable(initial_value=tf.truncated_normal(shape=[4096, 1000], mean=0.0, stddev=np.sqrt(1.0 / 1000), dtype=tf.float32),name='weights')
    biases = tf.Variable(initial_value=tf.constant(0.1, shape=[1000], name='biases'))
    dense3 = tf.nn.softmax(tf.matmul(dense2, weight) + biases, name=scope)
    print_activation(dense3)
    parameters += [weight, biases]

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Label_holder * tf.log(dense3),reduction_indices=[1]),name='cross_entropy')
trans_step = tf.train.AdamOptimizer(0.02).minimize(cross_entropy)

def run(max_epoches,Image,Label):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(max_epoches):
            _,loss = sess.run(fetches=[trans_step,cross_entropy],
                     feed_dict={
                         Image_holder:Image,
                         Label_holder:Label
                     })

