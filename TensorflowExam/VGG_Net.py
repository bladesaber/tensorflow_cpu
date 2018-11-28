import tensorflow as tf
import numpy as np

'''
VGG-16 Net
Conv Layer:     size:3*3  filter:64  strides:1
Conv Layer:     size:3*3  filter:64  strides:1
MaxPool Layer:  size:2*2             strides:2

Conv Layer:     size:3*3  filter:128 strides:1
Conv Layer:     size:3*3  filter:128 strides:1
MaxPool Layer:  size:2*2             strides:2

Conv Layer:     size:3*3  filter:256  strides:1
Conv Layer:     size:3*3  filter:256  strides:1
Conv Layer:     size:3*3  filter:256  strides:1
MaxPool Layer:  size:2*2             strides:2

Conv Layer:     size:3*3  filter:512  strides:1
Conv Layer:     size:3*3  filter:512  strides:1
Conv Layer:     size:3*3  filter:512  strides:1
MaxPool Layer:  size:2*2             strides:2

Conv Layer:     size:3*3  filter:512  strides:1
Conv Layer:     size:3*3  filter:512  strides:1
Conv Layer:     size:3*3  filter:512  strides:1
MaxPool Layer:  size:2*2             strides:2

Dense Layer:    size:4096
Dense Layer:    size:4096
Dense Layer:    size:1000

1，提出使用多个小的卷积核拥有比大卷积核更好的效果（重点是变形多），同时更少的参数量
   多个小的卷积核吸纳的视野更多而且参数更少，方便快速拟合
2，1*1 卷积核虽然有效，但不如 3*3 卷积核 ，应为没附带相对位置信息
3，提出网络越深，效果越好
'''

#--------------------------------------------     Model
def print_activation(t):
    print(t.op.name,' ',t.get_shape().as_list())

def conv_Layer(name,filter_size,kernel_size,strides,input,stddev,mean):
    with tf.name_scope(name=name) as scope:
        dim = input.get_shape()[-1].value
        shape = [kernel_size,kernel_size,dim,filter_size]
        kernel = tf.Variable(initial_value=tf.truncated_normal(shape=shape,mean=mean,stddev=stddev),dtype=tf.float32,name='weights')
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[filter_size]),name='biases')
        output = tf.nn.relu(tf.nn.conv2d(input=input,
                                       filter=kernel,
                                       strides=[1,strides,strides,1],
                                       padding='SAME') + biases,name=scope)
        print_activation(output)
        return output

def dense_Layer(name,filter_size,input,stddev,mean):
    with tf.name_scope(name=name) as scope:
        dim = input.get_shape()[-1].value
        shape = [dim,filter_size]
        kernel = tf.Variable(initial_value=tf.truncated_normal(shape=shape,mean=mean,stddev=stddev),dtype=tf.float32,name='weights')
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[filter_size]),name='biases')
        output = tf.nn.relu(tf.matmul(input,kernel) + biases,name=scope)
        print_activation(output)
        return output

#------------------------------------------------------------
batch_size = 32

Image_holder = tf.placeholder(dtype=tf.float32,shape=[batch_size,224,224,3])
Label_holder = tf.placeholder(dtype=tf.float32,shape=[batch_size,1000])

conv1 = conv_Layer(name='conv1',filter_size=64,kernel_size=3,strides=1,
           input=Image_holder,stddev=0.1,mean=0.0)
conv2 = conv_Layer(name='conv2',filter_size=64,kernel_size=3,strides=1,
           input=conv1,stddev=0.1,mean=0.0)
maxPool1 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print_activation(maxPool1)

conv3 = conv_Layer(name='conv3',filter_size=128,kernel_size=3,strides=1,
           input=maxPool1,stddev=0.1,mean=0.0)
conv4 = conv_Layer(name='conv4',filter_size=128,kernel_size=3,strides=1,
           input=conv3,stddev=0.1,mean=0.0)
maxPool2 = tf.nn.max_pool(conv4,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print_activation(maxPool2)

conv5 = conv_Layer(name='conv5',filter_size=256,kernel_size=3,strides=1,
           input=maxPool2,stddev=0.1,mean=0.0)
conv6 = conv_Layer(name='conv6',filter_size=256,kernel_size=3,strides=1,
           input=conv5,stddev=0.1,mean=0.0)
conv7 = conv_Layer(name='conv7',filter_size=256,kernel_size=3,strides=1,
           input=conv6,stddev=0.1,mean=0.0)
maxPool3 = tf.nn.max_pool(conv7,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print_activation(maxPool3)

conv8 = conv_Layer(name='conv8',filter_size=512,kernel_size=3,strides=1,
           input=maxPool3,stddev=0.1,mean=0.0)
conv9 = conv_Layer(name='conv9',filter_size=512,kernel_size=3,strides=1,
           input=conv8,stddev=0.1,mean=0.0)
conv10 = conv_Layer(name='conv10',filter_size=512,kernel_size=3,strides=1,
           input=conv9,stddev=0.1,mean=0.0)
maxPool4 = tf.nn.max_pool(conv10,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print_activation(maxPool4)

conv11 = conv_Layer(name='conv8',filter_size=512,kernel_size=3,strides=1,
           input=maxPool4,stddev=0.1,mean=0.0)
conv12 = conv_Layer(name='conv9',filter_size=512,kernel_size=3,strides=1,
           input=conv11,stddev=0.1,mean=0.0)
conv13 = conv_Layer(name='conv10',filter_size=512,kernel_size=3,strides=1,
           input=conv12,stddev=0.1,mean=0.0)
maxPool5 = tf.nn.max_pool(conv13,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
print_activation(maxPool5)

fc1_flatern = tf.reshape(tensor=maxPool5,shape=[batch_size,-1])
dense1 = dense_Layer(name='dense1',filter_size=4096,input=fc1_flatern,stddev=0.1,mean=0.0)

dropout1 = tf.nn.dropout(x=dense1,keep_prob=0.5,name='dropout1')
print(dropout1)

dense2 = dense_Layer(name='dense2',filter_size=4096,input=dropout1,stddev=0.1,mean=0.0)

dropout2 = tf.nn.dropout(x=dense2,keep_prob=0.5,name='dropout2')
print(dropout2)

with tf.name_scope('logits') as scope:
    kernel = tf.Variable(initial_value=tf.truncated_normal(shape=[4096,1000], mean=0.0, stddev=0.1), dtype=tf.float32,
                         name='weights')
    biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1000]), name='biases')

    logits = tf.nn.softmax(tf.matmul(dropout2, kernel) + biases, name=scope)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(Label_holder * tf.log(logits),reduction_indices=[1]),name='cross_entropy')
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)

#----------------------------------------     Train & Loss

def run(max_epoches,input,label):
    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in range(max_epoches):
            _,loss = sess.run(fetches=[train_step,cross_entropy],
                     feed_dict={
                         Image_holder:input,
                         Label_holder:label
                     })
            if epoch % 50:
                print('loss: ',loss)